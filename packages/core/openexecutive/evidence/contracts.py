"""The two construction paths, as types.

**Path A — untrusted proposals.** :class:`DocumentIngestProposal` and
:class:`SourceSpanProposal` are the *only* models built from wire or model JSON.
Every field is a bounded descriptive claim; they carry no id, hash, timestamp,
scope, logical-source key, lineage, status, quality, independence,
applicability, verification or authority field, and ``extra="forbid"`` turns an
attempt to supply one into a ``ValidationError`` rather than a silent drop. A
declared publisher, author, date, geography, period or methodology is a
*recorded claim*, not a finding: nothing parses or corroborates it — a declared
date stays a string so no reader mistakes a parsed ``datetime`` for a verified
one — and a missing value stays ``None``.

**Path B — trusted canonical records.** :class:`DocumentVersion`,
:class:`Extraction` and :class:`SourceSpan` derive every authoritative field and
may only be built inside ``trusted_construction``, which in production means
only :mod:`openexecutive.evidence.factory`. See ``_authority`` for that guard's
honest scope.

**Representation and offsets — the invariant this package exists to keep.**
``DocumentVersion.content_sha256`` hashes the immutable **raw document bytes**.
Raw extracted text and NFC text are two *distinct representations* with separate
hashes; ``nfc_text`` must equal ``unicodedata.normalize("NFC", raw_text)``.
:class:`SourceSpan` offsets address the **verified NFC text**. The offset unit is
a Python ``str`` index — a Unicode **code point**, never a UTF-8 byte position.
``start_char`` is inclusive, ``end_char`` is exclusive, ``quote`` is exactly
``trusted_nfc_text[start_char:end_char]``, and ``quote_sha256`` hashes that
exact stored substring with **no second normalization**.
"""
from __future__ import annotations

import unicodedata

from pydantic import Field, field_validator, model_validator

from openexecutive.evidence._authority import require_trusted_construction
from openexecutive.evidence._model import EvidenceModel

MAX_FILENAME_BYTES = 255
MAX_MEDIA_TYPE = 128
MAX_DECLARED = 512
MAX_QUOTE_CODE_POINTS = 4_096
UNSAFE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp", "Cs"})
_HEX64 = r"^[0-9a-f]{64}$"


def reject_lone_surrogates(text: str, label: str) -> str:
    """Reject text with no UTF-8 encoding. Applied everywhere, quotes included."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} contains a lone surrogate") from exc
    return text


def reject_unsafe_characters(text: str, label: str) -> str:
    """Reject NUL, control, format, separator and surrogate code points.

    A *safety* rule for identifier- and filename-shaped strings, deliberately
    **not** applied to :attr:`SourceSpanProposal.quote`: an evidence quote must
    survive byte-exact, and Arabic bidi marks, joiners and line separators are
    ``Cf``/``Zl`` characters that carry meaning. Stripping them corrupts it.
    """
    reject_lone_surrogates(text, label)
    if "\x00" in text:
        raise ValueError(f"{label} contains NUL")
    for char in text:
        if unicodedata.category(char) in UNSAFE_CATEGORIES:
            raise ValueError(f"{label} contains a disallowed control or format character")
    return text


class DocumentIngestProposal(EvidenceModel):
    """Untrusted description of a document offered for ingest.

    ``filename`` and every ``declared_*`` field are untrusted descriptive
    *observations* about an upload. They are not properties of the immutable
    content version and are never embedded in :class:`DocumentVersion`; a later
    append-only ingestion observation may record them against a version id.
    Phase 4A1 implements neither that model nor its persistence. ``filename``
    is never part of identity and never used here as a path.
    """

    filename: str
    media_type: str
    declared_publisher: str | None = None
    declared_author: str | None = None
    declared_published_at: str | None = None
    declared_geography: str | None = None
    declared_period: str | None = None
    declared_methodology: str | None = None

    @field_validator("filename")
    @classmethod
    def _check_filename(cls, value: str) -> str:
        reject_unsafe_characters(value, "filename")
        if not value:
            raise ValueError("filename must not be empty")
        if len(value.encode("utf-8")) > MAX_FILENAME_BYTES:
            raise ValueError(f"filename exceeds {MAX_FILENAME_BYTES} UTF-8 bytes")
        return value

    @field_validator("media_type")
    @classmethod
    def _check_media_type(cls, value: str) -> str:
        reject_unsafe_characters(value, "media_type")
        if not 1 <= len(value) <= MAX_MEDIA_TYPE:
            raise ValueError(f"media_type must be 1..{MAX_MEDIA_TYPE} characters")
        return value

    @field_validator(
        "declared_publisher",
        "declared_author",
        "declared_published_at",
        "declared_geography",
        "declared_period",
        "declared_methodology",
    )
    @classmethod
    def _check_declared(cls, value: str | None) -> str | None:
        if value is None:
            return None
        reject_lone_surrogates(value, "declared metadata")
        if "\x00" in value:
            raise ValueError("declared metadata contains NUL")
        if not 1 <= len(value) <= MAX_DECLARED:
            raise ValueError(f"declared metadata must be 1..{MAX_DECLARED} code points")
        return value


class SourceSpanProposal(EvidenceModel):
    """Untrusted offer of a span. Until ``factory.mint_source_span`` checks these
    offsets against the extraction's own text, they are *claims*, not addresses."""

    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    quote: str

    @field_validator("quote")
    @classmethod
    def _check_quote(cls, value: str) -> str:
        reject_lone_surrogates(value, "quote")
        if not 1 <= len(value) <= MAX_QUOTE_CODE_POINTS:
            raise ValueError(f"quote must be 1..{MAX_QUOTE_CODE_POINTS} code points")
        return value


class _CanonicalModel(EvidenceModel):
    """Base for records mintable only inside the trusted context."""

    @model_validator(mode="before")
    @classmethod
    def _require_trusted(cls, data: object) -> object:
        require_trusted_construction(cls.__name__)
        return data


class DocumentVersion(_CanonicalModel):
    """One immutable byte-content of one logical source, in one scope.

    Identity is ``(scope_id, logical_source_id, content_sha256)``, and every
    field here is determined by it: the same id always means the same complete
    record. Filename, declared metadata and upload time are deliberately absent
    — they describe an upload, not the content, and belong to a later
    append-only ingestion observation. Whether a repeated upload is a new event
    or a no-op is later repository policy; there is no ``ingest_event_id``.
    """

    document_version_id: str = Field(pattern=_HEX64)
    scope_id: str
    logical_source_id: str = Field(pattern=_HEX64)
    content_sha256: str = Field(pattern=_HEX64)
    byte_size: int = Field(ge=0)


class Extraction(_CanonicalModel):
    """One extractor's run over one document version, identified by its output.

    Every field is determined by ``extraction_id``; there is no run timestamp,
    because two runs producing the same output are the same extraction. Both
    output hashes participate in ``extraction_id``: ``raw_text_sha256``
    already distinguishes different valid raw outputs, since valid NFC text is
    deterministically derived from raw text, and ``nfc_text_sha256`` binds
    explicitly the canonical representation every span is checked against.
    """

    extraction_id: str = Field(pattern=_HEX64)
    document_version_id: str = Field(pattern=_HEX64)
    extractor_name: str
    extractor_version: str
    extractor_config_sha256: str = Field(pattern=_HEX64)
    raw_text_sha256: str = Field(pattern=_HEX64)
    nfc_text_sha256: str = Field(pattern=_HEX64)
    nfc_text_length: int = Field(ge=0)


class SourceSpan(_CanonicalModel):
    """A verified code-point range of one extraction's NFC text, with its quote.
    ``nfc_text_sha256`` repeats so a span carries the identity of the exact text
    its offsets address."""

    span_id: str = Field(pattern=_HEX64)
    extraction_id: str = Field(pattern=_HEX64)
    nfc_text_sha256: str = Field(pattern=_HEX64)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    quote: str
    quote_sha256: str = Field(pattern=_HEX64)
