"""The only production path that mints canonical evidence records.

Every function here is pure: no clock, no filesystem, no network, no ambient
configuration. Canonical records carry no timestamps at all — every field of a
record is determined by its id, so two mints from equal identity inputs yield
equal records.

This is the sole production site entering ``trusted_construction``; an
architectural test pins that, and the guard's honest scope is in ``_authority``.
A later persistence phase reconstructs trusted rows through this same path —
untrusted parsers never reach it, because untrusted input only ever becomes a
proposal type.

The ordering inside :func:`mint_source_span` is the load-bearing part: **every
verification runs before any identity is derived**, because a span id asserts
that the offsets were checked against real text.
"""
from __future__ import annotations

import unicodedata

from openexecutive.evidence import identity
from openexecutive.evidence._authority import trusted_construction
from openexecutive.evidence.contracts import (
    DocumentVersion,
    Extraction,
    SourceSpan,
    SourceSpanProposal,
    reject_unsafe_characters,
)
from openexecutive.evidence.extractor_config import ConfigValue, extractor_config_sha256

MAX_RAW_BYTES = 64 * 1024 * 1024
MAX_TEXT_CODE_POINTS = 2_000_000
MAX_SCOPE_ID = 128
MAX_LOGICAL_SOURCE_KEY = 200
MAX_EXTRACTOR_FIELD = 128


class EvidenceFactoryError(ValueError):
    """A trusted input to the factory was malformed or out of bounds."""


class NormalizationError(EvidenceFactoryError):
    """``nfc_text`` was not the NFC normalization of ``raw_text``."""


class SpanVerificationError(EvidenceFactoryError):
    """A proposed span failed verification against the extraction's own text.

    ``check`` is a stable literal naming the failed check. The message never
    echoes the quote or source text: failures get logged, and a rejected span
    is still untrusted input.
    """

    def __init__(self, check: str) -> None:
        self.check = check
        super().__init__(f"source span verification failed: {check}")


def _bounded_identifier(value: str, label: str, limit: int) -> str:
    if not isinstance(value, str):
        raise EvidenceFactoryError(f"{label} must be str")
    if not 1 <= len(value) <= limit:
        raise EvidenceFactoryError(f"{label} must be 1..{limit} code points")
    try:
        reject_unsafe_characters(value, label)
    except ValueError as exc:
        raise EvidenceFactoryError(str(exc)) from exc
    return value


def _bounded_text(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise EvidenceFactoryError(f"{label} must be str")
    if len(value) > MAX_TEXT_CODE_POINTS:
        raise EvidenceFactoryError(f"{label} exceeds {MAX_TEXT_CODE_POINTS} code points")
    return value


def mint_document_version(
    *,
    raw_bytes: bytes,
    scope_id: str,
    logical_source_key: str,
) -> DocumentVersion:
    """Mint the canonical version record for one document's raw bytes.

    ``logical_source_key`` is **trusted** input from the application or human
    workflow boundary — a registry entry supplied by the caller, never inferred
    from filename, model output, URL similarity or content hash, and never
    present on the proposal. It groups versions of one logical document; it does
    not prove lineage or independence.
    """
    if not isinstance(raw_bytes, bytes):
        raise EvidenceFactoryError("raw_bytes must be bytes")
    if len(raw_bytes) > MAX_RAW_BYTES:
        raise EvidenceFactoryError(f"raw_bytes exceeds {MAX_RAW_BYTES} bytes")
    scope_id = _bounded_identifier(scope_id, "scope_id", MAX_SCOPE_ID)
    logical_source_key = _bounded_identifier(
        logical_source_key, "logical_source_key", MAX_LOGICAL_SOURCE_KEY
    )

    logical_source_id = identity.mint_id(
        identity.TAG_LOGICAL_SOURCE, scope_id, logical_source_key
    )
    content_hash = identity.content_sha256(raw_bytes)
    version_id = identity.mint_id(
        identity.TAG_DOCUMENT_VERSION, scope_id, logical_source_id, content_hash
    )
    with trusted_construction():
        return DocumentVersion(
            document_version_id=version_id,
            scope_id=scope_id,
            logical_source_id=logical_source_id,
            content_sha256=content_hash,
            byte_size=len(raw_bytes),
        )


def mint_extraction(
    version: DocumentVersion,
    *,
    extractor_name: str,
    extractor_version: str,
    extractor_config: ConfigValue,
    raw_text: str,
    nfc_text: str,
) -> Extraction:
    """Mint the canonical extraction record, identified by its own output."""
    if not isinstance(version, DocumentVersion):
        raise EvidenceFactoryError("version must be a DocumentVersion")
    extractor_name = _bounded_identifier(extractor_name, "extractor_name", MAX_EXTRACTOR_FIELD)
    extractor_version = _bounded_identifier(
        extractor_version, "extractor_version", MAX_EXTRACTOR_FIELD
    )
    raw_text = _bounded_text(raw_text, "raw_text")
    nfc_text = _bounded_text(nfc_text, "nfc_text")
    if nfc_text != unicodedata.normalize("NFC", raw_text):
        raise NormalizationError("nfc_text is not the NFC normalization of raw_text")

    config_hash = extractor_config_sha256(extractor_config)
    raw_hash = identity.text_sha256(raw_text)
    nfc_hash = identity.text_sha256(nfc_text)
    extraction_id = identity.mint_id(
        identity.TAG_EXTRACTION,
        version.document_version_id,
        extractor_name,
        extractor_version,
        config_hash,
        raw_hash,
        nfc_hash,
    )
    with trusted_construction():
        return Extraction(
            extraction_id=extraction_id,
            document_version_id=version.document_version_id,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
            extractor_config_sha256=config_hash,
            raw_text_sha256=raw_hash,
            nfc_text_sha256=nfc_hash,
            nfc_text_length=len(nfc_text),
        )


def mint_source_span(
    proposal: SourceSpanProposal,
    *,
    extraction: Extraction,
    trusted_nfc_text: str,
) -> SourceSpan:
    """Verify a proposed span against the extraction's real text, then mint it.

    All four checks run before ``quote_sha256`` or ``span_id`` is derived:
    ``nfc_text_sha256`` (the supplied text is the extraction's text),
    ``nfc_text_length`` (and the length it recorded), ``offset_range`` (offsets
    inside it and non-empty), ``quote_mismatch`` (the quote is exactly the slice
    they address).
    """
    if not isinstance(proposal, SourceSpanProposal):
        raise EvidenceFactoryError("proposal must be a SourceSpanProposal")
    if not isinstance(extraction, Extraction):
        raise EvidenceFactoryError("extraction must be an Extraction")
    trusted_nfc_text = _bounded_text(trusted_nfc_text, "trusted_nfc_text")

    if identity.text_sha256(trusted_nfc_text) != extraction.nfc_text_sha256:
        raise SpanVerificationError("nfc_text_sha256")
    if len(trusted_nfc_text) != extraction.nfc_text_length:
        raise SpanVerificationError("nfc_text_length")
    if not 0 <= proposal.start_char < proposal.end_char <= len(trusted_nfc_text):
        raise SpanVerificationError("offset_range")
    if trusted_nfc_text[proposal.start_char : proposal.end_char] != proposal.quote:
        raise SpanVerificationError("quote_mismatch")

    quote_hash = identity.text_sha256(proposal.quote)
    span_id = identity.mint_id(
        identity.TAG_SOURCE_SPAN,
        extraction.extraction_id,
        extraction.nfc_text_sha256,
        proposal.start_char,
        proposal.end_char,
        quote_hash,
    )
    with trusted_construction():
        return SourceSpan(
            span_id=span_id,
            extraction_id=extraction.extraction_id,
            nfc_text_sha256=extraction.nfc_text_sha256,
            start_char=proposal.start_char,
            end_char=proposal.end_char,
            quote=proposal.quote,
            quote_sha256=quote_hash,
        )
