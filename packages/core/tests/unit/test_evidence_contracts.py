"""Proposal types: bounded, descriptive, and unable to claim authority."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from openexecutive.evidence._authority import trusted_construction
from openexecutive.evidence.contracts import (
    MAX_QUOTE_CODE_POINTS,
    DocumentIngestProposal,
    DocumentVersion,
    SourceSpanProposal,
)

HEX = "a" * 64
FORGED_KEYS = [
    "document_version_id", "logical_source_id", "extraction_id", "span_id",
    "content_sha256", "quote_sha256", "nfc_text_sha256", "scope_id",
    "logical_source_key", "recorded_at", "extracted_at", "status", "lineage",
    "quality", "independence", "applicability", "verification_status", "authority",
]


def _doc(**overrides: object) -> dict[str, object]:
    return {"filename": "q3.pdf", "media_type": "application/pdf", **overrides}


@pytest.mark.parametrize("key", FORGED_KEYS)
def test_document_proposal_rejects_forged_authority_key(key: str) -> None:
    with pytest.raises(ValidationError):
        DocumentIngestProposal.model_validate_json(json.dumps(_doc(**{key: "x"})))


@pytest.mark.parametrize("key", FORGED_KEYS)
def test_span_proposal_rejects_forged_authority_key(key: str) -> None:
    payload = {"start_char": 0, "end_char": 1, "quote": "a", key: "x"}
    with pytest.raises(ValidationError):
        SourceSpanProposal.model_validate_json(json.dumps(payload))


def test_untrusted_json_parses_only_into_proposal_types() -> None:
    parsed = DocumentIngestProposal.model_validate_json(json.dumps(_doc()))
    assert type(parsed) is DocumentIngestProposal
    assert not isinstance(parsed, DocumentVersion)
    assert set(DocumentIngestProposal.model_fields) & set(DocumentVersion.model_fields) == set()


def test_missing_declared_metadata_stays_none() -> None:
    parsed = DocumentIngestProposal.model_validate(_doc())
    assert parsed.declared_publisher is None
    assert parsed.declared_author is None
    assert parsed.declared_published_at is None
    assert parsed.declared_geography is None
    assert parsed.declared_period is None
    assert parsed.declared_methodology is None


def test_declared_published_at_is_recorded_verbatim_not_parsed() -> None:
    parsed = DocumentIngestProposal.model_validate(_doc(declared_published_at="not a date"))
    assert parsed.declared_published_at == "not a date"


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "a" * 256,
        "with\x00nul",
        "with\nnewline",
        "with\u200fformat",
        "with\u2028separator",
    ],
)
def test_filename_safety_rules(filename: str) -> None:
    with pytest.raises(ValidationError):
        DocumentIngestProposal.model_validate(_doc(filename=filename))


def test_filename_byte_bound_is_utf8_not_code_points() -> None:
    # 128 code points of a 2-byte character is 256 UTF-8 bytes.
    name = "\u00e9" * 128
    assert len(name) == 128 and len(name.encode("utf-8")) == 256
    with pytest.raises(ValidationError):
        DocumentIngestProposal.model_validate(_doc(filename=name))
    shorter = "\u00e9" * 127
    assert DocumentIngestProposal.model_validate(_doc(filename=shorter)).filename == shorter


def test_filename_rejects_lone_surrogate() -> None:
    with pytest.raises(ValidationError):
        DocumentIngestProposal.model_validate(_doc(filename="a\ud800b"))


@pytest.mark.parametrize("media_type", ["", "a" * 129, "text/\x00plain"])
def test_media_type_bounds(media_type: str) -> None:
    with pytest.raises(ValidationError):
        DocumentIngestProposal.model_validate(_doc(media_type=media_type))


@pytest.mark.parametrize("value", ["", "a" * 513, "pub\x00lisher"])
def test_declared_metadata_bounds(value: str) -> None:
    with pytest.raises(ValidationError):
        DocumentIngestProposal.model_validate(_doc(declared_publisher=value))


def test_quote_preserves_format_and_separator_characters() -> None:
    """The quote rule is fidelity, not filename safety: nothing is stripped."""
    quote = "a\u200fb\u200e\u200dc\u2028d"
    assert [ord(c) for c in quote] == [0x61, 0x200F, 0x62, 0x200E, 0x200D, 0x63, 0x2028, 0x64]
    parsed = SourceSpanProposal(start_char=0, end_char=len(quote), quote=quote)
    assert parsed.quote == quote
    assert [ord(c) for c in parsed.quote] == [ord(c) for c in quote]


@pytest.mark.parametrize("quote", ["", "a" * (MAX_QUOTE_CODE_POINTS + 1), "a\ud800b"])
def test_quote_bounds_and_lone_surrogate(quote: str) -> None:
    with pytest.raises(ValidationError):
        SourceSpanProposal(start_char=0, end_char=1, quote=quote)


def test_quote_at_maximum_length_is_accepted() -> None:
    quote = "a" * MAX_QUOTE_CODE_POINTS
    assert len(SourceSpanProposal(start_char=0, end_char=1, quote=quote).quote) == MAX_QUOTE_CODE_POINTS


@pytest.mark.parametrize("field", ["start_char", "end_char"])
def test_span_offsets_must_be_non_negative(field: str) -> None:
    payload = {"start_char": 0, "end_char": 1, "quote": "a", field: -1}
    with pytest.raises(ValidationError):
        SourceSpanProposal.model_validate(payload)


def test_proposals_are_frozen() -> None:
    parsed = DocumentIngestProposal.model_validate(_doc())
    with pytest.raises(ValidationError):
        parsed.filename = "other.pdf"  # type: ignore[misc]


def test_model_copy_update_is_revalidated() -> None:
    parsed = DocumentIngestProposal.model_validate(_doc())
    assert parsed.model_copy(update={"filename": "ok.pdf"}).filename == "ok.pdf"
    with pytest.raises(ValidationError):
        parsed.model_copy(update={"filename": "bad\x00.pdf"})


def test_canonical_model_rejects_extra_key_even_inside_trusted_context() -> None:
    payload = {
        "document_version_id": HEX,
        "scope_id": "acme",
        "logical_source_id": HEX,
        "content_sha256": HEX,
        "byte_size": 1,
        "verification_status": "VERIFIED",
    }
    with trusted_construction(), pytest.raises(ValidationError):
        DocumentVersion.model_validate(payload)
