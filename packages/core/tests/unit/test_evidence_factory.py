"""Factory invariants: identity outcomes, NFC binding, and span verification order.

Every non-ASCII literal in this file is written as a ``\\uXXXX`` escape and
pinned with an ``ord()`` assertion. Literal glyphs have been silently rewritten
in transit before, which would make a fidelity test pass against the wrong text.
"""
from __future__ import annotations

import unicodedata
from typing import Any

import pytest

from openexecutive.evidence import factory, identity
from openexecutive.evidence.contracts import DocumentVersion, Extraction, SourceSpanProposal
from openexecutive.evidence.extractor_config import ExtractorConfigError, extractor_config_sha256
from openexecutive.evidence.factory import (
    MAX_RAW_BYTES,
    MAX_TEXT_CODE_POINTS,
    EvidenceFactoryError,
    NormalizationError,
    SpanVerificationError,
    mint_document_version,
    mint_extraction,
    mint_source_span,
)
from openexecutive.evidence.identity import TAG_EXTRACTION, text_sha256

CONFIG: dict[str, Any] = {"ocr": True, "dpi": 300}
DOCUMENT_VERSION_FIELDS = {
    "document_version_id", "scope_id", "logical_source_id", "content_sha256", "byte_size",
}
EXTRACTION_FIELDS = {
    "extraction_id", "document_version_id", "extractor_name", "extractor_version",
    "extractor_config_sha256", "raw_text_sha256", "nfc_text_sha256", "nfc_text_length",
}


def version(raw: bytes = b"bytes", scope: str = "acme", key: str = "reg-1") -> DocumentVersion:
    return mint_document_version(raw_bytes=raw, scope_id=scope, logical_source_key=key)


def extraction(raw_text: str, nfc_text: str | None = None, **kw: Any) -> Extraction:
    nfc = unicodedata.normalize("NFC", raw_text) if nfc_text is None else nfc_text
    return mint_extraction(
        kw.pop("version", None) or version(),
        extractor_name=kw.pop("extractor_name", "pdftext"),
        extractor_version=kw.pop("extractor_version", "1.2.0"),
        extractor_config=kw.pop("extractor_config", CONFIG),
        raw_text=raw_text,
        nfc_text=nfc,
    )


def test_same_version_id_implies_equal_complete_record() -> None:
    """Two mints from equal identity inputs are equal in every field, not just id."""
    left, right = version(), version()
    assert left.document_version_id == right.document_version_id
    assert left == right
    assert left.model_dump() == right.model_dump()


def test_document_version_field_set_is_exactly_identity_determined() -> None:
    """Reintroducing a timestamp, declared proposal or any other non-identity field fails here."""
    assert set(DocumentVersion.model_fields) == DOCUMENT_VERSION_FIELDS


def test_changed_bytes_give_a_different_version() -> None:
    left, right = version(raw=b"a"), version(raw=b"b")
    assert left.document_version_id != right.document_version_id
    assert left != right


def test_identical_bytes_in_different_scopes_give_different_ids() -> None:
    left, right = version(scope="acme"), version(scope="other")
    assert left.document_version_id != right.document_version_id
    assert left.logical_source_id != right.logical_source_id
    assert left.content_sha256 == right.content_sha256


def test_different_logical_keys_in_one_scope_give_different_ids() -> None:
    assert version(key="reg-1").document_version_id != version(key="reg-2").document_version_id


def test_version_records_byte_size() -> None:
    assert version(raw=b"12345").byte_size == 5


def test_same_extraction_id_implies_equal_complete_record() -> None:
    left, right = extraction("alpha"), extraction("alpha")
    assert left.extraction_id == right.extraction_id
    assert left == right
    assert left.model_dump() == right.model_dump()


def test_extraction_field_set_is_exactly_identity_determined() -> None:
    assert set(Extraction.model_fields) == EXTRACTION_FIELDS


def test_different_raw_text_gives_a_different_extraction() -> None:
    left, right = extraction("a"), extraction("b")
    assert left.extraction_id != right.extraction_id
    assert left != right


def test_decomposed_and_composed_share_nfc_hash_but_not_extraction_id() -> None:
    """Pins ``raw_text_sha256``'s presence in the identity."""
    decomposed, composed = "e\u0301", "\u00e9"
    assert [ord(c) for c in decomposed] == [0x65, 0x301]
    assert [ord(c) for c in composed] == [0xE9]
    left, right = extraction(decomposed), extraction(composed)
    assert left.nfc_text_sha256 == right.nfc_text_sha256
    assert left.raw_text_sha256 != right.raw_text_sha256
    assert left.extraction_id != right.extraction_id


def test_extraction_id_equals_the_documented_component_recomputation() -> None:
    """Pins both output hashes and their documented order."""
    raw = "e\u0301"
    base = version()
    minted = extraction(raw, version=base)
    expected = identity.mint_id(
        TAG_EXTRACTION,
        base.document_version_id,
        "pdftext",
        "1.2.0",
        extractor_config_sha256(CONFIG),
        text_sha256(raw),
        text_sha256(unicodedata.normalize("NFC", raw)),
    )
    assert text_sha256(raw) != text_sha256(unicodedata.normalize("NFC", raw))
    assert minted.extraction_id == expected


def test_mint_id_receives_both_output_hashes_in_the_documented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []
    real = identity.mint_id

    def spy(tag: bytes, *parts: Any) -> str:
        calls.append((tag, *parts))
        return real(tag, *parts)

    monkeypatch.setattr(identity, "mint_id", spy)
    raw = "e\u0301"
    minted = extraction(raw)
    extraction_calls = [c for c in calls if c[0] == TAG_EXTRACTION]
    assert len(extraction_calls) == 1
    assert extraction_calls[0][5] == minted.raw_text_sha256 == text_sha256(raw)
    assert extraction_calls[0][6] == minted.nfc_text_sha256
    assert extraction_calls[0][5] != extraction_calls[0][6]


@pytest.mark.parametrize(
    "field,value",
    [
        ("extractor_name", "other"),
        ("extractor_version", "9.9.9"),
        ("extractor_config", {"ocr": False, "dpi": 300}),
    ],
)
def test_extractor_identity_and_config_participate(field: str, value: Any) -> None:
    assert extraction("alpha").extraction_id != extraction("alpha", **{field: value}).extraction_id


def test_same_extraction_across_different_document_versions_differs() -> None:
    left = extraction("alpha", version=version(raw=b"one"))
    right = extraction("alpha", version=version(raw=b"two"))
    assert left.extraction_id != right.extraction_id


def test_nfc_text_length_is_recorded_in_code_points() -> None:
    minted = extraction("caf\u00e9 latte")
    assert minted.nfc_text_length == 10


def test_nfc_text_must_be_the_nfc_normalization_of_raw_text() -> None:
    with pytest.raises(NormalizationError):
        extraction("e\u0301", nfc_text="e\u0301")


def test_invalid_config_is_rejected_before_an_extraction_exists() -> None:
    with pytest.raises(ExtractorConfigError):
        extraction("alpha", extractor_config={"bad": float("nan")})


def test_verified_span_round_trips_the_quote_and_binds_the_text() -> None:
    text = "alpha beta"
    minted = extraction(text)
    span = mint_source_span(
        SourceSpanProposal(start_char=6, end_char=10, quote="beta"),
        extraction=minted,
        trusted_nfc_text=text,
    )
    assert span.quote == text[6:10]
    assert span.quote_sha256 == text_sha256("beta")
    assert span.nfc_text_sha256 == minted.nfc_text_sha256
    assert span.extraction_id == minted.extraction_id


def test_offsets_address_nfc_text_not_raw_text() -> None:
    raw = "cafe\u0301 latte"
    nfc = unicodedata.normalize("NFC", raw)
    assert len(raw) == 11 and len(nfc) == 10
    assert raw.index("latte") == 6 and nfc.index("latte") == 5
    span = mint_source_span(
        SourceSpanProposal(start_char=5, end_char=10, quote="latte"),
        extraction=extraction(raw),
        trusted_nfc_text=nfc,
    )
    assert span.start_char == 5 and span.quote == "latte"


def test_arabic_bidi_marks_and_separators_round_trip_byte_exact() -> None:
    raw = "\u200f\u0645\u0631\u062d\u0628\u0627\u200e \u0645\u200d\u062d\u2028\u0628\u064a\u062a"
    marks = {0x200F, 0x200E, 0x200D, 0x2028}
    assert marks <= {ord(c) for c in raw}
    nfc = unicodedata.normalize("NFC", raw)
    span = mint_source_span(
        SourceSpanProposal(start_char=0, end_char=len(nfc), quote=nfc),
        extraction=extraction(raw),
        trusted_nfc_text=nfc,
    )
    assert span.quote == nfc
    assert [ord(c) for c in span.quote] == [ord(c) for c in nfc]
    assert marks <= {ord(c) for c in span.quote}


def test_hash_mismatch_names_the_failed_check() -> None:
    with pytest.raises(SpanVerificationError) as excinfo:
        mint_source_span(
            SourceSpanProposal(start_char=0, end_char=5, quote="alpha"),
            extraction=extraction("alpha beta"),
            trusted_nfc_text="alpha other",
        )
    assert excinfo.value.check == "nfc_text_sha256"


def test_length_mismatch_names_the_failed_check() -> None:
    """A forged Extraction: correct hash, wrong recorded length."""
    from openexecutive.evidence._authority import trusted_construction

    text = "alpha beta"
    honest = extraction(text)
    with trusted_construction():
        forged = honest.model_copy(update={"nfc_text_length": honest.nfc_text_length + 1})
    assert forged.nfc_text_sha256 == text_sha256(text)
    with pytest.raises(SpanVerificationError) as excinfo:
        mint_source_span(
            SourceSpanProposal(start_char=0, end_char=5, quote="alpha"),
            extraction=forged,
            trusted_nfc_text=text,
        )
    assert excinfo.value.check == "nfc_text_length"


def test_a_proposal_for_one_text_cannot_be_minted_against_another_extraction() -> None:
    other = "gamma delta"
    with pytest.raises(SpanVerificationError) as excinfo:
        mint_source_span(
            SourceSpanProposal(start_char=6, end_char=10, quote="beta"),
            extraction=extraction(other),
            trusted_nfc_text=other,
        )
    assert excinfo.value.check == "quote_mismatch"


@pytest.mark.parametrize(
    "start,end,quote",
    [(0, 99, "alpha"), (5, 5, "alpha"), (7, 3, "alpha"), (10, 11, "a")],
)
def test_out_of_range_and_inverted_offsets_fail(start: int, end: int, quote: str) -> None:
    text = "alpha beta"
    with pytest.raises(SpanVerificationError) as excinfo:
        mint_source_span(
            SourceSpanProposal(start_char=start, end_char=end, quote=quote),
            extraction=extraction(text),
            trusted_nfc_text=text,
        )
    assert excinfo.value.check == "offset_range"


def test_quote_that_is_not_the_slice_fails() -> None:
    text = "alpha beta"
    with pytest.raises(SpanVerificationError) as excinfo:
        mint_source_span(
            SourceSpanProposal(start_char=0, end_char=5, quote="alph"),
            extraction=extraction(text),
            trusted_nfc_text=text,
        )
    assert excinfo.value.check == "quote_mismatch"


@pytest.mark.parametrize(
    "start,end,quote,text",
    [
        (0, 5, "alpha", "alpha other"),
        (0, 99, "alpha", "alpha beta"),
        (0, 5, "alph!", "alpha beta"),
    ],
)
def test_no_identity_is_derived_on_any_failed_verification(
    start: int, end: int, quote: str, text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    minted = extraction("alpha beta")
    calls: list[Any] = []
    monkeypatch.setattr(identity, "mint_id", lambda *a, **k: calls.append(a))
    with pytest.raises(SpanVerificationError):
        mint_source_span(
            SourceSpanProposal(start_char=start, end_char=end, quote=quote),
            extraction=minted,
            trusted_nfc_text=text,
        )
    assert calls == []


def test_verification_error_never_echoes_the_quote_or_source_text() -> None:
    secret = "confidential-payroll-figure"
    text = "alpha " + secret
    with pytest.raises(SpanVerificationError) as excinfo:
        mint_source_span(
            SourceSpanProposal(start_char=0, end_char=5, quote=secret[:5]),
            extraction=extraction(text),
            trusted_nfc_text=text,
        )
    message = str(excinfo.value)
    assert secret not in message and text not in message
    assert message == "source span verification failed: quote_mismatch"


@pytest.mark.parametrize("scope", ["", "a" * 129, "bad\x00scope", "bad\u200fscope"])
def test_scope_id_bounds(scope: str) -> None:
    with pytest.raises(EvidenceFactoryError):
        version(scope=scope)


@pytest.mark.parametrize("key", ["", "a" * 201, "bad\x00key", "bad\u2028key"])
def test_logical_source_key_bounds(key: str) -> None:
    with pytest.raises(EvidenceFactoryError):
        version(key=key)


@pytest.mark.parametrize("field", ["extractor_name", "extractor_version"])
@pytest.mark.parametrize("value", ["", "a" * 129, "bad\x00value"])
def test_extractor_field_bounds(field: str, value: str) -> None:
    with pytest.raises(EvidenceFactoryError):
        extraction("alpha", **{field: value})


def test_raw_bytes_size_bound_is_enforced_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert MAX_RAW_BYTES == 64 * 1024 * 1024
    monkeypatch.setattr(factory, "MAX_RAW_BYTES", 4)
    monkeypatch.setattr(
        identity, "content_sha256", lambda _: pytest.fail("hashed an oversized input")
    )
    with pytest.raises(EvidenceFactoryError):
        version(raw=b"12345")


def test_text_length_bound_is_enforced() -> None:
    assert MAX_TEXT_CODE_POINTS == 2_000_000
    with pytest.raises(EvidenceFactoryError):
        extraction("a" * (MAX_TEXT_CODE_POINTS + 1))


@pytest.mark.parametrize("bad", ["not bytes", None, bytearray(b"x")])
def test_raw_bytes_must_be_bytes(bad: object) -> None:
    with pytest.raises(EvidenceFactoryError):
        mint_document_version(raw_bytes=bad, scope_id="acme", logical_source_key="k")  # type: ignore[arg-type]


def test_factories_reject_the_wrong_model_types() -> None:
    with pytest.raises(EvidenceFactoryError):
        extraction("alpha", version="not a version")
    with pytest.raises(EvidenceFactoryError):
        mint_source_span(
            SourceSpanProposal(start_char=0, end_char=1, quote="a"),
            extraction="not an extraction",  # type: ignore[arg-type]
            trusted_nfc_text="a",
        )
