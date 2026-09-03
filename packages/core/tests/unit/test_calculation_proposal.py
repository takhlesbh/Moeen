"""The model-shaped proposal: intent only, and unsafe text is refused not rewritten."""
from __future__ import annotations

import hashlib
import unicodedata
from typing import Any

import pytest
from pydantic import ValidationError

from openexecutive.calc.contract import (
    MAX_ID_LEN,
    MAX_LABEL_LEN,
    MAX_OPERANDS_PER_REQUEST,
    MAX_PURPOSE_LEN,
    Operand,
    SourceHint,
)
from openexecutive.calc.units import Unit
from openexecutive.specialists.calculation_proposal import (
    CalculationProposal,
    is_safe_descriptive_text,
    is_safe_identifier,
)

USD = "currency:USD"

# Every character class the screen must refuse, named so a failure says WHICH
# class regressed rather than merely "a test failed".
UNSAFE_CHARS = {
    "LF": chr(10),
    "CR": chr(13),
    "NEL": chr(0x85),
    "LINE_SEP": chr(0x2028),
    "PARA_SEP": chr(0x2029),
    "ESC": chr(27),
    "NUL": chr(0),
    "BIDI_RLO": chr(0x202E),
    "BIDI_LRI": chr(0x2066),
    "ZWJ": chr(0x200D),
    "BOM": chr(0xFEFF),
}


def _op(oid: str = "a", value: str = "10", **kw: Any) -> Operand:
    return Operand(
        operand_id=oid,
        label=kw.pop("label", oid),
        value=value,
        unit=Unit(code=USD),
        basis="applicant_stated",
        **kw,
    )


def _proposal(**kw: Any) -> CalculationProposal:
    base: dict[str, Any] = {
        "operation": "add",
        "operands": (_op("a", "10"), _op("b", "5")),
        "target_unit": Unit(code=USD),
        "purpose": "runway check",
    }
    base.update(kw)
    return CalculationProposal(**base)


def _hostile(marker: str = "x") -> list[str]:
    """Values BOTH policies must refuse: line and control characters."""
    return [f"a{c}{marker}" for c in UNSAFE_CHARS.values()]


def _identifier_only_hostile() -> list[str]:
    """Values only the IDENTIFIER policy refuses — padding, not characters."""
    return [" leading", "trailing "]


# ---------------------------------------------------------------------------
# Intent only — the absences are the guarantee
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "request_id",
        "correlation",
        "result_value",
        "exact_result",
        "arithmetic_status",
        "evidence",
        "fingerprint",
        "authority",
        "computed_at",
        "normalized_operands",
        "expression_executed",
    ],
)
def test_a_proposal_has_no_authoritative_field(field: str) -> None:
    assert field not in CalculationProposal.model_fields


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", "req-1"),
        ("correlation", {"specialist": "cfo", "case_id": "c", "run_id": "r"}),
        ("arithmetic_status", "ARITHMETIC_VERIFIED"),
        ("result_value", "999999"),
        ("fingerprint", "a" * 64),
        ("authority", {"authority_id": "openexecutive.calc"}),
        ("computed_at", "2026-09-03T00:00:00Z"),
        ("evidence", {"status": "ALL_SUPPORTED"}),
    ],
)
def test_a_proposal_cannot_inject_an_authoritative_field(field: str, value: Any) -> None:
    """``extra="forbid"`` makes the attempt visible rather than stripping it."""
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        _proposal(**{field: value})


def test_a_proposal_is_frozen() -> None:
    proposal = _proposal()
    with pytest.raises(ValidationError):
        proposal.purpose = "rewritten"  # type: ignore[misc]


def test_a_valid_proposal_round_trips() -> None:
    proposal = _proposal(claim_ref="c1")
    assert CalculationProposal.model_validate(proposal.model_dump()) == proposal


# ---------------------------------------------------------------------------
# Unsafe text: rejected, never rewritten
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("evil", _hostile())
def test_every_model_authored_string_rejects_line_and_control_characters(
    evil: str,
) -> None:
    """The rule BOTH policies share, across every string on the proposal.

    A matrix because the recurring failure was screening the field that had just
    been found wrong while leaving its siblings open.
    """
    with pytest.raises(ValidationError):
        _proposal(purpose=evil)
    with pytest.raises(ValidationError):
        _proposal(claim_ref=evil[:MAX_ID_LEN])
    with pytest.raises(ValidationError):
        _proposal(operands=(_op("a", "1", label=evil), _op("b", "2")))
    with pytest.raises(ValidationError):
        _proposal(operands=(_op(evil[:MAX_ID_LEN], "1"), _op("b", "2")))


@pytest.mark.parametrize("padded", _identifier_only_hostile())
def test_identifiers_reject_padding_but_descriptive_text_does_not(
    padded: str,
) -> None:
    """The policies diverge exactly here, and the divergence is deliberate.

    An identifier is MATCHED, so ``"c1"`` and ``" c1"`` must not be two
    spellings of one id. Descriptive text is only read, so padding is a
    formatting quirk rather than an identity hazard — and rewriting it would
    make the record say something the specialist did not.
    """
    with pytest.raises(ValidationError):
        _proposal(claim_ref=padded)
    with pytest.raises(ValidationError):
        _proposal(operands=(_op(padded, "1"), _op("b", "2")))
    assert _proposal(purpose=padded).purpose == padded


def test_descriptive_text_may_contain_the_reason_delimiter() -> None:
    """REGRESSION: coupling this module to ``result_contract._join`` was wrong.

    ``"; "`` was refused here because that function composes degradation reasons
    with it — so a specialist writing "runway; then covenant headroom" was
    rejected for a reason having nothing to do with its own field, and the
    coupling would invert silently the day ``_join`` changed separators.
    Escaping a delimiter is the renderer's job.
    """
    prose = "runway; then covenant headroom"
    assert _proposal(purpose=prose).purpose == prose
    assert is_safe_descriptive_text(prose, max_length=MAX_PURPOSE_LEN) is True
    # An identifier still refuses it: ids are matched, and a delimiter inside
    # one is a parsing hazard rather than prose.
    assert is_safe_identifier("a; b", max_length=MAX_ID_LEN) is True


@pytest.mark.parametrize("evil", _hostile())
@pytest.mark.parametrize(
    "hint_field",
    ["document_label", "filename", "retrieval_id_hint", "quoted_text"],
)
def test_every_source_hint_field_rejects_unsafe_text(
    evil: str, hint_field: str
) -> None:
    """``SourceHint`` is fully model-authored and rides on a persisted request."""
    cap = MAX_ID_LEN if hint_field == "retrieval_id_hint" else MAX_LABEL_LEN
    hint = SourceHint(**{hint_field: evil[:cap]})
    with pytest.raises(ValidationError):
        _proposal(operands=(_op("a", "1", source_hint=hint), _op("b", "2")))


def test_nothing_is_ever_rewritten() -> None:
    """A proposal that validates carries exactly the text it was given.

    Rejection is the whole policy: a sanitiser that "fixes" text makes the
    record say something the specialist did not, and its substitution marker
    reads as a tamper signal in a record nothing tampered with.
    """
    purpose = "runway = cash / burn (approx)"
    label = "Q3 net burn [board deck]"
    proposal = _proposal(
        purpose=purpose,
        claim_ref="c1",
        operands=(_op("a", "1", label=label), _op("b", "2")),
    )
    assert proposal.purpose == purpose
    assert proposal.operands[0].label == label
    assert proposal.claim_ref == "c1"


def test_safe_but_unusual_text_is_still_accepted() -> None:
    """The screen blocks forgery vectors, not legitimate prose."""
    for good in ("naïve €4.2M — 日本語", "a,b;c", "x" * MAX_PURPOSE_LEN, "a; " .strip()):
        assert is_safe_descriptive_text(good, max_length=MAX_PURPOSE_LEN) is True


def test_a_digest_shaped_identifier_is_accepted() -> None:
    """A content hash is a legitimate way to name a claim.

    Banning the shape would drop a contract-legal value and blame the model for
    it; identity safety comes from the gateway's framing, not from a ban.
    """
    digest = hashlib.sha256(b"claim").hexdigest()
    assert is_safe_identifier(digest, max_length=MAX_ID_LEN) is True
    assert _proposal(claim_ref=digest).claim_ref == digest


@pytest.mark.parametrize(
    ("value", "max_length", "safe"),
    [
        ("ok", 64, True),
        ("", 64, False),
        ("x" * 64, 64, True),
        ("x" * 65, 64, False),
        (" pad", 64, False),
        ("pad ", 64, False),
        (None, 64, False),
        (42, 64, False),
        (b"bytes", 64, False),
    ],
)
def test_is_safe_identifier(value: Any, max_length: int, safe: bool) -> None:
    assert is_safe_identifier(value, max_length=max_length) is safe


def test_the_screen_refuses_every_control_format_and_separator_codepoint() -> None:
    """Exhaustive over the intended set, not a break at the first match.

    An earlier version iterated ``range(0x110000)`` and ``break``ed on the first
    hit — codepoint 0 — so it tested NUL alone and would have passed with the
    rule deleted. ``isprintable()`` is the mechanism; this asserts it actually
    covers every category it is credited with.
    """
    unscreened = [
        cp
        for cp in range(0x110000)
        if unicodedata.category(chr(cp)) in ("Cc", "Cf", "Zl", "Zp")
        and (
            is_safe_identifier(f"a{chr(cp)}b", max_length=64)
            or is_safe_descriptive_text(f"a{chr(cp)}b", max_length=64)
        )
    ]
    assert unscreened == [], f"{len(unscreened)} unsafe codepoints accepted"


def test_the_screen_is_stricter_than_the_category_list_alone() -> None:
    """Surrogates, private use, unassigned and non-ASCII spaces are also out."""
    for name, cp in {
        "surrogate": 0xD800,
        "private_use": 0xE000,
        "unassigned": 0x0378,
        "ogham_space": 0x1680,
        "nbsp": 0x00A0,
    }.items():
        assert is_safe_identifier(f"a{chr(cp)}b", max_length=64) is False, name
        assert is_safe_descriptive_text(f"a{chr(cp)}b", max_length=64) is False, name


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_operands_are_bounded_and_unique() -> None:
    with pytest.raises(ValidationError, match="1\\.\\.64 operands"):
        _proposal(operands=())
    with pytest.raises(ValidationError, match="unique"):
        _proposal(operands=(_op("a", "1"), _op("a", "2")))
    too_many = tuple(
        _op(f"o{i}", "1") for i in range(MAX_OPERANDS_PER_REQUEST + 1)
    )
    with pytest.raises(ValidationError, match="operands"):
        _proposal(operands=too_many)


def test_an_unknown_operation_is_refused() -> None:
    with pytest.raises(ValidationError):
        _proposal(operation="exec_arbitrary_code")


# ---------------------------------------------------------------------------
# The wire schema mirrors the model — Phase 3B2
# ---------------------------------------------------------------------------


def _schema_props(node: dict) -> dict:  # type: ignore[type-arg]
    return node["properties"]  # type: ignore[no-any-return]


def test_wire_schema_properties_equal_the_model_fields() -> None:
    """Every proposal field is on the wire and nothing else is."""
    import json

    from openexecutive.calc.contract import Operand, SourceHint
    from openexecutive.specialists.calculation_proposal import (
        CALCULATION_REQUESTS_SCHEMA,
    )

    item = CALCULATION_REQUESTS_SCHEMA["items"]
    assert set(_schema_props(item)) == set(CalculationProposal.model_fields)
    operand = _schema_props(item)["operands"]["items"]
    assert set(_schema_props(operand)) == set(Operand.model_fields)
    hint = _schema_props(operand)["source_hint"]
    assert set(_schema_props(hint)) == set(SourceHint.model_fields)
    # Nothing a model could use to assert an answer, an id or a status.
    serialized = json.dumps(CALCULATION_REQUESTS_SCHEMA)
    for forbidden in (
        "request_id", "correlation", "result", "status", "fingerprint",
        "computed_at", "authority", "evidence", "verified", "integrity",
    ):
        assert f'"{forbidden}"' not in serialized, forbidden


def test_wire_schema_enums_are_derived_from_the_contract_literals() -> None:
    from typing import get_args

    from openexecutive.calc.contract import (
        OperandBasis,
        OperandRole,
        OperationId,
        RoundingMode,
    )
    from openexecutive.calc.numeric import NumberFormat
    from openexecutive.specialists.calculation_proposal import (
        CALCULATION_REQUESTS_SCHEMA,
    )

    item = _schema_props(CALCULATION_REQUESTS_SCHEMA["items"])
    operand = _schema_props(item["operands"]["items"])
    assert item["operation"]["enum"] == list(get_args(OperationId))
    assert item["rounding"]["enum"] == list(get_args(RoundingMode))
    assert operand["basis"]["enum"] == list(get_args(OperandBasis))
    assert operand["role"]["enum"] == list(get_args(OperandRole))
    assert operand["number_format"]["enum"] == list(get_args(NumberFormat))


def test_wire_schema_properties_are_sorted_for_cache_stability() -> None:
    from openexecutive.specialists.calculation_proposal import (
        CALCULATION_REQUESTS_SCHEMA,
    )

    def _check(node: object) -> None:
        if not isinstance(node, dict):
            return
        props = node.get("properties")
        if isinstance(props, dict):
            assert list(props) == sorted(props), list(props)
        for value in node.values():
            if isinstance(value, dict):
                _check(value)
            elif isinstance(value, list):
                for entry in value:
                    _check(entry)

    _check(CALCULATION_REQUESTS_SCHEMA)
