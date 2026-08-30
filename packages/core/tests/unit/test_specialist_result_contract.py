"""Tests for the specialist result contract.

The contract exists to stop three specific failure modes, and the tests are
organised around them rather than around the class list:

* fabricated provenance — a page/URL/date that was never observed
* calculation-authority leakage — a model's arithmetic presented as checked
* dishonest degradation — structure invented from prose when parsing failed
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from openexecutive.specialists.result_contract import (
    EMIT_SPECIALIST_RESULT_TOOL,
    Attribution,
    CalculationProvenance,
    Claim,
    ClaimType,
    ConfidenceLevel,
    EvidenceKind,
    EvidenceRef,
    SpecialistResult,
    VerificationStatus,
    parse_specialist_result,
    render_for_executive,
)

TOOL_NAME = EMIT_SPECIALIST_RESULT_TOOL["name"]


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use(payload: Any, name: str = TOOL_NAME) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id="tu1", name=name, input=payload)


def _message(*blocks: Any) -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks))


def _claim(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"claim_id": "c1", "text": "t", "claim_type": "assessment"}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# claim types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "claim_type",
    ["source_fact", "assessment", "unsupported", "conflict"],
)
def test_non_calculation_claim_types_round_trip(claim_type: str) -> None:
    claim = Claim(claim_id="c1", text="t", claim_type=claim_type)  # type: ignore[arg-type]
    assert claim.claim_type == claim_type
    assert Claim.model_validate(claim.model_dump()) == claim


def test_derived_calculation_round_trips_with_provenance() -> None:
    claim = Claim(
        claim_id="c1",
        text="Burn multiple is roughly 1.4",
        claim_type="derived_calculation",
        calculation=CalculationProvenance(
            inputs=["net burn $410k/mo", "net new ARR $290k/mo"],
            method="net burn / net new ARR",
            model_stated_result="1.4",
        ),
    )
    assert Claim.model_validate(claim.model_dump()) == claim


def test_unknown_claim_type_rejected() -> None:
    with pytest.raises(ValidationError):
        Claim(claim_id="c1", text="t", claim_type="vibes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# provenance: missing stays missing
# ---------------------------------------------------------------------------


def test_evidence_with_only_label_and_filename_leaves_the_rest_none() -> None:
    """The retrieval layer supplies filename and nothing else today."""
    ref = EvidenceRef(kind="document", label="[Q3-board-deck.pdf]", filename="Q3-board-deck.pdf")
    assert ref.page is None
    assert ref.sheet is None
    assert ref.cell_range is None
    assert ref.url is None
    assert ref.retrieved_at is None
    assert ref.chunk_index is None
    assert ref.provenance_note is None


def test_no_provenance_field_has_a_non_none_default() -> None:
    """A default other than None would manufacture provenance on every ref."""
    ref = EvidenceRef(kind="none", label="(none)")
    optional = (
        "filename", "chunk_index", "page", "sheet",
        "cell_range", "url", "retrieved_at", "provenance_note",
    )
    for field in optional:
        assert getattr(ref, field) is None, field


def test_provenance_note_round_trips() -> None:
    ref = EvidenceRef(
        kind="document",
        label="[deck.pdf]",
        provenance_note="page unavailable: PDF pages flattened before chunking",
    )
    assert EvidenceRef.model_validate(ref.model_dump()) == ref


def test_evidence_rejects_unknown_fields() -> None:
    """extra='forbid' stops a typo'd field from silently vanishing."""
    with pytest.raises(ValidationError):
        EvidenceRef(kind="document", label="x", pge=4)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# calculation authority
# ---------------------------------------------------------------------------


def test_model_stated_result_does_not_imply_verified_result() -> None:
    calc = CalculationProvenance(
        inputs=["a", "b"], method="a / b", model_stated_result="1.4"
    )
    assert calc.model_stated_result == "1.4"
    assert calc.verified_result is None
    assert calc.verification_status == "unverified"


def test_verified_result_cannot_be_set_while_unverified() -> None:
    """The core ADR guard: no verified figure without verified status."""
    with pytest.raises(ValidationError):
        CalculationProvenance(
            inputs=["a"], method="m", verified_result="1.4"
        )


def test_a_matched_verified_pair_is_still_rejected() -> None:
    """The rule is absolute, not lockstep — this is the shape an attacker writes.

    A lockstep rule ("the two fields must agree") constrains the pair without
    constraining who set it, so writing both together produced a valid object
    indistinguishable from a checked figure. Since no authority exists in this
    slice, no verified value is legitimate at all.
    """
    with pytest.raises(ValidationError):
        CalculationProvenance(
            inputs=["a"], method="m",
            model_stated_result="1.4",
            verified_result="1.38",
            verification_status="verified",
        )


def test_verified_status_alone_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CalculationProvenance(
            inputs=["a"], method="m", verification_status="verified"
        )


@pytest.mark.parametrize("status", ["unverified", "refuted", "not_applicable"])
def test_non_verified_statuses_remain_expressible(status: str) -> None:
    """The field is not inert — only 'verified' is unreachable."""
    calc = CalculationProvenance(
        inputs=["a"], method="m", verification_status=status  # type: ignore[arg-type]
    )
    assert calc.verification_status == status
    assert calc.verified_result is None


def test_derived_calculation_requires_provenance() -> None:
    with pytest.raises(ValidationError):
        Claim(claim_id="c1", text="t", claim_type="derived_calculation")


def test_non_calculation_claim_may_not_carry_calculation_provenance() -> None:
    with pytest.raises(ValidationError):
        Claim(
            claim_id="c1", text="t", claim_type="source_fact",
            calculation=CalculationProvenance(inputs=[], method="m"),
        )


def test_tool_schema_does_not_expose_verification_fields_to_the_model() -> None:
    """Structural guarantee: the model has no field in which to claim verification."""
    schema = json.dumps(EMIT_SPECIALIST_RESULT_TOOL)
    assert "verified_result" not in schema
    assert "verification_status" not in schema
    assert "model_stated_result" in schema


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------


def test_attribution_defaults_to_unknown_not_to_evidence() -> None:
    """Defaulting to independent_evidence would silently upgrade every claim."""
    assert Claim(claim_id="c1", text="t", claim_type="assessment").attribution == "unknown"


@pytest.mark.parametrize(
    "attribution",
    ["applicant_asserted", "independent_evidence", "specialist_judgement", "unknown"],
)
def test_attribution_values_round_trip(attribution: str) -> None:
    claim = Claim(
        claim_id="c1", text="t", claim_type="source_fact", attribution=attribution,  # type: ignore[arg-type]
    )
    assert claim.attribution == attribution


def test_applicant_asserted_is_distinguishable_from_independent_evidence() -> None:
    promoter = Claim(
        claim_id="c1", text="Revenue is $4M", claim_type="source_fact",
        attribution="applicant_asserted",
    )
    audited = Claim(
        claim_id="c2", text="Revenue is $4M", claim_type="source_fact",
        attribution="independent_evidence",
    )
    assert promoter.text == audited.text
    assert promoter.attribution != audited.attribution


# ---------------------------------------------------------------------------
# claim ids and conflicts
# ---------------------------------------------------------------------------


def _result(claims: list[Claim]) -> SpecialistResult:
    return SpecialistResult(specialist="cfo", narrative="n", claims=claims)


def test_duplicate_claim_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate claim_id"):
        _result([
            Claim(claim_id="c1", text="a", claim_type="assessment"),
            Claim(claim_id="c1", text="b", claim_type="assessment"),
        ])


def test_conflicts_with_resolves_to_an_existing_claim_id() -> None:
    result = _result([
        Claim(claim_id="c1", text="a", claim_type="conflict", conflicts_with=["c2"]),
        Claim(claim_id="c2", text="b", claim_type="conflict", conflicts_with=["c1"]),
    ])
    assert result.claims[0].conflicts_with == ("c2",)


def test_unknown_conflict_target_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown claim_id"):
        _result([
            Claim(claim_id="c1", text="a", claim_type="conflict", conflicts_with=["nope"]),
        ])


def test_self_conflict_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot conflict with itself"):
        Claim(claim_id="c1", text="a", claim_type="conflict", conflicts_with=["c1"])


def test_empty_claim_id_rejected() -> None:
    with pytest.raises(ValidationError):
        Claim(claim_id="", text="a", claim_type="assessment")


def test_claim_ids_are_local_so_two_results_may_reuse_them() -> None:
    a = _result([Claim(claim_id="c1", text="a", claim_type="assessment")])
    b = _result([Claim(claim_id="c1", text="b", claim_type="assessment")])
    assert a.claims[0].claim_id == b.claims[0].claim_id == "c1"


# ---------------------------------------------------------------------------
# parsing — happy path
# ---------------------------------------------------------------------------


def test_parses_a_well_formed_tool_use_block() -> None:
    message = _message(_tool_use({
        "narrative": "Runway is tighter than reported.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "Cash on hand is $3.03m",
                "claim_type": "source_fact",
                "attribution": "independent_evidence",
                "confidence": "high",
                "evidence": [
                    {"kind": "document", "label": "[Q3-deck.pdf]", "filename": "Q3-deck.pdf"}
                ],
            },
            {
                "claim_id": "c2",
                "text": "Runway is 7.4 months",
                "claim_type": "derived_calculation",
                "calculation": {
                    "inputs": ["cash $3.03m", "net burn $410k/mo"],
                    "method": "cash / net burn",
                    "model_stated_result": "7.4 months",
                },
            },
        ],
    }))

    result = parse_specialist_result(message, specialist="cfo", model="qwen3.5:latest")

    assert result.degraded is False
    assert result.degraded_reason is None
    assert result.specialist == "cfo"
    assert result.model == "qwen3.5:latest"
    assert result.narrative == "Runway is tighter than reported."
    assert [c.claim_id for c in result.claims] == ["c1", "c2"]
    assert result.claims[0].evidence[0].filename == "Q3-deck.pdf"
    assert result.claims[1].calculation is not None
    assert result.claims[1].calculation.model_stated_result == "7.4 months"
    assert result.claims[1].calculation.verified_result is None
    assert result.claims[1].calculation.verification_status == "unverified"


def test_parses_tool_input_delivered_as_a_json_string() -> None:
    message = _message(_tool_use(json.dumps({"narrative": "n", "claims": []})))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is False
    assert result.narrative == "n"


def test_parser_discards_model_supplied_verification_fields() -> None:
    """Defence in depth: the schema omits them, but a model may go off-schema."""
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{
            "claim_id": "c1",
            "text": "IRR is 24%",
            "claim_type": "derived_calculation",
            "calculation": {
                "inputs": ["cash flows"],
                "method": "IRR",
                "model_stated_result": "24%",
                "verified_result": "24%",
                "verification_status": "verified",
            },
        }],
    }))

    result = parse_specialist_result(message, specialist="cfo")

    calc = result.claims[0].calculation
    assert calc is not None
    assert calc.model_stated_result == "24%"
    assert calc.verified_result is None
    assert calc.verification_status == "unverified"
    # Stripping alone is silent; the attempt is also reported, so a model
    # trying to self-certify is visible in production rather than merely
    # corrected.
    assert result.degraded is True
    assert "verification fields" in (result.degraded_reason or "")


def test_parser_discards_model_invented_provenance() -> None:
    """A model cannot know a page it was never shown."""
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{
            "claim_id": "c1",
            "text": "t",
            "claim_type": "source_fact",
            "evidence": [{
                "kind": "document",
                "label": "[deck.pdf]",
                "filename": "deck.pdf",
                "page": 12,
                "url": "https://example.invalid/deck.pdf",
                "retrieved_at": "2026-08-29",
                "sheet": "Sheet1",
                "cell_range": "B2:B9",
                "chunk_index": 3,
            }],
        }],
    }))

    result = parse_specialist_result(message, specialist="cfo")

    ref = result.claims[0].evidence[0]
    assert ref.filename == "deck.pdf"
    assert ref.page is None
    assert ref.url is None
    assert ref.retrieved_at is None
    assert ref.sheet is None
    assert ref.cell_range is None
    assert ref.chunk_index is None


# ---------------------------------------------------------------------------
# parsing — degraded fallback
# ---------------------------------------------------------------------------


def test_text_only_response_degrades_and_keeps_the_prose() -> None:
    """The expected case on backends that silently drop tool_choice."""
    message = _message(_text("Here is my analysis of the runway."))

    result = parse_specialist_result(message, specialist="cfo")

    assert result.degraded is True
    assert result.degraded_reason
    assert result.claims == ()
    assert result.narrative == "Here is my analysis of the runway."


def test_degraded_parsing_never_creates_claims() -> None:
    message = _message(_text("Cash is $3.03m and runway is 7.4 months."))
    assert parse_specialist_result(message, specialist="cfo").claims == ()


def test_wrong_tool_name_degrades() -> None:
    message = _message(
        _text("prose"),
        _tool_use({"narrative": "n"}, name="some_other_tool"),
    )
    result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is True
    assert result.claims == ()


def test_non_object_tool_input_degrades() -> None:
    message = _message(_text("prose"), _tool_use("not json at all"))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is True
    assert result.narrative == "prose"


def test_malformed_claims_degrade_rather_than_raise() -> None:
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{"claim_id": "c1", "text": "t", "claim_type": "not_a_type"}],
    }))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is True
    assert result.claims == ()
    assert "validation" in (result.degraded_reason or "")


def test_problems_survive_a_later_validation_failure() -> None:
    """Losses already detected must not be discarded by a subsequent failure.

    Otherwise an operator sees one claim's dangling reference and never learns
    another claim's evidence was dropped on the way there.
    """
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [
            {"claim_id": "c1", "text": "t1", "claim_type": "source_fact",
             "evidence": {"kind": "document", "label": "x"}},
            {"claim_id": "c2", "text": "t2", "claim_type": "conflict",
             "conflicts_with": ["ghost"]},
        ],
    }))
    result = parse_specialist_result(message, specialist="cfo")
    reason = result.degraded_reason or ""
    assert result.degraded is True
    assert "evidence was dict" in reason
    assert "failed validation" in reason


def test_model_level_validation_failure_names_the_object_not_nothing() -> None:
    """A ``mode="after"`` validator has an empty loc; "at: " alone is useless."""
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [
            {"claim_id": "c1", "text": "a", "claim_type": "assessment"},
            {"claim_id": "c1", "text": "b", "claim_type": "assessment"},
        ],
    }))
    reason = parse_specialist_result(message, specialist="cfo").degraded_reason or ""
    assert "<result>" in reason
    assert not reason.rstrip().endswith("at:")


def test_dangling_conflict_from_the_model_degrades() -> None:
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{
            "claim_id": "c1", "text": "t", "claim_type": "conflict",
            "conflicts_with": ["ghost"],
        }],
    }))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is True
    assert result.claims == ()


def test_empty_message_degrades_without_raising() -> None:
    result = parse_specialist_result(_message(), specialist="cfo")
    assert result.degraded is True
    assert result.claims == ()
    assert result.narrative == ""


def test_parser_never_raises_on_hostile_payloads() -> None:
    """Nothing a model returns may take down the Executive's tool loop."""
    hostile: list[Any] = [
        _message(_tool_use(None)),
        _message(_tool_use([1, 2, 3])),
        _message(_tool_use({"narrative": 42})),
        _message(_tool_use({"narrative": "n", "claims": "not-a-list"})),
        _message(_tool_use({"narrative": "n", "claims": [None, 7, "x"]})),
        _message(_tool_use({})),
        SimpleNamespace(content=None),
    ]
    for message in hostile:
        result = parse_specialist_result(message, specialist="cfo")
        assert isinstance(result, SpecialistResult)


def test_degraded_requires_a_reason() -> None:
    with pytest.raises(ValidationError, match="degraded_reason"):
        SpecialistResult(specialist="cfo", narrative="n", degraded=True)


def test_narrative_falls_back_to_message_text_when_field_is_unusable() -> None:
    """The prose survives, and the fallback is reported rather than hidden."""
    message = _message(_text("real prose"), _tool_use({"narrative": "   ", "claims": []}))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.narrative == "real prose"
    assert result.degraded is True
    assert "narrative" in (result.degraded_reason or "")


# ---------------------------------------------------------------------------
# immutability — the invariants must hold for the object's whole lifetime
# ---------------------------------------------------------------------------


def test_verified_result_cannot_be_assigned_after_construction() -> None:
    """The laundering path: construct clean, then mutate into 'verified'.

    Pydantic ``mode='after'`` validators run on construction only, so without
    frozen models a caller could set both fields post-hoc and re-serialize a
    model's guess as a checked figure.
    """
    calc = CalculationProvenance(inputs=["a"], method="m", model_stated_result="1.4")
    with pytest.raises(ValidationError):
        calc.verified_result = "1.4"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        calc.verification_status = "verified"  # type: ignore[misc]
    assert calc.verified_result is None
    assert calc.verification_status == "unverified"


def test_evidence_provenance_cannot_be_assigned_after_construction() -> None:
    ref = EvidenceRef(kind="document", label="[deck.pdf]")
    for field, value in (("page", 12), ("url", "https://example.invalid"),
                         ("provenance_note", "page 12 of the audited accounts")):
        with pytest.raises(ValidationError):
            setattr(ref, field, value)
        assert getattr(ref, field) is None


def test_conflicts_and_claims_are_immutable_collections() -> None:
    """Tuples, so an in-place append cannot smuggle a dangling reference in."""
    result = SpecialistResult(
        specialist="cfo",
        narrative="n",
        claims=[Claim(claim_id="c1", text="a", claim_type="assessment")],
    )
    assert isinstance(result.claims, tuple)
    assert isinstance(result.claims[0].conflicts_with, tuple)
    with pytest.raises(AttributeError):
        result.claims[0].conflicts_with.append("ghost")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        result.claims.append(Claim(claim_id="c1", text="b", claim_type="assessment"))  # type: ignore[attr-defined]


def test_degraded_flag_cannot_be_flipped_after_construction() -> None:
    result = SpecialistResult(specialist="cfo", narrative="n")
    with pytest.raises(ValidationError):
        result.degraded = True  # type: ignore[misc]
    assert result.degraded is False


def test_model_copy_update_cannot_launder_a_verified_result() -> None:
    """``frozen=True`` blocks setattr but NOT ``model_copy(update=...)``.

    Stock pydantic writes the update unvalidated, which is enough to turn a
    model's guess into a figure that then survives re-validation. The
    ``_ContractModel`` override routes the update back through validation.
    """
    calc = CalculationProvenance(inputs=["a"], method="m", model_stated_result="1.4")
    for update in (
        {"verified_result": "1.4"},
        {"verified_result": "1.4", "verification_status": "unverified"},
        # The matched pair — the only update an attacker would actually write,
        # and the one a lockstep rule lets through.
        {"verified_result": "1.4", "verification_status": "verified"},
        {"verification_status": "verified"},
    ):
        with pytest.raises(ValidationError):
            calc.model_copy(update=update)
    assert calc.verified_result is None


def test_nested_model_copy_cannot_launder_a_verified_calculation() -> None:
    """The same route one level down, via the parent Claim."""
    claim = Claim(
        claim_id="c1", text="IRR is 24%", claim_type="derived_calculation",
        calculation=CalculationProvenance(
            inputs=["cash flows"], method="IRR", model_stated_result="24%"
        ),
    )
    with pytest.raises(ValidationError):
        claim.model_copy(update={"calculation": {
            "inputs": ("cash flows",), "method": "IRR",
            "model_stated_result": "24%",
            "verified_result": "24%", "verification_status": "verified",
        }})


def test_model_copy_update_cannot_fabricate_provenance_or_dangling_conflicts() -> None:
    ref = EvidenceRef(kind="document", label="[deck.pdf]")
    # A legal-but-system-authored field still validates; the guard is that the
    # copy is validated at all, which the dangling-conflict case below proves.
    assert ref.model_copy(update={"page": 12}).page == 12

    result = SpecialistResult(
        specialist="cfo",
        narrative="n",
        claims=[Claim(claim_id="c1", text="a", claim_type="assessment")],
    )
    with pytest.raises(ValidationError):
        result.model_copy(update={
            "claims": (
                Claim(claim_id="c1", text="a", claim_type="conflict",
                      conflicts_with=("ghost",)),
            )
        })
    with pytest.raises(ValidationError):
        result.model_copy(update={"degraded": True})


def test_model_copy_without_update_is_unchanged() -> None:
    result = SpecialistResult(specialist="cfo", narrative="n")
    assert result.model_copy() == result
    assert result.model_copy(deep=True) == result


# ---------------------------------------------------------------------------
# honest degradation — partial loss must be reported, not absorbed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "claims_value",
    [{"c1": {"text": "t"}}, '[{"claim_id":"c1"}]', 42],
)
def test_unreadable_claims_container_degrades(claims_value: Any) -> None:
    message = _message(_tool_use({"narrative": "n", "claims": claims_value}))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is True
    assert "claims" in (result.degraded_reason or "")


def test_dropped_claim_entries_are_reported() -> None:
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [_claim(), "x", None, 5],
    }))
    result = parse_specialist_result(message, specialist="cfo")
    assert len(result.claims) == 1
    assert result.degraded is True
    assert "claim entr" in (result.degraded_reason or "")


def test_unreadable_evidence_container_degrades_the_result() -> None:
    """A sourced claim arriving with zero refs must not look evidence-free.

    Otherwise a claim marked ``independent_evidence`` reaches the Executive
    with no evidence and no sign that any was lost.
    """
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{
            "claim_id": "c1",
            "text": "Revenue is $4M",
            "claim_type": "source_fact",
            "attribution": "independent_evidence",
            "evidence": {"kind": "document", "label": "[deck.pdf]"},
        }],
    }))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.claims[0].evidence == ()
    assert result.degraded is True
    assert "evidence" in (result.degraded_reason or "")


def test_dropped_evidence_entries_are_reported() -> None:
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{
            "claim_id": "c1", "text": "t", "claim_type": "source_fact",
            "evidence": ["[deck.pdf]", {"kind": "document", "label": "[deck.pdf]"}],
        }],
    }))
    result = parse_specialist_result(message, specialist="cfo")
    assert len(result.claims[0].evidence) == 1
    assert result.degraded is True


def test_provenance_note_from_the_model_is_stripped() -> None:
    """The free-text channel that would otherwise bypass the field scrubber."""
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{
            "claim_id": "c1", "text": "t", "claim_type": "source_fact",
            "evidence": [{
                "kind": "document",
                "label": "[deck.pdf]",
                "provenance_note": "page 12, sheet FY24, from https://registry.invalid",
            }],
        }],
    }))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.claims[0].evidence[0].provenance_note is None


def test_multiple_tool_blocks_are_reported_not_silently_dropped() -> None:
    message = _message(
        _tool_use({"narrative": "first", "claims": []}),
        _tool_use({"narrative": "second", "claims": []}),
    )
    result = parse_specialist_result(message, specialist="cfo")
    assert result.narrative == "first"
    assert result.degraded is True
    assert "2 emit_specialist_result blocks" in (result.degraded_reason or "")


def test_degraded_reason_distinguishes_unreadable_payload_from_absent_block() -> None:
    """An operator debugging a truncating provider must not be told
    'the model never called the tool'."""
    absent = parse_specialist_result(_message(_text("p")), specialist="cfo")
    assert "no emit_specialist_result tool_use block" in (absent.degraded_reason or "")

    broken = parse_specialist_result(
        _message(_tool_use("{not json")), specialist="cfo"
    )
    assert "not valid JSON" in (broken.degraded_reason or "")

    wrong_type = parse_specialist_result(
        _message(_tool_use([1, 2, 3])), specialist="cfo"
    )
    assert "expected object" in (wrong_type.degraded_reason or "")


@pytest.mark.parametrize("calculation", ["garbage", 42, [1, 2, 3], True])
def test_unreadable_calculation_degrades(calculation: Any) -> None:
    """Same silent-loss class as evidence, on the calculation path.

    A well-formed calculation on a non-derived claim fails loudly in the Claim
    validator, so a malformed one must not be treated more leniently.
    """
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{
            "claim_id": "c1", "text": "Revenue is $4M", "claim_type": "assessment",
            "calculation": calculation,
        }],
    }))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.claims[0].calculation is None
    assert result.degraded is True
    assert "calculation" in (result.degraded_reason or "")


def test_stripped_fabricated_provenance_is_reported() -> None:
    """Correcting a hallucination silently would make it undetectable in prod."""
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{
            "claim_id": "c1", "text": "t", "claim_type": "source_fact",
            "evidence": [{
                "kind": "document", "label": "[deck.pdf]",
                "page": 12, "url": "https://example.invalid",
            }],
        }],
    }))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.claims[0].evidence[0].page is None
    assert result.degraded is True
    reason = result.degraded_reason or ""
    assert "discarded model-asserted provenance" in reason
    assert "page" in reason and "url" in reason


@pytest.mark.parametrize("payload", [{"claims": []}, {"narrative": None, "claims": []}])
def test_absent_or_null_narrative_is_reported(payload: dict[str, Any]) -> None:
    """``narrative`` is required in the tool schema; violating that must show."""
    message = _message(_text("fallback prose"), _tool_use(payload))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.narrative == "fallback prose"
    assert result.degraded is True
    assert "narrative" in (result.degraded_reason or "")


def test_repeated_problem_categories_are_deduplicated() -> None:
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [
            {"claim_id": "c1", "text": "t", "claim_type": "source_fact",
             "evidence": {"kind": "document", "label": "x"}},
            {"claim_id": "c2", "text": "t", "claim_type": "source_fact",
             "evidence": {"kind": "document", "label": "y"}},
        ],
    }))
    reason = parse_specialist_result(message, specialist="cfo").degraded_reason or ""
    assert reason.count("evidence was dict") == 1


def test_degraded_reason_never_echoes_a_model_supplied_field_name() -> None:
    """An off-schema KEY is model-authored text and must not be echoed.

    ``extra="forbid"`` makes this the dominant failure mode, and the offending
    key lands in the pydantic error path — so the path leaf, not just the
    value, has to be scrubbed.
    """
    secret_key = "REVENUE_IS_4200000_ACME_CONFIDENTIAL"
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{
            "claim_id": "c1", "text": "t", "claim_type": "source_fact",
            secret_key: 1,
        }],
    }))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is True
    assert secret_key not in (result.degraded_reason or "")
    assert "<extra>" in (result.degraded_reason or "")


def test_degraded_reason_cannot_be_used_to_forge_log_lines() -> None:
    """A newline in a model-influenced key would otherwise inject a log line."""
    forged = "x\n2026-01-01 CRITICAL audit: verified_result=1.4 PASSED"
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{"claim_id": "c1", "text": "t", "claim_type": "source_fact",
                    forged: 1}],
    }))
    reason = parse_specialist_result(message, specialist="cfo").degraded_reason or ""
    assert "\n" not in reason
    assert "CRITICAL" not in reason


@pytest.mark.parametrize(
    "separator",
    [" ", " ", "", "", "‮"],
)
def test_non_ascii_line_terminators_are_stripped(separator: str) -> None:
    """``ch >= " "`` compares codepoints and keeps these.

    U+2028/U+2029/U+0085 are line breaks to ``str.splitlines`` — enough to forge
    a log line. U+009B is an ANSI escape introducer for anyone tailing logs, and
    U+202E reorders display in any UI that renders the field.
    """
    forged = f"a{separator}VERIFIED: calculation checked by authority"
    message = _message(_tool_use({"narrative": "n", "claims": [], forged: 1}))
    reason = parse_specialist_result(message, specialist="cfo").degraded_reason or ""
    assert separator not in reason
    assert len(reason.splitlines()) == 1


def test_aggregate_degraded_reason_is_bounded() -> None:
    """Each contributor is capped; the joined string needs its own cap.

    Per-claim counts differ, so dedup cannot collapse them.
    """
    claims = [
        {"claim_id": f"c{i}", "text": "t", "claim_type": "source_fact",
         "evidence": [None] * (i + 1)}
        for i in range(400)
    ]
    message = _message(_tool_use({"narrative": "n", "claims": claims}))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is True
    assert len(result.degraded_reason or "") <= 420
    assert len(result.claims) == 400  # the claims themselves are all kept


def test_validation_failure_is_not_logged_with_raw_model_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``str(ValidationError)`` embeds raw loc components and input_value.

    Sanitising ``degraded_reason`` and then logging the raw exception one line
    above would undo the whole defence.
    """
    secret = "FOUNDER_OWNS_62_PERCENT_CONFIDENTIAL"
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{"claim_id": "c1", "text": secret, "claim_type": "bogus_type"}],
    }))
    with caplog.at_level("WARNING"):
        result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is True
    assert secret not in caplog.text


def test_forged_nested_calculation_is_rejected_by_the_parent() -> None:
    """``revalidate_instances='always'`` re-checks a nested model at each boundary.

    Without it pydantic trusts an already-constructed child, so a
    ``model_construct``-forged provenance would ride into a fully-validating
    parent untouched.
    """
    forged = CalculationProvenance.model_construct(
        inputs=(), method="m",
        verified_result="42", verification_status="verified",
    )
    with pytest.raises(ValidationError):
        Claim(
            claim_id="c1", text="t", claim_type="derived_calculation",
            calculation=forged,
        )


def test_text_block_with_non_string_text_is_counted_as_unreadable() -> None:
    """Readable-but-unusable prose is the same loss as a block that raised."""
    message = SimpleNamespace(content=[SimpleNamespace(type="text", text=["prose"])])
    result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is True
    assert "could not be read" in (result.degraded_reason or "")


def test_unknown_top_level_keys_are_counted_not_echoed() -> None:
    """A payload key is model-authored text; the claim path already refuses to
    echo one, and `;` here would forge a second reason segment."""
    secret = "REVENUE_IS_4200000_ACME_CONFIDENTIAL"
    forged = "x; VERIFIED by calculation authority"
    message = _message(_tool_use({"narrative": "n", secret: 1, forged: 2}))
    reason = parse_specialist_result(message, specialist="cfo").degraded_reason or ""
    assert secret not in reason
    assert "VERIFIED" not in reason
    assert "2 unrecognized top-level payload key(s)" in reason


def test_lone_surrogate_key_cannot_poison_the_reason() -> None:
    """A lone surrogate raises on ``.encode('utf-8')`` — it would crash whoever
    persists the result. Reachable via the JSON-string payload path."""
    payload = json.dumps({"narrative": "n", "claims": []})[:-1] + ', "\\ud800BAD": 1}'
    result = parse_specialist_result(
        _message(_tool_use(payload)), specialist="cfo"
    )
    (result.degraded_reason or "").encode("utf-8")  # must not raise


def test_problems_survive_the_no_usable_narrative_exit() -> None:
    """The sibling of the branch above — it dropped everything detected so far.

    The operator would be told the payload had no narrative and never learn the
    parser used the first of three tool blocks, when a later one carried the
    real answer.
    """
    message = _message(
        _tool_use({"Narrative": "n", "Claims": []}),
        _tool_use({"narrative": "the real answer", "claims": []}),
    )
    result = parse_specialist_result(message, specialist="cfo")
    reason = result.degraded_reason or ""
    assert result.degraded is True
    assert "2 emit_specialist_result blocks" in reason
    assert "unrecognized top-level payload key" in reason
    assert "no usable narrative" in reason


def test_no_payload_branch_reason_is_bounded() -> None:
    """The branch that fires when the response is MOST broken was the one
    unbounded reason site."""
    blocks = [_tool_use([1, 2, 3]) for _ in range(500)]
    result = parse_specialist_result(_message(*blocks), specialist="cfo")
    assert result.degraded is True
    assert len(result.degraded_reason or "") <= 420


def test_dict_shaped_content_blocks_are_read() -> None:
    """Raw provider JSON / a model_dump() / a replayed cache entry.

    ``getattr`` returns the default for a dict, so every block failed the type
    check: narrative and claims both vanished while the reason blamed the model
    for never calling the tool.
    """
    message = SimpleNamespace(content=[
        {"type": "text", "text": "prose"},
        {"type": "tool_use", "id": "t1", "name": TOOL_NAME,
         "input": {"narrative": "n",
                   "claims": [{"claim_id": "c1", "text": "t",
                               "claim_type": "source_fact"}]}},
    ])
    result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is False
    assert result.narrative == "n"
    assert [c.claim_id for c in result.claims] == ["c1"]


def test_degraded_reason_is_length_bounded() -> None:
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{"claim_id": "c1", "text": "t", "claim_type": "source_fact",
                    "K" * 5000: 1}],
    }))
    reason = parse_specialist_result(message, specialist="cfo").degraded_reason or ""
    assert len(reason) < 500


def test_degraded_reason_never_echoes_model_supplied_values() -> None:
    """degraded_reason is logged and persisted; it must not carry document text."""
    secret = "ACME-CONFIDENTIAL-REVENUE-4200000"
    message = _message(_tool_use({
        "narrative": "n",
        "claims": [{"claim_id": "c1", "text": secret, "claim_type": "not_a_type"}],
    }))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is True
    assert secret not in (result.degraded_reason or "")
    assert "claims.0.claim_type" in (result.degraded_reason or "")


# ---------------------------------------------------------------------------
# provider message shapes
# ---------------------------------------------------------------------------


def test_plain_string_content_is_read_as_narrative() -> None:
    """The OpenAI message shape: content is a str, not a block list.

    Iterating it character by character would match no block and silently
    discard the specialist's entire answer.
    """
    message = SimpleNamespace(content="Runway is 7.4 months at current burn.")
    result = parse_specialist_result(message, specialist="cfo")
    assert result.narrative == "Runway is 7.4 months at current burn."
    assert result.degraded is True
    assert result.claims == ()


@pytest.mark.parametrize("content", [42, object(), 3.5, True])
def test_non_iterable_content_degrades_without_raising(content: Any) -> None:
    result = parse_specialist_result(SimpleNamespace(content=content), specialist="cfo")
    assert isinstance(result, SpecialistResult)
    assert result.degraded is True


def test_generator_content_is_not_exhausted_before_the_tool_block_is_found() -> None:
    """Iterating content twice would make the tool block invisible."""
    blocks = [_text("prose"), _tool_use({"narrative": "n", "claims": []})]
    message = SimpleNamespace(content=(b for b in blocks))
    result = parse_specialist_result(message, specialist="cfo")
    assert result.degraded is False
    assert result.narrative == "n"


@pytest.mark.parametrize(
    "payload",
    [
        {"Claims": [{"claim_id": "c1", "text": "t", "claim_type": "source_fact"}],
         "narrative": "n"},
        {"claim": [{"claim_id": "c1", "text": "t", "claim_type": "source_fact"}],
         "narrative": "n"},
        {"narrative": "n", "claims": [], "recommendations": ["x"]},
    ],
)
def test_unrecognized_top_level_payload_keys_are_reported(payload: dict[str, Any]) -> None:
    """Claim-level extras hit extra='forbid'; the top level is read key-by-key.

    Without this check a model emitting findings under ``Claims`` — a
    capitalisation slip, or an instruction planted in an indexed document —
    hands the Executive ``claims=()`` with ``degraded=False``, which reads as
    "the specialist genuinely made no claims".
    """
    result = parse_specialist_result(_message(_tool_use(payload)), specialist="cfo")
    assert result.degraded is True
    assert "unrecognized top-level payload key" in (result.degraded_reason or "")


def test_known_payload_keys_alone_do_not_degrade() -> None:
    result = parse_specialist_result(
        _message(_tool_use({"narrative": "n", "claims": []})), specialist="cfo"
    )
    assert result.degraded is False


@pytest.mark.parametrize(
    ("specialist", "model"),
    [("", ""), ("   ", ""), (None, ""), ("cfo", None), (None, None)],
)
def test_bad_caller_arguments_degrade_instead_of_raising(
    specialist: Any, model: Any
) -> None:
    """The last-resort path must not be able to double-fault.

    ``_degraded`` builds a validated model, so an unusable ``specialist`` would
    raise inside the try, be caught, and raise again from the retry — escaping
    the 'never raises' contract exactly when something has already gone wrong.
    """
    result = parse_specialist_result(
        _message(_text("prose")), specialist=specialist, model=model
    )
    assert isinstance(result, SpecialistResult)
    assert result.degraded is True
    assert result.specialist  # coerced to a usable placeholder


def test_unreadable_content_blocks_are_reported() -> None:
    """Prose we could not read is lost structure like any other."""

    class Hostile:
        @property
        def type(self) -> str:
            raise RuntimeError("boom")

    message = SimpleNamespace(content=[_text("kept"), Hostile()])
    result = parse_specialist_result(message, specialist="cfo")
    assert result.narrative == "kept"
    assert result.degraded is True
    assert "could not be read" in (result.degraded_reason or "")


def test_block_raising_on_attribute_access_does_not_escape() -> None:
    class Hostile:
        @property
        def type(self) -> str:
            raise RuntimeError("boom")

    result = parse_specialist_result(
        SimpleNamespace(content=[Hostile()]), specialist="cfo"
    )
    assert isinstance(result, SpecialistResult)
    assert result.degraded is True


# ---------------------------------------------------------------------------
# compatibility renderer
# ---------------------------------------------------------------------------


def test_render_for_executive_is_exactly_the_narrative() -> None:
    result = SpecialistResult(
        specialist="cfo",
        narrative="Runway is 7.4 months at current burn.",
        claims=[Claim(claim_id="c1", text="t", claim_type="assessment")],
    )
    assert render_for_executive(result) == "Runway is 7.4 months at current burn."


def test_render_adds_nothing_when_claims_are_present() -> None:
    """Appending claims here would change the Executive prompt — out of scope."""
    plain = SpecialistResult(specialist="cfo", narrative="same text")
    annotated = SpecialistResult(
        specialist="cfo",
        narrative="same text",
        claims=[
            Claim(claim_id="c1", text="a", claim_type="source_fact"),
            Claim(claim_id="c2", text="b", claim_type="unsupported"),
        ],
    )
    assert render_for_executive(plain) == render_for_executive(annotated)


def test_render_of_a_degraded_result_is_still_the_prose() -> None:
    message = _message(_text("prose only"))
    result = parse_specialist_result(message, specialist="cfo")
    assert render_for_executive(result) == "prose only"


# ---------------------------------------------------------------------------
# tool schema hygiene
# ---------------------------------------------------------------------------


def test_tool_schema_is_json_serialisable() -> None:
    assert json.loads(json.dumps(EMIT_SPECIALIST_RESULT_TOOL)) == EMIT_SPECIALIST_RESULT_TOOL


def test_tool_schema_properties_are_name_sorted_for_cache_stability() -> None:
    """The cached tool block must be byte-stable across processes."""

    def _assert_sorted(node: Any) -> None:
        if not isinstance(node, dict):
            return
        props = node.get("properties")
        if isinstance(props, dict):
            assert list(props) == sorted(props), list(props)
        for value in node.values():
            if isinstance(value, dict):
                _assert_sorted(value)
            elif isinstance(value, list):
                for entry in value:
                    _assert_sorted(entry)

    _assert_sorted(EMIT_SPECIALIST_RESULT_TOOL["input_schema"])


def test_tool_schema_requires_narrative() -> None:
    assert EMIT_SPECIALIST_RESULT_TOOL["input_schema"]["required"] == ["narrative"]


def test_every_tool_schema_enum_matches_its_contract_literal() -> None:
    """No enum may drift between the tool schema and the pydantic models.

    Derived from the ``Literal`` aliases with ``get_args`` rather than
    hardcoded, so adding a value to a Literal fails here until the schema is
    updated too. Hardcoding the expected sets would let the pair drift together
    away from the contract without any test noticing.

    Drift is not cosmetic: the tool block is prompt-cached, so a schema that
    permits a value the model then emits — and the contract rejects — turns
    into a degraded result rather than a loud error.
    """
    claim_schema = (
        EMIT_SPECIALIST_RESULT_TOOL["input_schema"]["properties"]["claims"]["items"]
    )
    evidence_schema = claim_schema["properties"]["evidence"]["items"]

    pairs = [
        ("claim_type", claim_schema["properties"]["claim_type"], ClaimType),
        ("attribution", claim_schema["properties"]["attribution"], Attribution),
        ("confidence", claim_schema["properties"]["confidence"], ConfidenceLevel),
        ("evidence.kind", evidence_schema["properties"]["kind"], EvidenceKind),
    ]
    for name, node, literal in pairs:
        assert set(node["enum"]) == set(get_args(literal)), name


def test_verification_status_literal_is_absent_from_the_tool_schema() -> None:
    """The one contract enum the model must NOT be offered.

    Every other Literal is mirrored into the schema; this one is deliberately
    withheld so the model has no field in which to assert verification.
    """
    schema = json.dumps(EMIT_SPECIALIST_RESULT_TOOL)
    for value in get_args(VerificationStatus):
        assert f'"{value}"' not in schema, value


# ---------------------------------------------------------------------------
# isolation: this slice is not wired into production
# ---------------------------------------------------------------------------


def test_only_cfo_imports_the_contract() -> None:
    """CFO is the single wired specialist; the migration must not spread.

    Slice 1 shipped this module as dead code and asserted zero importers. The
    CFO wiring slice made ``agents/finance.py`` the one exception, and this test
    now pins that exact set — so a second specialist (or a workflow, or the
    Executive) importing the contract fails here rather than quietly widening
    the migration.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "openexecutive"
    importers = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "specialists/" not in path.as_posix()
        and "result_contract" in path.read_text(encoding="utf-8")
    )
    assert importers == ["agents/finance.py"], importers
