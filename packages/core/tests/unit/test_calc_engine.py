"""Engine semantics: arity, dimensions, conversions, batching, and containment.

The GreenHarvest fixtures prove the engine gets the measured case right. These
prove it *refuses* everything it should — which is the larger half, since a
calculator that answers every question is worse than none.
"""
from __future__ import annotations

import decimal
import inspect
from datetime import UTC
from decimal import Decimal, localcontext
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from openexecutive.calc import contract as contract_mod
from openexecutive.calc import engine as engine_mod
from openexecutive.calc import fingerprint as fp_mod
from openexecutive.calc.authority import current_authority
from openexecutive.calc.contract import (
    CalculationBatch,
    CalculationRequest,
    Correlation,
    InputEvidenceSummary,
    Operand,
    OperationId,
)
from openexecutive.calc.engine import (
    LIMITS,
    EngineError,
    execute,
    execute_batch,
    signature_for,
)
from openexecutive.calc.numeric import NumericPolicyError, parse_numeric
from openexecutive.calc.units import Unit, known_unit_codes

AT = "2026-09-02T00:00:00Z"
TND = "currency:TND"

_V1_OPERATIONS = (
    "add", "subtract", "multiply", "divide", "sum_components", "percentage_of",
    "percentage_point_difference", "ratio", "weighted_average", "variance",
    "convert_unit", "interval_implied_total",
)


def _corr(case: str = "c1", run: str = "r1", claim: str | None = None) -> Correlation:
    return Correlation(specialist="cfo", case_id=case, run_id=run, claim_id=claim)


def _op(oid: str, value: str, unit: str, role: str = "input") -> Operand:
    return Operand(operand_id=oid, label=oid, value=value, unit=Unit(code=unit),
                   basis="applicant_stated", role=role)  # type: ignore[arg-type]


def _request(operation: str, operands: list[Operand], target: str | None = "one",
             scale: int = 2, **kw: object) -> CalculationRequest:
    return CalculationRequest(
        request_id=kw.pop("request_id", "r1"),  # type: ignore[arg-type]
        operation=operation,  # type: ignore[arg-type]
        operands=tuple(operands),
        target_unit=Unit(code=target) if target else None,
        scale=scale, purpose="test", correlation=kw.pop("correlation", _corr()),  # type: ignore[arg-type]
        **kw,  # type: ignore[arg-type]
    )


def _run(operation: str, operands: list[Operand], target: str | None = "one",
         scale: int = 2, **kw: object):
    exec_kw = {k: kw.pop(k) for k in ("evidence", "time_conversion_policy", "weight_policy")
               if k in kw}
    request = _request(operation, operands, target, scale, **kw)
    return execute(request, computed_at=AT, **exec_kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Signatures and arity
# ---------------------------------------------------------------------------


def test_exactly_the_twelve_authorised_operations_have_signatures() -> None:
    assert set(OperationId.__args__) == set(_V1_OPERATIONS)  # type: ignore[attr-defined]
    for name in _V1_OPERATIONS:
        assert signature_for(name) is not None, name


def test_unknown_operation_resolves_to_none_never_a_callable() -> None:
    for name in ("irr", "npv", "xirr", "eval", "__import__", ""):
        assert signature_for(name) is None


def test_unknown_operation_is_reported_not_coerced() -> None:
    """The contract rejects it first; the engine has its own guard behind that."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _request("irr", [_op("o1", "1", "one")])
    signature = signature_for("irr")
    assert signature is None


@pytest.mark.parametrize(
    "operation,too_few,too_many",
    [
        ("add", 1, 3), ("subtract", 1, 3), ("multiply", 1, 3), ("divide", 1, 3),
        ("percentage_of", 1, 3), ("percentage_point_difference", 1, 3),
        ("ratio", 1, 3), ("variance", 2, 3), ("convert_unit", 2, 3),
        ("interval_implied_total", 3, 5),
    ],
)
def test_wrong_arity_is_rejected_for_every_operation(
    operation: str, too_few: int, too_many: int
) -> None:
    """Every operation refuses both too few and too many input operands."""
    for count in (too_few, too_many):
        # A request with zero operands is refused by the contract before the
        # engine is reached, so arity cases here start at one.
        operands = [_op(f"o{i}", "1", "one") for i in range(count)]
        result = _run(operation, operands, "one")
        assert result.arithmetic_status in ("INVALID_INPUT", "UNIT_MISMATCH"), (
            f"{operation} with {count} operands: {result.arithmetic_status}"
        )
        assert result.result_value is None


def test_sum_components_accepts_one_to_sixty_four() -> None:
    one = _run("sum_components", [_op("o0", "5", "pct")], "pct", scale=0)
    assert one.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert one.result_value == "5"
    full = _run("sum_components",
                [_op(f"o{i}", "1", "pct") for i in range(LIMITS.max_operands)],
                "pct", scale=0)
    assert full.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert full.result_value == str(LIMITS.max_operands)


def test_too_many_operands_is_rejected_by_the_contract() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _request("sum_components",
                 [_op(f"o{i}", "1", "pct") for i in range(LIMITS.max_operands + 1)], "pct")


# ---------------------------------------------------------------------------
# Target unit
# ---------------------------------------------------------------------------


def test_missing_target_unit_is_rejected() -> None:
    result = _run("add", [_op("a", "1", "one"), _op("b", "2", "one")], None)
    assert result.arithmetic_status == "INVALID_INPUT"
    assert "target_unit" in result.errors[0].detail


def test_unexpected_target_dimension_is_rejected() -> None:
    result = _run("add", [_op("a", "1", "kg"), _op("b", "2", "kg")], TND)
    assert result.arithmetic_status == "UNIT_MISMATCH"


def test_engine_never_fabricates_a_target_unit() -> None:
    result = _run("convert_unit", [_op("a", "11", "ha")], None)
    assert result.arithmetic_status == "INVALID_INPUT"
    assert result.result_unit is None


# ---------------------------------------------------------------------------
# Dimensional rules
# ---------------------------------------------------------------------------


def test_cross_currency_addition_is_refused() -> None:
    result = _run("add", [_op("a", "1", TND), _op("b", "1", "currency:EUR")], TND)
    assert result.arithmetic_status == "UNIT_MISMATCH"
    assert "exchange-rate authority" in result.errors[0].detail


def test_same_currency_division_yields_a_dimensionless_ratio() -> None:
    result = _run("divide", [_op("a", "10", TND), _op("b", "4", TND)], "one", scale=2)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert result.result_value == "2.50"


def test_cross_currency_division_is_refused() -> None:
    result = _run("divide", [_op("a", "10", TND), _op("b", "4", "currency:EUR")], "one")
    assert result.arithmetic_status == "UNIT_MISMATCH"


def test_mass_plus_area_is_refused() -> None:
    result = _run("add", [_op("a", "1", "kg"), _op("b", "1", "m2")], "kg")
    assert result.arithmetic_status == "UNIT_MISMATCH"


def test_pct_and_pct_point_never_mix() -> None:
    added = _run("add", [_op("a", "60", "pct"), _op("b", "25", "pct_point")], "pct")
    assert added.arithmetic_status == "UNIT_MISMATCH"
    converted = _run("convert_unit", [_op("a", "60", "pct")], "pct_point")
    assert converted.arithmetic_status == "UNIT_MISMATCH"
    assert "not scale variants" in converted.errors[0].detail


def test_percentage_point_difference_requires_percentage_inputs() -> None:
    result = _run("percentage_point_difference",
                  [_op("a", "60", "kg"), _op("b", "35", "kg")], "pct_point")
    assert result.arithmetic_status == "UNIT_MISMATCH"


def test_percentage_of_may_not_produce_percentage_points() -> None:
    result = _run("percentage_of", [_op("a", "1", "kg"), _op("b", "2", "kg")], "pct_point")
    assert result.arithmetic_status == "UNIT_MISMATCH"


def test_undeclared_multiplication_is_refused() -> None:
    result = _run("multiply", [_op("a", "2", "kg"), _op("b", "3", "t")], "kg")
    assert result.arithmetic_status == "UNIT_MISMATCH"
    assert "no declared result dimension" in result.errors[0].detail


def test_undeclared_division_is_refused() -> None:
    result = _run("divide", [_op("a", "2", "m2"), _op("b", "3", "kg")], "one")
    assert result.arithmetic_status == "UNIT_MISMATCH"


def test_declared_multiplication_and_its_inverse_round_trip() -> None:
    product = _run("multiply", [_op("y", "52", "kg_per_m2"), _op("a", "110000", "m2")],
                   "kg", scale=0)
    assert product.result_value == "5720000"
    back = _run("divide", [_op("m", "5720000", "kg"), _op("a", "110000", "m2")],
                "kg_per_m2", scale=0)
    assert back.result_value == "52"


def test_dimensionless_scaling_is_allowed_in_both_directions() -> None:
    scaled = _run("multiply", [_op("v", "1000", TND), _op("k", "3", "one")], TND, scale=0)
    assert scaled.result_value == "3000"
    divided = _run("divide", [_op("v", "3000", TND), _op("k", "3", "one")], TND, scale=0)
    assert divided.result_value == "1000"


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,source,target,expected",
    [("11", "ha", "m2", "110000"), ("110000", "m2", "ha", "11"),
     ("5720000", "kg", "t", "5720"), ("5720", "t", "kg", "5720000"),
     ("42", TND, TND, "42")],
)
def test_exact_conversions(value: str, source: str, target: str, expected: str) -> None:
    result = _run("convert_unit", [_op("v", value, source)], target, scale=0)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert result.result_value == expected


def test_month_to_year_requires_an_explicit_policy() -> None:
    without = _run("convert_unit", [_op("v", "12", "month")], "year", scale=0)
    assert without.arithmetic_status == "INVALID_INPUT"
    assert "time_conversion_policy" in without.errors[0].detail
    with_policy = _run("convert_unit", [_op("v", "12", "month")], "year", scale=0,
                       time_conversion_policy="calendar_12_months_per_year")
    assert with_policy.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert with_policy.result_value == "1"


def test_monthly_and_annual_cannot_be_added_without_a_policy() -> None:
    result = _run("add", [_op("a", "12", "month"), _op("b", "1", "year")], "month", scale=0)
    assert result.arithmetic_status == "INVALID_INPUT"


def test_currency_conversion_is_prohibited() -> None:
    result = _run("convert_unit", [_op("v", "1", TND)], "currency:EUR")
    assert result.arithmetic_status == "UNIT_MISMATCH"


# ---------------------------------------------------------------------------
# Operation-specific semantics
# ---------------------------------------------------------------------------


def test_subtract_preserves_operand_order() -> None:
    forward = _run("subtract", [_op("a", "10", "one"), _op("b", "3", "one")], "one", scale=0)
    backward = _run("subtract", [_op("a", "3", "one"), _op("b", "10", "one")], "one", scale=0)
    assert forward.result_value == "7"
    assert backward.result_value == "-7"
    assert forward.fingerprint != backward.fingerprint


def test_ratio_preserves_direction() -> None:
    result = _run("ratio", [_op("n", "10", "kg"), _op("d", "4", "kg")], "one", scale=2)
    assert result.result_value == "2.50"
    flipped = _run("ratio", [_op("n", "4", "kg"), _op("d", "10", "kg")], "one", scale=2)
    assert flipped.result_value == "0.40"


def test_division_by_zero_is_typed_for_every_dividing_operation() -> None:
    for operation, target in (("divide", "one"), ("ratio", "one"), ("percentage_of", "pct")):
        result = _run(operation, [_op("n", "1", "kg"), _op("d", "0", "kg")], target)
        assert result.arithmetic_status == "DIVISION_BY_ZERO", operation
        assert result.result_value is None


def test_weighted_average_uses_roles_not_positions() -> None:
    result = _run(
        "weighted_average",
        [_op("v1", "4.80", TND), _op("v2", "3.80", TND),
         _op("w1", "65", "pct", role="stated_comparison"),
         _op("w2", "35", "pct", role="stated_comparison")],
        TND, scale=4,
    )
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    # 4.80*65 + 3.80*35 = 312 + 133 = 445; / 100 = 4.45
    assert result.result_value == "4.4500"


def test_weighted_average_rejects_mismatched_counts_and_mixed_weight_units() -> None:
    mismatched = _run("weighted_average",
                      [_op("v1", "1", TND), _op("v2", "2", TND),
                       _op("w1", "1", "one", role="stated_comparison")], TND)
    assert mismatched.arithmetic_status == "INVALID_INPUT"
    mixed = _run("weighted_average",
                 [_op("v1", "1", TND), _op("v2", "2", TND),
                  _op("w1", "1", "one", role="stated_comparison"),
                  _op("w2", "50", "pct", role="stated_comparison")], TND)
    assert mixed.arithmetic_status == "UNIT_MISMATCH"


def test_weighted_average_rejects_zero_total_weight() -> None:
    """Zero weights, all non-negative — the sign guard cannot mask this one."""
    result = _run("weighted_average",
                  [_op("v1", "1", TND), _op("v2", "2", TND),
                   _op("w1", "0", "one", role="stated_comparison"),
                   _op("w2", "0", "one", role="stated_comparison")], TND)
    assert result.arithmetic_status == "DIVISION_BY_ZERO"


def test_weights_need_not_total_one_hundred() -> None:
    result = _run("weighted_average",
                  [_op("v1", "10", TND), _op("v2", "20", TND),
                   _op("w1", "1", "one", role="stated_comparison"),
                   _op("w2", "1", "one", role="stated_comparison")], TND, scale=0)
    assert result.result_value == "15"


@pytest.mark.parametrize(
    "stated,calculated,expected",
    [("100", "100", "EXACT_MATCH"), ("100", "100.00001", "WITHIN_TOLERANCE"),
     ("100", "150", "CONFLICT_DETECTED"), ("572", "5720", "ORDER_OF_MAGNITUDE"),
     ("5720", "572", "ORDER_OF_MAGNITUDE"), ("100", "-100", "SIGN_MISMATCH")],
)
def test_variance_classification(stated: str, calculated: str, expected: str) -> None:
    result = _run("variance",
                  [_op("calc", calculated, "t"),
                   _op("stated", stated, "t", role="stated_comparison")], "t", scale=5)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert result.conflict == expected


def test_variance_requires_compatible_units() -> None:
    result = _run("variance",
                  [_op("calc", "5720", "t"),
                   _op("stated", "572", TND, role="stated_comparison")], "t")
    assert result.arithmetic_status == "UNIT_MISMATCH"


def test_variance_keeps_both_values_and_does_not_pick_a_winner() -> None:
    result = _run("variance",
                  [_op("calc", "5720", "t"),
                   _op("stated", "572", "t", role="stated_comparison")], "t", scale=0)
    assert result.stated_value == "572"
    assert result.result_value == "5148"
    assert result.ratio == "10"
    # Nothing in the record asserts which figure is correct.
    assert result.conflict == "ORDER_OF_MAGNITUDE"


def test_interval_rejects_bad_ordering_and_non_positive_coverage() -> None:
    reversed_volume = _run("interval_implied_total",
                           [_op("a", "3500", "t"), _op("b", "3200", "t"),
                            _op("c", "55", "pct"), _op("d", "60", "pct")], "t")
    assert reversed_volume.arithmetic_status == "INVALID_INPUT"
    reversed_coverage = _run("interval_implied_total",
                             [_op("a", "3200", "t"), _op("b", "3500", "t"),
                              _op("c", "60", "pct"), _op("d", "55", "pct")], "t")
    assert reversed_coverage.arithmetic_status == "INVALID_INPUT"
    for low, high in (("0", "60"), ("-5", "60"), ("-10", "-5")):
        crossing = _run("interval_implied_total",
                        [_op("a", "3200", "t"), _op("b", "3500", "t"),
                         _op("c", low, "pct"), _op("d", high, "pct")], "t")
        assert crossing.arithmetic_status == "INVALID_INPUT", (low, high)


# ---------------------------------------------------------------------------
# Result shape and authority
# ---------------------------------------------------------------------------


def test_result_carries_exact_and_rounded_values_with_their_policy() -> None:
    result = _run("divide", [_op("a", "1", "one"), _op("b", "3", "one")], "one", scale=4)
    assert result.result_value == "0.3333"
    assert result.exact_result is not None and result.exact_result.startswith("0.3333333333")
    assert result.scale_applied == 4
    assert result.rounding_applied == "ROUND_HALF_EVEN"


def test_engine_stamps_its_own_authority() -> None:
    from openexecutive.calc.authority import AUTHORITY_ID, AUTHORITY_VERSION

    result = _run("add", [_op("a", "1", "one"), _op("b", "1", "one")], "one")
    assert result.authority.authority_id == AUTHORITY_ID
    assert result.authority.authority_version == AUTHORITY_VERSION


def test_execute_accepts_no_authority_field_from_the_caller() -> None:
    """The signature is the boundary: there is nowhere to put a forged stamp."""
    parameters = set(inspect.signature(execute).parameters)
    for forbidden in ("fingerprint", "arithmetic_status", "authority", "authority_version",
                      "expression_executed", "verified_result", "result_value"):
        assert forbidden not in parameters


def test_failures_carry_no_result_and_no_fingerprint() -> None:
    result = _run("add", [_op("a", "1", "kg"), _op("b", "1", "m2")], "kg")
    assert result.arithmetic_status == "UNIT_MISMATCH"
    assert result.result_value is None
    assert result.exact_result is None
    assert result.expression_executed is None
    assert result.fingerprint is None
    assert result.conflict == "NONE"


# ---------------------------------------------------------------------------
# Evidence separation
# ---------------------------------------------------------------------------


def test_arithmetic_verified_defaults_to_evidence_unavailable() -> None:
    result = _run("add", [_op("a", "1", "one"), _op("b", "1", "one")], "one")
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert result.evidence.status == "EVIDENCE_UNAVAILABLE"
    assert result.is_verified_evidence() is False


def test_model_source_hints_cannot_elevate_evidence_status() -> None:
    """A retrieval id that exists only as model text stays untrusted."""
    from openexecutive.calc.contract import SourceHint

    hinted = Operand(
        operand_id="a", label="a", value="1", unit=Unit(code="one"),
        basis="applicant_stated",
        source_hint=SourceHint(document_label="proposal.pdf", retrieval_id_hint="rid-1"),
    )
    result = _run("add", [hinted, _op("b", "1", "one")], "one")
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert result.evidence.status == "EVIDENCE_UNAVAILABLE"
    assert result.evidence.bound_operand_ids == ()
    assert result.is_verified_evidence() is False


def test_application_supplied_binding_is_the_only_route_to_supported() -> None:
    evidence = InputEvidenceSummary(status="ALL_SUPPORTED", bound_operand_ids=("a", "b"))
    result = _run("add", [_op("a", "1", "one"), _op("b", "1", "one")], "one",
                  evidence=evidence)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert result.is_verified_evidence() is True


def test_conflicting_source_bindings_stay_visible() -> None:
    evidence = InputEvidenceSummary(status="CONFLICTING_SOURCES", unbound_operand_ids=("a",))
    result = _run("add", [_op("a", "1", "one"), _op("b", "1", "one")], "one",
                  evidence=evidence)
    assert result.evidence.status == "CONFLICTING_SOURCES"
    assert result.is_verified_evidence() is False


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def _batch_request(index: int, operation: str, operands: list[Operand], target: str):
    return _request(operation, operands, target, request_id=f"r{index}")


def test_batch_returns_one_result_per_request_in_order() -> None:
    batch = CalculationBatch(requests=tuple(
        _batch_request(i, "add", [_op("a", str(i), "one"), _op("b", "1", "one")], "one")
        for i in range(5)
    ))
    results = execute_batch(batch, computed_at=AT)
    assert len(results) == 5
    assert [r.request_id for r in results] == [f"r{i}" for i in range(5)]
    assert [r.result_value for r in results] == [f"{i + 1}.00" for i in range(5)]


def test_one_failure_does_not_erase_its_siblings() -> None:
    """Phase 1 measured the alternative: one malformed claim discarded all 11."""
    batch = CalculationBatch(requests=(
        _batch_request(0, "add", [_op("a", "1", "one"), _op("b", "1", "one")], "one"),
        _batch_request(1, "add", [_op("a", "1", "kg"), _op("b", "1", "m2")], "kg"),
        _batch_request(2, "add", [_op("a", "2", "one"), _op("b", "2", "one")], "one"),
    ))
    results = execute_batch(batch, computed_at=AT)
    assert [r.arithmetic_status for r in results] == [
        "ARITHMETIC_VERIFIED", "UNIT_MISMATCH", "ARITHMETIC_VERIFIED",
    ]
    assert results[0].result_value == "2.00"
    assert results[2].result_value == "4.00"


def test_batch_size_is_bounded_by_the_contract() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CalculationBatch(requests=tuple(
            _batch_request(i, "add", [_op("a", "1", "one"), _op("b", "1", "one")], "one")
            for i in range(LIMITS.max_requests + 1)
        ))


def test_engine_holds_no_global_mutable_state() -> None:
    """Two identical batches must not be able to observe each other."""
    batch = CalculationBatch(requests=(
        _batch_request(0, "add", [_op("a", "1", "one"), _op("b", "1", "one")], "one"),
    ))
    first = execute_batch(batch, computed_at=AT)
    second = execute_batch(batch, computed_at=AT)
    assert first[0].result_value == second[0].result_value
    assert first[0].fingerprint == second[0].fingerprint
    module_state = [
        name for name, value in vars(engine_mod).items()
        if isinstance(value, (list, dict, set)) and not name.startswith("__")
        and name not in ("_SIGNATURES", "_DISPATCH", "_ROUNDING")
    ]
    assert module_state == [], f"mutable module state: {module_state}"


def test_dispatch_tables_are_closed_and_match_the_signatures() -> None:
    assert set(engine_mod._DISPATCH) == set(_V1_OPERATIONS)
    assert set(engine_mod._SIGNATURES) == set(_V1_OPERATIONS)


def test_limits_match_the_phase_one_contract() -> None:
    from openexecutive.calc import contract as contract_mod
    from openexecutive.calc import numeric as numeric_mod

    assert LIMITS.max_operands == contract_mod.MAX_OPERANDS_PER_REQUEST == 64
    assert LIMITS.max_requests == contract_mod.MAX_REQUESTS_PER_BATCH == 32
    assert LIMITS.max_precision == numeric_mod.MAX_PRECISION_REQUEST == 50
    assert LIMITS.max_adjusted_exponent == numeric_mod.MAX_ADJUSTED_EXPONENT == 30
    assert LIMITS.nested_operations == 0


def test_max_precision_request_is_no_longer_inert() -> None:
    """Phase 1 declared it and nothing read it; the engine is now its consumer."""
    assert engine_mod.ENGINE_PRECISION == 50
    # Behavioural, not a source-text grep: the engine's context must actually
    # carry the precision, and a quotient must actually be computed to it.
    assert engine_mod.engine_context().prec == 50
    result = _run("divide", [_op("a", "1", "one"), _op("b", "3", "one")], "one", scale=28)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert len(result.exact_result or "") > 28


def test_every_registry_unit_can_at_least_identity_convert() -> None:
    for code in known_unit_codes():
        value = "50" if code in ("pct", "pct_point") else "2"
        result = _run("convert_unit", [_op("v", value, code)], code, scale=0)
        assert result.arithmetic_status == "ARITHMETIC_VERIFIED", code
        assert result.result_value == value


def test_engine_does_no_io_and_reads_no_clock_for_identity() -> None:
    """AST-scoped, not a text grep.

    A substring scan for "requests" matches the legitimate ``max_requests``
    field — a security check that cries wolf teaches the reader to ignore it.
    """
    import ast

    tree = ast.parse(inspect.getsource(engine_mod))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("urllib", "socket", "subprocess", "requests", "httpx", "pathlib",
                   "os", "shutil", "pickle", "importlib"):
        assert banned not in imported, banned
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for banned in ("eval", "exec", "compile", "open", "__import__", "getattr"):
        assert banned not in called, banned
    # ``time`` is imported, but only for budgets — never for the record: the
    # timestamp on a result is the caller's, so tests can pin it.
    attrs = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "monotonic" in attrs
    assert "time" not in attrs


# ---------------------------------------------------------------------------
# Review round 1 regressions — each reproduces a reported exploit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,left_unit,right,right_unit,target,expected",
    [
        # HIGH-1: the 10^4 error, verbatim. Un-normalized operands — the path
        # every earlier fixture skipped by converting ha->m2 first.
        ("52", "kg_per_m2", "11", "ha", "kg", "5720000"),
        ("31", "kg_per_m2", "7", "ha", "kg", "2170000"),
        ("5", "t", "1", "one", "kg", "5000"),
        ("1", "one", "5", "t", "kg", "5000"),
    ],
)
def test_multiply_normalizes_before_multiplying(
    left: str, left_unit: str, right: str, right_unit: str, target: str, expected: str
) -> None:
    """``52 kg/m2 x 11 ha`` must be 5,720,000 kg, not 572.

    Validating the target by dimension alone accepted the bare coefficients and
    returned the exact figure this package was built to prevent — with a
    verified status, an authority stamp and a fingerprint, which is strictly
    worse than a model saying it.
    """
    result = _run("multiply", [_op("a", left, left_unit), _op("b", right, right_unit)],
                  target, scale=0)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert result.result_value == expected


@pytest.mark.parametrize(
    "num,num_unit,den,den_unit,target,expected",
    [
        ("1", "t", "500", "kg", "one", "2"),          # HIGH-1: was 0.00
        ("5720", "t", "11", "ha", "kg_per_m2", "52"),  # HIGH-1: was 520
        ("5720000", "kg", "110000", "m2", "kg_per_m2", "52"),
    ],
)
def test_divide_normalizes_before_dividing(
    num: str, num_unit: str, den: str, den_unit: str, target: str, expected: str
) -> None:
    result = _run("divide", [_op("n", num, num_unit), _op("d", den, den_unit)],
                  target, scale=0)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert result.result_value == expected


def test_divide_and_ratio_agree_on_identical_operands() -> None:
    """Two operations that must agree, and once did not: 0.00 versus 2.00."""
    operands = [_op("n", "1", "t"), _op("d", "500", "kg")]
    divided = _run("divide", operands, "one", scale=2)
    ratioed = _run("ratio", operands, "one", scale=2)
    assert divided.result_value == ratioed.result_value == "2.00"


@pytest.mark.parametrize("target", ["t"])
def test_multiplicative_result_must_use_its_dimension_base_unit(target: str) -> None:
    """``-> t`` for a kilogram result was the other half of the 10^4 bug.

    A caller wanting tonnes issues an explicit ``convert_unit``, which is
    auditable, rather than having the engine silently pick a scale.
    """
    result = _run("multiply", [_op("a", "52", "kg_per_m2"), _op("b", "11", "ha")],
                  target, scale=0)
    assert result.arithmetic_status == "UNIT_MISMATCH"
    assert "base unit" in result.errors[0].detail
    # A wrong *dimension* is caught earlier, by its own rule.
    wrong_dimension = _run("multiply",
                           [_op("a", "52", "kg_per_m2"), _op("b", "11", "ha")],
                           "ha", scale=0)
    assert wrong_dimension.arithmetic_status == "UNIT_MISMATCH"
    assert "produces mass" in wrong_dimension.errors[0].detail


@pytest.mark.parametrize("operation", ["multiply", "divide"])
def test_dimensionless_scaling_cannot_relabel_a_currency(operation: str) -> None:
    """HIGH-2: ``1,000,000 TND x 1 -> EUR`` was verified at par.

    The engine refused the honest conversion and permitted the dishonest one,
    because the target check compared only ``"currency" != "currency"``.
    """
    result = _run(operation,
                  [_op("v", "1000000", "currency:TND"), _op("k", "1", "one")],
                  "currency:EUR", scale=0)
    assert result.arithmetic_status == "UNIT_MISMATCH"
    assert "exchange-rate authority" in result.errors[0].detail


def test_same_currency_scaling_still_works() -> None:
    result = _run("multiply",
                  [_op("v", "1000", "currency:TND"), _op("k", "3", "one")],
                  "currency:TND", scale=0)
    assert result.result_value == "3000"


def test_batch_evidence_is_scoped_to_one_request() -> None:
    """HIGH-3: a shared summary stamped unvalidated siblings as verified.

    ``operand_id`` is unique only *within* a request and models reuse generic
    ids, so evidence validated for r1 landed on r2's invented operands and
    reached ``is_verified_evidence()``.
    """
    batch = CalculationBatch(requests=(
        _request("add", [_op("a", "10", "kg"), _op("b", "20", "kg")], "kg",
                 request_id="r1"),
        _request("add", [_op("a", "999999", "kg"), _op("b", "888888", "kg")], "kg",
                 request_id="r2"),
    ))
    validated = InputEvidenceSummary(status="ALL_SUPPORTED",
                                     bound_operand_ids=("a", "b"))
    results = execute_batch(batch, computed_at=AT,
                            evidence_by_request={"r1": validated})
    assert results[0].evidence.status == "ALL_SUPPORTED"
    assert results[0].is_verified_evidence() is True
    assert results[1].evidence.status == "EVIDENCE_UNAVAILABLE"
    assert results[1].is_verified_evidence() is False


def test_execute_batch_takes_no_batch_wide_evidence() -> None:
    parameters = set(inspect.signature(execute_batch).parameters)
    assert "evidence" not in parameters
    assert "evidence_by_request" in parameters


def test_batch_evidence_must_name_a_request_in_the_batch() -> None:
    """Silently ignoring an unknown key would let a caller believe a binding
    was applied when it was not."""
    batch = CalculationBatch(requests=(
        _request("add", [_op("a", "1", "kg"), _op("b", "1", "kg")], "kg",
                 request_id="r1"),
    ))
    summary = InputEvidenceSummary(status="ALL_SUPPORTED", bound_operand_ids=("a", "b"))
    with pytest.raises(ValueError, match="not in this batch"):
        execute_batch(batch, computed_at=AT, evidence_by_request={"ghost": summary})
    with pytest.raises(TypeError):
        execute_batch(batch, computed_at=AT,
                      evidence_by_request={"r1": "ALL_SUPPORTED"})  # type: ignore[dict-item]


def test_engine_results_cannot_collide_with_contract_phase_ones() -> None:
    """MEDIUM-4: the stamp still read ``0.1.0-contract`` when the engine shipped.

    Asserted against the *real* authority, not a hypothetical substitution — a
    draft test substituted ``9.9.9-hypothetical-engine`` into the payload and
    was therefore vacuous with respect to the shipped code.
    """
    from openexecutive.calc.authority import AUTHORITY_VERSION, current_authority
    from openexecutive.calc.contract import ApplicationAuthority
    from openexecutive.calc.fingerprint import build_payload, compute_fingerprint

    assert AUTHORITY_VERSION == "0.2.0-engine"
    assert current_authority().authority_version == "0.2.0-engine"

    computed = _run("add", [_op("a", "1", "kg"), _op("b", "2", "kg")], "kg")
    normalized = computed.normalized_operands
    contract_era = compute_fingerprint(build_payload(
        operation="add", normalized_operands=normalized, target_unit=Unit(code="kg"),
        scale=2, rounding="ROUND_HALF_EVEN",
        authority=ApplicationAuthority(authority_id="openexecutive.calc",
                                       authority_version="0.1.0-contract"),
    ))
    assert computed.fingerprint != contract_era


def test_a_near_match_variance_reports_rather_than_failing() -> None:
    """MEDIUM-6: 50-digit intermediates overflowed 64-character fields.

    Figures agreeing to fifteen significant digits returned *no result*, and the
    pydantic error text — including a truncated echo of the computed value —
    was stored in the durable record.
    """
    result = _run("variance",
                  [_op("calc", "30140000.00", TND),
                   _op("stated", "30140000.0000001", TND, role="stated_comparison")],
                  TND, scale=2)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert result.conflict == "WITHIN_TOLERANCE"
    assert result.percentage_difference is not None
    # The finding survives: trimming precision, not quantizing to scale 2, which
    # would have rendered it "0.00" and erased it.
    assert Decimal(result.percentage_difference) != 0
    for field in (result.percentage_difference, result.ratio, result.exact_result):
        assert field is None or len(field) <= 64


def test_no_framework_error_text_reaches_the_record() -> None:
    for result in (
        _run("ratio", [_op("a", "1e30", "kg"), _op("b", "1e-30", "kg")], "one"),
        _run("multiply", [_op("a", "1e30", "one"), _op("b", "1e30", "one")], "one"),
    ):
        for error in result.errors:
            assert "ValidationError" not in error.detail
            assert "Traceback" not in error.detail
            assert "pydantic" not in error.detail.lower()


def test_an_unconsumed_time_policy_does_not_change_identity() -> None:
    """MEDIUM-7: a kwarg that touched nothing split the identity of an
    identical calculation — the dedup failure the allowlist exists to prevent."""
    plain = _run("add", [_op("a", "10", "kg"), _op("b", "20", "kg")], "kg")
    with_policy = _run("add", [_op("a", "10", "kg"), _op("b", "20", "kg")], "kg",
                       time_conversion_policy="calendar_12_months_per_year")
    assert plain.fingerprint == with_policy.fingerprint


def test_a_consumed_time_policy_does_change_identity() -> None:
    converted = _run("convert_unit", [_op("a", "12", "month")], "year", scale=0,
                     time_conversion_policy="calendar_12_months_per_year")
    assert converted.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert converted.result_value == "1"
    identity = _run("convert_unit", [_op("a", "12", "month")], "month", scale=0)
    assert identity.fingerprint != converted.fingerprint


@pytest.mark.parametrize(
    "weights",
    [
        # Every pair here sums to a POSITIVE total, so the later
        # ``total_weight <= 0`` branch cannot fire: only the sign guard can
        # reject them. A draft parametrization used pairs that all summed to
        # <= 0, so deleting the sign guard left the whole suite green while
        # ``w1=-2, w2=+5`` returned 26.67 as the "average" of 10 and 20 —
        # outside the convex hull, verified and fingerprinted.
        ("-2", "5"), ("-1", "10"), ("3", "-1"), ("-100", "101"),
    ],
)
def test_negative_weights_are_rejected(weights: tuple[str, str]) -> None:
    """MEDIUM-8: a negative weight puts the "average" outside its inputs."""
    result = _run("weighted_average",
                  [_op("v1", "10", "kg"), _op("v2", "20", "kg"),
                   _op("w1", weights[0], "one", role="stated_comparison"),
                   _op("w2", weights[1], "one", role="stated_comparison")],
                  "kg", scale=4)
    assert result.arithmetic_status == "INVALID_INPUT"
    assert result.errors[0].code == "negative_weight"
    assert result.result_value is None


def test_zero_total_weight_is_rejected_separately_from_the_sign_guard() -> None:
    """Non-negative weights summing to zero — only the total branch can fire."""
    result = _run("weighted_average",
                  [_op("v1", "10", "kg"), _op("v2", "20", "kg"),
                   _op("w1", "0", "one", role="stated_comparison"),
                   _op("w2", "0", "one", role="stated_comparison")],
                  "kg", scale=4)
    assert result.arithmetic_status == "DIVISION_BY_ZERO"


def test_a_weighted_average_stays_inside_its_inputs() -> None:
    result = _run("weighted_average",
                  [_op("v1", "10", "kg"), _op("v2", "20", "kg"),
                   _op("w1", "1", "one", role="stated_comparison"),
                   _op("w2", "3", "one", role="stated_comparison")],
                  "kg", scale=2)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert Decimal("10") <= Decimal(result.result_value) <= Decimal("20")


def test_out_of_range_results_report_the_limit_not_the_input() -> None:
    """LOW-9: a reviewer was told the input was bad when the limit was ours."""
    result = _run("ratio", [_op("a", "1e30", "kg"), _op("b", "1e-30", "kg")], "one")
    assert result.arithmetic_status == "RESOURCE_LIMIT_EXCEEDED"
    assert result.errors[0].code in ("result_out_of_range", "result_too_long",
                                     "result_exponent")


# ---------------------------------------------------------------------------
# Review round 2 regressions
# ---------------------------------------------------------------------------

_TIME_POLICY = "calendar_12_months_per_year"


@pytest.mark.parametrize(
    "operation,operands,target,scale,expected",
    [
        ("divide", [("a", "12", "month"), ("b", "1", "year")], "one", 2, "1.00"),
        ("ratio", [("a", "12", "month"), ("b", "1", "year")], "one", 2, "1.00"),
        ("multiply", [("a", "2", "year"), ("b", "3", "one")], "month", 0, "72"),
        ("multiply", [("a", "3", "one"), ("b", "2", "year")], "month", 0, "72"),
        ("divide", [("a", "2", "year"), ("b", "3", "one")], "month", 0, "8"),
    ],
)
def test_multiplicative_operations_honour_the_time_policy(
    operation: str, operands: list, target: str, scale: int, expected: str
) -> None:
    """MEDIUM-1: the caller's policy was hard-coded to ``None`` in
    ``_normalize_factor``.

    ``divide 12 month / 1 year`` refused the exact quotient ``ratio`` computed
    from the same operands; ``multiply`` with a ``year`` operand was unreachable
    in every target shape; and the error told the integrator to supply a
    parameter they had already supplied and the code discarded.
    """
    result = _run(operation, [_op(*o) for o in operands], target, scale=scale,
                  time_conversion_policy=_TIME_POLICY)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert result.result_value == expected


def test_divide_and_ratio_agree_across_a_time_conversion() -> None:
    operands = [_op("a", "12", "month"), _op("b", "1", "year")]
    divided = _run("divide", operands, "one", scale=2,
                   time_conversion_policy=_TIME_POLICY)
    ratioed = _run("ratio", operands, "one", scale=2,
                   time_conversion_policy=_TIME_POLICY)
    assert divided.result_value == ratioed.result_value == "1.00"


@pytest.mark.parametrize(
    "operation,operands,target",
    [
        ("divide", [("a", "12", "month"), ("b", "1", "year")], "one"),
        # ``multiply`` needs a dimensionless partner: month x year is a
        # time x time product with no declared composition, so it is refused on
        # dimension before the policy is ever consulted.
        ("multiply", [("a", "2", "year"), ("b", "3", "one")], "month"),
    ],
)
def test_multiplicative_time_conversion_still_needs_the_policy(
    operation: str, operands: list, target: str
) -> None:
    """Threading the policy must not make it optional."""
    result = _run(operation, [_op(*o) for o in operands], target, scale=2)
    assert result.arithmetic_status == "INVALID_INPUT"
    assert "time_conversion_policy" in result.errors[0].detail


@pytest.mark.parametrize(
    "operation,operands,target",
    [
        ("ratio", [("a", "6", "month"), ("b", "1", "year")], "one"),
        ("percentage_of", [("a", "6", "month"), ("b", "1", "year")], "pct"),
        ("divide", [("a", "6", "month"), ("b", "1", "year")], "one"),
        ("multiply", [("a", "2", "year"), ("b", "3", "one")], "month"),
    ],
)
def test_a_consumed_policy_enters_the_fingerprint_for_every_operation(
    operation: str, operands: list, target: str
) -> None:
    """LOW-2: the gate was wired for only two of the seven consuming operations.

    Harmless while one policy literal exists; a real collision the moment a
    second is added, since two arithmetically different results would share one
    identity and a dedup index would call the second "already computed".
    """
    converting = _run(operation, [_op(*o) for o in operands], target, scale=4,
                      time_conversion_policy=_TIME_POLICY)
    assert converting.arithmetic_status == "ARITHMETIC_VERIFIED", converting.errors
    assert converting.fingerprint is not None

    payload = fp_mod.build_payload(
        operation=operation,  # type: ignore[arg-type]
        normalized_operands=converting.normalized_operands,
        target_unit=converting.result_unit, scale=4, rounding="ROUND_HALF_EVEN",
        authority=current_authority(), stated_value=converting.stated_value,
        time_conversion_policy=_TIME_POLICY,
    )
    assert payload.get("parameters", {}).get("time_conversion_policy") == _TIME_POLICY
    assert fp_mod.compute_fingerprint(payload) == converting.fingerprint


def test_signature_declares_no_unread_order_metadata() -> None:
    """LOW-5: ``order_matters`` was declared, exported, and never read.

    A Phase 3 consumer reading ``signature_for("add").order_matters is False``
    would reasonably conclude the engine canonicalises commutative operand
    order. It does not — labels and provenance ride with position — and the
    architecture page documents the opposite.
    """
    assert not hasattr(signature_for("add"), "order_matters")
    forward = _run("add", [_op("a", "1", "kg"), _op("b", "2", "kg")], "kg")
    swapped = _run("add", [_op("a", "2", "kg"), _op("b", "1", "kg")], "kg")
    assert forward.result_value == swapped.result_value
    assert forward.fingerprint != swapped.fingerprint


def test_declared_import_surface_matches_the_code() -> None:
    """LOW-3: the prose still claimed the Phase 1 import list after Phase 2
    added ``hashlib`` and ``time``."""
    import openexecutive.calc as calc_pkg

    doc = calc_pkg.__doc__ or ""
    assert "hashlib" in doc and "time" in doc


# ---------------------------------------------------------------------------
# Review round 3 regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    ["actual/360", "30/360", "'; DROP TABLE calc--", "x" * 5000, "",
     "CALENDAR_12_MONTHS_PER_YEAR"],
)
def test_unsanctioned_time_policies_are_refused(policy: str) -> None:
    """``Literal`` is erased at runtime, so the annotation stopped nothing.

    An unvalidated ``"actual/360"`` was accepted, the registry's exact ``x1/12``
    factor was applied regardless, and the durable record then asserted a basis
    the engine had never used — "annualised without saying how", inverted: it
    said *how*, and the *how* was false. Every distinct string also produced a
    distinct fingerprint for byte-identical arithmetic.
    """
    with pytest.raises(ValueError, match="unsupported time_conversion_policy"):
        _run("convert_unit", [_op("a", "12", "month")], "year", scale=0,
             time_conversion_policy=policy)


@pytest.mark.parametrize("policy", ["must_sum_to_one", "raw", ""])
def test_unsanctioned_weight_policies_are_refused(policy: str) -> None:
    with pytest.raises(ValueError, match="unsupported weight_policy"):
        _run("weighted_average",
             [_op("v1", "10", "kg"), _op("w1", "1", "one", role="stated_comparison")],
             "kg", weight_policy=policy)


def test_the_sanctioned_policies_still_work() -> None:
    converted = _run("convert_unit", [_op("a", "12", "month")], "year", scale=0,
                     time_conversion_policy="calendar_12_months_per_year")
    assert converted.result_value == "1"
    averaged = _run("weighted_average",
                    [_op("v1", "10", "kg"), _op("v2", "20", "kg"),
                     _op("w1", "1", "one", role="stated_comparison"),
                     _op("w2", "1", "one", role="stated_comparison")],
                    "kg", scale=0, weight_policy="normalized_by_engine")
    assert averaged.result_value == "15"


def test_the_policy_allowlists_are_closed_and_match_the_literals() -> None:
    assert frozenset(
        {"calendar_12_months_per_year"}
    ) == engine_mod.SUPPORTED_TIME_CONVERSION_POLICIES
    assert frozenset({"normalized_by_engine"}) == engine_mod.SUPPORTED_WEIGHT_POLICIES


@pytest.mark.parametrize(
    "operation,operands,target",
    [
        ("subtract", [("a", "1." + "0" * 30 + "1", "kg"),
                      ("b", "1." + "0" * 30 + "1", "kg")], "kg"),
        ("percentage_point_difference", [("a", "1." + "0" * 40, "pct"),
                                         ("b", "1." + "0" * 40, "pct")], "pct_point"),
    ],
)
def test_an_exact_zero_result_is_reported_as_zero(
    operation: str, operands: list, target: str
) -> None:
    """``v - v`` is zero. It returned no result at all.

    ``_render``'s exponent guard read ``value != 0 and abs(adjusted()) > 30``,
    and ``Decimal("0E-50")`` compares equal to zero — the "check unless it is
    zero" shape ``numeric.py`` names as the one that matters. It expanded
    positionally, the contract rejected it at -50, and the reviewer-facing
    message named nothing.
    """
    result = _run(operation, [_op(*o) for o in operands], target, scale=2)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert Decimal(result.result_value) == 0


def test_a_variance_of_identical_high_precision_figures_reports_exact_match() -> None:
    value = "1." + "0" * 40
    result = _run("variance",
                  [_op("calc", value, "kg"),
                   _op("stated", value, "kg", role="stated_comparison")],
                  "kg", scale=2)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert result.conflict == "EXACT_MATCH"
    assert Decimal(result.result_value) == 0


def test_interval_bounds_are_written_in_canonical_positional_form() -> None:
    """A draft rendered the upper bound as ``5.0000E+5``.

    Every numeric *field* in this contract is canonical positional form; the
    interval's upper bound lives only in free text, and it was the one figure
    printed in exponent notation — asking a reviewer to re-read the exact
    magnitude this package exists to protect.
    """
    result = _run("interval_implied_total",
                  [_op("vl", "100000", "kg"), _op("vh", "200000", "kg"),
                   _op("cl", "40", "pct"), _op("ch", "60", "pct")],
                  "kg", scale=4)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert result.warnings[0] == "upper bound 500000"
    for text in (result.expression_executed or "", result.warnings[0]):
        assert "E+" not in text and "E-" not in text


# ---------------------------------------------------------------------------
# Review round 4 regressions
# ---------------------------------------------------------------------------

_ZERO_SPELLINGS = ["0", "0.0", "0.00", "-0.00", "0.000000", "-0"]

_ZERO_OPERAND_CASES = [
    ("add", [("a", "1000.00", TND), ("z", "ZERO", TND)], TND, "1000.00"),
    ("subtract", [("a", "1000.00", TND), ("z", "ZERO", TND)], TND, "1000.00"),
    ("sum_components",
     [("a", "1000.00", TND), ("z", "ZERO", TND), ("c", "250.00", TND)], TND, "1250.00"),
    ("multiply", [("z", "ZERO", "m2"), ("y", "5", "kg_per_m2")], "kg", "0.00"),
    ("divide", [("z", "ZERO", "kg"), ("d", "5", "kg")], "one", "0.00"),
    ("ratio", [("z", "ZERO", "kg"), ("d", "5", "kg")], "one", "0.00"),
    ("percentage_of", [("z", "ZERO", "kg"), ("w", "5", "kg")], "pct", "0.00"),
    ("percentage_point_difference",
     [("z", "ZERO", "pct"), ("b", "5.00", "pct")], "pct_point", "-5.00"),
    ("variance",
     [("z", "ZERO", "kg"), ("s", "5.00", "kg")], "kg", "-5.00"),
    ("weighted_average",
     [("z", "ZERO", "kg"), ("v", "20.00", "kg")], "kg", "10.00"),
    ("interval_implied_total",
     [("z", "ZERO", "kg"), ("vh", "200000", "kg")], "kg", "0.00"),
]


@pytest.mark.parametrize("zero", _ZERO_SPELLINGS)
@pytest.mark.parametrize(
    "operation,operands,target,expected",
    _ZERO_OPERAND_CASES,
    ids=[c[0] for c in _ZERO_OPERAND_CASES],
)
def test_scaled_zero_operands_are_accepted_by_every_operation(
    zero: str, operation: str, operands: list, target: str, expected: str
) -> None:
    """A ``0.00`` operand must not break the calculation.

    A draft normalised every zero to ``"0"`` regardless of its stated scale.
    ``NormalizedOperand`` then refused it — "a conversion that does not convert
    units cannot alter the number" — and eleven of the twelve operations failed
    on any scaled zero, including a cost stack with one ``0.00`` line. Worse,
    the outcome depended on an irrelevant spelling detail: ``0.00 ha`` survived
    (the unit changed) while ``0.00 m2`` did not.
    """
    built: list[Operand] = []
    for oid, value, unit in operands:
        role = "input"
        if operation == "variance" and oid == "s":
            role = "stated_comparison"
        built.append(_op(oid, zero if value == "ZERO" else value, unit, role=role))
    if operation == "weighted_average":
        built += [_op("w1", "1", "one", role="stated_comparison"),
                  _op("w2", "1", "one", role="stated_comparison")]
    if operation == "interval_implied_total":
        built += [_op("cl", "40", "pct"), _op("ch", "60", "pct")]
    result = _run(operation, built, target, scale=2)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", (
        f"{operation} with zero spelled {zero!r}: {result.errors}"
    )
    assert result.result_value == expected


@pytest.mark.parametrize("zero", _ZERO_SPELLINGS)
def test_a_zero_operand_keeps_its_own_scale_in_the_record(zero: str) -> None:
    """The stored operand must equal what the contract canonicalised."""
    operand = _op("z", zero, TND)
    result = _run("add", [operand, _op("b", "1.00", TND)], TND, scale=2)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    recorded = next(o for o in result.normalized_operands if o.operand_id == "z")
    assert recorded.normalized_value == operand.value
    assert recorded.original_value == operand.value


def test_a_computed_zero_below_the_contract_bound_is_clamped_not_refused() -> None:
    """``v - v`` for a 31-decimal ``v`` is ``0E-31`` — beyond the numeric bound.

    No *operand* can reach that exponent (``parse_numeric`` refuses one, which
    is what closed the 4 GB positional-expansion DoS), so this is only ever a
    computed value, and a zero has no magnitude to lose.
    """
    value = "1." + "0" * 30 + "1"
    for scale, expected in ((0, "0"), (2, "0.00"), (6, "0.000000")):
        result = _run("subtract", [_op("a", value, "kg"), _op("b", value, "kg")],
                      "kg", scale=scale)
        assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
        assert result.result_value == expected


@pytest.mark.parametrize(
    "literal", ["0e-2000000000", "-0e-2000000000", "0e2000000000", "0e-99999999"],
)
def test_the_hostile_zero_exponent_dos_stays_closed(literal: str) -> None:
    """Preserving zero's scale must not re-open the expansion hazard.

    ``format(Decimal("0E-2000000000"), "f")`` really does produce two billion
    characters — measured while diagnosing this. The bound is computed from the
    exponent and never by rendering first.
    """
    import resource
    import time

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.monotonic()
    with pytest.raises(NumericPolicyError):
        parse_numeric(literal)
    with pytest.raises(ValidationError):
        _op("z", literal, "kg")
    assert time.monotonic() - started < 1.0
    grew = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    assert grew < 256 * 1024 * 1024


def test_an_unrepresentable_derived_field_does_not_erase_the_result() -> None:
    """F-C: the largest possible discrepancy produced no record at all.

    ``_bounded``'s last-resort quantize escaped as ``INVALID_INPUT``, blaming
    two inputs that were both perfectly valid, and discarded a primary result
    that was representable.
    """
    result = _run("variance",
                  [_op("calc", "1e30", "kg"),
                   _op("stated", "1e-30", "kg", role="stated_comparison")],
                  "kg", scale=2)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert result.result_value is not None
    assert result.conflict == "ORDER_OF_MAGNITUDE"
    # The derived fields that cannot fit are absent, and say so.
    assert result.ratio is None
    assert result.percentage_difference is None
    # The precise warning set, not merely the word "unavailable". This case
    # emits two warnings, so `any(...)` passed even with one of them suppressed
    # — a partial pin that mutation testing exposed.
    assert set(result.warnings) == {
        "percentage_difference unavailable: the value cannot be represented "
        "within the field's limits",
        "ratio unavailable: the value cannot be represented within the "
        "field's limits",
    }
    for error in result.errors:
        assert "ValidationError" not in error.detail
        assert "pydantic" not in error.detail.lower()


def test_the_declared_fingerprint_fields_match_the_real_payload() -> None:
    """F-D: ``parameters`` was a live identity key absent from the declaration.

    Set equality against a payload that actually carries a consumed policy —
    the pinning test built one with no parameters, so the drift went unseen.
    """
    from openexecutive.calc.contract import (
        FINGERPRINT_INCLUDED_FIELDS,
        FINGERPRINT_OPTIONAL_FIELDS,
    )

    converting = _run("convert_unit", [_op("a", "12", "month")], "year", scale=0,
                      time_conversion_policy="calendar_12_months_per_year")
    assert converting.arithmetic_status == "ARITHMETIC_VERIFIED", converting.errors
    with_policy = fp_mod.build_payload(
        operation="convert_unit",
        normalized_operands=converting.normalized_operands,
        target_unit=converting.result_unit, scale=0, rounding="ROUND_HALF_EVEN",
        authority=current_authority(),
        time_conversion_policy="calendar_12_months_per_year",
    )
    assert set(with_policy) == set(FINGERPRINT_INCLUDED_FIELDS) | set(
        FINGERPRINT_OPTIONAL_FIELDS
    )
    plain = fp_mod.build_payload(
        operation="add", normalized_operands=converting.normalized_operands,
        target_unit=converting.result_unit, scale=0, rounding="ROUND_HALF_EVEN",
        authority=current_authority(),
    )
    assert set(plain) == set(FINGERPRINT_INCLUDED_FIELDS)
    assert "parameters" not in plain


@pytest.mark.parametrize(
    "table", ["_SIGNATURES", "_DISPATCH", "_ROUNDING"],
)
def test_engine_lookup_tables_are_immutable(table: str) -> None:
    """F-G: ``units.py`` hardened its registry and argued that hardening one
    table while leaving its sibling open "reads as a guarantee that does not
    hold" — and ``_DISPATCH["add"] = evil`` was a one-liner."""
    mapping = getattr(engine_mod, table)
    assert isinstance(mapping, MappingProxyType)
    with pytest.raises(TypeError):
        mapping["add"] = None  # type: ignore[index]
    assert not hasattr(engine_mod, f"_MUTABLE{table}")


def test_the_result_exponent_guard_is_load_bearing() -> None:
    """F-F: the guard survived mutation to ``if False:`` with the suite green."""
    result = _run("convert_unit", [_op("a", "1e30", "ha")], "m2", scale=0)
    assert result.arithmetic_status == "RESOURCE_LIMIT_EXCEEDED"
    assert result.errors[0].code in ("result_exponent", "result_too_long")
    assert result.result_value is None


def test_the_result_length_guard_is_load_bearing() -> None:
    """``1e-20 / 3`` renders to 72 positional characters — past the 64-char field.

    Two earlier attempts missed this guard entirely. ``1 / 3e-25`` renders 51
    characters and fits; ``1e-30 / 3`` has an adjusted exponent of -31, so the
    *exponent* guard fires first and the length guard is never reached. The
    input has to sit inside the magnitude bound while still rendering long —
    which is exactly the case the length guard exists for, since it is the
    leading zeros plus fifty significant digits that overflow, not the
    magnitude.
    """
    from openexecutive.calc.engine import _render

    # At the engine's own 50-digit working precision, not the default 28.
    with localcontext() as ctx:
        ctx.prec = engine_mod.ENGINE_PRECISION
        quotient = Decimal("1e-20") / Decimal(3)
    assert abs(quotient.adjusted()) <= 30, "must be inside the exponent bound"
    assert len(f"{quotient:f}") > 64, "must be long enough to reach the guard"
    with pytest.raises(EngineError) as caught:
        _render(quotient, "exact_result", 28)
    assert caught.value.code == "result_too_long"
    assert caught.value.status == "RESOURCE_LIMIT_EXCEEDED"
    # End to end the engine keeps it typed, and no over-length string reaches a
    # 64-character field.
    result = _run("divide", [_op("a", "1e-20", "one"), _op("b", "3", "one")],
                  "one", scale=28)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert len(result.result_value or "") <= 64
    assert len(result.exact_result or "") <= 64


def test_the_hostile_zero_width_guard_is_load_bearing() -> None:
    """A zero at exponent -63 would render as 65 characters.

    The width is computed from the exponent and never by rendering first —
    ``format(Decimal("0E-2000000000"), "f")`` genuinely produces two billion
    characters. This pins the boundary at the point where a *computed* zero
    would overflow its field: no operand can reach it, because the numeric
    boundary refuses one.
    """
    from openexecutive.calc.engine import _render

    # Inside the contract's numeric bound the zero keeps its own scale...
    assert _render(Decimal(0).scaleb(-30), "exact_result", 2) == "0." + "0" * 30
    # ...beyond it, a *computed* zero is clamped to the caller's scale...
    assert _render(Decimal(0).scaleb(-62), "exact_result", 2) == "0.00"
    # ...and past the field width even that is refused, without rendering.
    with pytest.raises(EngineError) as caught:
        _render(Decimal(0).scaleb(-63), "exact_result", 2)
    assert caught.value.code == "result_too_long"
    assert "exponent -63" in caught.value.detail
    # The two-billion-exponent form cannot even be constructed via ``scaleb``
    # (CPython raises ``InvalidOperation`` first), which is itself part of the
    # defence. Construct it literally, and confirm the width check refuses it
    # without ever rendering — ``format(v, "f")`` on this value really does
    # produce two billion characters, measured while diagnosing this.
    with pytest.raises(EngineError) as hostile:
        _render(Decimal("0E-2000000000"), "exact_result", 2)
    assert hostile.value.code == "result_too_long"
    assert "2000000000" in hostile.value.detail


def test_an_unexpected_stated_comparison_operand_is_refused() -> None:
    """F-F: without this guard a stray ``stated_comparison`` operand is silently
    dropped from ``normalized_operands`` and from the fingerprint, because
    ``_inputs()`` filters on role."""
    result = _run("add",
                  [_op("a", "1", "kg"), _op("b", "2", "kg"),
                   _op("stray", "999", "kg", role="stated_comparison")],
                  "kg", scale=0)
    assert result.arithmetic_status == "INVALID_INPUT"
    assert result.errors[0].code == "unexpected_stated_value"
    assert result.result_value is None


# ---------------------------------------------------------------------------
# Review round 5 regressions — ambient Decimal context isolation
# ---------------------------------------------------------------------------

_HOSTILE_CONTEXTS: list[tuple[str, object]] = [
    ("pristine", lambda ctx: None),
    ("ROUND_UP", lambda ctx: setattr(ctx, "rounding", decimal.ROUND_UP)),
    ("ROUND_FLOOR", lambda ctx: setattr(ctx, "rounding", decimal.ROUND_FLOOR)),
    ("ROUND_CEILING", lambda ctx: setattr(ctx, "rounding", decimal.ROUND_CEILING)),
    ("prec=5", lambda ctx: setattr(ctx, "prec", 5)),
    ("prec=200", lambda ctx: setattr(ctx, "prec", 200)),
    ("narrow exponent range",
     lambda ctx: (setattr(ctx, "Emin", -10), setattr(ctx, "Emax", 10))),
    ("clamp=1", lambda ctx: setattr(ctx, "clamp", 1)),
    ("Inexact trapped",
     lambda ctx: ctx.traps.__setitem__(decimal.Inexact, True)),
    ("Rounded trapped",
     lambda ctx: ctx.traps.__setitem__(decimal.Rounded, True)),
    ("every optional trap",
     lambda ctx: [ctx.traps.__setitem__(signal, True) for signal in
                  (decimal.Inexact, decimal.Rounded, decimal.Subnormal,
                   decimal.Underflow, decimal.Clamped)]),
]

_DIVISION_CASES = [
    ("divide", [("a", "1", "one"), ("b", "3", "one")], "one", {}),
    ("percentage_of", [("a", "1", "kg"), ("b", "3", "kg")], "pct", {}),
    ("ratio", [("a", "2", "kg"), ("b", "3", "kg")], "one", {}),
    ("convert_unit", [("a", "1", "month")], "year",
     {"time_conversion_policy": "calendar_12_months_per_year"}),
]


@pytest.fixture
def pristine_decimal_context():
    """Restore the process Decimal context after a test perturbs it."""
    saved = decimal.getcontext().copy()
    try:
        yield
    finally:
        decimal.setcontext(saved)


@pytest.mark.parametrize(
    "operation,operands,target,kwargs", _DIVISION_CASES,
    ids=[c[0] for c in _DIVISION_CASES],
)
def test_records_are_identical_under_every_hostile_ambient_context(
    operation: str, operands: list, target: str, kwargs: dict,
    pristine_decimal_context: None,
) -> None:
    """Byte-identical requests must produce byte-identical records.

    ``localcontext()`` with no argument COPIES the caller's thread-local
    context. A draft did that and overrode only ``prec`` and three traps, so
    ``getcontext().rounding = ROUND_UP`` — a legal global any dependency may
    set — changed the 50th digit of a normalized operand, and that digit is a
    fingerprint payload field: byte-identical input produced two fingerprints.
    """
    seen: set[tuple] = set()
    for label, mutate in _HOSTILE_CONTEXTS:
        decimal.setcontext(decimal.Context())
        mutate(decimal.getcontext())  # type: ignore[operator]
        result = _run(operation, [_op(*o) for o in operands], target, scale=2,
                      **kwargs)  # type: ignore[arg-type]
        assert result.arithmetic_status == "ARITHMETIC_VERIFIED", (
            f"{label}: {result.errors}"
        )
        seen.add((
            result.fingerprint, result.exact_result, result.result_value,
            tuple(o.normalized_value for o in result.normalized_operands),
        ))
    assert len(seen) == 1, f"{operation} varied across ambient contexts: {seen}"


def test_a_trapped_inexact_signal_cannot_deny_service(
    pristine_decimal_context: None,
) -> None:
    """``getcontext().traps[Inexact] = True`` once broke every division.

    This repository's own suite sets that trap (the contract layer's
    no-arithmetic test), so the hazard was already in-tree rather than
    hypothetical.
    """
    decimal.setcontext(decimal.Context())
    decimal.getcontext().traps[decimal.Inexact] = True
    decimal.getcontext().traps[decimal.Rounded] = True
    result = _run("divide", [_op("a", "1", "one"), _op("b", "3", "one")],
                  "one", scale=2)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert result.result_value == "0.33"


def test_the_engine_context_is_fully_specified_and_not_inherited(
    pristine_decimal_context: None,
) -> None:
    """Every field pinned explicitly — none left to inheritance."""
    decimal.setcontext(decimal.Context())
    hostile = decimal.getcontext()
    hostile.prec = 7
    hostile.rounding = decimal.ROUND_UP
    hostile.Emin = -5
    hostile.Emax = 5
    hostile.clamp = 1
    for signal in (decimal.Inexact, decimal.Rounded, decimal.Subnormal):
        hostile.traps[signal] = True

    ctx = engine_mod.engine_context()
    assert ctx.prec == engine_mod.ENGINE_PRECISION == 50
    assert ctx.rounding == decimal.ROUND_HALF_EVEN
    assert ctx.Emin == -999999 and ctx.Emax == 999999
    assert ctx.clamp == 0
    # The signals that must fire as typed statuses...
    for signal in (decimal.InvalidOperation, decimal.DivisionByZero,
                   decimal.Overflow, decimal.FloatOperation):
        assert ctx.traps[signal], signal
    # ...and the ones that must not, being the normal condition of division.
    for signal in (decimal.Inexact, decimal.Rounded, decimal.Subnormal,
                   decimal.Underflow, decimal.Clamped):
        assert not ctx.traps[signal], signal


def test_the_engine_does_not_disturb_the_callers_context(
    pristine_decimal_context: None,
) -> None:
    """Isolation runs both ways: the caller's context must survive untouched."""
    decimal.setcontext(decimal.Context())
    caller = decimal.getcontext()
    caller.prec = 7
    caller.rounding = decimal.ROUND_UP
    before = (caller.prec, caller.rounding, caller.Emin, caller.Emax, caller.clamp)
    _run("divide", [_op("a", "1", "one"), _op("b", "3", "one")], "one", scale=2)
    after = decimal.getcontext()
    assert (after.prec, after.rounding, after.Emin, after.Emax, after.clamp) == before


def test_failure_records_never_carry_framework_text() -> None:
    """Exercises the failure handler and asserts what it wrote.

    The predecessor set hostile ambient traps and then looped over
    ``result.errors`` — but with the context fix in place those calls succeed
    with zero errors, so the loop body never ran and the test asserted nothing.
    It would have passed unchanged if the handler went back to echoing
    ``str(exc)``. These inputs actually produce failures.
    """
    failures = [
        _run("divide", [_op("a", "1", "one"), _op("b", "0", "one")], "one"),
        _run("add", [_op("a", "1", "kg"), _op("b", "1", "m2")], "kg"),
        _run("convert_unit", [_op("a", "1e30", "ha")], "m2", scale=0),
        _run("weighted_average",
             [_op("v1", "1", "kg"), _op("v2", "2", "kg"),
              _op("w1", "-1", "one", role="stated_comparison"),
              _op("w2", "5", "one", role="stated_comparison")], "kg"),
        execute(_request("add", [_op("a", "1", "one"), _op("b", "2", "one")], "one"),
                computed_at="not-a-timestamp"),
    ]
    for result in failures:
        assert result.arithmetic_status != "ARITHMETIC_VERIFIED"
        assert result.errors, "a failure must say why"
        for error in result.errors:
            assert error.code and error.code.replace("_", "").isalnum()
            for banned in ("ValidationError", "pydantic", "Traceback", "<class",
                           "decimal.", "Inexact", "self.", "File \""):
                assert banned not in error.detail, error.detail


def test_equivalent_normalized_conversions_share_a_fingerprint() -> None:
    """A recorded decision, not an accident.

    The payload carries only the NORMALIZED operand, so two routes to the same
    computed figure share an identity. The result is a pure function of the
    payload, so two calculations with DIFFERING results cannot collide — what
    collides is provenance, and a consumer needing that reads the record.
    """
    from_kg = _run("convert_unit", [_op("a", "100", "kg")], "t", scale=4)
    from_t = _run("convert_unit", [_op("a", "0.1", "t")], "t", scale=4)
    assert from_kg.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert from_kg.result_value == from_t.result_value
    assert from_kg.fingerprint == from_t.fingerprint
    # The record still distinguishes them, which is where provenance lives.
    assert from_kg.normalized_operands[0].original_unit.code == "kg"
    assert from_t.normalized_operands[0].original_unit.code == "t"

# ---------------------------------------------------------------------------
# Review round 6 regressions — computed_at recovery
# ---------------------------------------------------------------------------

_ACCEPTED_TIMESTAMPS = [
    ("Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    ("offset +00:00", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00Z"),
    ("offset +0000", "2026-01-01T00:00:00+0000", "2026-01-01T00:00:00Z"),
    ("microseconds Z", "2026-01-01T00:00:00.123456Z", "2026-01-01T00:00:00.123456Z"),
    ("microseconds offset", "2026-01-01T00:00:00.123456+00:00",
     "2026-01-01T00:00:00.123456Z"),
    ("negative zero offset", "2026-01-01T00:00:00-00:00", "2026-01-01T00:00:00Z"),
]

_REJECTED_TIMESTAMPS = [
    ("empty", ""),
    ("naive", "2026-01-01T00:00:00"),
    ("malformed", "not-a-timestamp"),
    ("date only", "2026-01-01"),
    ("500 characters", "x" * 500),
    ("non-zero offset", "2026-01-01T00:00:00+02:00"),
]


@pytest.mark.parametrize(
    "label,supplied,canonical", _ACCEPTED_TIMESTAMPS,
    ids=[c[0] for c in _ACCEPTED_TIMESTAMPS],
)
def test_valid_timestamps_are_canonicalised_not_rejected(
    label: str, supplied: str, canonical: str
) -> None:
    """``datetime.now(timezone.utc).isoformat()`` produces ``+00:00``.

    That is the single most likely value an integrator passes, and the
    contract's ``…Z`` pattern rejected it — so *every* call raised, including
    calls that were going to fail anyway. A zero offset is the same instant
    written the way the standard library writes it.
    """
    result = execute(
        _request("add", [_op("a", "1", "one"), _op("b", "2", "one")], "one"),
        computed_at=supplied,
    )
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert result.computed_at == canonical


def test_a_real_isoformat_timestamp_is_accepted() -> None:
    from datetime import datetime

    stamp = datetime.now(UTC).isoformat()
    assert stamp.endswith("+00:00"), "precondition: this is the risky form"
    result = execute(
        _request("add", [_op("a", "1", "one"), _op("b", "2", "one")], "one"),
        computed_at=stamp,
    )
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert result.computed_at.endswith("Z")


@pytest.mark.parametrize(
    "label,supplied", _REJECTED_TIMESTAMPS, ids=[c[0] for c in _REJECTED_TIMESTAMPS],
)
@pytest.mark.parametrize(
    "operation,operands", [("add", [("a", "1"), ("b", "2")]),
                           ("divide", [("a", "1"), ("b", "0")])],
    ids=["success-shaped", "divide-by-zero"],
)
def test_an_unusable_timestamp_returns_a_typed_result_never_an_exception(
    label: str, supplied: str, operation: str, operands: list
) -> None:
    """The recovery handler passed the rejected clock value back into
    ``_failure``, which re-raised the very ``ValidationError`` it existed to
    convert — so the exception escaped ``execute`` untyped, carrying pydantic's
    traceback-shaped text.
    """
    request = _request(operation, [_op(oid, value, "one") for oid, value in operands],
                       "one")
    result = execute(request, computed_at=supplied)
    assert result.arithmetic_status == "INVALID_INPUT"
    assert result.errors[0].code == "invalid_computed_at"
    assert result.result_value is None
    assert result.fingerprint is None
    assert result.computed_at == engine_mod.FALLBACK_COMPUTED_AT
    for banned in ("ValidationError", "pydantic", "Traceback", "<class"):
        assert banned not in result.errors[0].detail


@pytest.mark.parametrize(
    "label,supplied", _REJECTED_TIMESTAMPS, ids=[c[0] for c in _REJECTED_TIMESTAMPS],
)
def test_a_bad_timestamp_does_not_abort_a_batch(label: str, supplied: str) -> None:
    """The escape aborted the whole batch, destroying failure isolation."""
    batch = CalculationBatch(requests=(
        _request("add", [_op("a", "1", "one"), _op("b", "2", "one")], "one",
                 request_id="ok"),
        _request("divide", [_op("a", "1", "one"), _op("b", "0", "one")], "one",
                 request_id="boom"),
    ))
    results = execute_batch(batch, computed_at=supplied)
    assert [r.request_id for r in results] == ["ok", "boom"]
    for result in results:
        assert result.arithmetic_status == "INVALID_INPUT"
        assert result.errors[0].code == "invalid_computed_at"


def test_a_valid_timestamp_batch_still_isolates_failures() -> None:
    from datetime import datetime

    batch = CalculationBatch(requests=(
        _request("add", [_op("a", "1", "one"), _op("b", "2", "one")], "one",
                 request_id="ok"),
        _request("divide", [_op("a", "1", "one"), _op("b", "0", "one")], "one",
                 request_id="boom"),
        _request("add", [_op("a", "5", "one"), _op("b", "5", "one")], "one",
                 request_id="ok2"),
    ))
    results = execute_batch(batch, computed_at=datetime.now(UTC).isoformat())
    assert [r.arithmetic_status for r in results] == [
        "ARITHMETIC_VERIFIED", "DIVISION_BY_ZERO", "ARITHMETIC_VERIFIED",
    ]
    assert results[0].result_value == "3.00"
    assert results[2].result_value == "10.00"


def test_canonicalisation_never_raises_for_any_input() -> None:
    """It is the boundary that must not be able to abort a calculation."""
    for value in ("", "x" * 5000, "2026-01-01T00:00:00", None, 42, object(),
                  "2026-13-45T99:99:99Z", "\x00", "Z"):
        canonical, reason = engine_mod.canonical_computed_at(value)
        assert canonical is None or isinstance(canonical, str)
        assert (canonical is None) != (reason is None)


def test_bounded_trimming_uses_the_engine_context(
    pristine_decimal_context: None,
) -> None:
    """The second context site inside the round-5 fix, pinned.

    Reverting it to a bare ``localcontext()`` survived the whole suite, yet it
    is not inert: it feeds ``exact_result``, ``ratio``, ``percentage_difference``
    and the interval bound text — and none of those enter the fingerprint. A
    regression there yields two records with *different reported answers under
    one identity*, which is worse for a dedup index than a fingerprint split.
    """
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for rounding in (decimal.ROUND_HALF_EVEN, decimal.ROUND_UP, decimal.ROUND_FLOOR,
                     decimal.ROUND_CEILING, decimal.ROUND_05UP):
        decimal.setcontext(decimal.Context())
        decimal.getcontext().rounding = rounding
        decimal.getcontext().prec = 7
        result = _run("variance",
                      [_op("calc", "1", "kg"),
                       _op("stated", "3", "kg", role="stated_comparison")],
                      "kg", scale=20)
        assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
        seen.add((result.exact_result, result.ratio, result.percentage_difference))
    assert len(seen) == 1, f"derived text varied with ambient rounding: {seen}"


# ---------------------------------------------------------------------------
# Review round 7 regressions — Unicode digits in timestamps
# ---------------------------------------------------------------------------

_UNICODE_DIGIT_TIMESTAMPS = [
    ("arabic-indic", "\u0662\u0660\u0662\u0666-\u0660\u0661-\u0660\u0661"
                     "T\u0660\u0660:\u0660\u0660:\u0660\u0660Z"),
    ("fullwidth", "\uff12\uff10\uff12\uff16-\uff10\uff11-\uff10\uff11"
                  "T\uff10\uff10:\uff10\uff10:\uff10\uff10Z"),
    ("devanagari", "\u0968\u0966\u0968\u096c-\u0966\u0967-\u0966\u0967"
                   "T\u0966\u0966:\u0966\u0966:\u0966\u0966Z"),
    ("mixed ascii and arabic", "202\u0666-01-01T00:00:00Z"),
    ("mathematical digit", "\U0001d7da026-09-02T00:00:00Z"),
    ("arabic offset", "2026-01-01T00:00:00+\u0660\u0660:\u0660\u0660"),
    ("arabic fraction", "2026-01-01T00:00:00.\u0661\u0662\u0663Z"),
]


@pytest.mark.parametrize(
    "label,supplied", _UNICODE_DIGIT_TIMESTAMPS,
    ids=[c[0] for c in _UNICODE_DIGIT_TIMESTAMPS],
)
@pytest.mark.parametrize(
    "operation,operands", [("add", [("a", "1"), ("b", "2")]),
                           ("divide", [("a", "1"), ("b", "0")])],
    ids=["success-shaped", "divide-by-zero"],
)
def test_unicode_digit_timestamps_return_typed_failures(
    label: str, supplied: str, operation: str, operands: list
) -> None:
    """``\\d`` matches every Unicode decimal digit; ``[0-9]`` does not.

    Writing the engine's timestamp patterns with ``\\d`` while the contract used
    ``[0-9]`` meant this module ACCEPTED a timestamp the contract then refused.
    The recovery path re-canonicalised the same value successfully, so the epoch
    fallback never fired, the constructor raised again, and the
    ``ValidationError`` escaped ``execute`` untyped — carrying pydantic's
    traceback-shaped text to the caller.

    Not a fuzz artefact: any integrator formatting through a locale-aware
    formatter under ``ar-EG``, ``fa-IR``, ``hi-IN`` or a fullwidth CJK locale
    emits exactly these digits.
    """
    request = _request(operation, [_op(oid, value, "one") for oid, value in operands],
                       "one")
    result = execute(request, computed_at=supplied)
    assert result.arithmetic_status == "INVALID_INPUT"
    assert result.errors[0].code == "invalid_computed_at"
    assert result.computed_at == engine_mod.FALLBACK_COMPUTED_AT
    assert result.result_value is None
    assert result.fingerprint is None
    for banned in ("ValidationError", "pydantic", "Traceback", "<class"):
        assert banned not in result.errors[0].detail


@pytest.mark.parametrize(
    "label,supplied", _UNICODE_DIGIT_TIMESTAMPS,
    ids=[c[0] for c in _UNICODE_DIGIT_TIMESTAMPS],
)
def test_unicode_digit_timestamps_do_not_abort_a_batch(
    label: str, supplied: str
) -> None:
    """One character in the caller's clock string once destroyed all 32 results."""
    batch = CalculationBatch(requests=(
        _request("add", [_op("a", "1", "one"), _op("b", "2", "one")], "one",
                 request_id="ok"),
        _request("divide", [_op("a", "1", "one"), _op("b", "0", "one")], "one",
                 request_id="boom"),
    ))
    results = execute_batch(batch, computed_at=supplied)
    assert [r.request_id for r in results] == ["ok", "boom"]
    for result in results:
        assert result.arithmetic_status == "INVALID_INPUT"
        assert result.errors[0].code == "invalid_computed_at"


@pytest.mark.parametrize(
    "candidate",
    [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:00-0000",
        "2026-01-01T00:00:00.123456Z",
        "2026-01-01T00:00:00.1+00:00",
        "\u0662\u0660\u0662\u0666-\u0660\u0661-\u0660\u0661"
        "T\u0660\u0660:\u0660\u0660:\u0660\u0660Z",
        "\uff12\uff10\uff12\uff16-\uff10\uff11-\uff10\uff11"
        "T\uff10\uff10:\uff10\uff10:\uff10\uff10Z",
        "202\u0666-01-01T00:00:00Z",
        "2026-01-01T00:00:00",
        "2026-01-01T00:00:00+02:00",
        "2026-01-01",
        "",
        "x" * 500,
        "not-a-timestamp",
        "2026-01-01t00:00:00z",
        " 2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z ",
        "2026-01-01T00:00:00Z\n",
        "2026-01-01T00:00:00\x00Z",
    ],
)
def test_engine_acceptance_agrees_with_the_contract_validator(candidate: str) -> None:
    """Whatever the engine canonicalises, the contract must accept.

    This is the invariant whose absence produced the escape: the two regexes
    were written independently, one with ``\\d`` and one with ``[0-9]``, and
    nothing checked that they agreed. Asserting agreement directly is durable in
    a way that enumerating hostile inputs is not — it catches the next
    divergence too.
    """
    canonical, reason = engine_mod.canonical_computed_at(candidate)
    assert (canonical is None) != (reason is None), "exactly one of the two"
    if canonical is None:
        return
    assert contract_mod._ISO_UTC_RE.fullmatch(canonical), (
        f"engine canonicalised {candidate!r} to {canonical!r}, which the "
        "contract validator refuses — the two patterns have diverged"
    )
    # And the contract really does accept a record carrying it.
    result = execute(
        _request("add", [_op("a", "1", "one"), _op("b", "2", "one")], "one"),
        computed_at=candidate,
    )
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert result.computed_at == canonical


def test_the_two_timestamp_patterns_use_ascii_digits_only() -> None:
    """Pin the character class itself, not only its observable effect.

    ``numeric.py`` states this rule for numeric literals and the architecture
    notes repeat it; the timestamp patterns were the one place that did not
    follow it.
    """
    for pattern in (engine_mod._ISO_UTC_CANONICAL, engine_mod._ISO_UTC_OFFSET,
                    contract_mod._ISO_UTC_RE):
        assert "\\d" not in pattern.pattern, pattern.pattern
        assert "[0-9]" in pattern.pattern


def test_the_over_budget_batch_path_canonicalises_its_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``execute_batch``'s over-budget path calls ``_failure`` with the RAW value.

    A comment once claimed every ``_failure`` call site ran after
    canonicalisation, and carried a ``# pragma: no cover`` on the fallback. Both
    were wrong: this path bypasses ``execute`` entirely, so the fallback is
    reachable and is what keeps the record constructible.
    """
    monkeypatch.setattr(engine_mod, "PER_BATCH_BUDGET_S", -1.0)
    batch = CalculationBatch(requests=(
        _request("add", [_op("a", "1", "one"), _op("b", "2", "one")], "one",
                 request_id="first"),
        _request("add", [_op("a", "3", "one"), _op("b", "4", "one")], "one",
                 request_id="second"),
    ))
    results = execute_batch(batch, computed_at="not-a-timestamp")
    assert [r.request_id for r in results] == ["first", "second"]
    for result in results:
        assert result.computed_at == engine_mod.FALLBACK_COMPUTED_AT
    assert results[1].arithmetic_status == "RESOURCE_LIMIT_EXCEEDED"
