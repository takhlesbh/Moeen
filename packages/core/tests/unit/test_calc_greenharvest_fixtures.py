"""Acceptance fixtures: the calculations this package was built to get right.

These values are the measured GreenHarvest case. They live **here, in tests,
and nowhere else** — no fixture figure appears in production code or in any
prompt, because an engine tuned to one case proves nothing about the next one.

The two that matter most:

* ``52 kg/m2 x 110,000 m2 = 5,720,000 kg`` — the yield a specialist reported as
  ~678 tonnes after correctly stating both operands and the formula;
* ``5,720,000 kg / 110,000 m2 = 52 kg/m2`` — its inverse, which a draft
  dimensional table in Phase 1 rejected outright, which is why that table was
  removed.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from openexecutive.calc.contract import (
    CalculationRequest,
    Correlation,
    Operand,
    OperationId,
    RoundingMode,
)
from openexecutive.calc.engine import execute
from openexecutive.calc.units import Unit

AT = "2026-09-02T00:00:00Z"


def _corr() -> Correlation:
    return Correlation(specialist="cfo", case_id="greenharvest", run_id="acceptance")


def _op(
    oid: str, value: str, unit: str, role: str = "input", label: str | None = None
) -> Operand:
    return Operand(
        operand_id=oid, label=label or oid, value=value, unit=Unit(code=unit),
        basis="applicant_stated", role=role,  # type: ignore[arg-type]
    )


def _run(
    operation: str,
    operands: list[Operand],
    target: str,
    scale: int = 2,
    rounding: str = "ROUND_HALF_EVEN",
    **kw: object,
):
    request = CalculationRequest(
        request_id=f"gh-{operation}", operation=operation,  # type: ignore[arg-type]
        operands=tuple(operands), target_unit=Unit(code=target), scale=scale,
        rounding=rounding,  # type: ignore[arg-type]
        purpose="GreenHarvest acceptance", correlation=_corr(),
    )
    return execute(request, computed_at=AT, **kw)  # type: ignore[arg-type]


def _verified(result) -> None:
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED", result.errors
    assert result.fingerprint is not None and len(result.fingerprint) == 64
    # Arithmetic verified says nothing about the inputs: no evidence layer ran.
    assert result.evidence.status == "EVIDENCE_UNAVAILABLE"
    assert result.is_verified_evidence() is False


# ---------------------------------------------------------------------------
# Sources and uses
# ---------------------------------------------------------------------------

TND = "currency:TND"


def test_sources_minus_uses_surplus() -> None:
    """42,000,000 - 30,140,000 = 11,860,000 TND.

    The specialist held both totals in adjacent claims and never subtracted.
    """
    r = _run("subtract", [_op("sources", "42000000", TND), _op("uses", "30140000", TND)], TND)
    _verified(r)
    assert r.result_value == "11860000.00"
    assert r.exact_result == "11860000"
    assert r.result_unit is not None and r.result_unit.code == TND


@pytest.mark.parametrize(
    "part,whole,expected_prefix",
    [
        ("11860000", "30140000", "39.3497013935"),   # surplus / uses
        ("11860000", "42000000", "28.2380952381"),   # surplus / sources
        ("18000000", "30140000", "59.7213005972"),   # equity as % of capex
        ("18000000", "42000000", "42.8571428571"),   # equity as % of sources
    ],
)
def test_sources_and_uses_percentages(part: str, whole: str, expected_prefix: str) -> None:
    """The pair that exposes the mislabelled denominator.

    "43% of total capex" is 42.86% of *sources*; against capex it is 59.72%.
    """
    r = _run("percentage_of", [_op("part", part, TND), _op("whole", whole, TND)], "pct", scale=10)
    _verified(r)
    assert r.result_value == expected_prefix
    assert r.result_unit is not None and r.result_unit.code == "pct"


def test_percentage_rounds_only_at_the_requested_boundary() -> None:
    """The exact value is kept alongside the rounded one, not instead of it."""
    r = _run("percentage_of", [_op("part", "11860000", TND), _op("whole", "30140000", TND)],
             "pct", scale=2)
    _verified(r)
    assert r.result_value == "39.35"
    assert r.exact_result is not None
    assert r.exact_result.startswith("39.3497013934970139349701393497")
    assert Decimal(r.exact_result) != Decimal(r.result_value)


# ---------------------------------------------------------------------------
# Gross margin
# ---------------------------------------------------------------------------


def test_operating_cost_stack_sums_to_65_percent() -> None:
    components = ["15", "22", "8", "6", "10", "4"]
    r = _run("sum_components",
             [_op(f"c{i}", v, "pct", label=f"component {v}%") for i, v in enumerate(components)],
             "pct", scale=0)
    _verified(r)
    assert r.result_value == "65"
    # Component identity survives into the record: a 65% total is only
    # auditable if the reader can see the six lines behind it.
    assert r.expression_executed is not None
    for value in components:
        assert f"={value}" in r.expression_executed


def test_margin_is_one_hundred_minus_the_cost_stack() -> None:
    r = _run("subtract", [_op("total", "100", "pct"), _op("opex", "65", "pct")], "pct", scale=0)
    _verified(r)
    assert r.result_value == "35"


def test_claimed_sixty_versus_calculated_thirty_five_is_25_points() -> None:
    """25 **percentage points**, not 25 percent — the unit is the finding."""
    r = _run("percentage_point_difference",
             [_op("claimed", "60", "pct"), _op("calculated", "35", "pct")],
             "pct_point", scale=0)
    _verified(r)
    assert r.result_value == "25"
    assert r.result_unit is not None and r.result_unit.code == "pct_point"


# ---------------------------------------------------------------------------
# Production — tomato and pepper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hectares,square_metres,yield_kg_m2,kilograms,tonnes,stated_tonnes,difference",
    [
        ("11", "110000", "52", "5720000", "5720", "572", "5148"),
        ("7", "70000", "31", "2170000", "2170", "217", "1953"),
    ],
)
def test_production_chain_and_ten_times_conflict(
    hectares: str, square_metres: str, yield_kg_m2: str, kilograms: str,
    tonnes: str, stated_tonnes: str, difference: str,
) -> None:
    """Area -> production -> tonnes -> variance, the full measured chain."""
    area = _run("convert_unit", [_op("area", hectares, "ha")], "m2", scale=0)
    _verified(area)
    assert area.result_value == square_metres

    produced = _run("multiply",
                    [_op("yield", yield_kg_m2, "kg_per_m2"), _op("area", square_metres, "m2")],
                    "kg", scale=0)
    _verified(produced)
    assert produced.result_value == kilograms

    in_tonnes = _run("convert_unit", [_op("mass", kilograms, "kg")], "t", scale=0)
    _verified(in_tonnes)
    assert in_tonnes.result_value == tonnes

    variance = _run("variance",
                    [_op("calculated", tonnes, "t"),
                     _op("stated", stated_tonnes, "t", role="stated_comparison")],
                    "t", scale=0)
    _verified(variance)
    assert variance.result_value == difference
    assert variance.stated_value == stated_tonnes
    assert variance.ratio == "10"
    assert variance.conflict == "ORDER_OF_MAGNITUDE"


def test_the_motivating_inverse_division_is_recordable() -> None:
    """5,720,000 kg / 110,000 m2 = 52 kg/m2.

    A draft dimensional table in Phase 1 rejected exactly this as a "dimension
    mismatch: mass vs area". Its removal is why the engine owns operation
    semantics, and this test is the guard on that decision.
    """
    r = _run("divide", [_op("mass", "5720000", "kg"), _op("area", "110000", "m2")],
             "kg_per_m2", scale=0)
    _verified(r)
    assert r.result_value == "52"
    assert r.result_unit is not None and r.result_unit.code == "kg_per_m2"


# ---------------------------------------------------------------------------
# Offtake interval
# ---------------------------------------------------------------------------


def test_offtake_implied_production_interval_is_cross_paired() -> None:
    """[3200, 3500] t at [55, 60]% implies [5333.33…, 6363.64…] t.

    Division is decreasing in its denominator, so the bounds are 3200/0.60 and
    3500/0.55 — cross-paired. Same-index pairing gives [5818.18, 5833.33], a
    spuriously narrow band that *excludes* the correct 5,720 t and would let a
    cross-check pass when it should fail.
    """
    r = _run("interval_implied_total",
             [_op("vol_low", "3200", "t"), _op("vol_high", "3500", "t"),
              _op("cov_low", "55", "pct"), _op("cov_high", "60", "pct")],
             "t", scale=4)
    _verified(r)
    assert r.result_value == "5333.3333"
    assert r.warnings and r.warnings[0].startswith("upper bound 6363.63636363")

    lower = Decimal(r.exact_result or "0")
    upper = Decimal(r.warnings[0].split()[-1])
    assert lower < Decimal("5720") < upper, "the correct figure must fall inside"
    assert not (lower < Decimal("572") < upper), "the stated figure must fall outside"
    naive_low, naive_high = Decimal(3200) / Decimal("0.55"), Decimal(3500) / Decimal("0.60")
    assert not (naive_low < Decimal("5720") < naive_high), (
        "the naive same-index band must exclude the correct value — that is why "
        "the cross-paired band is required"
    )


def test_stated_volume_against_stated_production_is_impossible() -> None:
    """3,200 t as a share of a 572 t crop is 559.44% — physically impossible."""
    r = _run("percentage_of", [_op("offtake", "3200", "t"), _op("crop", "572", "t")],
             "pct", scale=4)
    _verified(r)
    assert r.result_value == "559.4406"
    assert Decimal(r.result_value) > 100


# ---------------------------------------------------------------------------
# Magnitude protection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("energy", ["650000", "750000"])
def test_energy_cost_keeps_its_magnitude(energy: str) -> None:
    """650,000–750,000 TND must never read as 65–75M or 650–750M.

    The measured failure quoted "~TND 75M/year" from a source saying
    650,000–750,000. Nothing here can rescale: a magnitude change requires an
    explicit registered conversion, and no currency scale unit exists.
    """
    r = _run("convert_unit", [_op("energy", energy, TND)], TND, scale=2)
    _verified(r)
    assert r.result_value == f"{energy}.00"
    assert Decimal(r.result_value) == Decimal(energy)
    assert Decimal(r.result_value).adjusted() == 5


def test_no_currency_scale_conversion_exists() -> None:
    """There is no ``currency:TND_millions`` to convert into."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Unit(code="currency:TND_millions")


@pytest.mark.parametrize(
    "operation,rounding,expected",
    [("subtract", "ROUND_HALF_EVEN", "0.12"), ("subtract", "ROUND_HALF_UP", "0.13"),
     ("subtract", "ROUND_DOWN", "0.12")],
)
def test_rounding_mode_is_honoured_at_the_boundary(
    operation: OperationId, rounding: RoundingMode, expected: str
) -> None:
    r = _run(operation, [_op("a", "0.125", "one"), _op("b", "0", "one")],
             "one", scale=2, rounding=rounding)
    _verified(r)
    assert r.result_value == expected
    assert r.exact_result == "0.125"
    assert r.rounding_applied == rounding
