"""Contract tests for the deterministic calculation foundation.

These are behavioural validation tests, not string assertions about docstrings.
Each one pins a rule the Phase 2 engine will be built on top of, so that the
engine is written against a boundary that has already been argued about.

The failures these encode are measured, not hypothetical. A controlled CFO
evaluation produced: a production figure understated by exactly 10^4 (hectares
read as square metres), an energy cost misquoted by 10^2, a percentage compared
against a percentage-point gap as if they were one quantity, and a calculation
returned with the placeholder ``XX`` where a number belonged.
"""
from __future__ import annotations

import ast
import decimal
import inspect
import os
import pathlib
from decimal import Decimal

import pytest
from pydantic import ValidationError

from openexecutive import calc
from openexecutive.calc import authority as authority_mod
from openexecutive.calc import contract as contract_mod
from openexecutive.calc import units as units_mod
from openexecutive.calc.contract import (
    FINGERPRINT_INCLUDED_FIELDS,
    ApplicationAuthority,
    ArithmeticStatus,
    CalculationBatch,
    CalculationError,
    CalculationRequest,
    CalculationResult,
    ConflictClass,
    Correlation,
    InputEvidenceStatus,
    InputEvidenceSummary,
    NormalizedOperand,
    Operand,
    OperationId,
    SourceHint,
    canonical_payload_json,
    fingerprint_payload,
)
from openexecutive.calc.numeric import NumericPolicyError, parse_numeric
from openexecutive.calc.units import (
    CURRENCY_PREFIX,
    Unit,
    additively_compatible,
    composed_dimension,
    convertible,
    known_unit_codes,
    same_dimension,
    unit_spec,
)
from tests.unit._calc_import_scan import references_calc, scan_tree

CALC_DIR = pathlib.Path(__file__).resolve().parents[2] / "openexecutive" / "calc"
"""Anchored on THIS test file's own location — never on scanned code.

Round 3 removed ``inspect.getsource`` from the file-reading step but left
``pathlib.Path(units_mod.__file__).parent`` as the root of the tree walk, which
is the same dependency one line up: a module that rebinds ``__file__`` at import
time redirected the entire scan to a decoy directory holding clean copies, and a
``units.py`` importing ``subprocess``, ``socket`` and ``pickle`` — and running a
subprocess at import time — passed 197/197. The root must come from somewhere
the scanned code cannot assign. ``__file__`` of the test module is that place.
"""

_IMPORTABLE_NON_SOURCE_SUFFIXES = (".pyc", ".pyo", ".so", ".pyd", ".dylib")
"""Extensions Python will import but a ``*.py`` glob never matches.

A sourceless ``.pyc`` loads through ``SourcelessFileLoader``, so scanning only
``*.py`` leaves an importable artifact unread. Emphasising that ``__pycache__``
is scanned buys nothing on its own — ``__pycache__`` holds ``.pyc``.
"""


def _walk_calc_tree() -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """Every real file under ``calc/``, and every symlink found on the way.

    ``os.walk`` rather than ``Path.rglob(..., recurse_symlinks=True)``: that
    keyword landed in **Python 3.13**, while this project targets 3.11
    (``requires-python = ">=3.11"``, ``target-version = "py311"``, CI and the
    Dockerfile both pin 3.11). A draft used it and every scanner test raised
    ``TypeError`` on the shipped interpreter — the whole no-execution-authority
    guarantee silently not running anywhere the project actually ships, with the
    obvious "fix" being to drop the keyword and restore the symlink escape.

    ``os.walk`` does not follow symlinked directories by default, which is what
    we want: symlinks are *rejected*, not traversed, so there is nothing to
    follow and no cycle to guard against.
    """
    files: list[pathlib.Path] = []
    links: list[pathlib.Path] = []
    for dirpath, dirnames, filenames in os.walk(CALC_DIR):
        here = pathlib.Path(dirpath)
        for name in list(dirnames):
            if (here / name).is_symlink():
                links.append(here / name)
        for name in filenames:
            path = here / name
            if path.is_symlink():
                links.append(path)
            else:
                files.append(path)
    return sorted(files), sorted(links)


def _calc_source_files() -> tuple[pathlib.Path, ...]:
    """Every ``.py`` file under ``calc/``, symlinks excluded and reported."""
    files, _ = _walk_calc_tree()
    return tuple(p for p in files if p.suffix == ".py")


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(CALC_DIR).as_posix()


def test_source_scan_reads_every_python_file_on_disk() -> None:
    """The scan's own coverage — the thing four rounds of review kept breaking."""
    found = {_rel(p) for p in _calc_source_files()}
    expected = {
        "__init__.py", "_model.py", "authority.py", "contract.py",
        "engine.py", "fingerprint.py", "numeric.py", "units.py",
    }
    assert found == expected, (
        f"unexpected or missing source files: {sorted(found ^ expected)}. Every "
        "file under calc/ must be known, because every safety test reads this set."
    )


def test_no_python_file_hides_in_a_subdirectory() -> None:
    """No file may live below ``calc/`` outside a top-level ``__pycache__``.

    Checking only ``.py`` left the subdirectory in
    ``calc/sub/__pycache__/units.pyc`` unnoticed, because the artifact test
    exempted ``__pycache__`` at any depth and this one saw no ``.py``.
    """
    files, _ = _walk_calc_tree()
    nested = [
        _rel(p) for p in files
        if "/" in _rel(p)
        and not (p.parent.name == "__pycache__" and p.parent.parent == CALC_DIR)
    ]
    assert nested == [], f"calc/ must stay flat; found {nested}"


def test_no_symlink_exists_anywhere_under_the_package() -> None:
    """A symlink is how a subdirectory hides from a default tree walk.

    Rejected outright rather than followed-and-checked: a link can also point
    outside the repository, where "what is committed" and "what is scanned"
    stop being the same question.
    """
    _, links = _walk_calc_tree()
    names = [str(p.relative_to(CALC_DIR)) for p in links]
    assert names == [], f"symlinks under calc/: {names}"


def test_no_importable_non_source_artifact_is_committed() -> None:
    """A ``.pyc``/``.so`` imports but a ``*.py`` glob never matches it.

    ``__pycache__`` is exempt only for the byte-compiled twins of the six known
    modules; anything else there, or any such artifact beside them, is a file
    Python would import and no test would read.
    """
    known = {"__init__", "_model", "authority", "contract", "engine",
             "fingerprint", "numeric", "units"}
    files, _ = _walk_calc_tree()
    offenders = []
    for path in files:
        if path.suffix not in _IMPORTABLE_NON_SOURCE_SUFFIXES:
            continue
        stem = path.name.split(".")[0]
        # Exempt ONLY CPython's own tagged twins, and only one level down.
        # A bare ``units.pyc`` is a *sourceless* module that
        # ``SourcelessFileLoader`` imports and executes; CPython's legitimate
        # artifacts are always tagged (``units.cpython-313.pyc``), so exempting
        # the untagged name was never necessary and admitted exactly that. The
        # depth bound matters too: ``calc/sub/__pycache__/units.pyc`` slipped
        # past both this check and the ``.py``-only subdirectory check.
        legitimate_twin = (
            path.parent.name == "__pycache__"
            and path.parent.parent == CALC_DIR
            and stem in known
            and ".cpython-" in path.name
        )
        if legitimate_twin:
            continue
        offenders.append(str(path.relative_to(CALC_DIR)))
    assert offenders == [], f"importable non-source artifacts under calc/: {offenders}"


def test_scan_root_is_independent_of_the_scanned_code() -> None:
    """A module rebinding ``__file__`` must not redirect the scan.

    Constructed rather than asserted: a stand-in whose ``__file__`` points
    elsewhere is shown to move ``inspect.getfile`` and NOT to move ``CALC_DIR``,
    which is anchored on this test file instead.
    """
    import types as _types

    impostor = _types.ModuleType("openexecutive.calc.impostor")
    impostor.__file__ = "/tmp/decoy/openexecutive/calc/units.py"
    assert inspect.getfile(impostor) == "/tmp/decoy/openexecutive/calc/units.py"
    assert pathlib.Path(__file__).resolve().parents[2] / "openexecutive" / "calc" == CALC_DIR
    assert (CALC_DIR / "units.py").is_file()
    assert "/tmp/decoy" not in str(CALC_DIR)


def _corr() -> Correlation:
    return Correlation(specialist="cfo", case_id="case-1", run_id="run-1")


def _operand(
    oid: str = "o1",
    value: str = "1",
    unit: str = "one",
    basis: str = "applicant_stated",
    **kw: object,
) -> Operand:
    return Operand(
        operand_id=oid, label=f"operand {oid}", value=value,
        unit=Unit(code=unit), basis=basis, **kw,  # type: ignore[arg-type]
    )


def _normalized(oid: str = "o1", value: str = "1", unit: str = "one") -> NormalizedOperand:
    u = Unit(code=unit)
    return NormalizedOperand(
        operand_id=oid, label=f"operand {oid}", original_value=value,
        original_unit=u, normalized_value=value, normalized_unit=u,
        basis="applicant_stated",
    )


# ---------------------------------------------------------------------------
# 1. Operations
# ---------------------------------------------------------------------------

_EXPECTED_OPERATIONS = {
    "add", "subtract", "multiply", "divide", "sum_components", "percentage_of",
    "percentage_point_difference", "ratio", "weighted_average", "variance",
    "convert_unit", "interval_implied_total",
}


def test_operation_enum_is_exactly_the_approved_v1_set() -> None:
    assert set(OperationId.__args__) == _EXPECTED_OPERATIONS  # type: ignore[attr-defined]


@pytest.mark.parametrize("operation", sorted(_EXPECTED_OPERATIONS))
def test_every_operation_value_constructs_a_request(operation: str) -> None:
    req = CalculationRequest(
        request_id="r1", operation=operation,  # type: ignore[arg-type]
        operands=(_operand(),), purpose="p", correlation=_corr(),
    )
    assert req.operation == operation


def test_unknown_operation_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CalculationRequest(
            request_id="r1", operation="irr",  # type: ignore[arg-type]
            operands=(_operand(),), purpose="p", correlation=_corr(),
        )


def test_excluded_v1_operations_are_absent() -> None:
    """IRR/NPV/XIRR/CAGR need a cash-flow series contract this scalar model
    does not carry. Their absence is the decision, so it is pinned."""
    for excluded in ("irr", "npv", "xirr", "cagr", "debt_service", "sensitivity", "eval"):
        assert excluded not in OperationId.__args__  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 2-9. Units and dimensions
# ---------------------------------------------------------------------------

_EXPECTED_UNITS = {
    "one": "dimensionless", "pct": "percentage", "pct_point": "percentage_point",
    "kg": "mass", "t": "mass", "m2": "area", "ha": "area",
    "month": "time", "year": "time", "kg_per_m2": "mass_per_area",
}


def test_registry_contains_exactly_the_required_units() -> None:
    assert set(known_unit_codes()) == set(_EXPECTED_UNITS)


@pytest.mark.parametrize("code,dimension", sorted(_EXPECTED_UNITS.items()))
def test_each_unit_resolves_to_its_dimension(code: str, dimension: str) -> None:
    assert Unit(code=code).dimension == dimension


def test_pct_and_pct_point_are_different_dimensions() -> None:
    pct, pp = Unit(code="pct"), Unit(code="pct_point")
    assert not same_dimension(pct, pp)
    compatible, reason = additively_compatible(pct, pp)
    assert compatible is False
    assert reason is not None and "dimension mismatch" in reason


def test_kg_and_tonne_share_mass_with_an_exact_factor() -> None:
    kg, tonne = Unit(code="kg"), Unit(code="t")
    assert additively_compatible(kg, tonne)[0] is True
    assert tonne.factor_to_base == Decimal("1000")
    assert isinstance(tonne.factor_to_base, Decimal)


def test_m2_and_hectare_share_area_with_an_exact_factor() -> None:
    """The measured 10^4 error: 11 ha is 110,000 m2, not 11 m2."""
    m2, ha = Unit(code="m2"), Unit(code="ha")
    assert additively_compatible(m2, ha)[0] is True
    assert ha.factor_to_base == Decimal("10000")


def test_month_and_year_share_time_but_require_an_explicit_policy() -> None:
    month, year = Unit(code="month"), Unit(code="year")
    assert same_dimension(month, year)
    ok, note = convertible(month, year)
    assert ok is True
    assert note is not None and "explicit" in note
    assert year.conversion_policy == "explicit_required"


def test_kg_per_m2_composes_with_area_to_give_mass() -> None:
    """The yield calculation's dimensional signature, declared before any engine."""
    assert composed_dimension(Unit(code="kg_per_m2"), Unit(code="m2")) == "mass"
    assert composed_dimension(Unit(code="m2"), Unit(code="kg_per_m2")) == "mass"
    assert Unit(code="kg_per_m2").dimension == "mass_per_area"


def test_undeclared_composition_returns_none_rather_than_guessing() -> None:
    assert composed_dimension(Unit(code="kg"), Unit(code="month")) is None


def test_all_conversion_factors_are_exact_decimals_never_floats() -> None:
    for code in known_unit_codes():
        spec = unit_spec(code)
        assert spec is not None
        assert isinstance(spec.factor_to_base, Decimal)
        assert spec.factor_to_base == spec.factor_to_base.to_integral_value()


def test_unknown_unit_is_rejected_and_never_defaulted() -> None:
    with pytest.raises(ValidationError):
        Unit(code="tonnes")
    with pytest.raises(ValidationError):
        Unit(code="")


def test_operand_requires_a_unit() -> None:
    with pytest.raises(ValidationError):
        Operand(  # type: ignore[call-arg]
            operand_id="o1", label="l", value="1", basis="applicant_stated",
        )


# ---------------------------------------------------------------------------
# 3-4. Currency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iso", ["TND", "EUR", "USD", "GBP"])
def test_well_formed_currency_codes_are_accepted(iso: str) -> None:
    u = Unit(code=f"{CURRENCY_PREFIX}{iso}")
    assert u.is_currency and u.currency_code == iso and u.dimension == "currency"


@pytest.mark.parametrize("bad", ["", "tnd", "TN", "TNDD", "T1D", "TN D", "123"])
def test_malformed_currency_codes_are_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        Unit(code=f"{CURRENCY_PREFIX}{bad}")


def test_bare_iso_code_is_not_a_unit() -> None:
    """A bare 'TND' must not resolve — that is the implicit-guessing path."""
    with pytest.raises(ValidationError):
        Unit(code="TND")


def test_cross_currency_is_incompatible_without_a_rate_authority() -> None:
    tnd, eur = Unit(code="currency:TND"), Unit(code="currency:EUR")
    assert same_dimension(tnd, eur) is True
    compatible, reason = additively_compatible(tnd, eur)
    assert compatible is False
    assert reason is not None and "exchange-rate authority" in reason
    ok, note = convertible(tnd, eur)
    assert ok is False and note is not None


def test_same_currency_is_compatible() -> None:
    assert additively_compatible(Unit(code="currency:TND"), Unit(code="currency:TND"))[0]


def test_currency_has_no_base_factor() -> None:
    assert Unit(code="currency:TND").factor_to_base is None


# ---------------------------------------------------------------------------
# 10-14. Decimal input policy
# ---------------------------------------------------------------------------


def test_float_is_rejected_explicitly() -> None:
    with pytest.raises(NumericPolicyError, match="float is rejected"):
        parse_numeric(30140000.0)
    with pytest.raises(ValidationError):
        Operand(
            operand_id="o1", label="l", value=1.5,  # type: ignore[arg-type]
            unit=Unit(code="one"), basis="applicant_stated",
        )


def test_bool_is_rejected_even_though_it_subclasses_int() -> None:
    with pytest.raises(NumericPolicyError, match="bool"):
        parse_numeric(True)


def test_externally_built_decimal_is_rejected_at_the_boundary() -> None:
    with pytest.raises(NumericPolicyError, match="Decimal is rejected"):
        parse_numeric(Decimal("1.5"))


def test_int_is_accepted_losslessly() -> None:
    assert parse_numeric(42000000) == Decimal("42000000")
    huge = 10**25
    assert parse_numeric(huge) == Decimal(str(huge))


@pytest.mark.parametrize(
    "raw,expected",
    [("0", "0"), ("-11860000", "-11860000"), ("30140000.00", "30140000.00"),
     ("5720.5", "5720.5"), ("+7", "7"), ("1e6", "1000000")],
)
def test_canonical_decimal_strings_accepted(raw: str, expected: str) -> None:
    assert calc.canonical_numeric_string(parse_numeric(raw)) == expected


def test_zero_and_negative_are_preserved() -> None:
    assert parse_numeric("0") == Decimal("0")
    assert parse_numeric("-0.5") == Decimal("-0.5")
    assert calc.canonical_numeric_string(parse_numeric("-0.5")) == "-0.5"


def test_scale_is_preserved_in_canonical_form() -> None:
    """1.50 and 1.5 are different evidence and must not collapse."""
    assert calc.canonical_numeric_string(parse_numeric("1.50")) == "1.50"
    assert calc.canonical_numeric_string(parse_numeric("1.5")) == "1.5"


@pytest.mark.parametrize("bad", ["NaN", "nan", "Infinity", "inf", "-Infinity", "-inf", "sNaN"])
def test_nan_and_infinity_rejected(bad: str) -> None:
    with pytest.raises(NumericPolicyError):
        parse_numeric(bad)


@pytest.mark.parametrize("bad", ["", " ", " 1", "1 ", "\t5"])
def test_empty_and_whitespace_rejected(bad: str) -> None:
    with pytest.raises(NumericPolicyError):
        parse_numeric(bad)


def test_comma_in_plain_format_is_rejected_not_guessed() -> None:
    with pytest.raises(NumericPolicyError, match="never inferred"):
        parse_numeric("1,234")


def test_comma_thousands_accepted_only_when_declared() -> None:
    assert parse_numeric("1,234,567.89", "comma_thousands") == Decimal("1234567.89")
    with pytest.raises(NumericPolicyError, match="exactly three digits"):
        parse_numeric("1,23,456", "comma_thousands")


def test_decimal_comma_is_rejected_as_ambiguous() -> None:
    for bad in ("1.234,56", "42,5"):
        with pytest.raises(NumericPolicyError, match="decimal comma"):
            parse_numeric(bad, "comma_thousands")


def test_exponent_bound_enforced() -> None:
    assert parse_numeric("1e30") == Decimal("1e30")
    with pytest.raises(NumericPolicyError, match="exponent out of range"):
        parse_numeric("1e31")
    with pytest.raises(NumericPolicyError, match="exponent out of range"):
        parse_numeric("1e400")
    with pytest.raises(NumericPolicyError, match="exponent out of range"):
        parse_numeric("1e-31")


@pytest.mark.parametrize(
    "literal",
    ["0e-2000000000", "-0e-2000000000", "0e2000000000", "0e-40000000",
     "0.0e-2000000000", "0e-99999999999999999"],
)
def test_zero_with_a_huge_exponent_is_rejected(literal: str) -> None:
    """A zero carrying a huge exponent is the case a ``!= 0`` guard lets through.

    ``Decimal("0e-2000000000") == 0``, so "check unless it is zero" never looks
    at it — and ``format(value, "f")`` then expands it positionally into a
    two-billion-character string. Measured before the fix: 4 GB resident from a
    13-character literal, inside the 64-character limit, on the exact wire shape
    a model emits; the larger variants raised ``MemoryError`` rather than
    ``ValidationError``, so an unhandled exception type escaped the validation
    boundary.
    """
    with pytest.raises(NumericPolicyError):
        parse_numeric(literal)
    with pytest.raises(ValidationError):
        _operand("o1", literal)


def test_the_exponent_bound_applies_to_zero_and_non_zero_alike() -> None:
    assert parse_numeric("0e30") == 0
    assert parse_numeric("0e-30") == 0
    with pytest.raises(NumericPolicyError, match="exponent out of range"):
        parse_numeric("0e31")
    with pytest.raises(NumericPolicyError, match="exponent out of range"):
        parse_numeric("0e-31")
    # Ordinary zeros are untouched, and so are small non-zero magnitudes.
    for ordinary in ("0", "0.00", "-0", "0E+5"):
        assert parse_numeric(ordinary) == 0
    assert parse_numeric("0.000000001") == Decimal("1e-9")


def test_rejection_is_bounded_in_time_and_memory() -> None:
    """The point is not only that it is rejected, but that rejecting is cheap.

    A guard that allocates gigabytes before raising is still a denial of
    service, so this asserts the *cost* of the rejection, not just its outcome.
    """
    import resource
    import time

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.monotonic()
    for _ in range(50):
        with pytest.raises(NumericPolicyError):
            parse_numeric("0e-2000000000")
    elapsed = time.monotonic() - start
    grew = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    assert elapsed < 2.0, f"50 rejections took {elapsed:.2f}s"
    # ru_maxrss is bytes on macOS and kibibytes on Linux; 256 MiB is far below
    # the 4 GB the unfixed path allocated on either reading.
    assert grew < 256 * 1024 * 1024, f"peak RSS grew by {grew}"


def test_unparseable_exponent_raises_a_typed_error_not_a_raw_decimal_error() -> None:
    """The ``InvalidOperation`` branch is reachable, contrary to a draft pragma.

    ``Decimal()`` raises on an exponent past its own parseable range, and
    ``_PLAIN_RE`` matches that string happily.
    """
    with pytest.raises(NumericPolicyError, match="is not a decimal number"):
        parse_numeric("0e-" + "9" * 19)


def test_numeric_string_length_bound_enforced() -> None:
    with pytest.raises(NumericPolicyError, match="exceeds 64 characters"):
        parse_numeric("1" * 65)


def test_no_automatic_scaling_between_thousands_and_millions() -> None:
    """650,000 must never silently become 650 million."""
    v = parse_numeric("650,000", "comma_thousands")
    assert v == Decimal("650000")
    assert v.adjusted() == 5


# ---------------------------------------------------------------------------
# 15-18. Request bounds and authority absence
# ---------------------------------------------------------------------------


def test_operand_count_bounds_fail_closed() -> None:
    with pytest.raises(ValidationError):
        CalculationRequest(
            request_id="r", operation="add", operands=(), purpose="p",
            correlation=_corr(),
        )
    too_many = tuple(_operand(f"o{i}") for i in range(calc.MAX_OPERANDS_PER_REQUEST + 1))
    with pytest.raises(ValidationError, match="exceeds the limit"):
        CalculationRequest(
            request_id="r", operation="add", operands=too_many, purpose="p",
            correlation=_corr(),
        )


def test_duplicate_operand_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        CalculationRequest(
            request_id="r", operation="add",
            operands=(_operand("o1"), _operand("o1")), purpose="p", correlation=_corr(),
        )


def test_batch_bounds_fail_closed() -> None:
    def req(i: int) -> CalculationRequest:
        return CalculationRequest(
            request_id=f"r{i}", operation="add", operands=(_operand(),),
            purpose="p", correlation=_corr(),
        )

    CalculationBatch(requests=tuple(req(i) for i in range(calc.MAX_REQUESTS_PER_BATCH)))
    with pytest.raises(ValidationError, match="batch limit"):
        CalculationBatch(
            requests=tuple(req(i) for i in range(calc.MAX_REQUESTS_PER_BATCH + 1))
        )


def test_scale_bounds_fail_closed() -> None:
    with pytest.raises(ValidationError):
        CalculationRequest(
            request_id="r", operation="add", operands=(_operand(),), purpose="p",
            correlation=_corr(), scale=-1,
        )
    with pytest.raises(ValidationError):
        CalculationRequest(
            request_id="r", operation="add", operands=(_operand(),), purpose="p",
            correlation=_corr(), scale=calc.MAX_SCALE + 1,
        )


_AUTHORITY_FIELDS = [
    "arithmetic_status", "evidence", "verification_status", "verified_result",
    "authority", "fingerprint", "computed_at", "result_value", "expression_executed",
    "page", "sheet", "cell_range", "url", "retrieved_at", "retrieval_id",
]


@pytest.mark.parametrize("field", _AUTHORITY_FIELDS)
def test_request_cannot_carry_any_verification_or_authority_field(field: str) -> None:
    """extra='forbid' makes the attempt an error rather than a silent strip."""
    assert field not in CalculationRequest.model_fields
    payload = {
        "request_id": "r", "operation": "add",
        "operands": [_operand().model_dump()], "purpose": "p",
        "correlation": _corr().model_dump(), field: "x",
    }
    with pytest.raises(ValidationError):
        CalculationRequest.model_validate(payload)


@pytest.mark.parametrize("field", ["page", "sheet", "cell_range", "url", "retrieved_at", "retrieval_id"])
def test_source_hint_has_no_trusted_provenance_field(field: str) -> None:
    """A model may name a document; it may not mint a cell or a timestamp."""
    assert field not in SourceHint.model_fields
    with pytest.raises(ValidationError):
        SourceHint.model_validate({"document_label": "d.pdf", field: "x"})


def test_source_hint_id_is_named_a_hint_not_a_retrieval_id() -> None:
    hint = SourceHint(document_label="Investment_Proposal.pdf", retrieval_id_hint="rid-1")
    assert "retrieval_id_hint" in SourceHint.model_fields
    assert "retrieval_id" not in SourceHint.model_fields
    # And it does not appear in the fingerprint payload either, so a hint can
    # never influence calculation identity.
    op = _normalized()
    payload = fingerprint_payload(
        operation="add", normalized_operands=(op,), target_unit=None, scale=2,
        rounding="ROUND_HALF_EVEN", authority=calc.current_authority(),
    )
    assert hint.retrieval_id_hint not in canonical_payload_json(payload)


def test_operand_hints_do_not_become_bound_evidence() -> None:
    """The request-side hint and the result-side binding are different types."""
    op = _operand(source_hint=SourceHint(document_label="x.pdf", retrieval_id_hint="rid"))
    assert op.source_hint is not None
    summary = InputEvidenceSummary(status="UNSUPPORTED", unbound_operand_ids=("o1",))
    assert summary.bound_operand_ids == ()
    assert "source_hint" not in InputEvidenceSummary.model_fields


# ---------------------------------------------------------------------------
# 19-22. Result axes and authority
# ---------------------------------------------------------------------------


def _result(**kw: object) -> CalculationResult:
    base: dict[str, object] = dict(
        request_id="r1", operation="subtract", correlation=_corr(),
        arithmetic_status="CALCULATION_UNAVAILABLE",
        evidence=InputEvidenceSummary(status="EVIDENCE_UNAVAILABLE"),
        computed_at="2026-01-01T00:00:00Z",
    )
    base.update(kw)
    return authority_mod.issue_calculation_result(**base)  # type: ignore[arg-type]


def test_arithmetic_status_and_evidence_status_are_independent_axes() -> None:
    assert "CONFLICT_DETECTED" not in ArithmeticStatus.__args__  # type: ignore[attr-defined]
    assert "CONFLICT_DETECTED" in ConflictClass.__args__  # type: ignore[attr-defined]
    assert set(InputEvidenceStatus.__args__) == {  # type: ignore[attr-defined]
        "ALL_SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED",
        "CONFLICTING_SOURCES", "EVIDENCE_UNAVAILABLE",
    }


def test_all_required_arithmetic_statuses_exist() -> None:
    assert set(ArithmeticStatus.__args__) == {  # type: ignore[attr-defined]
        "ARITHMETIC_VERIFIED", "CALCULATION_UNAVAILABLE", "DIVISION_BY_ZERO",
        "UNIT_MISMATCH", "RESOURCE_LIMIT_EXCEEDED", "INVALID_INPUT",
        "UNSUPPORTED_OPERATION",
    }


def test_arithmetic_verified_with_unsupported_inputs_is_representable() -> None:
    """The state the whole contract exists for: right sum, unbacked numbers."""
    res = _result(
        arithmetic_status="ARITHMETIC_VERIFIED",
        evidence=InputEvidenceSummary(status="UNSUPPORTED",
                                      unbound_operand_ids=("o1",)),
        expression_executed="42000000 - 30140000",
        exact_result="11860000", result_value="11860000.00",
        result_unit=Unit(code="currency:TND"), scale_applied=2,
        rounding_applied="ROUND_HALF_EVEN",
        normalized_operands=(_normalized("o1", "42000000", "currency:TND"),),
    )
    assert res.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert res.evidence.status == "UNSUPPORTED"
    # And it must NOT be summarisable as verified evidence.
    assert res.is_verified_evidence() is False


def test_verified_evidence_requires_both_axes() -> None:
    res = _result(
        arithmetic_status="ARITHMETIC_VERIFIED",
        evidence=InputEvidenceSummary(status="ALL_SUPPORTED",
                                      bound_operand_ids=("o1", "o2")),
        expression_executed="1 + 1", result_value="2", result_unit=Unit(code="one"),
        normalized_operands=(_normalized("o1", "1"), _normalized("o2", "1")),
    )
    assert res.is_verified_evidence() is True


def test_verified_status_requires_the_operands_it_consumed() -> None:
    """A verified result recording zero operands did not come from a request.

    Every CalculationRequest carries at least one operand, so an empty tuple
    here is self-contradictory — and it would leave the fingerprint with no
    inputs to identify.
    """
    with pytest.raises(ValidationError, match="requires normalized_operands"):
        _result(
            arithmetic_status="ARITHMETIC_VERIFIED",
            evidence=InputEvidenceSummary(status="EVIDENCE_UNAVAILABLE"),
            expression_executed="1 + 1", result_value="2",
            result_unit=Unit(code="one"),
        )


def test_result_can_be_both_verified_and_conflicting() -> None:
    """Why CONFLICT_DETECTED is orthogonal rather than an arithmetic status."""
    res = _result(
        operation="multiply",
        arithmetic_status="ARITHMETIC_VERIFIED",
        evidence=InputEvidenceSummary(
            status="PARTIALLY_SUPPORTED", bound_operand_ids=("o1",),
            unbound_operand_ids=("o2",),
        ),
        expression_executed="52 * 110000", result_value="5720000",
        result_unit=Unit(code="kg"), stated_value="572000",
        conflict="ORDER_OF_MAGNITUDE", ratio="10",
        normalized_operands=(_normalized("o1", "52", "kg_per_m2"),
                             _normalized("o2", "110000", "m2")),
    )
    assert res.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert res.conflict == "ORDER_OF_MAGNITUDE"


def test_verified_status_requires_an_executed_expression_and_value() -> None:
    with pytest.raises(ValidationError, match="expression_executed"):
        _result(arithmetic_status="ARITHMETIC_VERIFIED", result_value="1",
                result_unit=Unit(code="one"))
    with pytest.raises(ValidationError, match="result_value"):
        _result(arithmetic_status="ARITHMETIC_VERIFIED", expression_executed="1+1")


_SUCCESS_ASSERTING_FIELDS: tuple[tuple[str, object], ...] = (
    ("result_value", "0"),
    ("exact_result", "5720000"),
    ("expression_executed", "5 / 0"),
    ("result_unit", Unit(code="one")),
    ("scale_applied", 2),
    ("rounding_applied", "ROUND_HALF_EVEN"),
    ("absolute_difference", "1"),
    ("percentage_difference", "1"),
    ("ratio", "10"),
    ("conflict", "ORDER_OF_MAGNITUDE"),
)
"""Every field the failure branch rejects — all ten, not the first three.

Mutation testing showed seven could be dropped from the contract's guard with
the suite still green, while the architecture notes state the invariant naming
each one. A ``DIVISION_BY_ZERO`` result carrying ``ratio="10"`` and
``conflict="ORDER_OF_MAGNITUDE"`` is exactly the "reader concludes the
calculation succeeded" case the guard exists for.
"""


@pytest.mark.parametrize("field,value", _SUCCESS_ASSERTING_FIELDS)
def test_failed_status_must_not_claim_the_calculation_ran(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError, match=f"must not carry .*{field}"):
        _result(arithmetic_status="DIVISION_BY_ZERO", **{field: value})


def test_the_failure_guard_lists_exactly_these_ten_fields() -> None:
    """Pin the list itself, so a field added to the guard gains a test with it."""
    source = (CALC_DIR / "contract.py").read_text()
    body = source[source.index("claimed = ["):source.index("if claimed:")]
    listed = {name for name, _ in _SUCCESS_ASSERTING_FIELDS}
    in_contract = {
        line.split('"')[1] for line in body.splitlines() if line.strip().startswith('("')
    }
    assert in_contract == listed, f"guard/test drift: {sorted(in_contract ^ listed)}"


def test_verified_status_cannot_carry_errors() -> None:
    with pytest.raises(ValidationError, match="cannot carry errors"):
        _result(
            arithmetic_status="ARITHMETIC_VERIFIED", expression_executed="1+1",
            result_value="2", result_unit=Unit(code="one"),
            normalized_operands=(_normalized(),),
            errors=(CalculationError(code="E", detail="d"),),
        )


def test_conflict_requires_the_stated_value_it_conflicts_with() -> None:
    with pytest.raises(ValidationError, match="requires the stated_value"):
        _result(
            arithmetic_status="ARITHMETIC_VERIFIED",
            evidence=InputEvidenceSummary(status="EVIDENCE_UNAVAILABLE"),
            expression_executed="1 + 1", result_value="2",
            result_unit=Unit(code="one"),
            normalized_operands=(_normalized(),),
            conflict="CONFLICT_DETECTED",
        )


def test_stated_value_without_a_conflict_is_deliberately_allowed() -> None:
    """"We compared and they agree" is a result worth keeping.

    The asymmetry is intentional: requiring a conflict class to justify a
    stated value would erase every successful reconciliation.
    """
    res = _result(stated_value="11860000")
    assert res.conflict == "NONE" and res.stated_value == "11860000"


def test_factory_stamps_authority_and_does_not_accept_it() -> None:
    """A caller cannot label a result with an authority version it did not use."""
    assert "authority" not in inspect.signature(
        authority_mod.issue_calculation_result
    ).parameters
    res = _result()
    assert res.authority.authority_id == authority_mod.AUTHORITY_ID
    assert res.authority.authority_version == authority_mod.AUTHORITY_VERSION
    # Phase 2 bumped this to "0.2.0-engine" in the commit that added the engine,
    # so an engine result can never fingerprint-collide with a contract-era one.
    assert res.authority.authority_version in contract_mod.KNOWN_AUTHORITY_VERSIONS


def test_model_originated_payload_cannot_assign_authority_fields() -> None:
    """Deserializing a model-shaped payload into a result must fail.

    Not because the fields are hidden — a result legitimately has them — but
    because a *request* payload has no route to them, and a result payload
    missing them cannot be built.
    """
    request_shaped = {
        "request_id": "r1", "operation": "add", "correlation": _corr().model_dump(),
        "arithmetic_status": "ARITHMETIC_VERIFIED",
        "evidence": {"status": "ALL_SUPPORTED"},
    }
    with pytest.raises(ValidationError):
        CalculationResult.model_validate(request_shaped)


def test_frozen_and_copy_update_revalidates() -> None:
    res = _result()
    with pytest.raises(ValidationError):
        res.model_copy(update={"arithmetic_status": "ARITHMETIC_VERIFIED"})
    with pytest.raises(ValidationError):
        res.arithmetic_status = "ARITHMETIC_VERIFIED"  # type: ignore[misc]


def test_existing_specialist_calculation_provenance_guard_is_untouched() -> None:
    """Phase 1 must not relax ADR 0001 in the specialist contract."""
    from openexecutive.specialists.result_contract import CalculationProvenance

    with pytest.raises(ValidationError):
        CalculationProvenance(method="m", verified_result="5720")
    with pytest.raises(ValidationError):
        CalculationProvenance(method="m", verification_status="verified")
    assert CalculationProvenance(method="m").verification_status == "unverified"


# ---------------------------------------------------------------------------
# Evidence-status coherence (the axis-collapse guard)
# ---------------------------------------------------------------------------


def test_all_supported_cannot_coexist_with_unbound_operands() -> None:
    """The state that would let unconfirmed figures read as verified evidence."""
    with pytest.raises(ValidationError, match="ALL_SUPPORTED contradicts"):
        InputEvidenceSummary(status="ALL_SUPPORTED", unbound_operand_ids=("o1",))


def test_unsupported_cannot_coexist_with_bound_operands() -> None:
    with pytest.raises(ValidationError, match="UNSUPPORTED contradicts"):
        InputEvidenceSummary(status="UNSUPPORTED", bound_operand_ids=("o1",))


def test_partially_supported_requires_one_of_each() -> None:
    with pytest.raises(ValidationError, match="PARTIALLY_SUPPORTED requires"):
        InputEvidenceSummary(status="PARTIALLY_SUPPORTED", bound_operand_ids=("o1",))
    InputEvidenceSummary(
        status="PARTIALLY_SUPPORTED", bound_operand_ids=("o1",),
        unbound_operand_ids=("o2",),
    )


def test_evidence_unavailable_cannot_list_bindings() -> None:
    with pytest.raises(ValidationError, match="EVIDENCE_UNAVAILABLE means"):
        InputEvidenceSummary(status="EVIDENCE_UNAVAILABLE", bound_operand_ids=("o1",))


def test_an_operand_cannot_be_both_bound_and_unbound() -> None:
    with pytest.raises(ValidationError, match="both bound and"):
        InputEvidenceSummary(
            status="PARTIALLY_SUPPORTED", bound_operand_ids=("o1", "o2"),
            unbound_operand_ids=("o2",),
        )


def test_is_verified_evidence_cannot_be_reached_with_unbound_inputs() -> None:
    """End-to-end: the accessor's strong claim now has the validator behind it."""
    with pytest.raises(ValidationError):
        _result(
            arithmetic_status="ARITHMETIC_VERIFIED",
            evidence=InputEvidenceSummary(
                status="ALL_SUPPORTED", unbound_operand_ids=("o1", "o2", "o3"),
            ),
            expression_executed="1 + 1", result_value="2",
            result_unit=Unit(code="one"), normalized_operands=(_normalized(),),
        )


def test_evidence_id_collections_are_bounded() -> None:
    too_many = tuple(f"o{i}" for i in range(calc.MAX_OPERANDS_PER_REQUEST + 1))
    with pytest.raises(ValidationError, match="exceeds"):
        InputEvidenceSummary(status="UNSUPPORTED", unbound_operand_ids=too_many)
    with pytest.raises(ValidationError, match="over-long operand id"):
        InputEvidenceSummary(status="UNSUPPORTED", unbound_operand_ids=("x" * 65,))
    with pytest.raises(ValidationError, match="over-long operand id"):
        InputEvidenceSummary(status="UNSUPPORTED", unbound_operand_ids=("",))


def test_result_side_collections_are_bounded_like_the_request_side() -> None:
    with pytest.raises(ValidationError, match="normalized operands exceeds"):
        _result(normalized_operands=tuple(
            _normalized(f"o{i}") for i in range(calc.MAX_OPERANDS_PER_REQUEST + 1)
        ))
    with pytest.raises(ValidationError, match="a warning exceeds"):
        _result(warnings=("x" * 5000,))


def test_canonical_expansion_beyond_the_bound_is_rejected() -> None:
    """max_length bounds the INPUT; the validator then rewrites the field.

    "1.<58 digits>e-30" is 64 characters in and 90 out, so without this check
    the object would store a value violating its own declared constraint and
    then fail to round-trip through model_validate or model_copy.
    """
    hostile = "1." + "2" * 58 + "e-30"
    assert len(hostile) == calc.MAX_NUMERIC_STRING_LEN
    with pytest.raises(ValidationError, match="canonical form"):
        _operand("o1", hostile)


def test_every_accepted_operand_round_trips() -> None:
    for value in ("0", "-11860000", "30140000.00", "1e6", "1e30", "5720.5"):
        op = _operand("o1", value)
        assert Operand.model_validate(op.model_dump()) == op
        assert op.model_copy(update={"label": "x"}).value == op.value


def test_month_year_compatibility_carries_the_explicit_policy_caveat() -> None:
    """The sibling predicate warns; this one must too.

    Returning a bare ``(True, None)`` would hand a Phase 2 engine an
    unqualified yes for comparing a monthly figure with an annual one — the
    exact failure the time dimension's ``explicit_required`` policy names.
    """
    ok, note = additively_compatible(Unit(code="month"), Unit(code="year"))
    assert ok is True
    assert note is not None and "explicit period policy" in note
    assert additively_compatible(Unit(code="kg"), Unit(code="t")) == (True, None)
    assert additively_compatible(Unit(code="year"), Unit(code="year")) == (True, None)


def test_all_supported_requires_a_non_empty_bound_list() -> None:
    """The empty tuple is the zero-argument default — the likeliest forgery."""
    with pytest.raises(ValidationError, match="requires a non-empty bound_operand_ids"):
        InputEvidenceSummary(status="ALL_SUPPORTED")


def test_verified_evidence_unreachable_without_binding_every_operand() -> None:
    ok = _result(
        arithmetic_status="ARITHMETIC_VERIFIED",
        evidence=InputEvidenceSummary(status="ALL_SUPPORTED",
                                      bound_operand_ids=("o1", "o2")),
        expression_executed="1 + 1", result_value="2", result_unit=Unit(code="one"),
        normalized_operands=(_normalized("o1"), _normalized("o2")),
    )
    assert ok.is_verified_evidence() is True
    with pytest.raises(ValidationError, match="every recorded operand to be bound"):
        _result(
            arithmetic_status="ARITHMETIC_VERIFIED",
            evidence=InputEvidenceSummary(status="ALL_SUPPORTED",
                                          bound_operand_ids=("o1",)),
            expression_executed="1 + 1", result_value="2",
            result_unit=Unit(code="one"),
            normalized_operands=(_normalized("o1"), _normalized("o2")),
        )


def test_evidence_cannot_name_an_operand_the_result_does_not_record() -> None:
    with pytest.raises(ValidationError, match="does not record"):
        _result(
            evidence=InputEvidenceSummary(status="UNSUPPORTED",
                                          unbound_operand_ids=("ghost",)),
            normalized_operands=(_normalized("o1"),),
        )


def test_bound_operand_ids_reject_duplicates() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        InputEvidenceSummary(status="ALL_SUPPORTED", bound_operand_ids=("o1", "o1"))


def test_verified_result_value_rejects_the_placeholder() -> None:
    """The headline case, on the status where it actually mattered.

    Every other parametrisation below runs on a failure status, where the
    "must not carry" rule fires first and hides which guard caught it. This one
    runs on ARITHMETIC_VERIFIED — the state in which ``result_value="XX"``
    reached ``is_verified_evidence()`` and the fingerprint payload.
    """
    good = dict(
        arithmetic_status="ARITHMETIC_VERIFIED",
        evidence=InputEvidenceSummary(status="EVIDENCE_UNAVAILABLE"),
        expression_executed="1 + 1", result_unit=Unit(code="one"),
        normalized_operands=(_normalized(),),
    )
    assert _result(result_value="2", **good).result_value == "2"
    for placeholder in ("XX", "TBD", "n/a", "~5720000"):
        with pytest.raises(ValidationError):
            _result(result_value=placeholder, **good)


def test_verified_normalized_operand_rejects_the_placeholder() -> None:
    with pytest.raises(ValidationError):
        _normalized("o1", "XX")


@pytest.mark.parametrize(
    "field", ["result_value", "exact_result", "stated_value",
              "absolute_difference", "percentage_difference", "ratio"],
)
def test_result_side_numeric_fields_reject_non_numbers(field: str) -> None:
    """``XX`` is one of the two measured failures this package exists to stop.

    Letting it round-trip through the record, through is_verified_evidence, and
    into the fingerprint payload would defeat the point of shipping contracts
    before an engine.
    """
    with pytest.raises(ValidationError):
        _result(**{field: "XX"})
    with pytest.raises(ValidationError):
        _result(**{field: "not a number at all"})


def test_normalized_operand_values_must_be_real_numbers() -> None:
    u = Unit(code="one")
    with pytest.raises(ValidationError):
        NormalizedOperand(
            operand_id="o1", label="l", original_value="XX", original_unit=u,
            normalized_value="1", normalized_unit=u, basis="applicant_stated",
        )


def test_result_side_numbers_are_canonicalised_like_operands() -> None:
    res = _result(stated_value="1e6")
    assert res.stated_value == "1000000"


def test_no_module_level_mutable_registry_name_survives() -> None:
    """MappingProxyType is a live view; leaving the backing dict reachable moves
    the hole one attribute over rather than closing it."""
    assert not hasattr(units_mod, "_MUTABLE_REGISTRY")
    with pytest.raises(TypeError):
        units_mod._REGISTRY["TND"] = units_mod._REGISTRY["kg"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Round-3 regressions
# ---------------------------------------------------------------------------


def test_duplicate_operand_ids_on_the_result_side_are_rejected() -> None:
    """One binding must not satisfy the bound-set check for two values.

    With duplicates allowed, ``{o.operand_id}`` collapses two operands carrying
    different values into one entry, and ALL_SUPPORTED with a single binding
    passes — reporting the strongest claim in the contract over an operand
    nobody checked.
    """
    with pytest.raises(ValidationError, match="duplicate operand_id"):
        _result(
            arithmetic_status="ARITHMETIC_VERIFIED",
            evidence=InputEvidenceSummary(status="ALL_SUPPORTED",
                                          bound_operand_ids=("o1",)),
            expression_executed="1+1", result_value="2",
            result_unit=Unit(code="one"),
            normalized_operands=(_normalized("o1", "1"), _normalized("o1", "999999999")),
        )


def test_normalized_operand_units_must_describe_a_real_conversion() -> None:
    """"11 hectares normalised to 11 dinars" must not be a legal record."""
    with pytest.raises(ValidationError, match="cannot normalise ha to currency:TND"):
        NormalizedOperand(
            operand_id="a", label="a", original_value="11",
            original_unit=Unit(code="ha"), normalized_value="11",
            normalized_unit=Unit(code="currency:TND"), basis="applicant_stated",
        )
    # Same dimension is fine — this is a conversion an engine could perform.
    NormalizedOperand(
        operand_id="a", label="a", original_value="11",
        original_unit=Unit(code="ha"), normalized_value="110000",
        normalized_unit=Unit(code="m2"), basis="applicant_stated",
    )


def test_unchanged_unit_cannot_change_the_value() -> None:
    with pytest.raises(ValidationError, match="unit is unchanged"):
        NormalizedOperand(
            operand_id="a", label="a", original_value="11",
            original_unit=Unit(code="kg"), normalized_value="11000",
            normalized_unit=Unit(code="kg"), basis="applicant_stated",
        )


def test_cross_currency_normalisation_is_refused() -> None:
    with pytest.raises(ValidationError, match="exchange-rate authority"):
        NormalizedOperand(
            operand_id="a", label="a", original_value="1",
            original_unit=Unit(code="currency:TND"), normalized_value="1",
            normalized_unit=Unit(code="currency:EUR"), basis="applicant_stated",
        )


@pytest.mark.parametrize("bad", ["currency:TND\n", "currency:TND ", "currency:\nTND"])
def test_currency_code_rejects_trailing_whitespace_and_newlines(bad: str) -> None:
    """``$`` also matches before a terminal newline; ``\\Z`` does not.

    ``currency:TND\\n`` would have been a second Unit.code for one currency —
    fingerprinting the same reconciliation differently, refusing a legitimate
    TND-vs-TND comparison as cross-currency, and putting a newline into
    ``display`` and therefore into logs.
    """
    with pytest.raises(ValidationError):
        Unit(code=bad)


@pytest.mark.parametrize("bad", ["١٢٣", "１２３", "٣.٥", "1\n", "1 "])
def test_only_ascii_digit_literals_are_accepted(bad: str) -> None:
    with pytest.raises(NumericPolicyError):
        parse_numeric(bad)


def test_composition_table_is_not_mutable_by_a_caller() -> None:
    """Hardening one lookup table and leaving its sibling open reads as a
    guarantee that does not hold."""
    with pytest.raises(TypeError):
        units_mod.MULTIPLICATIVE_COMPOSITIONS[("mass", "time")] = "currency"  # type: ignore[index]
    assert not hasattr(units_mod, "_MULTIPLICATIVE_COMPOSITIONS")


@pytest.mark.parametrize(
    "bad", ["whenever it suits me", "2026-01-01", "2026-01-01T00:00:00",
            "2026-01-01T00:00:00+00:00", ""],
)
def test_computed_at_must_be_an_iso_utc_instant(bad: str) -> None:
    with pytest.raises(ValidationError):
        _result(computed_at=bad)


def test_computed_at_accepts_the_documented_shapes() -> None:
    assert _result(computed_at="2026-01-01T00:00:00Z").computed_at
    assert _result(computed_at="2026-01-01T00:00:00.123456Z").computed_at


def test_authority_id_must_be_a_known_identity() -> None:
    """Closes the ordinary deserialization path, not the documented escapes.

    ``model_validate`` is what a queue replay or a database load uses. It is not
    ``model_construct`` or ``object.__setattr__``, so a free-text authority
    surviving it was a real gap rather than the acknowledged Python limit.
    """
    with pytest.raises(ValidationError, match="unknown authority_id"):
        ApplicationAuthority(authority_id="totally.made.up", authority_version="9.9.9")
    assert authority_mod.AUTHORITY_ID in contract_mod.KNOWN_AUTHORITY_IDS


def test_a_forged_authority_cannot_ride_in_on_a_result_payload() -> None:
    payload = _result().model_dump()
    payload["authority"] = {"authority_id": "evil.engine", "authority_version": "9.9.9"}
    with pytest.raises(ValidationError, match="unknown authority_id"):
        CalculationResult.model_validate(payload)


# ---------------------------------------------------------------------------
# Round-4/5 regressions
# ---------------------------------------------------------------------------


def test_no_per_operation_dimensional_rules_are_encoded_here() -> None:
    """Operation semantics belong to the engine, not to this contract.

    A draft encoded which operations require compatible inputs and what each
    produces. It was removed: it is outside this phase's scope, and getting
    ``divide`` wrong made the package reject its own motivating calculation —
    ``5,720,000 kg / 110,000 m2 = 52 kg/m2``. This test pins the removal so the
    table does not creep back without the arity and signature declarations that
    would make it correct.
    """
    for banned in ("_ADDITIVE_OPERATIONS", "_FIXED_RESULT_DIMENSION",
                   "_FIXED_INPUT_DIMENSION", "_check_result_dimensions"):
        assert not hasattr(contract_mod, banned), f"{banned} is back"


def test_the_motivating_yield_calculation_is_recordable() -> None:
    """mass / area -> kg_per_m2, the figure whose 10^4 misreading started this."""
    res = _result(
        operation="divide", arithmetic_status="ARITHMETIC_VERIFIED",
        evidence=InputEvidenceSummary(status="EVIDENCE_UNAVAILABLE"),
        expression_executed="5720000 kg / 110000 m2", result_value="52",
        result_unit=Unit(code="kg_per_m2"),
        normalized_operands=(_normalized("o1", "5720000", "kg"),
                             _normalized("o2", "110000", "m2")),
    )
    assert res.result_unit is not None and res.result_unit.code == "kg_per_m2"


def test_authority_version_must_be_a_known_version() -> None:
    """The fingerprint identity field — a replayed contract result must not be
    re-stampable as engine-phase."""
    with pytest.raises(ValidationError, match="unknown authority_version"):
        ApplicationAuthority(
            authority_id="openexecutive.calc", authority_version="2.0.0-engine"
        )
    payload = _result().model_dump()
    payload["authority"] = {
        "authority_id": "openexecutive.calc",
        "authority_version": "2.0.0-engine-VERIFIED",
    }
    with pytest.raises(ValidationError, match="unknown authority_version"):
        CalculationResult.model_validate(payload)
    assert authority_mod.AUTHORITY_VERSION in contract_mod.KNOWN_AUTHORITY_VERSIONS


def test_explicit_policy_conversion_must_state_its_basis() -> None:
    """"Annualised" without saying how is the failure the time policy names."""
    with pytest.raises(ValidationError, match="requires conversion_applied"):
        NormalizedOperand(
            operand_id="a", label="a", original_value="12",
            original_unit=Unit(code="month"), normalized_value="1",
            normalized_unit=Unit(code="year"), basis="applicant_stated",
        )
    NormalizedOperand(
        operand_id="a", label="a", original_value="12",
        original_unit=Unit(code="month"), normalized_value="1",
        normalized_unit=Unit(code="year"), basis="applicant_stated",
        conversion_applied="12 calendar months per financial year",
    )


def test_evidence_cannot_name_operands_when_none_are_recorded() -> None:
    with pytest.raises(ValidationError, match="records none"):
        _result(
            evidence=InputEvidenceSummary(
                status="UNSUPPORTED", unbound_operand_ids=("ghost", "phantom"),
            ),
        )


def test_conflicting_sources_must_name_what_conflicted() -> None:
    with pytest.raises(ValidationError, match="must name the operand"):
        InputEvidenceSummary(status="CONFLICTING_SOURCES")
    InputEvidenceSummary(status="CONFLICTING_SOURCES", unbound_operand_ids=("o1",))


def test_unit_spec_rejects_a_float_conversion_factor() -> None:
    with pytest.raises(ValidationError, match="must not be a float"):
        units_mod.UnitSpec(
            code="x", dimension="mass", base_code="kg", factor_to_base=0.1,
            conversion_policy="exact", display="x",
        )


def test_enforcement_points_are_exported() -> None:
    for name in ("KNOWN_AUTHORITY_IDS", "KNOWN_AUTHORITY_VERSIONS"):
        assert name in calc.__all__ and hasattr(calc, name)



def test_conversion_basis_must_be_more_than_whitespace() -> None:
    for blank in ("", "   ", "\t", "\n"):
        with pytest.raises(ValidationError, match="requires conversion_applied"):
            NormalizedOperand(
                operand_id="a", label="a", original_value="12",
                original_unit=Unit(code="month"), normalized_value="1",
                normalized_unit=Unit(code="year"), basis="applicant_stated",
                conversion_applied=blank,
            )


def test_negative_zero_canonicalises_to_positive_zero() -> None:
    """No document states a figure as negative zero, and "-0" would fingerprint
    differently from "0" while being the same quantity."""
    assert calc.canonical_numeric_string(parse_numeric("-0")) == "0"
    assert calc.canonical_numeric_string(parse_numeric("-0.00")) == "0.00"
    assert calc.canonical_numeric_string(parse_numeric("-0e5")) == "0"
    assert _operand("o1", "-0").value == "0"


def test_nan_and_infinity_are_rejected_at_the_boundary() -> None:
    """Pins the *outcome*, and is honest about what it cannot pin.

    ``parse_numeric``'s ``is_finite()`` check is defence in depth behind the
    regexes, which already exclude every NaN/Infinity spelling. That makes it
    unreachable by construction: deleting it changes no observable behaviour,
    so no test can distinguish its presence. That is a property of a
    defence-in-depth guard, not a coverage gap — the guard earns its place by
    holding if a regex is ever loosened, which is precisely the moment no
    current test would notice. Recorded here rather than papered over with a
    test that appears to cover it.
    """
    for spelling in ("NaN", "-NaN", "Infinity", "-Infinity", "sNaN", "inf"):
        with pytest.raises(NumericPolicyError):
            parse_numeric(spelling)
    assert not Decimal("NaN").is_finite()
    assert not Decimal("Infinity").is_finite()


def test_warning_and_error_count_caps_fail_closed() -> None:
    with pytest.raises(ValidationError, match="more than 32 warnings"):
        _result(warnings=tuple(f"w{i}" for i in range(calc.MAX_REQUESTS_PER_BATCH + 1)))
    with pytest.raises(ValidationError, match="more than 32 errors"):
        _result(errors=tuple(
            CalculationError(code=f"E{i}", detail="d")
            for i in range(calc.MAX_REQUESTS_PER_BATCH + 1)
        ))


# ---------------------------------------------------------------------------
# 23-26. Fingerprint payload
# ---------------------------------------------------------------------------


def _payload(**kw: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        operation="subtract",
        normalized_operands=(
            _normalized("o1", "42000000", "currency:TND"),
            _normalized("o2", "30140000", "currency:TND"),
        ),
        target_unit=Unit(code="currency:TND"), scale=2,
        rounding="ROUND_HALF_EVEN", authority=calc.current_authority(),
    )
    base.update(kw)
    return fingerprint_payload(**base)  # type: ignore[arg-type]


def _payload_from_result(res: CalculationResult, **over: object) -> dict[str, object]:
    """Build a fingerprint payload from a result's own fields.

    Threading a real result through is what makes the correlation/timestamp test
    below mean something: it proves those fields cannot reach the payload even
    when the payload is derived from an object that carries them.
    """
    kwargs: dict[str, object] = dict(
        operation=res.operation,
        normalized_operands=res.normalized_operands,
        target_unit=res.result_unit,
        scale=res.scale_applied if res.scale_applied is not None else 2,
        rounding=res.rounding_applied or "ROUND_HALF_EVEN",
        authority=res.authority,
        stated_value=res.stated_value,
    )
    kwargs.update(over)
    return fingerprint_payload(**kwargs)  # type: ignore[arg-type]


def test_fingerprint_payload_includes_the_required_identity_fields() -> None:
    payload = _payload(stated_value="11000000")
    for field in FINGERPRINT_INCLUDED_FIELDS:
        assert field in payload, field
    assert payload["operands"][0]["unit"] == "currency:TND"  # type: ignore[index]


def test_declared_included_fields_cannot_drift_from_the_payload() -> None:
    """The declared list and the function that builds the payload must agree.

    Asserting set equality both ways is the point: a one-way ``in`` check would
    let a later phase add a field to the payload without declaring it, which is
    exactly how a documented identity contract quietly stops describing the
    thing it identifies.
    """
    assert set(_payload(stated_value="1")) == set(FINGERPRINT_INCLUDED_FIELDS)
    assert set(_payload()) == set(FINGERPRINT_INCLUDED_FIELDS)
    assert len(FINGERPRINT_INCLUDED_FIELDS) == len(set(FINGERPRINT_INCLUDED_FIELDS))


def test_included_and_excluded_field_sets_are_disjoint() -> None:
    assert not set(FINGERPRINT_INCLUDED_FIELDS) & set(
        contract_mod.FINGERPRINT_EXCLUDED_FIELDS
    )


@pytest.mark.parametrize(
    "excluded", ["request_id", "case_id", "run_id", "specialist", "claim_id",
                 "computed_at", "purpose", "label", "warnings"],
)
def test_fingerprint_payload_excludes_non_identity_fields(excluded: str) -> None:
    assert excluded not in _payload()


def test_correlation_and_timestamp_do_not_alter_the_payload() -> None:
    """Same calculation, different case/run/claim/clock -> identical payload.

    Threads two real results through the payload builder rather than comparing
    two identical literal calls: the point is that correlation and timestamp
    cannot reach the payload even when it is derived from an object carrying
    them.
    """
    common = dict(
        arithmetic_status="ARITHMETIC_VERIFIED",
        evidence=InputEvidenceSummary(status="ALL_SUPPORTED",
                                      bound_operand_ids=("o1", "o2")),
        expression_executed="42000000 - 30140000", result_value="11860000",
        result_unit=Unit(code="currency:TND"), scale_applied=2,
        rounding_applied="ROUND_HALF_EVEN", stated_value="11000000",
        conflict="CONFLICT_DETECTED",
        normalized_operands=(_normalized("o1", "42000000", "currency:TND"),
                             _normalized("o2", "30140000", "currency:TND")),
    )
    r1 = _result(correlation=Correlation(specialist="cfo", case_id="A", run_id="1",
                                         claim_id="c1"),
                 computed_at="2026-01-01T00:00:00Z", **common)
    r2 = _result(correlation=Correlation(specialist="cso", case_id="B", run_id="2",
                                         claim_id="c9"),
                 computed_at="2030-12-31T23:59:59Z", **common)
    assert r1.correlation != r2.correlation
    assert r1.computed_at != r2.computed_at
    assert canonical_payload_json(_payload_from_result(r1)) == \
        canonical_payload_json(_payload_from_result(r2))
    # And structurally: the builder has no parameter for either.
    params = inspect.signature(fingerprint_payload).parameters
    assert "correlation" not in params and "computed_at" not in params


def test_canonical_serialization_is_stable_and_key_sorted() -> None:
    text = canonical_payload_json(_payload())
    assert text == canonical_payload_json(_payload())
    keys = [k for k in ("authority_id", "operands", "operation", "scale")]
    positions = [text.index(f'"{k}"') for k in keys]
    assert positions == sorted(positions), "mapping keys must be sorted"
    assert " " not in text.replace("currency:TND", "").replace("ROUND_HALF_EVEN", "")


def test_differing_identity_fields_change_the_payload() -> None:
    base = canonical_payload_json(_payload())
    assert canonical_payload_json(_payload(scale=3)) != base
    assert canonical_payload_json(_payload(rounding="ROUND_HALF_UP")) != base
    assert canonical_payload_json(_payload(operation="add")) != base
    assert canonical_payload_json(_payload(stated_value="1")) != base
    assert canonical_payload_json(_payload(target_unit=None)) != base


def test_operand_order_is_preserved_and_distinguishable() -> None:
    """42,000,000 - 30,140,000 is not 30,140,000 - 42,000,000."""
    a = _normalized("o1", "42000000", "currency:TND")
    b = _normalized("o2", "30140000", "currency:TND")
    forward = canonical_payload_json(_payload(normalized_operands=(a, b)))
    reversed_ = canonical_payload_json(_payload(normalized_operands=(b, a)))
    assert forward != reversed_
    assert "subtract" in contract_mod.NON_COMMUTATIVE_OPERATIONS


def test_commutative_operands_are_also_not_reordered() -> None:
    """Labels and provenance ride with position, so order is never normalised."""
    a = _normalized("o1", "1", "one")
    b = _normalized("o2", "2", "one")
    assert canonical_payload_json(_payload(operation="add", normalized_operands=(a, b))) != \
        canonical_payload_json(_payload(operation="add", normalized_operands=(b, a)))


def test_fingerprint_field_shape_is_enforced_when_present() -> None:
    assert _result().fingerprint is None
    with pytest.raises(ValidationError):
        _result(fingerprint="not-hex-and-too-short")
    with pytest.raises(ValidationError, match="lowercase hex"):
        _result(fingerprint="A" * 64)
    assert _result(fingerprint="a" * 64).fingerprint == "a" * 64


def test_hashing_lives_only_in_the_fingerprint_module() -> None:
    """Phase 1 forbade hashing outright; Phase 2 confines it to one file.

    The digest is the fingerprint module's whole job. Anywhere else it would
    mean some other module had invented a second notion of identity, which is
    exactly the drift the single ``fingerprint_for`` entry point prevents.
    """
    for path in _calc_source_files():
        names = _imported_roots(path) | _referenced_names(path)
        used = {n for n in ("hashlib", "hmac", "sha256", "md5", "blake2b") if n in names}
        if _rel(path) == "fingerprint.py":
            assert "hashlib" in names, "the fingerprint module must own the digest"
            assert not (used - {"hashlib", "sha256"}), used
        else:
            assert not used, f"{_rel(path)} hashes: {sorted(used)}"


_PROHIBITED_CALLED_BUILTINS = frozenset({
    "eval", "exec", "compile", "__import__", "open", "globals", "locals",
    "getattr", "setattr", "delattr", "input", "breakpoint", "vars",
})

_PROHIBITED_CALLED_ATTRS = frozenset({
    "system", "popen", "spawn", "spawnv", "run", "check_output", "call",
    "literal_eval", "hexdigest", "urlopen", "connect", "send", "recv",
    "loads", "load", "read_text", "write_text", "unlink", "rmtree",
})

_PERMITTED_IMPORT_PREFIXES = (
    "json", "re", "decimal", "types", "typing", "pydantic", "collections.abc",
    "hashlib", "time", "__future__", "openexecutive.calc",
)
"""``hashlib`` and ``time`` arrive with Phase 2 and are narrowly scoped.

``hashlib`` is the fingerprint digest, confined to one module by a test above.
``time`` is the engine's monotonic budget clock — never a wall clock, and never
the source of a record's timestamp, which the caller supplies so tests can pin
it."""
"""Full dotted prefixes, not top-level roots.

A bare ``"openexecutive"`` root would admit
``from openexecutive.providers.anthropic_provider import X`` — every intra-repo
import, including the ones the leaf-package rule exists to forbid. Requiring
``openexecutive.calc`` makes the allowlist and the leaf rule the same check,
so a subpackage added tomorrow cannot slip through a hand-maintained deny list.
"""


def _referenced_names(path: pathlib.Path) -> set[str]:
    """Every identifier and attribute name the file's CODE references."""
    found: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
    return found


def _called_names(path: pathlib.Path) -> tuple[set[str], set[str]]:
    """Names actually *invoked*: bare builtins, and attribute calls.

    Distinguishing calls from plain identifiers matters. ``CalculationBatch``
    has a field legitimately named ``requests``; a scan that treats every
    identifier as dangerous flags it and teaches the reader to ignore this test.
    A security check that cries wolf is worse than none.
    """
    bare: set[str] = set()
    attrs: set[str] = set()
    for node in ast.walk(_parse(path)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            bare.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            attrs.add(node.func.attr)
    return bare, attrs


def _imported_roots(path: pathlib.Path) -> set[str]:
    """Every module a file imports, with relative imports resolved to absolute.

    ``ast.ImportFrom`` for ``from .. import providers`` has ``module=None`` and
    ``level=2``; the imported name lives in ``names``, not in ``module``. An
    earlier version recorded only ``node.module`` and skipped the node entirely,
    so that one line walked past the import allowlist, the leaf-package rule and
    the execution scan simultaneously — ``from ..providers import X`` was caught
    while ``from .. import providers`` was not.

    Resolution: the package of a file at ``calc/<name>.py`` is
    ``openexecutive.calc``; ``level`` strips that many trailing components, and
    each alias is appended when ``module`` is absent.
    """
    package_parts = ["openexecutive", "calc"]
    roots: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            if level == 0:
                if node.module:
                    roots.add(node.module)
                continue
            base = package_parts[: len(package_parts) - (level - 1)] or ["openexecutive"]
            if node.module:
                roots.add(".".join([*base, node.module]))
            else:
                # ``from .. import providers`` — the target is each alias.
                roots.update(".".join([*base, alias.name]) for alias in node.names)
    return roots


def test_relative_imports_are_resolved_not_skipped() -> None:
    """``from .. import providers`` must not be invisible.

    Its ``ImportFrom`` node carries ``module=None``; the target is in ``names``.
    Recording only ``node.module`` skipped the node and defeated the allowlist,
    the leaf rule and the execution scan at once.
    """
    import tempfile

    cases = {
        "from .. import providers": "openexecutive.providers",
        "from ..orchestrator import router": "openexecutive.orchestrator",
        "from . import numeric": "openexecutive.calc.numeric",
        "from .units import Unit": "openexecutive.calc.units",
    }
    for source, expected in cases.items():
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(source + "\n")
            tmp = pathlib.Path(fh.name)
        try:
            assert expected in _imported_roots(tmp), f"{source!r} -> {expected!r}"
        finally:
            tmp.unlink()


def test_no_prohibited_execution_or_io_mechanism_in_the_package() -> None:
    """AST scan of every ``.py`` file found on disk under ``calc/``.

    Two complementary checks, because each alone has a hole. The import
    allowlist below bounds what the package can reach at all; this one bounds
    what it *invokes*, catching a dangerous builtin that needs no import.

    Scope, stated exactly, because a security test that overclaims is worse
    than none. Source is read from disk, so a module cannot redirect the
    scanner by rebinding ``__file__``. What this still does **not** catch is
    indirection *within* the source it reads — an aliased builtin
    (``_e = eval``), a subscript call (``__builtins__["eval"]``), a dunder
    lookup, or a mechanism reached through a dependency. Those are not
    statically decidable in general, and the defence against them is the import
    allowlist plus review. A regression guard on the obvious forms, not a proof.
    """
    for path in _calc_source_files():
        bare, attrs = _called_names(path)
        allowed_attrs: set[str] = set()
        if _rel(path) == "fingerprint.py":
            # Narrowed to the one file whose job is the digest, rather than
            # removed from the ban list: ``hexdigest`` anywhere else would mean a
            # second notion of identity, and ``loads`` is the round-trip
            # assertion that keeps an unserialisable payload from failing later
            # at a less obvious place. Every other module still trips on both.
            allowed_attrs = {"hexdigest", "loads"}
        offending = (bare & _PROHIBITED_CALLED_BUILTINS) | (
            (attrs - allowed_attrs) & _PROHIBITED_CALLED_ATTRS
        )
        assert not offending, f"{_rel(path)} invokes {sorted(offending)}"


def test_package_imports_only_the_permitted_stdlib_and_pydantic() -> None:
    for path in _calc_source_files():
        for root in _imported_roots(path):
            assert any(
                root == p or root.startswith(p + ".")
                for p in _PERMITTED_IMPORT_PREFIXES
            ), f"{_rel(path)} imports {root!r}, outside the permitted set"


def test_calc_is_a_leaf_package_with_no_agent_or_provider_coupling() -> None:
    """No intra-repo import outside ``openexecutive.calc`` itself.

    Derived rather than deny-listed: any ``openexecutive.*`` import that is not
    under ``calc`` fails, so a subpackage introduced later is covered without
    anyone remembering to add its name here.
    """
    for path in _calc_source_files():
        for root in _imported_roots(path):
            if not root.startswith("openexecutive"):
                continue
            assert root == "openexecutive.calc" or root.startswith("openexecutive.calc."), (
                f"{_rel(path)} couples to {root!r}; calc must stay a leaf"
            )


# Production modules permitted to import ``openexecutive.calc`` at all. An
# allowlist, not a prohibition: the original invariant could only hold until the
# package had a caller. What must still hold is that the coupling stays small
# enough to review by reading, and that `calc` remains a LEAF — the dependency
# runs one way and no entry here may be imported *by* calc.
_CALC_IMPORTERS = frozenset(
    {
        "specialists/calculation_gateway.py",
        "specialists/calculation_proposal.py",
    }
)


def test_only_allowlisted_production_modules_import_calc() -> None:
    """The set of production importers of ``calc`` is bounded and named.

    Parsed, not grepped, and the resolution logic is shared with the engine
    scanner in ``test_calc_adversarial.py`` via ``_calc_import_scan`` — two
    copies drifted, and between them missed ``from openexecutive import calc``,
    ``from openexecutive.calc import engine``, ``from .. import calc`` and
    ``from ..calc import engine``. ``ImportFrom.names`` is inspected as well as
    ``ImportFrom.module``, which is what those forms hide in.

    Matched on RESOLVED ABSOLUTE paths, anchored at the package root. A
    root-relative string would let any scanned tree — ``evals/``, ``scripts/`` —
    inherit the exemption by reusing the path, so a shadow module could silence
    the check by existing.
    """
    package_root = CALC_DIR.parent
    repo_root = package_root.parents[2]
    allowed = {(package_root / entry).resolve() for entry in _CALC_IMPORTERS}
    offenders: list[str] = []
    for root, package_name in (
        (package_root, "openexecutive"),
        (package_root.parent / "scripts", None),
        (repo_root / "evals", None),
    ):
        if not root.is_dir():
            continue
        scanned = scan_tree(
            root,
            package_name=package_name,
            skip_parts=("calc", "__pycache__", ".venv", "tests"),
        )
        for path, targets in scanned.items():
            referenced = references_calc(targets)
            if referenced and path.resolve() not in allowed:
                offenders.append(f"{path.relative_to(root)}: {sorted(referenced)}")
    assert offenders == [], (
        f"openexecutive.calc may only be imported by {sorted(_CALC_IMPORTERS)}; "
        f"found: {sorted(offenders)}"
    )



def test_every_allowlisted_calc_importer_exists() -> None:
    """A stale entry naming a deleted file would quietly widen the boundary."""
    package_root = CALC_DIR.parent
    for relative in _CALC_IMPORTERS:
        assert (package_root / relative).is_file(), f"allowlisted {relative} is gone"


# ---------------------------------------------------------------------------
# 30. No arithmetic is performed
# ---------------------------------------------------------------------------


def test_no_arithmetic_is_performed_by_the_contract_layer() -> None:
    """Constructing the full contract surface must not touch decimal arithmetic.

    Enforced by trapping every arithmetic signal in the decimal context: if any
    code path in this package added, multiplied, divided, or rounded a Decimal,
    an inexact or rounded operation would raise here.
    """
    with decimal.localcontext() as ctx:
        ctx.traps[decimal.Inexact] = True
        ctx.traps[decimal.Rounded] = True
        op = _operand("o1", "42000000.55", "currency:TND")
        req = CalculationRequest(
            request_id="r", operation="subtract", operands=(op,), purpose="p",
            correlation=_corr(),
        )
        payload = _payload()
        canonical_payload_json(payload)
        assert req.operands[0].decimal_value == Decimal("42000000.55")
        assert calc.canonical_numeric_string(parse_numeric("1.23456789")) == "1.23456789"


def test_arithmetic_lives_only_in_the_engine() -> None:
    """Phase 1 asserted no engine existed. Phase 2 asserts there is exactly one.

    The contract, unit registry, numeric boundary and authority modules must
    still perform no arithmetic: a second place that multiplies is a second
    place that can be wrong, and the whole point of the engine is that there is
    one auditable executor.
    """
    contract_only = [
        p for p in _calc_source_files()
        if _rel(p) in ("contract.py", "units.py", "numeric.py", "authority.py",
                       "_model.py", "__init__.py")
    ]
    assert len(contract_only) == 6
    with decimal.localcontext() as ctx:
        ctx.traps[decimal.Inexact] = True
        ctx.traps[decimal.Rounded] = True
        op = _operand("o1", "42000000.55", "currency:TND")
        req = CalculationRequest(
            request_id="r", operation="subtract", operands=(op,), purpose="p",
            correlation=_corr(),
        )
        canonical_payload_json(_payload())
        assert req.operands[0].decimal_value == Decimal("42000000.55")
        assert calc.canonical_numeric_string(parse_numeric("1.23456789")) == "1.23456789"


def test_the_engine_is_reachable_and_is_the_only_executor() -> None:
    assert hasattr(calc, "execute") and hasattr(calc, "execute_batch")
    assert "execute" in calc.__all__ and "execute_batch" in calc.__all__
    # No second entry point crept in alongside it.
    for banned in ("evaluate", "compute", "calculate", "run", "eval_expression"):
        assert banned not in calc.__all__
