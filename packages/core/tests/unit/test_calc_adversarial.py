"""Adversarial and resource-limit tests for the engine.

The governing rule: **no exception caused by request data may escape a public
entry point.** Every hostile input below must come back as a typed
``CalculationResult`` carrying a failure status — not as a raised exception, not
as a wrong number, and not after allocating gigabytes on the way.

Phase 1 found the sharpest instance of that last clause: ``"0e-2000000000"`` is
thirteen characters, sits inside the 64-character limit, compares equal to zero,
and — before the bound was applied to zero — expanded positionally into a
two-billion-character string, allocating 4 GB inside validation. It is pinned
here again from the engine's side.
"""
from __future__ import annotations

import ast
import inspect
import os
import pathlib
import resource
import time
from decimal import Decimal

import pytest
from pydantic import ValidationError

from openexecutive.calc import engine as engine_mod
from openexecutive.calc.contract import (
    CalculationBatch,
    CalculationRequest,
    Correlation,
    InputEvidenceSummary,
    Operand,
    SourceHint,
)
from openexecutive.calc.engine import execute, execute_batch
from openexecutive.calc.numeric import NumericPolicyError, parse_numeric
from openexecutive.calc.units import Unit
from tests.unit._calc_import_scan import reaches_execution, scan_tree

AT = "2026-09-02T00:00:00Z"
TND = "currency:TND"

_FAILURE_STATUSES = {
    "CALCULATION_UNAVAILABLE", "DIVISION_BY_ZERO", "UNIT_MISMATCH",
    "RESOURCE_LIMIT_EXCEEDED", "INVALID_INPUT", "UNSUPPORTED_OPERATION",
}


def _op(oid: str, value: str, unit: str, role: str = "input") -> Operand:
    return Operand(operand_id=oid, label=oid, value=value, unit=Unit(code=unit),
                   basis="applicant_stated", role=role)  # type: ignore[arg-type]


def _request(operation: str, operands: list[Operand], target: str | None = "one",
             scale: int = 2, request_id: str = "r1") -> CalculationRequest:
    return CalculationRequest(
        request_id=request_id, operation=operation,  # type: ignore[arg-type]
        operands=tuple(operands),
        target_unit=Unit(code=target) if target else None,
        scale=scale, purpose="adversarial",
        correlation=Correlation(specialist="cfo", case_id="c", run_id="r"),
    )


def _run(operation: str, operands: list[Operand], target: str | None = "one",
         scale: int = 2, **kw: object):
    return execute(_request(operation, operands, target, scale),
                   computed_at=AT, **kw)  # type: ignore[arg-type]


def _assert_typed_failure(result) -> None:
    assert result.arithmetic_status in _FAILURE_STATUSES, result.arithmetic_status
    assert result.result_value is None
    assert result.exact_result is None
    assert result.fingerprint is None
    assert result.errors, "a failure must say why"


# ---------------------------------------------------------------------------
# 1-3, 11-15. Hostile numeric forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "literal",
    ["0e-2000000000", "-0e-2000000000", "0e2000000000", "0e-40000000",
     "1e31", "1e-31", "1e400", "-1e400", "0e-99999999999999999"],
)
def test_hostile_exponents_are_rejected_before_expansion(literal: str) -> None:
    """Rejected, and rejected *cheaply* — the cost is the point.

    A guard that allocates gigabytes before raising is still a denial of
    service, so this asserts the memory and time of the rejection, not just its
    outcome.
    """
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.monotonic()
    with pytest.raises(NumericPolicyError):
        parse_numeric(literal)
    with pytest.raises(ValidationError):
        _op("o1", literal, "one")
    elapsed = time.monotonic() - started
    grew = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    assert elapsed < 1.0, f"rejection took {elapsed:.3f}s"
    # ru_maxrss is bytes on macOS and KiB on Linux; 256 MiB is far under the
    # 4 GB the unguarded path allocated on either reading.
    assert grew < 256 * 1024 * 1024, f"peak RSS grew by {grew}"


@pytest.mark.parametrize("literal", ["NaN", "-NaN", "sNaN", "Infinity", "-Infinity", "inf"])
def test_nan_and_infinity_never_enter(literal: str) -> None:
    with pytest.raises(NumericPolicyError):
        parse_numeric(literal)
    with pytest.raises(ValidationError):
        _op("o1", literal, "one")


@pytest.mark.parametrize("raw", [1.5, 0.1, True, False, Decimal("1.5"), Decimal(0.1)])
def test_float_bool_and_foreign_decimals_are_refused(raw: object) -> None:
    with pytest.raises(NumericPolicyError):
        parse_numeric(raw)
    with pytest.raises(ValidationError):
        Operand(operand_id="o1", label="l", value=raw,  # type: ignore[arg-type]
                unit=Unit(code="one"), basis="applicant_stated")


def test_decimal_built_from_a_float_is_refused_even_though_it_looks_fine() -> None:
    """``Decimal(0.1)`` is ``0.1000000000000000055511151231257827…``."""
    contaminated = Decimal(0.1)
    assert str(contaminated) != "0.1"
    with pytest.raises(NumericPolicyError):
        parse_numeric(contaminated)


@pytest.mark.parametrize("exponent,sign", [(4400, 1), (5000, 1), (4400, -1)])
def test_oversized_integers_raise_a_typed_error_not_a_raw_one(
    exponent: int, sign: int
) -> None:
    """CPython's 4,300-digit int-to-str limit must not leak a bare ValueError.

    ``NumericPolicyError`` subclasses ``ValueError``, so the assertion checks the
    concrete type rather than the base. The value is built inside the test
    rather than parametrized: pytest stringifies parameters to build test ids
    and would hit the very limit under test while collecting.
    """
    raw = sign * 10**exponent
    with pytest.raises(NumericPolicyError) as caught:
        parse_numeric(raw)
    assert type(caught.value) is NumericPolicyError


@pytest.mark.parametrize(
    "literal", ["1,234", "1.234,56", "42,5", "1 234", " 1", "1 ", "", "١٢٣", "１２３"],
)
def test_ambiguous_and_non_ascii_literals_are_refused(literal: str) -> None:
    with pytest.raises(NumericPolicyError):
        parse_numeric(literal)


def test_negative_and_negative_zero_are_handled_exactly() -> None:
    result = _run("subtract", [_op("a", "-5", "one"), _op("b", "3", "one")], "one", scale=0)
    assert result.result_value == "-8"
    zero = _run("add", [_op("a", "-0", "one"), _op("b", "0", "one")], "one", scale=0)
    assert zero.result_value == "0"
    assert not zero.result_value.startswith("-")


def test_excessive_scale_and_precision_are_refused() -> None:
    with pytest.raises(ValidationError):
        _request("add", [_op("a", "1", "one"), _op("b", "1", "one")], "one", scale=29)
    high = _run("divide", [_op("a", "1", "one"), _op("b", "3", "one")], "one", scale=28)
    assert high.arithmetic_status == "ARITHMETIC_VERIFIED"


# ---------------------------------------------------------------------------
# 16-22. Structural limits
# ---------------------------------------------------------------------------


def test_excessive_operands_and_batch_size_fail_closed() -> None:
    with pytest.raises(ValidationError):
        _request("sum_components", [_op(f"o{i}", "1", "pct") for i in range(65)], "pct")
    with pytest.raises(ValidationError):
        CalculationBatch(requests=tuple(
            _request("add", [_op("a", "1", "one"), _op("b", "1", "one")], "one",
                     request_id=f"r{i}")
            for i in range(33)
        ))


def test_unknown_operation_is_refused_by_contract_and_engine_alike() -> None:
    with pytest.raises(ValidationError):
        _request("irr", [_op("a", "1", "one")], "one")
    assert engine_mod.signature_for("irr") is None
    assert "irr" not in engine_mod._DISPATCH


def test_no_operand_is_ever_silently_skipped() -> None:
    """A bad operand fails the request; it does not vanish from the sum."""
    result = _run("sum_components",
                  [_op("a", "1", "pct"), _op("b", "2", "kg"), _op("c", "3", "pct")],
                  "pct", scale=0)
    _assert_typed_failure(result)
    assert result.arithmetic_status == "UNIT_MISMATCH"


def test_operand_order_is_never_rearranged() -> None:
    result = _run("subtract", [_op("first", "3", "one"), _op("second", "10", "one")],
                  "one", scale=0)
    assert result.result_value == "-7"
    assert [o.operand_id for o in result.normalized_operands] == ["first", "second"]


# ---------------------------------------------------------------------------
# 23-29. Semantic traps
# ---------------------------------------------------------------------------


def test_cross_currency_arithmetic_is_refused_in_every_shape() -> None:
    for operation, target in (("add", TND), ("subtract", TND), ("sum_components", TND),
                              ("ratio", "one"), ("percentage_of", "pct")):
        result = _run(operation, [_op("a", "1", TND), _op("b", "1", "currency:EUR")], target)
        _assert_typed_failure(result)


def test_pct_versus_pct_point_confusion_is_refused() -> None:
    for operation, target in (("add", "pct"), ("subtract", "pct"), ("ratio", "one")):
        result = _run(operation, [_op("a", "60", "pct"), _op("b", "25", "pct_point")], target)
        _assert_typed_failure(result)


def test_month_versus_year_without_a_policy_is_refused() -> None:
    result = _run("convert_unit", [_op("a", "12", "month")], "year", scale=0)
    _assert_typed_failure(result)
    assert "time_conversion_policy" in result.errors[0].detail


@pytest.mark.parametrize(
    "left,right,target",
    [("kg", "t", "kg"), ("m2", "ha", "m2"), ("pct", "pct", "pct"),
     ("month", "year", "month"), (TND, TND, TND)],
)
def test_undeclared_multiplication_compositions_are_refused(
    left: str, right: str, target: str
) -> None:
    result = _run("multiply", [_op("a", "2", left), _op("b", "3", right)], target)
    _assert_typed_failure(result)


def test_naive_same_index_interval_division_is_not_what_the_engine_does() -> None:
    """The cross-paired band must include 5,720 t; the naive one excludes it."""
    result = _run("interval_implied_total",
                  [_op("vl", "3200", "t"), _op("vh", "3500", "t"),
                   _op("cl", "55", "pct"), _op("ch", "60", "pct")], "t", scale=6)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    lower = Decimal(result.exact_result or "0")
    upper = Decimal(result.warnings[0].split()[-1])
    naive_lower = Decimal(3200) / Decimal("0.55")
    naive_upper = Decimal(3500) / Decimal("0.60")
    assert lower < naive_lower and upper > naive_upper
    assert lower < Decimal("5720") < upper
    assert not (naive_lower < Decimal("5720") < naive_upper)


def test_interval_crossing_or_touching_zero_is_refused() -> None:
    for low, high in (("0", "60"), ("-5", "5"), ("-10", "-1")):
        result = _run("interval_implied_total",
                      [_op("vl", "1", "t"), _op("vh", "2", "t"),
                       _op("cl", low, "pct"), _op("ch", high, "pct")], "t")
        _assert_typed_failure(result)


# ---------------------------------------------------------------------------
# 30-36. Authority and identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["fingerprint", "arithmetic_status", "evidence", "authority", "authority_version",
     "computed_at", "result_value", "expression_executed", "verified_result",
     "verification_status", "page", "sheet", "cell_range", "url", "retrieval_id"],
)
def test_a_request_cannot_carry_any_authority_field(field: str) -> None:
    assert field not in CalculationRequest.model_fields
    payload = {
        "request_id": "r", "operation": "add",
        "operands": [_op("a", "1", "one").model_dump()],
        "purpose": "p",
        "correlation": Correlation(specialist="cfo", case_id="c", run_id="r").model_dump(),
        field: "x",
    }
    with pytest.raises(ValidationError):
        CalculationRequest.model_validate(payload)


def test_a_model_source_hint_cannot_become_trusted_evidence() -> None:
    hinted = Operand(
        operand_id="a", label="a", value="1", unit=Unit(code="one"),
        basis="applicant_stated",
        source_hint=SourceHint(document_label="proposal.pdf",
                               retrieval_id_hint="rid-looks-real"),
    )
    result = _run("add", [hinted, _op("b", "1", "one")], "one")
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert result.evidence.status == "EVIDENCE_UNAVAILABLE"
    assert result.is_verified_evidence() is False
    # And the hint reaches neither the record's bindings nor the fingerprint.
    assert result.evidence.bound_operand_ids == ()
    assert "rid-looks-real" not in (result.fingerprint or "")


def test_correct_arithmetic_over_unsupported_inputs_stays_unsupported() -> None:
    """The state this whole package exists to keep representable."""
    evidence = InputEvidenceSummary(status="UNSUPPORTED", unbound_operand_ids=("a", "b"))
    result = _run("add", [_op("a", "1", "one"), _op("b", "1", "one")], "one",
                  evidence=evidence)
    assert result.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert result.evidence.status == "UNSUPPORTED"
    assert result.is_verified_evidence() is False


def test_a_caller_cannot_pass_an_evidence_status_string() -> None:
    """``evidence`` is a validated model; a bare status string is not one.

    The status vocabulary is closed, so an invented status cannot be constructed
    at all. And a caller passing a bare *string* where the model belongs is
    refused at the engine boundary: without that guard the string would be
    forwarded verbatim onto the record, giving ``.status`` a value no validator
    ever saw — on the one field that decides whether a figure counts as
    supported evidence.
    """
    with pytest.raises(ValidationError):
        InputEvidenceSummary(status="TOTALLY_FINE")  # type: ignore[arg-type]
    for forged in ("ALL_SUPPORTED", {"status": "ALL_SUPPORTED"}, 1):
        with pytest.raises(TypeError):
            _run("add", [_op("a", "1", "one"), _op("b", "1", "one")], "one",
                 evidence=forged)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 37-41. Batch and state
# ---------------------------------------------------------------------------


def test_a_failure_in_the_middle_leaves_its_siblings_intact() -> None:
    batch = CalculationBatch(requests=(
        _request("add", [_op("a", "1", "one"), _op("b", "1", "one")], "one", request_id="ok1"),
        _request("divide", [_op("a", "1", "one"), _op("b", "0", "one")], "one",
                 request_id="boom"),
        _request("add", [_op("a", "5", "one"), _op("b", "5", "one")], "one", request_id="ok2"),
    ))
    results = execute_batch(batch, computed_at=AT)
    assert [r.request_id for r in results] == ["ok1", "boom", "ok2"]
    assert results[0].result_value == "2.00"
    assert results[1].arithmetic_status == "DIVISION_BY_ZERO"
    assert results[2].result_value == "10.00"
    assert results[0].fingerprint and results[2].fingerprint
    assert results[1].fingerprint is None


def test_results_are_stable_over_repeated_execution() -> None:
    request = _request("divide", [_op("a", "22", "one"), _op("b", "7", "one")], "one", scale=20)
    seen = {(execute(request, computed_at=AT).result_value,
             execute(request, computed_at=AT).fingerprint) for _ in range(20)}
    assert len(seen) == 1


def test_the_engine_module_holds_no_mutable_state() -> None:
    mutable = [
        name for name, value in vars(engine_mod).items()
        if isinstance(value, (list, dict, set)) and not name.startswith("__")
        and name not in ("_SIGNATURES", "_DISPATCH", "_ROUNDING")
    ]
    assert mutable == []
    # The three tables that do exist are read-only by convention and by test:
    # replacing an entry would change dispatch, so their contents are pinned in
    # test_calc_engine.py rather than left to inspection.
    assert set(engine_mod._DISPATCH) == set(engine_mod._SIGNATURES)


# ---------------------------------------------------------------------------
# 42-45. Boundary integrity
# ---------------------------------------------------------------------------

_CALC_DIR = pathlib.Path(__file__).resolve().parents[2] / "openexecutive" / "calc"
"""Anchored on THIS test file, never on scanned code.

A draft of this module used ``pathlib.Path(engine_mod.__file__).parent``, which
re-opened the exact hole Phase 1 spent two rounds closing: a module that rebinds
``__file__`` at import time redirects the whole walk to a decoy tree, and a
``engine.py`` importing ``subprocess`` and a provider, with an ``eval`` call in
it, passed every scanner test in this file. The root has to come from somewhere
the scanned code cannot assign.
"""

_PRODUCTION_FILES = ("__init__.py", "_model.py", "authority.py", "contract.py",
                     "engine.py", "fingerprint.py", "numeric.py", "units.py")


def _calc_sources() -> list[pathlib.Path]:
    """Every ``.py`` under ``calc/``, read from disk.

    ``os.walk`` rather than a 3.13-only ``rglob`` keyword, because the project
    ships Python 3.11 — a draft used the keyword and would have raised
    ``TypeError`` on every scanner test in CI.
    """
    found: list[pathlib.Path] = []
    for dirpath, _dirnames, filenames in os.walk(_CALC_DIR):
        for name in filenames:
            if name.endswith(".py"):
                found.append(pathlib.Path(dirpath) / name)
    return sorted(found)


def test_the_new_engine_files_are_covered_by_the_scan() -> None:
    """A new production file must not be able to escape the security scan."""
    on_disk = {p.name for p in _calc_sources() if p.parent == _CALC_DIR}
    assert on_disk == set(_PRODUCTION_FILES)
    nested = [str(p.relative_to(_CALC_DIR)) for p in _calc_sources() if p.parent != _CALC_DIR]
    assert nested == [], f"calc/ must stay flat; found {nested}"


def test_no_calc_source_imports_or_invokes_a_forbidden_mechanism() -> None:
    banned_imports = {"subprocess", "socket", "pickle", "importlib", "urllib",
                      "requests", "httpx", "shutil", "ctypes", "marshal"}
    banned_calls = {"eval", "exec", "compile", "open", "__import__", "breakpoint",
                    "globals", "locals", "vars"}
    for path in _calc_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                level = node.level or 0
                if level == 0 and node.module:
                    imported.add(node.module.split(".")[0])
                elif level and node.module is None:
                    imported.update(a.name for a in node.names)
        assert not (imported & banned_imports), f"{path.name}: {imported & banned_imports}"
        called = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not (called & banned_calls), f"{path.name}: {called & banned_calls}"


def test_calc_never_imports_a_model_provider_or_agent_path() -> None:
    forbidden = ("agents", "providers", "orchestrator", "prompts", "specialists",
                 "memory", "api", "workflows", "knowledge")
    for path in _calc_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: set[str] = set()
            if isinstance(node, ast.Import):
                modules = {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = {node.module}
            for module in modules:
                if not module.startswith("openexecutive"):
                    continue
                assert module == "openexecutive.calc" or module.startswith(
                    "openexecutive.calc."
                ), f"{path.name} couples to {module}"
                for package in forbidden:
                    assert f"openexecutive.{package}" not in module


# The one production module permitted to EXECUTE a calculation or mint an
# authority stamp. One door is what makes "how many calculations ran" a
# countable question; a second entry here is a second place arithmetic can
# happen.
_ENGINE_IMPORTERS = frozenset({"specialists/calculation_gateway.py"})


def test_only_the_calculation_gateway_imports_the_engine() -> None:
    """Exactly one production module may reach an execution or authority surface.

    "Reach" is broader than "name". ``calc/__init__`` re-exports
    ``execute_batch`` and ``issue_calculation_result``, so a module holding the
    package object can execute without ever mentioning ``engine`` — and an
    earlier version of this scanner, which matched only the dotted string
    ``openexecutive.calc.engine`` in ``ImportFrom.module``, missed four ordinary
    forms including ``from openexecutive.calc import engine``. The resolution
    logic now lives in ``_calc_import_scan`` and is shared with the sibling
    scanner in ``test_calc_contract_foundation.py``, so the two cannot drift.

    Allowlist entries are resolved to ABSOLUTE paths: a root-relative string
    would let any scanned tree inherit the exemption by reusing the path.
    """
    root = _CALC_DIR.parent
    allowed = {(root / entry).resolve() for entry in _ENGINE_IMPORTERS}
    offenders: list[str] = []
    scanned = scan_tree(
        root, package_name="openexecutive", skip_parts=("calc", "__pycache__")
    )
    for path, targets in scanned.items():
        reaching = reaches_execution(targets)
        if reaching and path.resolve() not in allowed:
            offenders.append(f"{path.relative_to(root)}: {sorted(reaching)}")
    assert offenders == [], (
        f"execution and authority surfaces may only be reached by "
        f"{sorted(_ENGINE_IMPORTERS)}; found: {sorted(offenders)}"
    )



def test_the_allowlisted_engine_importer_exists() -> None:
    """An allowlist naming a deleted file silently permits everything."""
    root = _CALC_DIR.parent
    for relative in _ENGINE_IMPORTERS:
        assert (root / relative).is_file(), f"allowlisted {relative} is gone"


def test_the_whole_engine_surface_parses_under_python_3_11() -> None:
    """The project ships 3.11 — CI, Docker, ruff target, and mypy all pin it."""
    # From disk, for the same reason the root is: ``inspect.getsource`` resolves
    # a module's live ``__file__`` and would parse whatever it was pointed at.
    for path in _calc_sources():
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 11))
    assert {p.name for p in _calc_sources()} == set(_PRODUCTION_FILES)


def test_a_rebound_dunder_file_cannot_redirect_the_scan() -> None:
    """The scan root must be independent of the code being scanned.

    Constructed rather than asserted: a stand-in whose ``__file__`` points
    elsewhere is shown to move ``inspect.getfile`` and NOT to move ``_CALC_DIR``.
    """
    import types as _types

    impostor = _types.ModuleType("openexecutive.calc.impostor")
    impostor.__file__ = "/tmp/decoy/openexecutive/calc/engine.py"
    assert inspect.getfile(impostor) == "/tmp/decoy/openexecutive/calc/engine.py"
    assert (
        pathlib.Path(__file__).resolve().parents[2] / "openexecutive" / "calc"
    ) == _CALC_DIR
    assert (_CALC_DIR / "engine.py").is_file()
    assert "/tmp/decoy" not in str(_CALC_DIR)
    assert all("/tmp/decoy" not in str(p) for p in _calc_sources())


def test_no_3_13_only_api_is_used() -> None:
    """``rglob(recurse_symlinks=)`` and friends do not exist on 3.11."""
    banned = ("recurse_symlinks", "batched", "TypeIs")
    for path in _calc_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # AST-scoped: a text scan of this very file would match its own ban
        # list, which is the false positive that teaches a reader to ignore the
        # test.
        used = {
            node.arg for node in ast.walk(tree)
            if isinstance(node, ast.keyword) and node.arg
        } | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        for name in banned:
            assert name not in used, f"{path.name} uses a 3.12+/3.13+ API: {name}"


# ---------------------------------------------------------------------------
# 46. The scanner's own blind spots — every form that once slipped past
# ---------------------------------------------------------------------------

# Each entry evaded one or both allowlists before ``ImportFrom.names`` was
# resolved. They are exercised against the REAL scanners over a temporary tree
# laid out exactly like the repository, so a regression in the shared resolver
# fails here rather than in a review two rounds later.
_BYPASS_FORMS = {
    "from_package_import_calc": "from openexecutive import calc\n",
    "from_calc_import_engine": "from openexecutive.calc import engine\n",
    "from_calc_import_authority": "from openexecutive.calc import authority\n",
    "relative_from_parent_import_calc": "from .. import calc\n",
    "relative_from_parent_calc_import_engine": "from ..calc import engine\n",
    "import_dotted_package": "import openexecutive.calc\n",
    "from_calc_import_execute_batch": "from openexecutive.calc import execute_batch\n",
    "star_import_from_calc": "from openexecutive.calc import *\n",
}

# Dynamic imports with a LITERAL module name. The shared resolver reads these
# and only these spellings; a computed name, an aliased ``importlib`` or a
# relative literal are outside the static guard, and
# ``test_a_computed_module_name_is_outside_the_static_guard`` pins that
# boundary so the docs cannot claim more than the scan does.
_DYNAMIC_IMPORT_FORMS = {
    "importlib_import_module_calc": (
        'import importlib\ncalc = importlib.import_module("openexecutive.calc")\n'
    ),
    "importlib_import_module_engine": (
        'import importlib\n'
        'engine = importlib.import_module("openexecutive.calc.engine")\n'
    ),
    "importlib_import_module_authority": (
        'import importlib\n'
        'authority = importlib.import_module("openexecutive.calc.authority")\n'
    ),
    "bare_import_module_engine": (
        'from importlib import import_module\n'
        'engine = import_module("openexecutive.calc.engine")\n'
    ),
    "dunder_import_engine": 'engine = __import__("openexecutive.calc.engine")\n',
}

_ALL_BYPASS_FORMS = {**_BYPASS_FORMS, **_DYNAMIC_IMPORT_FORMS}


def _fake_repo(tmp_path: pathlib.Path, source: str) -> pathlib.Path:
    """A tree matching the real layout, holding one non-allowlisted module.

    The scanners derive their roots from ``CALC_DIR``, so the depth has to match
    the repository exactly for the production code to walk this tree.
    """
    package = tmp_path / "packages" / "core" / "openexecutive"
    (package / "calc").mkdir(parents=True)
    (package / "calc" / "__init__.py").write_text("", encoding="utf-8")
    (package / "orchestrator").mkdir(parents=True)
    (package / "orchestrator" / "__init__.py").write_text("", encoding="utf-8")
    (package / "orchestrator" / "second_door.py").write_text(source, encoding="utf-8")
    return package / "calc"


@pytest.mark.parametrize("form", sorted(_ALL_BYPASS_FORMS))
def test_the_engine_scanner_catches_every_bypass_form(
    form: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION: four of these reached execution with the scanner green.

    ``calc/__init__`` re-exports ``execute_batch`` and
    ``issue_calculation_result``, so binding the package is enough to execute —
    which is why a bare package binding counts here even though it never names
    the engine module.
    """
    calc_dir = _fake_repo(tmp_path, _ALL_BYPASS_FORMS[form])
    monkeypatch.setattr(
        "tests.unit.test_calc_adversarial._CALC_DIR", calc_dir, raising=True
    )
    with pytest.raises(AssertionError, match="second_door.py"):
        test_only_the_calculation_gateway_imports_the_engine()


@pytest.mark.parametrize("form", sorted(_ALL_BYPASS_FORMS))
def test_the_calc_scanner_catches_every_bypass_form(
    form: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tests.unit.test_calc_contract_foundation as foundation

    calc_dir = _fake_repo(tmp_path, _ALL_BYPASS_FORMS[form])
    monkeypatch.setattr(foundation, "CALC_DIR", calc_dir, raising=True)
    with pytest.raises(AssertionError, match="second_door.py"):
        foundation.test_only_allowlisted_production_modules_import_calc()


@pytest.mark.parametrize("ancestor", ["tests", "calc"])
def test_the_scanners_are_not_silenced_by_an_ancestor_named_like_a_skip_part(
    ancestor: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION (LOW): ``skip_parts`` was matched against the ABSOLUTE path.

    A scan root that merely lived under a directory named ``tests`` or ``calc``
    — a CI checkout, a ``tmp_path`` — skipped every file, and both boundary
    tests passed having scanned nothing. Filtering is now relative to the scan
    root, and this drives the REAL scanners over such a tree to prove it.
    """
    import tests.unit.test_calc_contract_foundation as foundation

    calc_dir = _fake_repo(
        tmp_path / ancestor, "from openexecutive.calc.engine import execute_batch\n"
    )
    assert ancestor in calc_dir.parts

    monkeypatch.setattr(
        "tests.unit.test_calc_adversarial._CALC_DIR", calc_dir, raising=True
    )
    with pytest.raises(AssertionError, match="second_door.py"):
        test_only_the_calculation_gateway_imports_the_engine()

    monkeypatch.setattr(foundation, "CALC_DIR", calc_dir, raising=True)
    with pytest.raises(AssertionError, match="second_door.py"):
        foundation.test_only_allowlisted_production_modules_import_calc()


def test_scan_tree_still_skips_matching_directories_below_the_root(
    tmp_path: pathlib.Path,
) -> None:
    """Root-relative filtering must not have thrown the filter away."""
    from tests.unit._calc_import_scan import scan_tree

    root = tmp_path / "tests" / "checkout" / "openexecutive"
    source = "from openexecutive.calc.engine import execute_batch\n"
    for relative in ("orchestrator/door.py", "tests/fixture.py", "calc/engine.py"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")

    scanned = scan_tree(root, package_name="openexecutive", skip_parts=("calc", "tests"))
    assert set(scanned) == {root / "orchestrator" / "door.py"}


def test_a_computed_module_name_is_outside_the_static_guard() -> None:
    """Scope, stated honestly: these are controlled by review, not by the scan.

    If the resolver ever starts seeing one of them, that is a capability change
    worth documenting — update the resolver's docstring and this test together
    rather than letting the docs lag the code in either direction.
    """
    import tests.unit._calc_import_scan as resolver
    from tests.unit._calc_import_scan import (
        import_targets,
        reaches_execution,
        references_calc,
    )

    undecidable = (
        'import importlib\nname = "openexecutive.calc"\nimportlib.import_module(name)\n',
        'import importlib as il\nil.import_module("openexecutive.calc.engine")\n',
        'import importlib\nimportlib.import_module("..calc", package=__package__)\n',
    )
    for source in undecidable:
        targets = import_targets(
            ast.parse(source),
            path=pathlib.Path("orchestrator/second_door.py"),
            root=pathlib.Path(),
            package_name="openexecutive",
        )
        assert references_calc(targets) == set(), f"{source!r} is documented as unseen"
        assert reaches_execution(targets) == set()

    doc = " ".join((resolver.__doc__ or "").split())
    assert "outside this guard" in doc
    assert "controlled by review" in doc
    assert "not statically decidable" in doc


def test_the_bypass_suite_fails_without_node_names_resolution() -> None:
    """Mutation-resistance, stated as a property of the resolver itself.

    Every form in ``_BYPASS_FORMS`` hides its target in ``ImportFrom.names``. A
    resolver that reads only ``ImportFrom.module`` returns nothing for them, so
    the tests above would stop failing and start passing vacuously — the exact
    regression that shipped. This pins the difference directly. (The dynamic
    forms are deliberately not here: they have no ``ImportFrom`` at all, and
    their mutation resistance is the scanner tests themselves failing when the
    ``Call`` branch is removed.)
    """
    from tests.unit._calc_import_scan import import_targets, reaches_execution

    for source in _BYPASS_FORMS.values():
        tree = ast.parse(source)
        # ``root`` is the PACKAGE root, so the path is relative to it and
        # carries no leading "openexecutive/" — the resolver prefixes the
        # package name itself.
        targets = import_targets(
            tree,
            path=pathlib.Path("orchestrator/second_door.py"),
            root=pathlib.Path(),
            package_name="openexecutive",
        )
        assert reaches_execution(targets), f"{source!r} must reach execution"

        module_only = {
            (node.module, False)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not reaches_execution(module_only), (
            f"{source!r} is invisible to a module-only resolver — which is why "
            "node.names must be resolved"
        )


def test_importing_a_calc_type_is_not_treated_as_reaching_execution() -> None:
    """The screen must not over-report, or it stops being obeyed.

    ``calculation_proposal.py`` imports calc TYPES and is allowlisted for the
    package, but must not be flagged as an executor.
    """
    from tests.unit._calc_import_scan import import_targets, reaches_execution

    tree = ast.parse("from openexecutive.calc.contract import Operand\n")
    targets = import_targets(
        tree,
        path=pathlib.Path("specialists/calculation_proposal.py"),
        root=pathlib.Path(),
        package_name="openexecutive",
    )
    assert reaches_execution(targets) == set()

    proposal = (
        _CALC_DIR.parent / "specialists" / "calculation_proposal.py"
    ).read_text(encoding="utf-8")
    proposal_targets = import_targets(
        ast.parse(proposal),
        path=pathlib.Path("specialists/calculation_proposal.py"),
        root=pathlib.Path(),
        package_name="openexecutive",
    )
    assert reaches_execution(proposal_targets) == set(), (
        "the proposal module imports types only and must not read as an executor"
    )
