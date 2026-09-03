"""Fingerprint identity: what changes it, what must not, and who owns it.

A fingerprint is only useful if it answers one question reliably — "is this the
same calculation?" — so the tests come in matched pairs: something that must
change it, and something that must not.

The cross-process test is the one that matters most. A hash that is stable
within a process but varies between them (through `PYTHONHASHSEED`, dict
ordering, or a locale-sensitive format) would look correct in every unit test
and be useless the moment two workers compare notes.
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from openexecutive.calc import fingerprint as fp_mod
from openexecutive.calc.authority import current_authority
from openexecutive.calc.contract import (
    FINGERPRINT_EXCLUDED_FIELDS,
    FINGERPRINT_INCLUDED_FIELDS,
    CalculationRequest,
    Correlation,
    NormalizedOperand,
    Operand,
)
from openexecutive.calc.engine import execute
from openexecutive.calc.fingerprint import (
    FAILURE_FINGERPRINT_RULE,
    FINGERPRINT_ALGORITHM,
    build_payload,
    canonical_json,
    compute_fingerprint,
    fingerprint_for,
)
from openexecutive.calc.units import Unit

AT = "2026-09-02T00:00:00Z"
TND = "currency:TND"


def _op(oid: str, value: str, unit: str, role: str = "input") -> Operand:
    return Operand(operand_id=oid, label=oid, value=value, unit=Unit(code=unit),
                   basis="applicant_stated", role=role)  # type: ignore[arg-type]


def _normalized(oid: str, value: str, unit: str) -> NormalizedOperand:
    u = Unit(code=unit)
    return NormalizedOperand(operand_id=oid, label=oid, original_value=value,
                             original_unit=u, normalized_value=value,
                             normalized_unit=u, basis="applicant_stated")


def _run(*, case: str = "c1", run: str = "r1", claim: str | None = None,
         request_id: str = "req-1", scale: int = 2, rounding: str = "ROUND_HALF_EVEN",
         values: tuple[str, str] = ("42000000", "30140000"),
         purpose: str = "reconcile", labels: tuple[str, str] = ("sources", "uses"),
         computed_at: str = AT, operation: str = "subtract"):
    request = CalculationRequest(
        request_id=request_id, operation=operation,  # type: ignore[arg-type]
        operands=(
            Operand(operand_id="o1", label=labels[0], value=values[0],
                    unit=Unit(code=TND), basis="applicant_stated"),
            Operand(operand_id="o2", label=labels[1], value=values[1],
                    unit=Unit(code=TND), basis="applicant_stated"),
        ),
        target_unit=Unit(code=TND), scale=scale,
        rounding=rounding,  # type: ignore[arg-type]
        purpose=purpose,
        correlation=Correlation(specialist="cfo", case_id=case, run_id=run,
                                claim_id=claim),
    )
    return execute(request, computed_at=computed_at)


# ---------------------------------------------------------------------------
# Authority: the engine computes, never accepts
# ---------------------------------------------------------------------------


def test_the_engine_generates_the_fingerprint() -> None:
    result = _run()
    assert result.fingerprint is not None
    assert len(result.fingerprint) == 64
    assert all(c in "0123456789abcdef" for c in result.fingerprint)


def test_no_public_entry_point_accepts_a_fingerprint() -> None:
    """Phase 1's ``issue_calculation_result`` takes one; the engine never does.

    That parameter was harmless while nothing consumed a fingerprint. It stops
    being harmless the moment one is used to recognise a repeated calculation,
    so the engine derives its own and forwards no caller value.
    """
    for func in (execute, fingerprint_for, build_payload):
        assert "fingerprint" not in inspect.signature(func).parameters


def test_a_caller_supplied_fingerprint_is_not_reused() -> None:
    """Even if a caller forges one on a request-shaped object, it cannot ride in.

    ``CalculationRequest`` has no fingerprint field at all — ``extra="forbid"``
    rejects the attempt rather than dropping it — so there is no route by which
    a caller value could reach the result.
    """
    from pydantic import ValidationError

    assert "fingerprint" not in CalculationRequest.model_fields
    with pytest.raises(ValidationError):
        CalculationRequest.model_validate({
            "request_id": "r", "operation": "add",
            "operands": [_op("o1", "1", "one").model_dump()],
            "purpose": "p",
            "correlation": Correlation(specialist="cfo", case_id="c",
                                       run_id="r").model_dump(),
            "fingerprint": "a" * 64,
        })


def test_failures_have_no_fingerprint_and_the_rule_is_documented() -> None:
    """A calculation that did not run has no identity to publish.

    Minting one would create a value that looks joinable with real results, and
    a dedup index keyed on it would return "already computed" for something
    never computed.
    """
    assert FAILURE_FINGERPRINT_RULE == "absent"
    failed = _run(values=("1", "0"), operation="divide")
    assert failed.arithmetic_status != "ARITHMETIC_VERIFIED"
    assert failed.fingerprint is None


def test_algorithm_is_named_and_is_sha256() -> None:
    assert FINGERPRINT_ALGORITHM == "sha256"
    payload = {"a": 1}
    import hashlib

    expected = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    assert compute_fingerprint(payload) == expected


# ---------------------------------------------------------------------------
# What must NOT change the fingerprint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"case": "other-case"},
        {"run": "other-run"},
        {"claim": "c99"},
        {"request_id": "some-other-request"},
        {"computed_at": "2030-12-31T23:59:59Z"},
        {"purpose": "an entirely different explanation of why"},
        {"labels": ("total inflows", "total outflows")},
    ],
)
def test_incidental_metadata_does_not_change_identity(kwargs: dict) -> None:
    """Same calculation in a different case, run, or minute — same fingerprint."""
    baseline = _run().fingerprint
    assert _run(**kwargs).fingerprint == baseline


def test_repeated_execution_is_stable() -> None:
    prints = {_run().fingerprint for _ in range(25)}
    assert len(prints) == 1


def test_excluded_fields_never_appear_in_the_payload() -> None:
    payload = build_payload(
        operation="subtract",
        normalized_operands=(_normalized("o1", "1", TND), _normalized("o2", "2", TND)),
        target_unit=Unit(code=TND), scale=2, rounding="ROUND_HALF_EVEN",
        authority=current_authority(),
    )
    text = canonical_json(payload)
    for excluded in FINGERPRINT_EXCLUDED_FIELDS:
        assert excluded not in payload
        assert f'"{excluded}"' not in text


def test_payload_keys_match_the_declared_contract() -> None:
    payload = build_payload(
        operation="subtract",
        normalized_operands=(_normalized("o1", "1", TND),),
        target_unit=Unit(code=TND), scale=2, rounding="ROUND_HALF_EVEN",
        authority=current_authority(), stated_value="1",
    )
    assert set(payload) == set(FINGERPRINT_INCLUDED_FIELDS)


def test_empty_operation_parameters_do_not_appear() -> None:
    """An ``add`` must not be separated from a ``subtract`` by an empty key."""
    payload = build_payload(
        operation="add",
        normalized_operands=(_normalized("o1", "1", TND),),
        target_unit=Unit(code=TND), scale=2, rounding="ROUND_HALF_EVEN",
        authority=current_authority(),
    )
    assert "parameters" not in payload


# ---------------------------------------------------------------------------
# What MUST change the fingerprint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"values": ("42000001", "30140000")},
        {"values": ("42000000", "30140001")},
        {"scale": 3},
        {"rounding": "ROUND_HALF_UP"},
        {"operation": "add"},
    ],
)
def test_calculation_defining_fields_change_identity(kwargs: dict) -> None:
    assert _run(**kwargs).fingerprint != _run().fingerprint


def test_operand_order_changes_identity_for_a_non_commutative_operation() -> None:
    forward = _run(values=("42000000", "30140000")).fingerprint
    reversed_ = _run(values=("30140000", "42000000")).fingerprint
    assert forward != reversed_


def test_operand_order_changes_identity_even_when_commutative() -> None:
    """Order is never canonicalised — labels and provenance ride with position.

    Reordering would let two requests that pair different labels with different
    provenance collapse onto one identity.
    """
    a = _run(operation="add", values=("1", "2"), labels=("first", "second")).fingerprint
    b = _run(operation="add", values=("2", "1"), labels=("first", "second")).fingerprint
    assert a != b


def test_unit_identity_changes_the_fingerprint() -> None:
    base = fingerprint_for(
        operation="convert_unit",
        normalized_operands=(_normalized("o1", "1", "kg"),),
        target_unit=Unit(code="t"), scale=2, rounding="ROUND_HALF_EVEN",
        authority=current_authority(),
    )
    other = fingerprint_for(
        operation="convert_unit",
        normalized_operands=(_normalized("o1", "1", "kg"),),
        target_unit=Unit(code="kg"), scale=2, rounding="ROUND_HALF_EVEN",
        authority=current_authority(),
    )
    assert base != other


def test_stated_comparison_value_changes_the_fingerprint() -> None:
    kwargs = dict(
        operation="variance",
        normalized_operands=(_normalized("o1", "5720", "t"),),
        target_unit=Unit(code="t"), scale=0, rounding="ROUND_HALF_EVEN",
        authority=current_authority(),
    )
    assert (fingerprint_for(**kwargs, stated_value="572")  # type: ignore[arg-type]
            != fingerprint_for(**kwargs, stated_value="571"))  # type: ignore[arg-type]


def test_operation_parameters_change_the_fingerprint() -> None:
    """A conversion made under a declared policy is a different calculation."""
    kwargs = dict(
        operation="convert_unit",
        normalized_operands=(_normalized("o1", "12", "month"),),
        target_unit=Unit(code="year"), scale=0, rounding="ROUND_HALF_EVEN",
        authority=current_authority(),
    )
    plain = fingerprint_for(**kwargs)  # type: ignore[arg-type]
    with_policy = fingerprint_for(  # type: ignore[arg-type]
        **kwargs, time_conversion_policy="calendar_12_months_per_year")
    assert plain != with_policy
    with_weight = fingerprint_for(**kwargs, weight_policy="normalized_by_engine")  # type: ignore[arg-type]
    assert with_weight not in (plain, with_policy)


def test_authority_version_changes_the_fingerprint() -> None:
    """A contract-phase result must not collide with an engine-phase one."""
    from openexecutive.calc.contract import KNOWN_AUTHORITY_VERSIONS, ApplicationAuthority

    assert len(KNOWN_AUTHORITY_VERSIONS) >= 1
    real = current_authority()
    payload_a = build_payload(
        operation="add", normalized_operands=(_normalized("o1", "1", TND),),
        target_unit=Unit(code=TND), scale=2, rounding="ROUND_HALF_EVEN", authority=real,
    )
    payload_b = dict(payload_a)
    payload_b["authority_version"] = "9.9.9-hypothetical-engine"
    assert compute_fingerprint(payload_a) != compute_fingerprint(payload_b)
    assert isinstance(real, ApplicationAuthority)


# ---------------------------------------------------------------------------
# Cross-process stability
# ---------------------------------------------------------------------------


_CHILD = """
import json, sys
sys.path.insert(0, {core!r})
from openexecutive.calc.contract import CalculationRequest, Correlation, Operand
from openexecutive.calc.engine import execute
from openexecutive.calc.units import Unit

request = CalculationRequest(
    request_id="req-1", operation="subtract",
    operands=(
        Operand(operand_id="o1", label="sources", value="42000000",
                unit=Unit(code="currency:TND"), basis="applicant_stated"),
        Operand(operand_id="o2", label="uses", value="30140000",
                unit=Unit(code="currency:TND"), basis="applicant_stated"),
    ),
    target_unit=Unit(code="currency:TND"), scale=2, purpose="reconcile",
    correlation=Correlation(specialist="cfo", case_id="c1", run_id="r1"),
)
result = execute(request, computed_at="2026-09-02T00:00:00Z")
print(json.dumps({{"fp": result.fingerprint, "value": result.result_value}}))
"""


def _child_fingerprint(hash_seed: str) -> dict:
    core = str(Path(__file__).resolve().parents[2])
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD.format(core=core)],
        capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": hash_seed,
             "PYTHONPATH": core},
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_fingerprint_is_stable_across_processes_and_hash_seeds() -> None:
    """Different interpreters, different ``PYTHONHASHSEED`` — same identity.

    A hash stable only within one process would pass every other test here and
    be useless the moment two workers compare results.
    """
    in_process = _run().fingerprint
    first = _child_fingerprint("0")
    second = _child_fingerprint("12345")
    assert first["fp"] == second["fp"] == in_process
    assert first["value"] == second["value"] == "11860000.00"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_canonical_json_is_key_sorted_and_compact() -> None:
    payload = build_payload(
        operation="subtract",
        normalized_operands=(_normalized("o1", "1", TND), _normalized("o2", "2", TND)),
        target_unit=Unit(code=TND), scale=2, rounding="ROUND_HALF_EVEN",
        authority=current_authority(),
    )
    text = canonical_json(payload)
    assert text == canonical_json(payload)
    positions = [text.index(f'"{k}"') for k in ("authority_id", "operands", "operation")]
    assert positions == sorted(positions)
    assert ", " not in text and '": ' not in text


def test_payload_round_trips_through_json() -> None:
    payload = build_payload(
        operation="add", normalized_operands=(_normalized("o1", "1", TND),),
        target_unit=Unit(code=TND), scale=2, rounding="ROUND_HALF_EVEN",
        authority=current_authority(),
    )
    assert json.loads(canonical_json(payload)) == payload


def test_fingerprint_module_performs_no_arithmetic_and_no_io() -> None:
    import ast

    tree = ast.parse(inspect.getsource(fp_mod))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"hashlib", "json", "typing", "openexecutive", "__future__"}
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for banned in ("eval", "exec", "open", "compile", "__import__"):
        assert banned not in called
