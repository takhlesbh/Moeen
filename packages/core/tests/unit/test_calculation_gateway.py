"""The gateway: minted identity, one engine call, and nothing else.

Phase 3B2 gave it exactly one production caller, ``FinanceAgent`` on the
structured routing path, behind a default-off setting. The structural tests
below pin the two one-door invariants that replaced the earlier "unwired"
assertions: only the gateway imports the engine, and only ``finance.py`` calls
the gateway.
"""
from __future__ import annotations

import ast
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openexecutive.calc.contract import (
    MAX_ID_LEN,
    MAX_REQUESTS_PER_BATCH,
    CalculationRequest,
    CalculationResult,
    Operand,
)
from openexecutive.calc.units import Unit
from openexecutive.specialists.calculation_gateway import (
    REQUEST_DOMAIN,
    SpecialistCalculations,
    execute_proposals,
    mint_request_id,
)
from openexecutive.specialists.calculation_proposal import CalculationProposal
from openexecutive.specialists.result_contract import (
    EMIT_SPECIALIST_RESULT_TOOL,
    SpecialistResult,
    parse_specialist_result,
    render_for_executive,
)
from tests.unit._calc_import_scan import import_targets, reaches_execution, scan_tree

AT = "2026-09-03T12:00:00Z"
USD = "currency:USD"
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "openexecutive"
REPO_ROOT = PACKAGE_ROOT.parents[2]


def _op(oid: str, value: str) -> Operand:
    return Operand(
        operand_id=oid,
        label=oid,
        value=value,
        unit=Unit(code=USD),
        basis="applicant_stated",
    )


def _proposal(value: str = "10", **kw: Any) -> CalculationProposal:
    base: dict[str, Any] = {
        "operation": "add",
        "operands": (_op("a", value), _op("b", "5")),
        "target_unit": Unit(code=USD),
        "purpose": "runway check",
    }
    base.update(kw)
    return CalculationProposal(**base)


def _run(**kw: Any) -> SpecialistCalculations:
    base: dict[str, Any] = {
        "specialist": "cfo",
        "proposals": [_proposal()],
        "case_id": "sess-1",
        "run_id": "turn-1",
        "computed_at": AT,
    }
    base.update(kw)
    return execute_proposals(**base)


# ---------------------------------------------------------------------------
# 1. The live CFO path is untouched
# ---------------------------------------------------------------------------


def _tool_message(payload: Any) -> SimpleNamespace:
    block = SimpleNamespace(
        type="tool_use", id="t1", name=EMIT_SPECIALIST_RESULT_TOOL["name"], input=payload
    )
    return SimpleNamespace(content=[block])


NARRATIVES = [
    "Runway is 7.4 months at current burn.",
    "",
    "   padded   ",
    "line one\nline two\n",
    "unicode: €4.2M — naïve 日本語",
    "a; b; c",
]


@pytest.mark.parametrize("narrative", NARRATIVES)
def test_the_cfo_render_path_is_byte_identical(narrative: str) -> None:
    """The narrative is the compatibility surface; nothing here may touch it."""
    parsed = parse_specialist_result(
        _tool_message({"narrative": narrative, "claims": []}),
        specialist="cfo",
        model="test",
    )
    assert parsed.narrative == narrative
    assert render_for_executive(parsed) == narrative
    assert render_for_executive(parsed).encode("utf-8") == narrative.encode("utf-8")


@pytest.mark.parametrize(
    "specialist",
    ["cfo", "  padded  ", "c" * 200, "cfo\nforged", "cfo; forged", ""],
)
def test_the_cfo_parser_never_raises(specialist: str) -> None:
    """Its recovery path coerces rather than validates, and must keep doing so.

    A validator added to ``SpecialistResult.specialist`` would turn this into a
    double fault: ``_degraded`` would raise, the outer handler would call it
    again, and the second raise would escape onto the live CFO path. Adding no
    field and no validator to that module is what keeps this true.
    """
    result = parse_specialist_result(object(), specialist=specialist, model="m")
    assert isinstance(result, SpecialistResult)
    assert result.degraded is True


def test_the_cfo_parser_reason_format_is_unchanged() -> None:
    """``_join`` composes with ``"; "``; nothing may rewrite its output."""
    parsed = parse_specialist_result(
        _tool_message(
            {
                "narrative": "n",
                "claims": [{"claim_id": "c1", "text": "t", "kind": "bogus"}, "x"],
                "unknown_key": 1,
            }
        ),
        specialist="cfo",
        model="test",
    )
    assert parsed.degraded_reason is not None
    assert "; " in parsed.degraded_reason
    assert "?" not in parsed.degraded_reason


def test_specialist_result_carries_intent_but_never_a_record() -> None:
    """``calculation_requests`` is model-owned INTENT and lives on the result;
    an engine record never does — those ride on the application-owned envelope."""
    assert "calculation_requests" in SpecialistResult.model_fields
    for forbidden in ("calculations", "results", "calculation_results", "requests"):
        assert forbidden not in SpecialistResult.model_fields


def test_the_tool_schema_exposes_no_calculation_surface() -> None:
    serialized = repr(EMIT_SPECIALIST_RESULT_TOOL)
    for forbidden in (
        "calculations",
        "calculation_proposals",
        "request_id",
        "arithmetic_status",
        "fingerprint",
        "authority",
        "computed_at",
        "verified_result",
    ):
        assert forbidden not in serialized


def test_the_protected_modules_declare_no_calculation_surface() -> None:
    """Durable structural assertions, not a diff against HEAD.

    An earlier version shelled out to ``git diff --name-only HEAD``, which is
    non-empty only while the change is uncommitted: the moment it lands the
    command returns nothing, the loop body never runs, and the test passes
    forever regardless of what the protected files contain. Enforcement that
    expires on commit is worse than none, because a doc cited it as the
    guarantee.

    These four assertions hold at any commit, in any checkout, with or without
    git installed.
    """
    # 1. The model-authored result type carries no calculation RECORD.
    for forbidden in ("calculations", "results", "calculation_results", "proposals"):
        assert forbidden not in SpecialistResult.model_fields

    # 2. Door B: finance.py reaches the gateway and ONLY the gateway. It never
    #    names the engine, the authority module, or the calc package at all.
    finance = (PACKAGE_ROOT / "agents" / "finance.py").read_text(encoding="utf-8")
    assert "calculation_gateway" in finance
    for absent in ("openexecutive.calc", "calc.engine", "calc.authority",
                   "execute_batch", "issue_calculation_result"):
        assert absent not in finance, f"finance.py references {absent}"
    # The router never touches either calc module: it dispatches by capability.
    router = (PACKAGE_ROOT / "orchestrator" / "router.py").read_text(encoding="utf-8")
    for absent in ("calculation_gateway", "calculation_proposal", "execute_proposals"):
        assert absent not in router, f"router.py references {absent}"

    # 3. calc stays a leaf: it imports nothing from specialists. Resolved with
    #    the shared resolver, so ``from .. import specialists`` is attributed to
    #    ``openexecutive.specialists`` rather than matched as a substring.
    specialists = "openexecutive.specialists"
    for path in (PACKAGE_ROOT / "calc").rglob("*.py"):
        if "__pycache__" in path.relative_to(PACKAGE_ROOT).parts:
            continue
        targets = import_targets(
            ast.parse(path.read_text(encoding="utf-8")),
            path=path,
            root=PACKAGE_ROOT,
            package_name="openexecutive",
        )
        leaked = {
            target
            for target, _binds in targets
            if target == specialists or target.startswith(f"{specialists}.")
        }
        assert not leaked, f"{path.name} imports {sorted(leaked)}"

    # 4. Exactly one production module reaches the engine.
    assert _engine_importers() == ["specialists/calculation_gateway.py"]


def test_the_specialist_parser_module_has_no_calculation_validator() -> None:
    """The read-only boundary that actually mattered, asserted on content.

    Six earlier attempts were rejected for editing ``result_contract.py``. Two
    HIGHs came from adding a validator to a type whose producers had not been
    enumerated: ``_degraded`` coerces ``specialist`` by truthiness, so any
    validator on that field makes the parser raise, be caught, and raise again
    from the retry — on the live CFO path.
    """
    source = (PACKAGE_ROOT / "specialists" / "result_contract.py").read_text(
        encoding="utf-8"
    )
    # It may name the PROPOSAL type (model-owned intent lives on the result);
    # it must never name the gateway, a record type, or the executor.
    for absent in (
        "calculation_gateway",
        "CalculationResult",
        "SpecialistCalculations",
        "execute_proposals",
        "is_safe_text",
    ):
        assert absent not in source, f"result_contract.py references {absent}"



# ---------------------------------------------------------------------------
# 2. Minted identity
# ---------------------------------------------------------------------------


def test_the_minted_id_is_a_digest_not_model_text() -> None:
    request = _run().requests[0]
    assert len(request.request_id) == 64
    assert all(c in "0123456789abcdef" for c in request.request_id)


def test_identity_is_deterministic_and_ignores_the_clock() -> None:
    a = _run()
    b = _run(computed_at="2027-01-01T00:00:00Z")
    assert a.requests[0].request_id == b.requests[0].request_id


def test_identity_is_stable_across_processes() -> None:
    """A replay a year from now must produce the same id, under any hash seed."""
    script = (
        "from openexecutive.calc.contract import Operand;"
        "from openexecutive.calc.units import Unit;"
        "from openexecutive.specialists.calculation_proposal import CalculationProposal;"
        "from openexecutive.specialists.calculation_gateway import execute_proposals;"
        "u=Unit(code='currency:USD');"
        "o=lambda i,v: Operand(operand_id=i,label=i,value=v,unit=u,basis='applicant_stated');"
        "p=CalculationProposal(operation='add',operands=(o('a','10'),o('b','5')),"
        "target_unit=u,purpose='runway check');"
        "print(execute_proposals(specialist='cfo',proposals=[p],case_id='sess-1',"
        "run_id='turn-1',computed_at='2026-09-03T12:00:00Z').requests[0].request_id)"
    )
    seen = set()
    for seed in ("1", "424242"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PACKAGE_ROOT.parent,
            capture_output=True,
            text=True,
            env=env,
        )
        assert out.returncode == 0, out.stderr[-800:]
        seen.add(out.stdout.strip())
    assert len(seen) == 1
    assert seen == {_run().requests[0].request_id}


@pytest.mark.parametrize(
    "override",
    [
        {"specialist": "cso"},
        {"case_id": "sess-2"},
        {"run_id": "turn-2"},
    ],
)
def test_the_frame_enters_identity(override: dict[str, str]) -> None:
    assert _run().requests[0].request_id != _run(**override).requests[0].request_id


def test_cross_specialist_isolation() -> None:
    """Two specialists proposing identical arithmetic get different addresses.

    Their fingerprints match — the arithmetic *is* identical — but the records
    are separately addressable, so neither can be mistaken for the other.
    """
    cfo = _run(specialist="cfo")
    cso = _run(specialist="cso")
    assert cfo.results[0].request_id != cso.results[0].request_id
    assert cfo.results[0].fingerprint == cso.results[0].fingerprint
    assert cfo.results[0].correlation.specialist == "cfo"
    assert cso.results[0].correlation.specialist == "cso"


def test_position_and_content_both_enter_identity() -> None:
    same = _run(proposals=[_proposal("10"), _proposal("10")])
    assert len({r.request_id for r in same.requests}) == 2
    differing = _run(proposals=[_proposal("10"), _proposal("20")])
    assert len({r.request_id for r in differing.requests}) == 2


def test_a_validated_claim_ref_enters_identity() -> None:
    """Two claims over identical arithmetic must not share one address."""
    known = frozenset({"c1", "c2"})
    one = _run(proposals=[_proposal(claim_ref="c1")], known_claim_ids=known)
    two = _run(proposals=[_proposal(claim_ref="c2")], known_claim_ids=known)
    bare = _run(proposals=[_proposal()])
    ids = {
        one.requests[0].request_id,
        two.requests[0].request_id,
        bare.requests[0].request_id,
    }
    assert len(ids) == 3
    assert one.results[0].correlation.claim_id == "c1"


def test_the_pre_image_is_length_prefixed_so_no_tuple_can_alias_another() -> None:
    """A plain separator join would let a shifted component forge another tuple."""
    shifted = mint_request_id(
        case_id="c", run_id="r", specialist="cfo\x1f0", position=1,
        claim_ref="x", content="C",
    )
    plain = mint_request_id(
        case_id="c", run_id="r", specialist="cfo", position=0,
        claim_ref="1\x1fx", content="C",
    )
    assert shifted != plain


def test_the_request_domain_is_tagged() -> None:
    assert REQUEST_DOMAIN.startswith("openexecutive.calc.request")


# ---------------------------------------------------------------------------
# 3. Evidence can never become verified
# ---------------------------------------------------------------------------


def test_no_result_ever_reports_verified_evidence() -> None:
    for proposals in ([_proposal()], [_proposal("10"), _proposal("20")]):
        out = _run(proposals=proposals)
        for record in out.results:
            assert record.evidence.status == "EVIDENCE_UNAVAILABLE"
            assert record.is_verified_evidence() is False


def test_the_gateway_exposes_no_evidence_parameter() -> None:
    import inspect

    parameters = inspect.signature(execute_proposals).parameters
    assert "evidence" not in parameters
    assert "evidence_by_request" not in parameters


def test_correct_arithmetic_over_unbacked_inputs_is_still_unbacked() -> None:
    record = _run().results[0]
    assert record.arithmetic_status == "ARITHMETIC_VERIFIED"
    assert record.result_value == "15.00"
    assert record.is_verified_evidence() is False


# ---------------------------------------------------------------------------
# 4. Isolation and failure
# ---------------------------------------------------------------------------


def test_one_bad_proposal_among_five_good_preserves_five_results() -> None:
    """The headline isolation property, stated over the count."""
    proposals = [_proposal(str(i + 1)) for i in range(5)]
    proposals.append(_proposal("99", claim_ref="ghost"))
    out = _run(proposals=proposals, known_claim_ids=frozenset({"c1"}))

    assert len(out.requests) == 5
    assert len(out.results) == 5
    assert out.dropped == ("proposal_5_unknown_claim_ref",)
    assert all(r.arithmetic_status == "ARITHMETIC_VERIFIED" for r in out.results)


def test_a_dropped_proposal_reports_a_bounded_code_not_model_text() -> None:
    secret = "ACQUISITION-PRICE-44000000"
    out = _run(
        proposals=[_proposal(claim_ref=secret[:MAX_ID_LEN])],
        known_claim_ids=frozenset(),
    )
    assert out.results == ()
    assert out.dropped == ("proposal_0_unknown_claim_ref",)
    for code in out.dropped:
        assert secret not in code
        assert code.isprintable()


def test_a_non_proposal_object_drops_only_itself() -> None:
    out = _run(proposals=[{"not": "a proposal"}, _proposal()])  # type: ignore[list-item]
    assert len(out.results) == 1
    assert out.dropped == ("proposal_0_not_a_proposal",)


def test_a_typed_arithmetic_failure_stays_a_visible_record() -> None:
    """Division by zero is a record a reader can see, not a silent absence."""
    bad = CalculationProposal(
        operation="divide",
        operands=(_op("a", "10"), _op("b", "0")),
        target_unit=Unit(code="one"),
        purpose="ratio",
    )
    out = _run(proposals=[bad])
    assert len(out.results) == 1
    assert out.results[0].arithmetic_status == "DIVISION_BY_ZERO"
    assert out.results[0].result_value is None
    assert out.results[0].fingerprint is None


def test_engine_failure_returns_typed_unavailable_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openexecutive.specialists.calculation_gateway as gw

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(gw, "execute_batch", _boom)
    out = gw.execute_proposals(
        specialist="cfo",
        proposals=[_proposal("10"), _proposal("20")],
        case_id="sess-1",
        run_id="turn-1",
        computed_at=AT,
    )
    assert len(out.results) == 2
    assert "batch_execution_failed" in out.dropped
    for record in out.results:
        assert record.arithmetic_status == "CALCULATION_UNAVAILABLE"
        assert record.result_value is None
        assert record.fingerprint is None
        assert record.is_verified_evidence() is False
        assert record.errors[0].code == "GATEWAY_UNAVAILABLE"


def test_no_proposals_returns_an_empty_record() -> None:
    out = _run(proposals=[])
    assert out == SpecialistCalculations(specialist="cfo")


@pytest.mark.parametrize("field", ["specialist", "case_id", "run_id"])
@pytest.mark.parametrize("evil", ["a\nb", " pad", "", "x" * 65])
def test_an_unusable_application_identifier_is_refused(field: str, evil: str) -> None:
    """These reach a persisted correlation and a log line."""
    with pytest.raises(ValueError, match="not a usable identifier"):
        _run(**{field: evil})


def test_proposals_over_the_batch_limit_are_refused() -> None:
    with pytest.raises(ValueError, match="batch limit"):
        _run(proposals=[_proposal(str(i + 1)) for i in range(MAX_REQUESTS_PER_BATCH + 1)])


# ---------------------------------------------------------------------------
# 5. The record itself
# ---------------------------------------------------------------------------


def test_the_record_is_frozen_yet_pydantic_reconstructs_it() -> None:
    """Frozen, no ``model_validate`` on the class — and still reconstructible.

    The invariant, stated precisely because two earlier docstrings got it wrong:

    * the record is frozen (attribute assignment raises);
    * it defines no class-level ``model_validate`` — it is a stdlib dataclass,
      not a ``BaseModel``;
    * **nevertheless** pydantic reconstructs it from an untrusted mapping,
      because pydantic builds validators for stdlib dataclasses;
    * therefore the type carries no provenance, and Phase 3B2 must never trust
      a value on type identity alone.

    The absence of ``model_validate`` prevents nothing. It is asserted here only
    so the reproduction below cannot be dismissed as "it is a BaseModel after
    all".
    """
    from pydantic import TypeAdapter

    out = _run()
    with pytest.raises(AttributeError):
        out.specialist = "cso"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        out.injected = "x"  # type: ignore[attr-defined]

    assert "model_validate" not in vars(SpecialistCalculations)
    assert not hasattr(SpecialistCalculations, "model_fields")

    # Positive reproduction: pydantic reconstructs the record with no help from
    # this module, from plain untrusted data.
    revived = TypeAdapter(SpecialistCalculations).validate_python(
        {"specialist": "cfo", "requests": [], "results": [], "dropped": ["forged"]}
    )
    assert isinstance(revived, SpecialistCalculations)
    assert revived.dropped == ("forged",)
    assert revived != out, "a reconstructed record is indistinguishable by type alone"


def test_every_result_answers_a_recorded_request() -> None:
    out = _run(proposals=[_proposal("10"), _proposal("20")])
    assert {r.request_id for r in out.results} == {r.request_id for r in out.requests}


def test_the_gateway_holds_no_mutable_module_state() -> None:
    import openexecutive.specialists.calculation_gateway as gw

    mutable = {
        name: value
        for name, value in vars(gw).items()
        if isinstance(value, set | list | dict | bytearray) and not name.startswith("__")
    }
    assert mutable == {}, f"module-level mutable state: {sorted(mutable)}"


def test_repeated_calls_are_independent() -> None:
    first, second = _run(), _run()
    assert first == second


def test_the_gateway_never_touches_a_narrative() -> None:
    source = (PACKAGE_ROOT / "specialists" / "calculation_gateway.py").read_text(
        encoding="utf-8"
    )
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr == "narrative":
            raise AssertionError("the gateway reads or writes .narrative")
    assert "render_for_executive" not in source
    assert "SpecialistResult" not in source


# ---------------------------------------------------------------------------
# 6. The import boundary
# ---------------------------------------------------------------------------


def _engine_importers() -> list[str]:
    """Every production module that can reach an execution or authority surface.

    Resolved by the shared ``_calc_import_scan`` resolver — the same one the
    adversarial and foundation scanners use — so the three cannot drift. This
    function was the third hand-rolled copy: it read only ``ImportFrom.module``,
    so ``from openexecutive.calc import engine`` and four other ordinary forms
    reached the engine while the one-door assertion below stayed green.
    ``test_engine_importers_catches_every_bypass_form`` now pins each of them.
    """
    scanned = scan_tree(
        PACKAGE_ROOT, package_name="openexecutive", skip_parts=("calc", "__pycache__")
    )
    return sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path, targets in scanned.items()
        if reaches_execution(targets)
    )



def test_exactly_one_production_module_imports_the_engine() -> None:
    assert _engine_importers() == ["specialists/calculation_gateway.py"]


def _second_door(tmp_path: Path, source: str) -> Path:
    """A package tree with one non-allowlisted module holding ``source``."""
    package = tmp_path / "openexecutive"
    (package / "calc").mkdir(parents=True)
    (package / "calc" / "__init__.py").write_text("", encoding="utf-8")
    (package / "orchestrator").mkdir()
    (package / "orchestrator" / "__init__.py").write_text("", encoding="utf-8")
    (package / "orchestrator" / "second_door.py").write_text(source, encoding="utf-8")
    return package


def _every_bypass_form() -> dict[str, str]:
    from tests.unit.test_calc_adversarial import _ALL_BYPASS_FORMS

    return _ALL_BYPASS_FORMS


@pytest.mark.parametrize("form", sorted(_every_bypass_form()))
def test_engine_importers_catches_every_bypass_form(
    form: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION (MEDIUM): five of these were invisible to this file's scanner.

    ``from openexecutive import calc``, ``from openexecutive.calc import engine``,
    ``from openexecutive.calc import authority``, ``from .. import calc`` and
    ``from ..calc import engine`` all reached the engine with
    ``test_exactly_one_production_module_imports_the_engine`` green. The direct
    dotted forms are in the set too, so the rewire is shown not to have lost
    what the old scanner did catch.
    """
    monkeypatch.setattr(
        "tests.unit.test_calculation_gateway.PACKAGE_ROOT",
        _second_door(tmp_path, _every_bypass_form()[form]),
        raising=True,
    )
    assert _engine_importers() == ["orchestrator/second_door.py"]


def test_engine_importers_ignores_a_type_only_importer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over-reporting would make the one-door list stop being obeyed."""
    monkeypatch.setattr(
        "tests.unit.test_calculation_gateway.PACKAGE_ROOT",
        _second_door(tmp_path, "from openexecutive.calc.contract import Operand\n"),
        raising=True,
    )
    assert _engine_importers() == []


def test_the_production_scanner_rejects_a_shadow_module_in_a_temp_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the REAL scanner over a throwaway tree containing a shadow.

    Two earlier versions each missed half of this. One wrote the shadow into the
    live checkout, so an aborted run left it behind and broke the allowlist for
    everyone. The next moved to ``tmp_path`` but asserted against a *local copy*
    of the scanner defined inside the test — so reverting the production
    allowlist to relative-string matching would not have failed it.

    This points the production scanner's roots at ``tmp_path`` and asserts it
    fails, which is the only form that catches a regression in the real code.
    """
    import tests.unit.test_calc_contract_foundation as foundation

    # The scanner derives its roots from CALC_DIR: package_root = CALC_DIR.parent
    # and repo_root = package_root.parents[2]. Mirror that depth exactly so the
    # real code walks the fake tree.
    fake_calc = tmp_path / "packages" / "core" / "openexecutive" / "calc"
    fake_calc.mkdir(parents=True)
    (fake_calc / "__init__.py").write_text("", encoding="utf-8")

    shadow = tmp_path / "evals" / "specialists" / "calculation_gateway.py"
    shadow.parent.mkdir(parents=True)
    shadow.write_text(
        "from openexecutive.calc.engine import execute_batch" + chr(10),
        encoding="utf-8",
    )
    # It reuses the allowlisted RELATIVE path — the whole point of the bypass.
    assert shadow.relative_to(tmp_path / "evals").as_posix() in (
        foundation._CALC_IMPORTERS
    )

    monkeypatch.setattr(foundation, "CALC_DIR", fake_calc, raising=True)
    with pytest.raises(AssertionError, match="calculation_gateway.py"):
        foundation.test_only_allowlisted_production_modules_import_calc()



def test_the_repository_allowlists_pass_unmodified() -> None:
    """The live scanners agree, without anything being written to the repo."""
    import tests.unit.test_calc_adversarial as adversarial
    import tests.unit.test_calc_contract_foundation as foundation

    foundation.test_only_allowlisted_production_modules_import_calc()
    foundation.test_every_allowlisted_calc_importer_exists()
    adversarial.test_only_the_calculation_gateway_imports_the_engine()
    adversarial.test_the_allowlisted_engine_importer_exists()



def test_the_gateway_has_exactly_the_expected_referrers() -> None:
    """Every production module that so much as names either calc-specialist file.

    ``result_contract`` names the proposal type (intent on the result),
    ``routed_output`` names the record container (the envelope), and
    ``finance`` is the one caller. Anything else appearing here is a widening
    that needs a review, not a green run.
    """
    referrers = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path.name in ("calculation_gateway.py", "calculation_proposal.py"):
            continue
        if "__pycache__" in path.parts:
            continue
        # calc/ is a leaf and cannot import specialists (assertion 3 above);
        # its package docstring merely NAMES the gateway as the one door.
        if path.relative_to(PACKAGE_ROOT).parts[0] == "calc":
            continue
        text = path.read_text(encoding="utf-8")
        if "calculation_gateway" in text or "calculation_proposal" in text:
            referrers.append(path.relative_to(PACKAGE_ROOT).as_posix())
    assert sorted(referrers) == [
        "agents/finance.py",
        "specialists/result_contract.py",
        "specialists/routed_output.py",
    ]


def _execute_proposals_callers() -> list[str]:
    """Door B, resolved on the AST: every production call of ``execute_proposals``.

    A call is ``execute_proposals(...)`` or ``<anything>.execute_proposals(...)``.
    Text search would also match the docstrings that explain the rule.
    """
    callers: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "calculation_gateway.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name == "execute_proposals":
                callers.append(path.relative_to(PACKAGE_ROOT).as_posix())
                break
    return callers


def test_exactly_one_production_module_calls_the_gateway() -> None:
    """Door B: ``agents/finance.py`` is the only caller of ``execute_proposals``."""
    assert _execute_proposals_callers() == ["agents/finance.py"]


def test_the_new_modules_stay_within_the_production_budget() -> None:
    lines = sum(
        len((PACKAGE_ROOT / "specialists" / name).read_text(encoding="utf-8").splitlines())
        for name in ("calculation_proposal.py", "calculation_gateway.py")
    )
    # Raised from 560 in Phase 3B2 for the proposal module's wire-schema
    # constant (~90 lines) and the gateway's one-line id screen.
    assert lines <= 680, f"{lines} production lines exceeds the budget"


def test_the_result_types_are_reused_not_redefined() -> None:
    """Three distinct types, and only the middle one is new here."""
    out = _run()
    assert isinstance(out.requests[0], CalculationRequest)
    assert isinstance(out.results[0], CalculationResult)
    assert CalculationRequest.__module__ == "openexecutive.calc.contract"
    assert CalculationResult.__module__ == "openexecutive.calc.contract"
    assert CalculationProposal.__module__ == (
        "openexecutive.specialists.calculation_proposal"
    )


# ---------------------------------------------------------------------------
# 7. Direct reproductions of every confirmed review finding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_stamp",
    [
        "not-a-timestamp",
        "",
        "x" * 5000,
        "2026-09-03T12:00:00",          # no zone
        "2026-09-03T12:00:00+05:30",    # non-zero offset
        "\u0662026-09-03T12:00:00Z",    # Unicode digit
        None,
        12345,
        object(),
    ],
)
def test_regression_a_malformed_computed_at_never_escapes(
    bad_stamp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION (MEDIUM): it raised from inside the engine-failure fallback.

    ``case_id``/``run_id``/``specialist`` were screened; ``computed_at`` was not.
    The engine defends itself, but ``_unavailable`` calls
    ``issue_calculation_result`` directly, so a bad stamp raised from inside the
    ``except`` block that exists to keep a failed batch typed — destroying every
    record instead of preserving them.
    """
    import openexecutive.specialists.calculation_gateway as gw

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("engine down")

    monkeypatch.setattr(gw, "execute_batch", _boom)
    out = gw.execute_proposals(
        specialist="cfo",
        proposals=[_proposal()],
        case_id="c",
        run_id="r",
        computed_at=bad_stamp,
    )
    assert len(out.results) == 1, "the fallback still produces a typed record"
    record = out.results[0]
    assert record.arithmetic_status == "CALCULATION_UNAVAILABLE"
    assert record.computed_at.endswith("Z")
    assert record.is_verified_evidence() is False


@pytest.mark.parametrize(
    "bad_stamp",
    ["not-a-timestamp", "", "x" * 5000, "\u0662026-09-03T12:00:00Z", None, 12345],
)
def test_regression_a_refused_clock_never_becomes_a_verified_record(
    bad_stamp: Any,
) -> None:
    """REGRESSION (MEDIUM): it returned ARITHMETIC_VERIFIED stamped 1970.

    Canonicalising before ``execute_batch`` meant the engine's own
    ``INVALID_INPUT`` / ``invalid_computed_at`` record could never fire: a
    refused clock came back verified at a fabricated instant with empty
    ``errors``, ``warnings`` and ``dropped``. The original value now reaches the
    engine, which rejects it properly.
    """
    out = _run(computed_at=bad_stamp)
    record = out.results[0]
    assert record.arithmetic_status == "INVALID_INPUT"
    assert record.result_value is None
    assert [e.code for e in record.errors] == ["invalid_computed_at"]
    assert record.is_verified_evidence() is False



def test_a_valid_computed_at_is_preserved_exactly() -> None:
    """Both accepted spellings survive; only the engine normalises ``+00:00``."""
    assert _run().results[0].computed_at == AT
    assert _run(computed_at="2026-09-03T12:00:00+00:00").results[0].computed_at == AT
    for out in (_run(), _run(computed_at="2026-09-03T12:00:00+00:00")):
        assert out.results[0].arithmetic_status == "ARITHMETIC_VERIFIED"
        assert out.dropped == ()


# --- known_claim_ids is an authorization boundary --------------------------


def test_regression_a_string_known_claim_ids_is_refused_not_coerced() -> None:
    """REGRESSION (MEDIUM): ``in`` on a str is SUBSTRING containment.

    ``known_claim_ids="c1,c9x,c42"`` authorised ``"c9"`` — a claim ref nobody
    issued. Refused rather than split, because coercing would guess at a
    delimiter the caller never specified.
    """
    with pytest.raises(TypeError, match="not a string"):
        _run(
            proposals=[_proposal(claim_ref="c9")],
            known_claim_ids="c1,c9x,c42",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "hostile",
    [
        b"c1c9",
        bytearray(b"c1"),
        {"c1": "claim"},
        123,
        None,
        object(),
    ],
)
def test_a_deceptive_authorization_set_is_refused(hostile: Any) -> None:
    with pytest.raises(TypeError):
        _run(proposals=[_proposal(claim_ref="c1")], known_claim_ids=hostile)


def test_a_one_shot_iterator_is_refused() -> None:
    """A generator would be consumed by the first membership test.

    Every later proposal would then read an empty set and be denied — a silent
    authorization failure that looks like a model mistake.
    """
    with pytest.raises(TypeError, match="consumed"):
        _run(
            proposals=[_proposal(claim_ref="c1")],
            known_claim_ids=iter(["c1"]),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_element", ["", " c1", "c1 ", "a\nb", 42, None])
def test_a_malformed_element_invalidates_the_authorization_set(
    bad_element: Any,
) -> None:
    with pytest.raises(ValueError, match="not a usable claim id"):
        _run(proposals=[_proposal(claim_ref="c1")], known_claim_ids={"c1", bad_element})


@pytest.mark.parametrize("collection", [{"c1"}, frozenset({"c1"}), ["c1"], ("c1",)])
def test_any_real_finite_collection_of_valid_ids_is_accepted(
    collection: Any,
) -> None:
    out = _run(proposals=[_proposal(claim_ref="c1")], known_claim_ids=collection)
    assert out.results[0].correlation.claim_id == "c1"
    assert out.dropped == ()


def test_membership_is_exact_never_substring() -> None:
    out = _run(
        proposals=[_proposal(claim_ref="c9")], known_claim_ids={"c1", "c9x", "c42"}
    )
    assert out.results == ()
    assert out.dropped == ("proposal_0_unknown_claim_ref",)


# --- the record makes no security claim ------------------------------------


def test_regression_the_record_makes_no_provenance_claim() -> None:
    """REGRESSION (MEDIUM): two docstrings here asserted a guarantee.

    The first said "no deserialization path". The second narrowed it to "no
    automatic pydantic deserializer" — also false: pydantic builds a validator
    for stdlib dataclasses, so a ``BaseModel`` field of this type deserializes
    an untrusted body straight into it.
    """
    import pickle

    from pydantic import BaseModel

    class Envelope(BaseModel):
        calcs: SpecialistCalculations

    forged = SpecialistCalculations(specialist="cfo", dropped=("hand_made",))
    assert pickle.loads(pickle.dumps(forged)) == forged, "picklable, as documented"

    revived = Envelope.model_validate(
        {"calcs": {"specialist": "cfo", "requests": [], "results": [], "dropped": []}}
    )
    assert isinstance(revived.calcs, SpecialistCalculations), (
        "pydantic DOES auto-deserialize it — the docstring must not deny this"
    )

    doc = SpecialistCalculations.__doc__ or ""
    assert "establishes **no provenance whatsoever**" in doc
    assert "pydantic builds a validator for stdlib dataclasses" in doc
    assert "must not treat this type alone as evidence" in doc
    assert "3B2" in doc, "the next phase is warned explicitly"
    for retracted in ("no deserialization path", "no automatic", "automatic pydantic"):
        assert retracted not in doc, f"a retracted claim survives: {retracted}"


def test_the_record_performs_no_validation_and_says_so() -> None:
    junk = SpecialistCalculations(
        specialist=None,  # type: ignore[arg-type]
        requests="not a tuple",  # type: ignore[arg-type]
    )
    assert junk.specialist is None
    doc = " ".join((SpecialistCalculations.__doc__ or "").split())
    assert "a frozen dataclass runs no validators" in doc, (
        "the absence of validation is documented"
    )




# --- log hygiene ------------------------------------------------------------


class _Capture(logging.Handler):
    """Attach directly to the gateway's logger.

    Deliberately not ``caplog``: another module in the suite calls
    ``logging.basicConfig`` at import time, which changes what the root-level
    fixture captures — so a ``caplog`` assertion here passed alone and failed in
    a full run. Attaching to the named logger measures what this module emits,
    independent of whatever global logging state the suite has reached.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def text(self) -> str:
        return "\n".join(
            r.getMessage() + ("|EXC" if r.exc_info else "") for r in self.records
        )


@contextmanager
def _captured_gateway_logs() -> Any:
    """Capture the gateway's own log records, restoring all state afterwards.

    A context manager rather than a bare helper: an earlier version set the
    logger to DEBUG and never put the level back, leaking global state into
    every later test in the session.
    """
    import openexecutive.specialists.calculation_gateway as gw

    handler = _Capture()
    previous_level = gw.logger.level
    gw.logger.addHandler(handler)
    gw.logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        gw.logger.removeHandler(handler)
        gw.logger.setLevel(previous_level)


def test_regression_a_failing_proposal_forces_the_handler_and_leaks_nothing() -> None:
    """REGRESSION: the previous version of this test never reached the handler.

    It tampered ``purpose`` to a 26-character printable string, which
    ``CalculationRequest`` accepts — so no exception was raised, the ``except``
    never ran, and every assertion passed against an empty log. Restoring
    ``logger.exception`` would not have failed it.

    This forces the path with a payload that genuinely exceeds a request-side
    bound, then asserts the log carries neither model text nor exception detail.
    """

    secret = "ACQUISITION-PRICE-44000000"
    proposal = _proposal()
    object.__setattr__(proposal, "purpose", secret + " " + "x" * 600)

    with _captured_gateway_logs() as handler:
        out = _run(proposals=[proposal, _proposal("20")])

    assert out.dropped == ("proposal_0_invalid_request",), "the handler ran"
    assert len(out.results) == 1, "the sibling survives"
    assert handler.records, "a log line was emitted"
    assert secret not in handler.text
    assert "44000000" not in handler.text
    assert "ValidationError" not in handler.text, "no exception class name"
    assert "EXC" not in handler.text, "no traceback"


def test_a_failed_batch_logs_a_literal_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import openexecutive.specialists.calculation_gateway as gw

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("engine down: 44000000")

    monkeypatch.setattr(gw, "execute_batch", _boom)
    with _captured_gateway_logs() as handler:
        out = gw.execute_proposals(
            specialist="cfo", proposals=[_proposal()], case_id="c", run_id="r",
            computed_at=AT,
        )

    assert "batch execution failed" in handler.text
    assert "44000000" not in handler.text
    assert "RuntimeError" not in handler.text, "no dynamic exception name"
    assert "EXC" not in handler.text, "no traceback"
    assert out.dropped == ("batch_execution_failed",)


def test_no_dynamic_exception_detail_appears_in_the_module_source() -> None:
    """Mutation-resistant: a future edit reintroducing it fails here.

    An exception class name is attacker-influenceable — it can carry a newline
    and forge an audit line — and ``logger.exception`` renders a pydantic
    ValidationError's ``input`` values, which are model text.
    """
    source = (PACKAGE_ROOT / "specialists" / "calculation_gateway.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "exception":
            raise AssertionError("logger.exception emits a traceback")
        if not (
            isinstance(node.func.value, ast.Name) and node.func.value.id == "logger"
        ):
            continue
        # No log CALL may interpolate anything dynamic: every argument must be a
        # literal or a bare position. (``type(x).__name__`` survives elsewhere in
        # one TypeError message, which reports a caller bug to a developer and
        # never reaches a durable record — the ``dropped`` codes carry those.)
        for argument in node.args[1:]:
            assert isinstance(argument, ast.Name | ast.Constant), (
                "a log call interpolates a computed value"
            )
        for argument in ast.walk(node):
            if isinstance(argument, ast.Attribute) and argument.attr == "__name__":
                raise AssertionError("a log call interpolates a dynamic type name")




# --- the screen's real mechanism -------------------------------------------


def test_regression_isprintable_is_the_character_rule_and_is_stricter() -> None:
    """REGRESSION (LOW): a redundant category clause was credited for this.

    ``isprintable()`` already excludes every Cc/Cf/Zl/Zp codepoint — the removed
    clause could never be the deciding term — and it also excludes surrogates,
    private use, unassigned and the other Unicode spaces.
    """
    import unicodedata

    from openexecutive.specialists.calculation_proposal import (
        is_safe_descriptive_text,
    )

    redundant = [
        cp
        for cp in range(0x110000)
        if chr(cp).isprintable()
        and unicodedata.category(chr(cp)) in ("Cc", "Cf", "Zl", "Zp")
    ]
    assert redundant == [], "the removed clause was never load-bearing"

    stricter = {
        "surrogate": 0xD800,
        "private_use": 0xE000,
        "unassigned": 0x0378,
        "ogham_space": 0x1680,
        "nbsp": 0x00A0,
    }
    for name, cp in stricter.items():
        assert is_safe_descriptive_text(f"a{chr(cp)}b", max_length=64) is False, name

    source = (PACKAGE_ROOT / "specialists" / "calculation_proposal.py").read_text(
        encoding="utf-8"
    )
    assert "unicodedata" not in source, "the dead clause and its import are gone"
    assert "SEGMENT_DELIMITER" not in source, (
        "the coupling to result_contract._join's delimiter is gone"
    )


# ---------------------------------------------------------------------------
# 8. Reproductions for the targeted repair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_stamp",
    ["not-a-timestamp", "", "x" * 5000, "٢026-09-03T12:00:00Z", None, 12345],
)
def test_a_refused_clock_on_the_failure_path_is_reported_not_silent(
    bad_stamp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback may substitute, but never silently.

    The engine cannot rule on the timestamp when it never ran, so this path does
    substitute — and surfaces ``invalid_computed_at`` so a reader can tell a
    record stamped at the epoch because the clock was refused from one genuinely
    computed then.
    """
    import openexecutive.specialists.calculation_gateway as gw

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("engine down")

    monkeypatch.setattr(gw, "execute_batch", _boom)
    out = gw.execute_proposals(
        specialist="cfo", proposals=[_proposal()], case_id="c", run_id="r",
        computed_at=bad_stamp,
    )
    assert len(out.results) == 1
    record = out.results[0]
    assert record.arithmetic_status == "CALCULATION_UNAVAILABLE"
    assert record.result_value is None
    assert record.computed_at.endswith("Z")
    assert "invalid_computed_at" in out.dropped
    assert "batch_execution_failed" in out.dropped


@pytest.mark.parametrize("good_stamp", [AT, "2026-09-03T12:00:00+00:00"])
def test_a_valid_clock_on_the_failure_path_reports_no_substitution(
    good_stamp: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openexecutive.specialists.calculation_gateway as gw

    monkeypatch.setattr(
        gw, "execute_batch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    )
    out = gw.execute_proposals(
        specialist="cfo", proposals=[_proposal()], case_id="c", run_id="r",
        computed_at=good_stamp,
    )
    assert out.dropped == ("batch_execution_failed",)
    assert "invalid_computed_at" not in out.dropped


# --- revalidation at the boundary ------------------------------------------


def _forged(**overrides: Any) -> CalculationProposal:
    """A proposal built past every validator, as ``model_construct`` allows."""
    base: dict[str, Any] = {
        "operation": "add",
        "operands": (_op("a", "10"), _op("b", "5")),
        "target_unit": Unit(code=USD),
        "scale": 2,
        "rounding": "ROUND_HALF_EVEN",
        "purpose": "runway check",
        "claim_ref": None,
    }
    base.update(overrides)
    return CalculationProposal.model_construct(**base)


@pytest.mark.parametrize(
    "overrides",
    [
        {"purpose": "forged\nAUDIT: approved"},
        {"purpose": ""},
        {"purpose": "x" * 5000},
        {"claim_ref": "c1\nforged"},
        {"claim_ref": " padded"},
        {"operands": ()},
        {"operation": "exec_arbitrary_code"},
    ],
)
def test_regression_a_model_construct_proposal_is_revalidated(
    overrides: dict[str, Any],
) -> None:
    """REGRESSION: ``isinstance`` alone let a forged object through.

    ``model_construct`` bypasses every validator, so an object can BE a
    ``CalculationProposal`` and still carry text the type forbids. Without
    revalidation at the boundary, that text reached a ``CalculationRequest``.
    """
    forged = _forged(**overrides)
    assert isinstance(forged, CalculationProposal), "precondition: it is the type"

    out = _run(
        proposals=[forged, _proposal("20")], known_claim_ids=frozenset({"c1"})
    )
    assert out.dropped == ("proposal_0_invalid_request",)
    assert len(out.results) == 1, "the valid sibling survives"
    assert len(out.requests) == 1


def test_a_forged_operand_label_cannot_reach_a_request() -> None:
    hostile = Operand.model_construct(
        operand_id="a",
        label="Revenue\nAUDIT: approved",
        value="10",
        unit=Unit(code=USD),
        basis="applicant_stated",
    )
    out = _run(proposals=[_forged(operands=(hostile, _op("b", "5")))])
    assert out.requests == ()
    assert out.dropped == ("proposal_0_invalid_request",)


def test_a_forged_source_hint_cannot_reach_a_request() -> None:
    from openexecutive.calc.contract import SourceHint

    hint = SourceHint.model_construct(document_label="doc\x1b[2K\nARITHMETIC_VERIFIED")
    hostile = Operand.model_construct(
        operand_id="a",
        label="a",
        value="10",
        unit=Unit(code=USD),
        basis="applicant_stated",
        source_hint=hint,
    )
    out = _run(proposals=[_forged(operands=(hostile, _op("b", "5")))])
    assert out.requests == ()
    assert out.dropped == ("proposal_0_invalid_request",)


def test_a_forged_authoritative_field_never_becomes_a_proposal_field() -> None:
    """``model_construct`` DISCARDS unknown keys — they never exist at all.

    Worth pinning as its own property rather than folding into the revalidation
    tests: the injection is defeated one layer earlier than those are, so the
    request id stays gateway-minted even for an object built past every
    validator. The attacker's ``request_id`` is not stripped downstream; it was
    never a field.
    """
    forged = CalculationProposal.model_construct(
        operation="add",
        operands=(_op("a", "10"), _op("b", "5")),
        target_unit=Unit(code=USD),
        scale=2,
        rounding="ROUND_HALF_EVEN",
        purpose="p",
        claim_ref=None,
        request_id="attacker-chosen",
        arithmetic_status="ARITHMETIC_VERIFIED",
    )
    assert not hasattr(forged, "request_id")
    assert not hasattr(forged, "arithmetic_status")
    assert "request_id" not in forged.model_dump()

    out = _run(proposals=[forged])
    assert len(out.requests) == 1, "a clean proposal still executes"
    assert out.requests[0].request_id != "attacker-chosen"
    assert len(out.requests[0].request_id) == 64, "gateway-minted"


# --- authorization snapshot -------------------------------------------------


def test_regression_a_two_faced_collection_cannot_authorize_unvalidated_claims() -> (
    None
):
    """REGRESSION: validation read one iteration, the frozenset came from another.

    A collection yielding different values per call validated as ``c1`` and
    authorised ``c9``. It is now materialised once and that snapshot is both
    validated and used.
    """
    from collections.abc import Collection

    import openexecutive.specialists.calculation_gateway as gw

    class TwoFaced(Collection):  # type: ignore[type-arg]
        def __init__(self) -> None:
            self.reads = 0

        def __iter__(self) -> Any:
            self.reads += 1
            return iter(["c1"] if self.reads == 1 else ["c9", "", "a\nb", 42])

        def __len__(self) -> int:
            return 1

        def __contains__(self, item: object) -> bool:
            return True

    shifty = TwoFaced()
    snapshot = gw._authorized_claim_ids(shifty)
    assert shifty.reads == 1, "materialised exactly once"
    assert snapshot == frozenset({"c1"})
    assert all(isinstance(entry, str) for entry in snapshot)


def test_a_lying_contains_cannot_authorize() -> None:
    """Membership runs against the built frozenset, not the caller's object."""
    from collections.abc import Collection

    class Liar(Collection):  # type: ignore[type-arg]
        def __iter__(self) -> Any:
            return iter(["c1"])

        def __len__(self) -> int:
            return 1

        def __contains__(self, item: object) -> bool:
            return True

    out = _run(proposals=[_proposal(claim_ref="c9")], known_claim_ids=Liar())
    assert out.results == ()
    assert out.dropped == ("proposal_0_unknown_claim_ref",)


# --- per-proposal isolation -------------------------------------------------


def test_regression_a_hostile_claim_ref_drops_only_its_own_proposal() -> None:
    """REGRESSION: ``Correlation`` was built OUTSIDE the per-proposal guard.

    A ``claim_ref`` satisfying frozenset membership but not a ``str`` raised
    uncaught and destroyed every sibling — breaking the isolation invariant this
    module documents.
    """

    class Sneaky:
        def __hash__(self) -> int:
            return hash("c1")

        def __eq__(self, other: object) -> bool:
            return True

    hostile = _forged(claim_ref=Sneaky())
    out = _run(
        proposals=[hostile, *[_proposal(str(i + 1)) for i in range(5)]],
        known_claim_ids=frozenset({"c1"}),
    )
    assert len(out.results) == 5, "five valid proposals survive one hostile sibling"
    assert out.dropped == ("proposal_0_invalid_request",)


def test_five_valid_plus_one_hostile_returns_five_results() -> None:
    """The instruction's headline case, stated as its own test."""
    proposals: list[Any] = [_proposal(str(i + 1)) for i in range(5)]
    proposals.insert(2, _forged(purpose="forged\nAUDIT"))
    out = _run(proposals=proposals)
    assert len(out.results) == 5
    assert len(out.requests) == 5
    assert out.dropped == ("proposal_2_invalid_request",)
    assert all(r.arithmetic_status == "ARITHMETIC_VERIFIED" for r in out.results)
