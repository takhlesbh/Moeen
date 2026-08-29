#!/usr/bin/env python3
"""Eval runner for Open Executive.

Usage:
    python run_evals.py --scenarios scenarios/ --output results/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core"))

# Judge model, imported (not redeclared) from the judges package so the model
# a provider is resolved FOR always matches the model requests are labelled
# WITH. Routed through ``openexecutive.providers.get_provider`` like every
# production call site, so an operator running the suite against OpenRouter or
# a local backend does not have the judge silently pinned to a direct
# Anthropic connection.
sys.path.insert(0, str(Path(__file__).parent))
from evidence import (  # noqa: E402
    OUTCOME_ERROR,
    OUTCOME_FAIL,
    OUTCOME_PASS,
    RunEvidence,
    harvest_audit,
    new_run_id,
)
from judges.base import JUDGE_MODEL, JudgeError, invoke_judge  # noqa: E402
from preflight import (  # noqa: E402
    RETRIEVAL_COLLECTIONS,
    PreflightError,
    assert_runtime_store_identity,
    load_inventory,
    observe_collections,
)
from preflight import required_collections_for as _required_collections  # noqa: E402
from preflight import resolve_store_path, validate_knowledge  # noqa: E402

PASS_THRESHOLD = 3.5
# Fraction of executed scenarios that must pass before the command reports
# success. Distinct from the infrastructure gates: this one is a judgement
# about the product, not the harness.
MIN_PASS_RATE = 0.8


async def run_eval(scenario: dict, executive) -> dict:
    from openexecutive.memory.company_profile import CompanyProfile
    from openexecutive.orchestrator.session import Session

    ctx = scenario.get("company_context", {})
    profile = CompanyProfile(
        name=ctx.get("name", "Eval Company"),
        industry=ctx.get("industry", ""),
        stage=ctx.get("stage", ""),
        headcount=ctx.get("headcount"),
        annual_revenue_arr=ctx.get("arr"),
    )

    if ctx.get("monthly_burn"):
        profile.financials.burn_rate_monthly = ctx["monthly_burn"]
    if ctx.get("runway_months"):
        profile.financials.runway_months = ctx["runway_months"]

    session = Session(company_profile=profile)

    response = await executive.chat(
        user_message=scenario["query"],
        session=session,
    )

    return {
        "id": scenario["id"],
        "domain": scenario["domain"],
        "query": scenario["query"],
        "response": response,
        "response_length": len(response),
        # Correlates this scenario to the audit rows the Executive wrote for
        # it — the only handle by which specialist consults and per-iteration
        # token usage can be recovered afterwards.
        "session_id": session.session_id,
    }


async def judge_response(scenario: dict, response: str, provider) -> dict:
    judge_prompt = f"""You are an evaluator for an AI executive advisory system.

Evaluate this response to an executive question on a 1-5 scale for each dimension.

QUESTION: {scenario['query']}

RESPONSE: {response}

Expected topics to cover: {', '.join(scenario.get('expected_topics', []))}

Rate each dimension (1=poor, 3=acceptable, 5=excellent):
1. persona_coherence: Does it sound like a senior executive, not a generic AI?
2. domain_accuracy: Is the advice factually correct and professionally sound?
3. actionability: Does it give concrete next steps with clear recommendations?
4. topic_coverage: Does it address the expected topics?
5. specificity: Is it specific to the situation, not generic advice?

Respond in JSON format:
{{"persona_coherence": N, "domain_accuracy": N, "actionability": N, "topic_coverage": N, "specificity": N, "overall": N, "notes": "brief explanation"}}"""

    return await invoke_judge(provider, judge_prompt)


async def run_workflow_eval(scenario: dict, store) -> dict:
    """Runs a workflow scenario through WORKFLOW_REGISTRY and returns the
    rendered artifact. The runner is intentionally minimal — it does not
    inject the scenario's ``company_context`` because workflows load their
    profile from ``COMPANY_PROFILE_PATH``; the scenario's ``workflow_inputs``
    already include the key metrics inline, so the artifact remains
    self-contained and judgeable.
    """
    from openexecutive.workflows import WORKFLOW_REGISTRY

    workflow_name = scenario.get("workflow")
    workflow = WORKFLOW_REGISTRY.get(workflow_name) if workflow_name else None
    if workflow is None:
        raise KeyError(f"unknown workflow: {workflow_name!r}")

    inputs = workflow.input_model()(**(scenario.get("workflow_inputs") or {}))

    artifact = ""
    async for event in workflow.run(inputs, store):
        if event.type == "artifact":
            artifact = event.content or ""
        elif event.type == "error":
            raise RuntimeError(f"workflow {workflow_name} errored: {event.message}")

    return {
        "id": scenario["id"],
        "domain": scenario["domain"],
        "workflow": workflow_name,
        "artifact": artifact,
        "artifact_length": len(artifact),
    }


async def judge_workflow(scenario: dict, artifact: str, provider) -> dict:
    expected_sections = scenario.get("expected_artifact_sections", []) or []
    quality_criteria = scenario.get("quality_criteria", {}) or {}

    judge_prompt = f"""You are evaluating an artifact produced by an AI executive workflow.

WORKFLOW: {scenario.get('workflow')}
DESCRIPTION: {scenario['description']}

ARTIFACT (Markdown):
{artifact}

Expected sections (should appear as headings or clear sections):
{', '.join(expected_sections) or 'n/a'}

Quality criteria the artifact should satisfy:
{json.dumps(quality_criteria, indent=2)}

Rate each dimension 1-5 (1=poor, 3=acceptable, 5=excellent):
1. structure: All expected sections present and well-organized.
2. specificity: Uses concrete facts from the inputs; no fabricated numbers.
3. actionability: Recommendations and decisions are clear and concrete.
4. coherence: Reads as a unified document, not stitched fragments.
5. completeness: Each section is substantive, not a stub.

Respond in JSON:
{{"structure": N, "specificity": N, "actionability": N, "coherence": N, "completeness": N, "overall": N, "notes": "brief"}}"""

    return await invoke_judge(provider, judge_prompt)


async def run_triage_eval(scenario: dict) -> dict:
    """Runs a triage scenario directly against the TriageAgent (no Executive loop).

    Takes no client/provider: ``TriageAgent.triage`` resolves its own backend
    via ``get_provider`` when none is injected, so the judge's provider must
    not be threaded in here (the agent and the judge can legitimately run on
    different models).

    Scenario YAML must include `event` + `context` (with `recent_alerts`,
    `muted_topics`, `active_initiatives`).
    """
    from openexecutive.agents.triage import TriageAgent
    from openexecutive.alerts.models import AlertEvent

    # Pydantic alias `from` → AlertEvent.from_ is handled automatically.
    event = AlertEvent(**scenario["event"])

    context = scenario.get("context", {}) or {}
    recent = context.get("recent_alerts", []) or []
    mutes = context.get("muted_topics", []) or []
    initiatives = context.get("active_initiatives", []) or []

    agent = TriageAgent()
    decision = await agent.triage(
        event,
        recent_alerts=recent,
        mute_patterns=mutes,
        active_initiatives=initiatives,
    )

    return {
        "id": scenario["id"],
        "domain": scenario["domain"],
        "decision": decision.model_dump(mode="json"),
    }



def _kind_from_args(args: argparse.Namespace) -> str:
    if args.triage:
        return "triage"
    if args.workflow:
        return "workflow"
    if args.mcp:
        return "mcp"
    return "chat"


def _judge_error_result(
    scenario: dict, partial: dict, exc: JudgeError, duration_ms: int
) -> dict:
    """A scenario whose verdict could not be obtained.

    Keeps the model's own output — it was produced successfully and is worth
    inspecting — but records NO score, because none exists.
    """
    return {
        **partial,
        "outcome": OUTCOME_ERROR,
        "error_kind": "judge_unparseable",
        "error": str(exc),
        "judge": None,
        "judge_raw": exc.raw,
        "duration_ms": duration_ms,
    }


async def _score(scenario: dict, partial: dict, verdict: dict, duration_ms: int) -> dict:
    raw = verdict.pop("_raw", "")
    overall = verdict.get("overall", 0.0)
    return {
        **partial,
        "outcome": OUTCOME_PASS if overall >= PASS_THRESHOLD else OUTCOME_FAIL,
        "judge": verdict,
        "judge_raw": raw,
        "judge_model": JUDGE_MODEL,
        "pass_threshold": PASS_THRESHOLD,
        "duration_ms": duration_ms,
    }


def _print_preflight(kind, source_dir, inventory, knowledge, census=None) -> None:
    """Echo what the run resolved to, so a wrong store or inventory is visible
    in the log before any money is spent."""
    print(f"run kind        : {kind}")
    print(f"scenario source : {source_dir}")
    print(
        f"discovered      : {inventory['discovered_count']} "
        f"(all kinds: {inventory['kind_totals']})"
    )
    print(f"knowledge store : {knowledge.persist_path}")
    for name, info in knowledge.collections.items():
        print(f"  {name}: {info['count']} chunks, dim={info.get('dimension')}")
    print(f"kb fingerprint  : {knowledge.fingerprint}")
    if knowledge.smoke_results:
        top = knowledge.smoke_results[0]
        print(f"retrieval smoke : OK — top={top['source']} d={top['distance']}")
    if census is not None:
        absent = [n for n, v in census.items() if not v["present"]]
        print(f"retrieval reads : {', '.join(census)}")
        if absent:
            print(f"  NOT in store  : {', '.join(absent)} (will be auto-created EMPTY)")



def _prepare_environment(args: argparse.Namespace) -> Path:
    """Pin every path to an absolute value BEFORE Settings is first read.

    This is the whole of defect 2: a relative ``VECTOR_STORE_PATH`` is
    re-resolved against ``Path.cwd()`` by ``Settings._resolve_paths``, so the
    Makefile's ``cd packages/core`` silently retargeted the knowledge store to
    a path that did not exist — which Chroma then created, empty.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if args.scenarios:
        os.environ["EVAL_SCENARIOS_PATH"] = str(
            Path(args.scenarios).expanduser().resolve()
        )
    store_path = resolve_store_path(args.vector_store, repo_root)
    os.environ["VECTOR_STORE_PATH"] = str(store_path)
    # packages/core/company/ — NOT <repo_root>/company/. Only the former is
    # gitignored, and CLAUDE.md designates it as where company data lives.
    # Pointing the harness at the repo root would put company-confidential
    # data one `git add -A` away from a commit.
    os.environ.setdefault(
        "COMPANY_PROFILE_PATH",
        str(repo_root / "packages" / "core" / "company" / "profile.yaml"),
    )
    return store_path


def _resolve_output_root(explicit: str | None) -> Path:
    """Where run evidence is written, refusing git-tracked destinations.

    Evidence embeds complete specialist prompts and responses — which carry the
    company profile and retrieved company_docs chunks — so a destination inside
    the repo but outside the gitignored ``evals/results/`` is one ``git add -A``
    away from committing confidential data. Both branches resolve fully first,
    so a symlinked ``evals/results`` cannot dodge the check.
    """
    repo_root = Path(__file__).resolve().parent.parent
    default_root = (repo_root / "evals" / "results").resolve()
    root = (
        Path(explicit).expanduser().resolve() if explicit else default_root
    )
    inside_repo = root == repo_root or repo_root in root.parents
    inside_default = root == default_root or default_root in root.parents
    if inside_repo and not inside_default:
        raise PreflightError(
            f"Refusing to write run evidence to {root}\n"
            f"It is inside the repository but outside the gitignored "
            f"{default_root}. Evidence contains full specialist prompts and "
            f"company context — writing it to a tracked path risks committing "
            f"it. Use the default, or an --output outside the repo."
        )
    return root


def _initialize_schemas() -> None:
    """Create the SQLite schemas the API server's lifespan normally creates.

    The chat path's RAG layer reads ``review_items``, and the workflow path
    additionally touches ``decisions`` / ``agent_overrides``. Without this every
    specialist consult dies with ``no such table``.
    """
    from openexecutive.agents.overrides import initialize_overrides_db
    from openexecutive.knowledge.review_store import ReviewStore
    from openexecutive.memory.episodic import initialize_db as initialize_episodic_db

    initialize_episodic_db()
    initialize_overrides_db()
    ReviewStore.initialize_db()



def _build_parser() -> argparse.ArgumentParser:
    """CLI surface. Kept separate so main() reads as a sequence of
    phases rather than fifty lines of flag declarations."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        help="Directory of scenario YAML files. Defaults to the packaged "
        "openexecutive/evals/_scenarios/ (exported as EVAL_SCENARIOS_PATH so "
        "the package loader — the single source of discovery — resolves it).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Root directory for run evidence. Defaults to <repo>/evals/results/, "
        "which is gitignored — evidence embeds full specialist prompts and "
        "responses, so a CWD-relative default risked committing them.",
    )
    parser.add_argument("--scenario-id", help="Run only this scenario ID")
    parser.add_argument(
        "--vector-store",
        help="Chroma persistence directory to evaluate against. Must already "
        "exist — the harness never creates one.",
    )
    parser.add_argument(
        "--triage",
        action="store_true",
        help="Run triage scenarios via TriageAgent + triage_judge (skip the Executive loop)",
    )
    parser.add_argument(
        "--workflow",
        action="store_true",
        help="Run type=workflow scenarios via WORKFLOW_REGISTRY (skip the Executive loop)",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help=(
            "Run requires_mcp scenarios. CI does not configure a gateway, "
            "so they are skipped unless this flag is set."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inventory + knowledge store and write the manifest, "
        "then exit without making any model call.",
    )
    return parser


async def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if sum([args.triage, args.workflow, args.mcp]) > 1:
        parser.error("--triage, --workflow, and --mcp are mutually exclusive")

    store_path = _prepare_environment(args)
    _initialize_schemas()
    kind = _kind_from_args(args)

    # ---- Preflight. Every failure below is fatal and non-zero. ----------
    try:
        # Validated first: a destination that would leak evidence into a
        # tracked path should fail before anything is discovered or opened.
        output_root = _resolve_output_root(args.output)
        scenarios, source_dir, inventory = load_inventory(kind, args.scenario_id)
        required = _required_collections(scenarios)
        knowledge = validate_knowledge(store_path, required)
        # Prove the store retrieval will actually open is the one just
        # validated, and census every collection retrieval can read so an
        # auto-created empty one is visible up front rather than silently
        # standing in for a validated source.
        runtime_path = assert_runtime_store_identity(store_path)
        retrieval_census = observe_collections(store_path, RETRIEVAL_COLLECTIONS)
    except PreflightError as exc:
        print(f"\nPREFLIGHT FAILED\n{exc}\n", file=sys.stderr)
        return 2

    _print_preflight(kind, source_dir, inventory, knowledge, retrieval_census)

    # ---- Evidence directory. Manifest exists BEFORE the first call. -----
    output_root.mkdir(parents=True, exist_ok=True)
    run = RunEvidence(output_root, new_run_id())
    run.open_manifest(
        kind=kind,
        inventory=inventory,
        knowledge=knowledge.to_dict()
        | {
            "runtime_path_verified": str(runtime_path),
            "retrieval_collections": retrieval_census,
        },
        config={
            "judge_model": JUDGE_MODEL,
            "pass_threshold": PASS_THRESHOLD,
            "scenario_source": str(source_dir),
            "vector_store_path": str(store_path),
            "scenario_id_filter": args.scenario_id,
            "argv": sys.argv[1:],
        },
    )
    print(f"run id          : {run.run_id}")
    print(f"evidence        : {run.run_dir}")

    if args.preflight_only:
        status = run.close()
        # The manifest is INCOMPLETE by construction — nothing was executed,
        # and it must never look like a finished baseline. The COMMAND still
        # exits 0 because the validation it was asked to perform succeeded;
        # that is what makes `make eval-preflight && make eval` usable.
        print("\npreflight-only: validation PASSED, no model calls made.")
        print(f"run manifest status={status} (nothing executed yet)")
        return 0

    from openexecutive.providers import get_provider

    judge_provider = get_provider(JUDGE_MODEL)
    await _execute(kind, scenarios, judge_provider, run)

    status = run.close()
    counts = run.manifest["counts"]
    print(
        f"\n{status}: executed {counts['executed']}/{counts['discovered']} | "
        f"pass {counts['passed']} | fail {counts['failed']} | error {counts['errored']}"
    )
    print(f"evidence: {run.run_dir}")

    if status != "COMPLETE":
        return 1
    # A COMPLETE run still fails the command when the product underperforms —
    # distinct from the infrastructure failures above.
    if counts["passed"] < counts["executed"] * MIN_PASS_RATE:
        print(f"WARNING: pass rate below {MIN_PASS_RATE:.0%}")
        return 1
    return 0


async def _run_and_judge(kind, scenario, judge_provider, executive, store, out: dict):
    """Run one scenario on its kind's path and obtain a judged verdict.

    ``out`` is filled with the run result AS SOON as the model answers, before
    the judge is called. That ordering matters: when the judge then fails, the
    caller still holds the response, session_id and audit trail — exactly the
    evidence needed to diagnose the judge outage. Returning a tuple instead
    would leave the caller with an empty stub, since the unpack never happens
    on the raising path.
    """
    if kind == "triage":
        from judges.triage_judge import judge_triage

        out.update(await run_triage_eval(scenario))
        return await judge_triage(scenario, out["decision"], judge_provider)

    if kind == "workflow":
        out.update(await run_workflow_eval(scenario, store))
        return await judge_workflow(scenario, out["artifact"], judge_provider)

    out.update(await run_eval(scenario, executive))
    # Read back what the Executive already recorded for this session. Purely
    # observational — see evidence.harvest_audit for what it can and cannot show.
    out["audit"] = harvest_audit(out["session_id"])
    return await judge_response(scenario, out["response"], judge_provider)



async def _execute(kind: str, scenarios: list, judge_provider, run: RunEvidence) -> None:
    """Run every discovered scenario, recording evidence for each.

    A scenario error never aborts the run — the remaining scenarios still
    produce evidence — but it does mark the run FAILED at close(), so a
    partial run can never be read as a baseline.
    """
    import time
    import traceback

    store = None
    executive = None
    if kind == "workflow":
        from openexecutive.config import get_settings
        from openexecutive.knowledge.store import ChromaDBStore

        store = ChromaDBStore(persist_directory=get_settings().vector_store_path)
    elif kind in ("chat", "mcp"):
        from openexecutive.orchestrator.executive import Executive

        executive = Executive()

    for scenario in scenarios:
        sid = scenario["id"]
        print(f"  [{sid}] {scenario.get('description', '')}")
        t0 = time.monotonic()
        partial: dict = {"id": sid, "domain": scenario.get("domain")}
        try:
            verdict = await _run_and_judge(
                kind, scenario, judge_provider, executive, store, partial
            )
            ms = int((time.monotonic() - t0) * 1000)
            result = await _score(scenario, partial, verdict, ms)
            print(f"    -> {result['outcome']} (overall {result['judge']['overall']:.1f}/5)")

        except JudgeError as exc:
            ms = int((time.monotonic() - t0) * 1000)
            result = _judge_error_result(scenario, partial, exc, ms)
            run.add_infrastructure_error(sid, "judge_unparseable", str(exc))
            print(f"    -> ERROR (judge): {exc}")
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            result = {
                **partial,
                "outcome": OUTCOME_ERROR,
                "error_kind": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "judge": None,
                "duration_ms": ms,
            }
            run.add_infrastructure_error(sid, type(exc).__name__, str(exc))
            print(f"    -> ERROR: {exc}")

        try:
            run.record(result)
        except Exception as exc:
            # One unwritable scenario must not deny the whole baseline. The
            # run still closes FAILED because this is an infrastructure error.
            run.add_infrastructure_error(sid, "evidence_write_failed", str(exc))
            print(f"    -> ERROR (evidence write): {exc}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
