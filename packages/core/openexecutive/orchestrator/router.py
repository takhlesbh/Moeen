from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from openexecutive.agents.base import BaseAgent

if TYPE_CHECKING:
    from openexecutive.knowledge.retriever import RetrievalSet
    from openexecutive.orchestrator.debug_events import DebugCollector
from openexecutive.agents.board_comms import BoardCommsAgent
from openexecutive.agents.finance import FinanceAgent
from openexecutive.agents.hr_talent import HRAgent
from openexecutive.agents.legal import LegalAgent
from openexecutive.agents.marketing import MarketingAgent
from openexecutive.agents.operations import OperationsAgent
from openexecutive.agents.product import ProductAgent
from openexecutive.agents.strategy import StrategyAgent
from openexecutive.agents.talent import TalentAgent
from openexecutive.agents.triage import TriageAgent
from openexecutive.specialists.routed_output import RoutedSpecialistOutput

SPECIALIST_REGISTRY: dict[str, BaseAgent] = {
    "cso": StrategyAgent(),
    "cfo": FinanceAgent(),
    "chro": HRAgent(),
    "gc": LegalAgent(),
    "coo": OperationsAgent(),
    "cmo": MarketingAgent(),
    "cpo": ProductAgent(),
    "board_comms": BoardCommsAgent(),
    "talent": TalentAgent(),
    "triage": TriageAgent(),
}

SPECIALIST_DESCRIPTIONS = {
    "cso": "Chief Strategy Officer — competitive analysis, M&A, market positioning, scenario planning, OKRs",
    "cfo": "Chief Financial Officer — financial modeling, unit economics, fundraising, cash flow, board finance",
    "chro": "Chief HR/People Officer — hiring, compensation, performance management, culture, org design",
    "gc": "General Counsel — contracts, IP, employment law basics, compliance (with appropriate disclaimers)",
    "coo": "Chief Operating Officer — process design, vendor management, operational scaling, metrics",
    "cmo": "Chief Marketing Officer — GTM strategy, brand, messaging, PR, crisis communications",
    "cpo": "Chief Product Officer — product roadmap, prioritization frameworks, product strategy",
    "board_comms": "Board Communications Director — board decks, investor relations, governance",
    "talent": "Head of Talent & Executive Search — candidate screening & fit scoring, executive sourcing, energy-sector talent-market mapping",
    "triage": "Chief of Staff — evaluates inbound events (email/Slack/docs) for significance and decides alerting",
}

SPECIALIST_TOOLS: list[dict[str, Any]] = [
    {
        "name": "consult_specialist",
        "description": (
            "Consult a specialist executive agent for domain-specific analysis. "
            "Use this to get deep expertise from the relevant functional leader. "
            "You may call this multiple times in parallel for cross-domain questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "specialist": {
                    "type": "string",
                    "enum": sorted(SPECIALIST_REGISTRY.keys()),
                    "description": f"Which specialist to consult. Options: {', '.join(f'{k} ({v})' for k, v in SPECIALIST_DESCRIPTIONS.items())}",
                },
                "query": {
                    "type": "string",
                    "description": "The specific question or task for the specialist. Be precise — they only see this query and the conversation context.",
                },
                "context": {
                    "type": "string",
                    "description": "Relevant context from the conversation that the specialist needs to give a good answer.",
                },
            },
            "required": ["specialist", "query"],
        },
    }
]


def _accepts_retrieval_set(agent: BaseAgent) -> bool:
    """Whether ``agent`` can validate evidence against a retrieval set.

    A class-attribute capability flag rather than an isinstance check, matching
    how the rest of the agent layer varies behaviour (``use_deep_reasoning``).
    That keeps this function free of any specific agent import, so the second
    specialist to opt in changes nothing here.
    """
    return bool(getattr(agent, "accepts_retrieval_set", False))


def _emits_structured_result(agent: BaseAgent) -> bool:
    """Whether ``agent`` exposes ``analyze_structured`` for ``route_parallel``.

    Same capability-flag idiom as :func:`_accepts_retrieval_set`, and for the
    same reason: no agent import here, so a second structured specialist is a
    one-line change on the agent and none on the router.
    """
    return bool(getattr(agent, "emits_structured_result", False))


async def route_to_specialist(
    specialist_name: str,
    query: str,
    context: str = "",
    retrieved_knowledge: str = "",
    episodic_context: str = "",
    failure_cases: str = "",
    department_memory: str = "",
    *,
    retrieval_set: RetrievalSet | None = None,
) -> str:
    """Dispatch one specialist call. Returns prose, as it always has.

    ``retrieval_set`` is keyword-only and defaults to ``None``, so the ~30
    workflow modules and the MCP tool that call this positionally are untouched.
    It is forwarded ONLY to agents that declare they can verify against it; for
    every other agent it is dropped here, because ``BaseAgent.analyze`` has no
    such parameter and gains none in this slice.

    The set is passed as an argument and never stored. ``SPECIALIST_REGISTRY``
    holds one shared instance per specialist, so parking it on the agent would
    let two concurrent turns validate against each other's retrieval — the exact
    cross-invocation confusion the token design exists to prevent.
    """
    agent = SPECIALIST_REGISTRY.get(specialist_name)
    if agent is None:
        return f"Unknown specialist: {specialist_name}"
    kwargs: dict[str, Any] = {}
    if retrieval_set is not None and _accepts_retrieval_set(agent):
        kwargs["retrieval_set"] = retrieval_set
    return await agent.analyze(
        query=query,
        context=context,
        retrieved_knowledge=retrieved_knowledge,
        episodic_context=episodic_context,
        failure_cases=failure_cases,
        department_memory=department_memory,
        **kwargs,
    )


async def route_to_specialist_structured(
    specialist_name: str,
    query: str,
    context: str = "",
    retrieved_knowledge: str = "",
    episodic_context: str = "",
    failure_cases: str = "",
    department_memory: str = "",
    *,
    retrieval_set: RetrievalSet | None = None,
) -> RoutedSpecialistOutput:
    """Dispatch one specialist call and return the application-owned envelope.

    Used by :func:`route_parallel` only. A specialist that declares
    ``emits_structured_result`` is dispatched through its ``analyze_structured``
    and its envelope is returned as-is. Every other specialist — and an unknown
    name — goes through :func:`route_to_specialist` itself, so the string in
    the envelope is produced by the legacy code path, byte for byte, rather
    than re-implemented here.

    Failure semantics are unchanged: nothing is caught. An exception from a
    specialist propagates exactly as it did when this function did not exist.
    """
    agent = SPECIALIST_REGISTRY.get(specialist_name)
    if agent is not None and _emits_structured_result(agent):
        kwargs: dict[str, Any] = {}
        if retrieval_set is not None and _accepts_retrieval_set(agent):
            kwargs["retrieval_set"] = retrieval_set
        return await agent.analyze_structured(  # type: ignore[attr-defined, no-any-return]
            query=query,
            context=context,
            retrieved_knowledge=retrieved_knowledge,
            episodic_context=episodic_context,
            failure_cases=failure_cases,
            department_memory=department_memory,
            **kwargs,
        )
    text = await route_to_specialist(
        specialist_name,
        query,
        context,
        retrieved_knowledge,
        episodic_context,
        failure_cases,
        department_memory,
        retrieval_set=retrieval_set,
    )
    return RoutedSpecialistOutput(specialist=specialist_name, narrative=text)


# Tool_result returned for consult_specialist calls past the per-turn fan-out
# cap, so the model sees an explicit acknowledgement and can re-ask next turn
# rather than the extra calls being silently dropped.
FANOUT_SKIP_MESSAGE = (
    "Skipped: this turn already dispatched the maximum number of parallel "
    "specialist consultations (cap={cap}). Ask again in a follow-up turn if "
    "this specialist's input is still needed."
)


def resolve_fanout_cap(max_parallel: int) -> int:
    """Effective per-turn specialist fan-out cap.

    ``max_parallel <= 0`` falls back to the specialist roster size, so the
    default (0) is inert — no real cross-domain turn consults more distinct
    specialists than exist. A positive value bounds pathological runaway. The
    floor of 1 keeps the cap from ever zeroing out dispatch (belt-and-suspenders
    against an empty roster).
    """
    return max_parallel if max_parallel > 0 else max(len(SPECIALIST_REGISTRY), 1)


def partition_specialist_fanout(
    tool_uses: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    max_parallel: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], int]:
    """Split a turn's specialist tool_uses/calls at the fan-out cap.

    ``tool_uses`` and ``calls`` must be 1:1 in the same order. Returns
    ``(run_tool_uses, run_calls, skipped_results, cap)``:

      - ``run_tool_uses`` / ``run_calls`` — the first ``cap`` to dispatch,
        aligned and equal length.
      - ``skipped_results`` — maps each over-cap tool_use id to the formatted
        skip message, so the caller can hand EVERY consult_specialist tool_use
        a tool_result (Anthropic requires one result per tool_use).
      - ``cap`` — the resolved cap, for instrumentation.
    """
    cap = resolve_fanout_cap(max_parallel)
    run_tool_uses = tool_uses[:cap]
    run_calls = calls[:cap]
    skipped_results = {
        tu["id"]: FANOUT_SKIP_MESSAGE.format(cap=cap) for tu in tool_uses[cap:]
    }
    return run_tool_uses, run_calls, skipped_results, cap


async def _retrieve_for_call(
    call: dict[str, str],
) -> tuple[str, RetrievalSet | None]:
    """Run a per-specialist, domain-filtered vector retrieval for one tool call.

    Specialists that can verify evidence take the structured path, which mints a
    fresh provenance token per retained chunk and tags it into the context the
    model reads. Everyone else takes the legacy path and gets the byte-identical
    string they get today — no token, and no set to validate against.
    """
    from openexecutive.knowledge.retriever import retrieve, retrieve_structured

    agent = SPECIALIST_REGISTRY.get(call["specialist"])
    if agent is not None and _accepts_retrieval_set(agent):
        return await asyncio.to_thread(
            retrieve_structured,
            query=call["query"],
            specialist_name=call["specialist"],
        )
    text = await asyncio.to_thread(
        retrieve, query=call["query"], specialist_name=call["specialist"]
    )
    return text, None


async def _retrieve_failures_for_call(call: dict[str, str]) -> str:
    """Domain-filtered failure case retrieval for one specialist call."""
    from openexecutive.knowledge.retriever import retrieve_failures

    return await asyncio.to_thread(
        retrieve_failures, query=call["query"], specialist_name=call["specialist"]
    )


async def _prefetch_department_for_call(
    call: dict[str, str], session_id: str | None
) -> str:
    """Department-memory prefetch for one specialist call.

    Resolves ``specialist → department_slug`` via the departments
    registry and queries the dept peer's Honcho representation. Returns
    "" when the specialist has no owning department (e.g. ``triage``)
    or when Honcho is disabled / fails — the wrapper's own degrade-on-
    failure semantics already audit the outcome.
    """
    from openexecutive.departments.registry import slug_for_specialist
    from openexecutive.memory.honcho_client import prefetch_department

    slug = slug_for_specialist(call["specialist"])
    if slug is None:
        return ""
    return await prefetch_department(
        query=call["query"],
        department_slug=slug,
        session_id=session_id,
    )


async def route_parallel(
    calls: list[dict[str, str]],
    retrieved_knowledge_map: dict[str, str] | None = None,
    episodic_context: str = "",
    session_id: str | None = None,
    debug_collector: DebugCollector | None = None,
) -> list[RoutedSpecialistOutput]:
    """Execute multiple specialist calls concurrently.

    Returns one :class:`RoutedSpecialistOutput` per call, in call order. The
    envelope is the asyncio **task result** of each gathered child — the one
    channel that reads correctly in the awaiting parent (a ContextVar set in a
    child is invisible here) and fails *empty* rather than *stale*. Its
    ``narrative`` is exactly the string this function returned before the
    envelope existed; the Executive reads that field and nothing else.

    Each specialist receives its own domain-filtered RAG context, fetched
    in parallel before the LLM calls fire. Callers may still supply a
    pre-built ``retrieved_knowledge_map`` (keyed by specialist name) to
    short-circuit the per-call retrieval — useful for tests or when the
    caller has already gathered shared context.

    ``episodic_context`` is per-turn (not per-specialist) and forwarded to
    every specialist in this batch.

    ``session_id`` (when provided) is threaded into the per-specialist
    department-memory prefetch for audit grouping. Specialists whose
    department has institutional Honcho memory receive a
    ``<department_memory>`` block synthesized from that dept peer's
    representation; specialists without an owning department (e.g.
    ``triage``) skip the prefetch entirely.

    Results are in the same order as ``calls`` so callers can zip with
    tool_use_ids. Exception handling is unchanged: ``gather`` without
    ``return_exceptions``, so one failing specialist still fails the batch.
    """
    retrieval_sets: list[RetrievalSet | None]
    if retrieved_knowledge_map is None:
        # Two nested gathers rather than one flat gather sliced at `mid`:
        # knowledge calls now return (text, set) pairs and failures return
        # plain strings, so a flat list would be heterogeneous and an
        # off-by-one in the slice would pair a specialist with another's
        # retrieval set. Both fan-outs still start together and run
        # concurrently — the outer gather is what preserves that.
        knowledge_pairs, failures_per_call = await asyncio.gather(
            asyncio.gather(*(_retrieve_for_call(c) for c in calls)),
            asyncio.gather(*(_retrieve_failures_for_call(c) for c in calls)),
        )
        knowledge_per_call = [text for text, _ in knowledge_pairs]
        retrieval_sets = [rset for _, rset in knowledge_pairs]
    else:
        knowledge_per_call = [
            retrieved_knowledge_map.get(c["specialist"], "") for c in calls
        ]
        failures_per_call = [""] * len(calls)
        # A caller-supplied map is plain text with no tokens in it. Handing a
        # set here would authorise references to chunks this call never saw.
        retrieval_sets = [None] * len(calls)

    # Fan out dept-memory prefetch alongside knowledge/failures. Each call
    # is cheap when Honcho is disabled or when the specialist has no
    # owning dept (returns "" immediately), so unconditionally gathering
    # keeps the per-call critical path uniform.
    dept_memory_per_call = list(
        await asyncio.gather(
            *(_prefetch_department_for_call(c, session_id) for c in calls)
        )
    )

    async def call_one(idx: int, call: dict[str, str]) -> RoutedSpecialistOutput:
        specialist = call["specialist"]
        if debug_collector:
            debug_collector.emit("specialist_start", {
                "specialist": specialist,
                "query": call["query"],
                "retrieved_chars": len(knowledge_per_call[idx]),
                "failures_chars": len(failures_per_call[idx]),
                "department_memory_chars": len(dept_memory_per_call[idx]),
            })
        t_start = time.monotonic()
        out = await route_to_specialist_structured(
            specialist_name=specialist,
            query=call["query"],
            context=call.get("context", ""),
            retrieved_knowledge=knowledge_per_call[idx],
            episodic_context=episodic_context,
            failure_cases=failures_per_call[idx],
            department_memory=dept_memory_per_call[idx],
            # Indexed by the same `idx` as the knowledge it was minted from, so
            # a specialist can only ever validate against its own retrieval.
            retrieval_set=retrieval_sets[idx],
        )
        if debug_collector:
            debug_collector.emit("specialist_done", {
                "specialist": specialist,
                "duration_ms": round((time.monotonic() - t_start) * 1000),
                "response_preview": out.narrative[:120],
                "response_length": len(out.narrative),
            })
        return out

    return list(await asyncio.gather(*(call_one(i, c) for i, c in enumerate(calls))))
