"""CFO specialist — the first agent wired to the structured result contract.

CFO is the pilot for replacing the plain-string specialist boundary
(``BaseAgent.analyze -> str``) with :class:`SpecialistResult`. The migration is
deliberately one agent wide: ``BaseAgent.analyze`` is untouched, the other nine
specialists keep using it, and ``route_to_specialist`` still returns ``str``, so
none of the ~100 call sites across workflows, the MCP tool, or the Executive
change.

Phase 3B2 adds a second public method, :meth:`FinanceAgent.analyze_structured`,
used only by ``route_parallel`` (via ``route_to_specialist_structured``). It
returns an application-owned envelope carrying the model's result AND, behind a
default-off setting, the deterministic calculation gateway's records. Both
public methods share one private core, :meth:`_analyze_result`, which is the
single place the provider is called and the parser is run — and which never
touches the gateway. ``analyze`` therefore cannot reach the gateway by any
path, because the only call to it lives in a method ``analyze`` never enters.

The override is additive in the strict sense: when the model does not emit the
tool — the expected case on backends that silently drop ``tool_choice``, which
includes Ollama — CFO returns the byte-identical string it returns today. The
structured path either yields claims or costs nothing.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from openexecutive.agents.base import BaseAgent
from openexecutive.audit.context import get_active_ids
from openexecutive.config import get_settings

if TYPE_CHECKING:
    from openexecutive.knowledge.retriever import RetrievalSet
from openexecutive.specialists.calculation_gateway import (
    MAX_PROPOSALS_PER_CALL,
    SpecialistCalculations,
    execute_proposals,
    is_usable_correlation_id,
)
from openexecutive.specialists.result_contract import (
    SpecialistResult,
    emit_specialist_result_tool,
    parse_specialist_result,
    render_for_executive,
)
from openexecutive.specialists.routed_output import (
    CALC_CLAIM_SET_UNSAFE,
    CALC_CONTEXT_UNAVAILABLE,
    CALC_DISABLED,
    CALC_GATEWAY_RAISED,
    CALC_NO_PROPOSALS,
    CALC_SKIPPED_STRUCTURE_LOST,
    CALC_TOO_MANY_PROPOSALS,
    CorrelationFrame,
    RoutedSpecialistOutput,
)

logger = logging.getLogger(__name__)

# BaseAgent.analyze gives a deep-reasoning agent 16000 output tokens and every
# other agent 4096. CFO sets use_deep_reasoning=True, so it gets 16000 today —
# and it must keep getting 16000 even where `thinking` itself is unrepresentable.
# analyze_with_tools only raises the budget inside its `if use_deep:` branch, so
# turning deep reasoning off to silence the gate would ALSO cut CFO's answer
# budget by 4x on every local/OpenRouter call. The budget and the thinking flag
# are separate concerns; this keeps them separate.
_CFO_DEEP_MAX_TOKENS = 16000
_CFO_MAX_TOKENS = 4096


class FinanceAgent(BaseAgent):
    name = "cfo"
    domain = "finance"
    use_deep_reasoning = True

    # Capability flag read by ``orchestrator.router``. Declaring it opts this
    # agent into the structured retrieval path — it receives context tagged with
    # per-invocation provenance tokens and a set to validate them against.
    # Agents that do not declare it keep the legacy string path unchanged.
    accepts_retrieval_set = True

    # Capability flag read by ``orchestrator.router``: ``route_parallel``
    # dispatches this agent through ``analyze_structured`` and receives the
    # envelope. ``route_to_specialist`` — every workflow, the MCP tool, the
    # Council sandbox — still calls ``analyze`` and still gets a string.
    emits_structured_result = True

    @property
    def model(self) -> str:  # type: ignore[override]
        return get_settings().deep_reasoning_model

    def get_system_prompt(self) -> str:
        from openexecutive.prompts.domain_prompts import CFO_PROMPT

        return CFO_PROMPT

    # ------------------------------------------------------------------
    # structured path — legacy string boundary
    # ------------------------------------------------------------------

    async def analyze(  # type: ignore[override]
        self,
        query: str,
        context: str = "",
        retrieved_knowledge: str = "",
        episodic_context: str = "",
        failure_cases: str = "",
        department_memory: str = "",
        *,
        system_prompt_override: str | None = None,
        model_override: str | None = None,
        deep_reasoning_override: bool | None = None,
        retrieval_set: RetrievalSet | None = None,
    ) -> str:
        """Run CFO's analysis structurally, returning the legacy string.

        Return type matches ``BaseAgent.analyze`` exactly, because
        ``route_to_specialist`` (``orchestrator/router.py``) and the Council
        sandbox (``api/routes/agents.py``) call it positionally and by keyword
        and both expect a ``str``. ``retrieval_set`` is an additive keyword-only
        parameter the base class does not have; the router only passes it to
        agents that declare ``accepts_retrieval_set``, so no other caller is
        affected and CFO still works when it is omitted.

        ``retrieval_set`` is the provenance authority for this one call. It is
        read into a local, used to validate the model's evidence references, and
        then dropped. It is deliberately NOT stored on ``self``: this class is
        instantiated once in ``SPECIALIST_REGISTRY`` and shared across every
        concurrent turn, so an instance attribute would let one turn's CFO call
        validate against another turn's retrieval — a cross-request provenance
        forgery that needs no hostile model to trigger, just two users at once.

        This method never advertises ``calculation_requests`` and never calls
        the calculation gateway: it hands ``_analyze_result`` a hard ``False``,
        so the parser treats the key as unknown exactly as it does today, and
        the only gateway call in this class lives in ``analyze_structured``.
        """
        if system_prompt_override is not None:
            # analyze_with_tools takes an *addendum* appended to this agent's own
            # prompt; there is no way to replace the prompt through it. Appending
            # would make the Council sandbox preview
            # (api/routes/agents.py -> POST /agents/{id}/test) run the stock CFO
            # prompt IN FRONT OF the operator's edit — so an operator loosening a
            # guardrail would see the original constraints still holding, judge
            # the edit safe, and save a prompt that then runs without them.
            # Failing in the unsafe direction on a preview tool is not acceptable,
            # so a prompt override takes the legacy path, which runs exactly the
            # prompt the operator typed. Previewing a prompt is what that endpoint
            # is for; exercising the structured wiring is not.
            return await super().analyze(
                query,
                context,
                retrieved_knowledge,
                episodic_context,
                failure_cases,
                department_memory,
                system_prompt_override=system_prompt_override,
                model_override=model_override,
                deep_reasoning_override=deep_reasoning_override,
            )

        result, message = await self._analyze_result(
            query,
            context,
            retrieved_knowledge,
            episodic_context,
            failure_cases,
            department_memory,
            model_override=model_override,
            deep_reasoning_override=deep_reasoning_override,
            retrieval_set=retrieval_set,
            calculation_requests=False,
        )
        return self._narrative_for(result, message)

    # ------------------------------------------------------------------
    # structured path — envelope boundary (route_parallel only)
    # ------------------------------------------------------------------

    async def analyze_structured(
        self,
        query: str,
        context: str = "",
        retrieved_knowledge: str = "",
        episodic_context: str = "",
        failure_cases: str = "",
        department_memory: str = "",
        *,
        model_override: str | None = None,
        deep_reasoning_override: bool | None = None,
        retrieval_set: RetrievalSet | None = None,
    ) -> RoutedSpecialistOutput:
        """Run CFO's analysis and return the application-owned envelope.

        Called only by ``route_to_specialist_structured``. There is deliberately
        no ``system_prompt_override`` parameter: the router has none to pass,
        and the Council sandbox — the one caller that overrides prompts — uses
        ``analyze``.

        ``narrative`` is produced by the same rule ``analyze`` uses, so the
        Executive and the Committee receive exactly the bytes they receive from
        the legacy path. The calculation gateway runs only when every gate in
        :meth:`_gated_calculations` passes, and its records are returned in the
        envelope — the asyncio task result — never via shared state.
        """
        enabled = bool(get_settings().calc_cfo_structured_enabled)
        result, message = await self._analyze_result(
            query,
            context,
            retrieved_knowledge,
            episodic_context,
            failure_cases,
            department_memory,
            model_override=model_override,
            deep_reasoning_override=deep_reasoning_override,
            retrieval_set=retrieval_set,
            calculation_requests=enabled,
        )
        narrative = self._narrative_for(result, message)
        calculations, frame, diagnostics = self._gated_calculations(result, enabled)
        return RoutedSpecialistOutput(
            specialist=self.name,
            narrative=narrative,
            specialist_result=result,
            calculations=calculations,
            frame=frame,
            diagnostics=diagnostics,
        )

    @property
    def last_result(self) -> SpecialistResult | None:
        """The structured result from the most recent :meth:`analyze` call.

        Read-only, and a debugging/test affordance only — never a data channel.
        Two properties a caller must know: ``SPECIALIST_REGISTRY`` holds one
        shared ``FinanceAgent``, so concurrent turns overwrite this and whichever
        finishes last wins (the returned strings are unaffected — all real state
        is call-local); and the last result, which may carry company-confidential
        figures, is retained on that process-global object until the next CFO
        call replaces it. Nothing in production reads it, and no route serializes
        agent state, so it is not exposed today.
        """
        return getattr(self, "_last_result", None)

    # ------------------------------------------------------------------
    # shared core
    # ------------------------------------------------------------------

    async def _analyze_result(
        self,
        query: str,
        context: str,
        retrieved_knowledge: str,
        episodic_context: str,
        failure_cases: str,
        department_memory: str,
        *,
        model_override: str | None,
        deep_reasoning_override: bool | None,
        retrieval_set: RetrievalSet | None,
        calculation_requests: bool,
    ) -> tuple[SpecialistResult, Any]:
        """The ONE provider call, then the parser. Never the gateway.

        ``calculation_requests`` is threaded to both the tool factory and the
        parser so the schema the model saw and the keys the parser accepts can
        never disagree. Returns the raw message alongside the result because
        the legacy narrative rule needs the message's first text block.
        """
        message = await self.analyze_with_tools(
            self._build_user_content(
                query,
                context=context,
                retrieved_knowledge=retrieved_knowledge,
                episodic_context=episodic_context,
                failure_cases=failure_cases,
                department_memory=department_memory,
            ),
            tools=[
                emit_specialist_result_tool(
                    include_calculation_requests=calculation_requests
                )
            ],
            max_tokens=self._max_tokens(),
            model_override=model_override,
            deep_reasoning_override=self._effective_deep_reasoning(
                model_override, deep_reasoning_override
            ),
        )

        # ``is not None``, matching analyze_with_tools' own resolution rule: with
        # ``or``, a caller passing model_override="" would label the result with
        # a different model than the request actually used.
        resolved_model = (
            model_override if model_override is not None else self.effective_model()
        )
        # Local, per-invocation. `None` when no set was supplied (workflows, the
        # MCP tool, the CLI, every caller that predates this path), which makes
        # the parser strip every retrieval_id — no set, no provenance.
        allowed_ids = retrieval_set.allowed_ids() if retrieval_set is not None else None
        result = parse_specialist_result(
            message,
            specialist=self.name,
            model=resolved_model,
            allowed_retrieval_ids=allowed_ids,
            accept_calculation_requests=calculation_requests,
        )
        self._last_result = result
        return result, message

    def _narrative_for(self, result: SpecialistResult, message: Any) -> str:
        """The legacy string for this call. One rule, shared by both paths."""
        if result.degraded:
            logger.info(
                "cfo: structured output degraded (%s); returning legacy text",
                result.degraded_reason,
            )
            # Legacy semantics, deliberately: BaseAgent.analyze returns the
            # FIRST text block, while the contract's parser joins them all.
            # Preserving the old rule keeps this a migration slice rather than a
            # silent behaviour change for every existing CFO caller; the join is
            # a separate decision for a separate slice.
            legacy = _first_text_block(message)
            if legacy:
                return legacy

            # No text block to preserve. A live Qwen run showed why this branch
            # has to exist: the model reliably calls the tool and puts its ENTIRE
            # answer in the payload, leaving no prose behind. When such a payload
            # then fails validation, returning the legacy "" would hand the
            # Executive an empty answer where today it would have received prose
            # — a real regression introduced by wiring, not by the model.
            #
            # Falling back to the narrative is safe precisely here: with no text
            # blocks there is no "first block" to preserve, so this cannot be the
            # join that legacy semantics forbid. It can only ever replace an
            # empty string with the model's own words.
            return render_for_executive(result)

        return render_for_executive(result)

    def _gated_calculations(
        self, result: SpecialistResult, enabled: bool
    ) -> tuple[SpecialistCalculations | None, CorrelationFrame | None, tuple[str, ...]]:
        """Run the gateway iff every gate passes; otherwise say which one did not.

        Gates, in order, each failing CLOSED with a literal diagnostic:

        * the setting is off;
        * the parser reached its lost-structure exit — no claim set exists to
          authorize against, and no proposal survives that exit anyway;
        * the model proposed nothing;
        * the model proposed more than the engine's batch limit. A declared
          condition, reported under its own code: the gateway would raise on
          it, and a model-controlled count must not read as an engine crash;
        * the audit context holds no usable session/turn identity. There is no
          constant fallback: a request id minted under ``"unbound_run"`` would be
          an audit record that looks correlated and is not;
        * a final ``claim_id`` is not a usable identifier. The whole set is
          refused rather than filtered — partial authorization is not a thing
          this method offers. (``Claim.claim_id`` is only ``min_length=1``; the
          gateway would raise on such a member, and that must never become a
          way for model output to fail the turn.)

        ``known_claim_ids`` is derived from ``result.claims`` — the FINAL,
        constructor-validated set — never from the raw payload, so a
        ``claim_ref`` naming a claim the parser dropped cannot be authorized.
        The gateway call is wrapped so that a raise there is contained as a
        diagnostic rather than propagating into ``route_parallel``, which today
        has no handler and would fail the whole turn.
        """
        if not enabled:
            return None, None, (CALC_DISABLED,)
        if result.integrity == "lost":
            return None, None, (CALC_SKIPPED_STRUCTURE_LOST,)
        if not result.calculation_requests:
            return None, None, (CALC_NO_PROPOSALS,)
        if len(result.calculation_requests) > MAX_PROPOSALS_PER_CALL:
            return None, None, (CALC_TOO_MANY_PROPOSALS,)
        case_id, run_id = get_active_ids()
        if not (is_usable_correlation_id(case_id) and is_usable_correlation_id(run_id)):
            return None, None, (CALC_CONTEXT_UNAVAILABLE,)
        assert case_id is not None and run_id is not None  # narrowed above
        known = frozenset(claim.claim_id for claim in result.claims)
        if not all(is_usable_correlation_id(claim_id) for claim_id in known):
            return None, None, (CALC_CLAIM_SET_UNSAFE,)
        frame = CorrelationFrame(
            case_id=case_id,
            run_id=run_id,
            # Passed to the engine AS GIVEN; a clock the engine refuses comes
            # back as its own typed INVALID_INPUT record, never a substitute.
            computed_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        try:
            calculations = execute_proposals(
                specialist=self.name,
                proposals=result.calculation_requests,
                case_id=frame.case_id,
                run_id=frame.run_id,
                computed_at=frame.computed_at,
                known_claim_ids=known,
            )
        except Exception:  # noqa: BLE001 - never a new turn-failure path
            # A literal only. No exception text and no model text reach the
            # log from here, matching the gateway's own logging rule.
            logger.warning("cfo: calculation gateway raised")
            return None, frame, (CALC_GATEWAY_RAISED,)
        return calculations, frame, ()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _effective_deep_reasoning(
        self, model_override: str | None, deep_reasoning_override: bool | None
    ) -> bool | None:
        """Deep reasoning, off when the resolved model cannot represent it.

        CFO sets ``use_deep_reasoning = True``, so ``analyze_with_tools`` would
        send ``thinking`` / ``output_config``. Those have no representation in
        the OpenAI ``/chat/completions`` body, so on a local or OpenRouter model
        the provider's feature gate strips them and WARN-logs the removal on
        every single call. That is pure noise, and it obscures real capability
        warnings.

        An explicit caller override always wins. Otherwise this consults the
        provider's own capability spec rather than hardcoding a provider name,
        so no provider-specific knowledge leaks into this agent — if a backend
        later gains thinking support, this follows automatically.
        """
        if deep_reasoning_override is not None:
            return deep_reasoning_override
        model = (
            model_override if model_override is not None else self.effective_model()
        )
        return None if _model_supports_thinking(model) else False

    def _max_tokens(self) -> int:
        """CFO's output budget, matching ``BaseAgent.analyze`` exactly.

        Keyed on the agent's own deep-reasoning setting, NOT on whether the
        backend can represent ``thinking``. Those coincide in ``BaseAgent``
        because one ``if use_deep:`` branch sets both; here they must not, or
        turning thinking off to quiet the provider gate would also quarter the
        answer budget on every local call.
        """
        if self.effective_use_deep_reasoning():
            return _CFO_DEEP_MAX_TOKENS
        return _CFO_MAX_TOKENS

    def _build_user_content(
        self,
        query: str,
        *,
        context: str,
        retrieved_knowledge: str,
        episodic_context: str,
        failure_cases: str,
        department_memory: str,
    ) -> str:
        """Wrap the query in the same context blocks ``BaseAgent.analyze`` uses.

        Ordering is copied from ``BaseAgent.analyze`` verbatim and matters: the
        blocks are layered outward from the query so ``past_decisions`` stays
        closest to it for cache stability across turns.
        """
        user_content = query
        if context:
            user_content = (
                f"<conversation_context>\n{context}\n</conversation_context>\n\n"
                f"{user_content}"
            )
        if retrieved_knowledge:
            user_content = (
                f"<relevant_knowledge>\n{retrieved_knowledge}\n</relevant_knowledge>"
                f"\n\n{user_content}"
            )
        if failure_cases:
            user_content = (
                f"<failure_cases>\n{failure_cases}\n</failure_cases>\n\n{user_content}"
            )
        if episodic_context:
            user_content = (
                f"<past_decisions>\n{episodic_context}\n</past_decisions>\n\n"
                f"{user_content}"
            )
        if department_memory:
            user_content = (
                f"<department_memory>\n{department_memory}\n</department_memory>\n\n"
                f"{user_content}"
            )
        return user_content


def _first_text_block(message: Any) -> str:
    """The legacy return value: the FIRST text block, or ``""``.

    Mirrors ``BaseAgent.analyze``'s final two lines, including the empty-string
    fallback, so a degraded CFO call returns what every existing caller receives
    today. One deliberate divergence: a text block whose ``.text`` is not a
    ``str`` is skipped rather than returned, because this function is annotated
    ``-> str`` and the base would hand back the non-string verbatim. Not
    reachable through the real translator, and safer where it is.
    """
    try:
        blocks = getattr(message, "content", None) or []
        for block in blocks:
            if getattr(block, "type", "") == "text":
                text = getattr(block, "text", "")
                if isinstance(text, str):
                    return text
    except Exception:  # noqa: BLE001 - a hostile provider object must not raise
        logger.warning("cfo: could not read text blocks from provider response")
    return ""


def _model_supports_thinking(model: str) -> bool:
    """Whether the backend serving ``model`` can represent Anthropic thinking.

    Asks the provider for its own ``FeatureSpec`` rather than testing its class,
    so this agent holds no provider names, model lists, or backend URLs — a
    backend that later gains thinking support is picked up with no change here.
    A provider that exposes no spec resolver is Anthropic-direct, which never
    runs the feature gate and does support thinking.

    Defaults to True on any failure: that is the pre-existing behaviour — send
    thinking and let the gate decide — so a registry error degrades to today's
    log noise rather than silently dropping deep reasoning on Anthropic.
    """
    try:
        from openexecutive.providers import get_provider

        resolve = getattr(get_provider(model), "_resolve", None)
        if resolve is None:
            return True
        _slug, spec = resolve(model)
        return bool(spec.supports_thinking)
    except Exception:  # noqa: BLE001 - never block a specialist call on this
        logger.debug("cfo: could not resolve thinking support for %r", model)
        return True
