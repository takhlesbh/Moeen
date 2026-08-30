"""CFO specialist — the first agent wired to the structured result contract.

CFO is the pilot for replacing the plain-string specialist boundary
(``BaseAgent.analyze -> str``) with :class:`SpecialistResult`. The migration is
deliberately one agent wide: ``BaseAgent.analyze`` is untouched, the other nine
specialists keep using it, and ``route_to_specialist`` still returns ``str``, so
none of the ~90 call sites across workflows, the MCP tool, or the Executive
change.

The override is additive in the strict sense: when the model does not emit the
tool — the expected case on backends that silently drop ``tool_choice``, which
includes Ollama — CFO returns the byte-identical string it returns today. The
structured path either yields claims or costs nothing.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from openexecutive.agents.base import BaseAgent
from openexecutive.config import get_settings

if TYPE_CHECKING:
    from openexecutive.knowledge.retriever import RetrievalSet
from openexecutive.specialists.result_contract import (
    SpecialistResult,
    emit_specialist_result_tool,
    parse_specialist_result,
    render_for_executive,
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

    @property
    def model(self) -> str:  # type: ignore[override]
        return get_settings().deep_reasoning_model

    def get_system_prompt(self) -> str:
        from openexecutive.prompts.domain_prompts import CFO_PROMPT

        return CFO_PROMPT

    # ------------------------------------------------------------------
    # structured path
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

        The structured result is parsed and kept for the duration of the call —
        :attr:`last_result` — but only its narrative crosses the boundary. The
        Executive's ``results_by_id`` and ``tool_result`` content are unchanged;
        exposing claims is a later slice.
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

        message = await self.analyze_with_tools(
            self._build_user_content(
                query,
                context=context,
                retrieved_knowledge=retrieved_knowledge,
                episodic_context=episodic_context,
                failure_cases=failure_cases,
                department_memory=department_memory,
            ),
            tools=[emit_specialist_result_tool()],
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
        )
        self._last_result = result

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
