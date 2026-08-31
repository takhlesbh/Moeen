"""Shared helpers for workflows that run an Executive tool-use loop.

Both ``executive_reflection`` and ``executive_research`` drive the
Executive through a short tool-use loop with the full outbound toolkit.
The two helpers below are the bits both loops share:

- ``execute_tool_calls`` dispatches any tool_use blocks in a provider
  response against ``_ALL_SKILL_HANDLERS`` and returns one summary per
  call (tool, input_preview, result_preview, ok). Failures are caught +
  summarized so the loop survives a single tool crash.
- ``extract_artifact_from_response`` pulls the concatenated text-block
  content from a provider response — the model's narrative summary
  emitted alongside or after its tool calls.

Promoting these out of ``executive_reflection`` removes the private-
import tripwire ``executive_research`` would otherwise carry — a future
refactor renaming or moving them could silently break the other
workflow.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from openexecutive.clients.context_guard import (
    CompanyContext,
    StaleCompanyContextError,
)

logger = logging.getLogger(__name__)


# Tools that only READ. Everything else is treated as company-bound and runs
# under the company guard when an origin context is supplied.
#
# An allowlist of reads rather than a denylist of writes, deliberately: a
# handler added to `_ALL_SKILL_HANDLERS` later is guarded by DEFAULT. A denylist
# would leave every future handler silently unprotected, which is the failure
# mode this whole line of work exists to prevent. Verified by inspecting each
# handler's source for write/send operations; a test pins the fail-closed rule.
READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "ask_about_person",
    "get_candidate",
    "get_onboarding_plan",
    "list_candidates",
    "list_department_goals",
    "list_engagements",
    "list_offers",
    "list_onboarding_plans",
    "list_onboarding_templates",
    "list_people",
    "list_watchlist",
    "list_workflows",
    "load_skill",
    "lookup_person",
    "match_candidates",
    "propose_form_values",
    "search_skills",
})


@asynccontextmanager
async def _tool_company_guard(
    origin: CompanyContext | None, tool_name: str
) -> AsyncIterator[None]:
    """Serialise ONE company-bound handler invocation against slot switches.

    Passthrough when no origin was supplied (every caller outside the research
    path) or when the tool only reads — read-only lookups must not queue behind
    a slot switch.

    The critical section is exactly one handler call. It must never widen to the
    provider await, the surrounding iteration, or the loop: those run for
    minutes, and holding the company-state lock across them would stall the very
    rotation that switches the client.
    """
    if origin is None or tool_name in READ_ONLY_TOOLS:
        yield
        return
    from openexecutive.clients.context_guard import company_mutation_guard

    async with company_mutation_guard(
        origin, operation=f"tool call {tool_name}"
    ):
        yield


async def execute_tool_calls(
    response: Any,
    handlers: dict[str, Any],
    *,
    budget_remaining: int | None = None,
    free_tools: frozenset[str] | None = None,
    origin_company_context: CompanyContext | None = None,
) -> list[dict[str, Any]]:
    """Invoke any tool_use blocks in the LLM response.

    Returns one summary entry per call:
    ``{tool, input_preview, result_preview, ok}``. Failures are caught
    + summarized so a single tool crash does not abort the workflow's
    iteration loop.

    ``budget_remaining`` (optional) caps how many handlers actually
    run. Once budget hits zero, remaining tool_use blocks are NOT
    invoked — their summary entries are still appended with
    ``result_preview = 'over budget — skipped'`` so the model sees a
    tool_result acknowledgement on the next turn. Default ``None``
    leaves the loop unmetered (executive_reflection's behaviour).

    ``free_tools`` (optional) names tools that are exempt from the
    budget — read-only context lookups (e.g. ``lookup_person``) the
    model must call to PREPARE a routing decision. They always execute
    and never decrement ``budget_remaining``, so the Executive can't
    spend its outbound budget on lookups and then have no budget left
    to actually route (the "looks up people but never DMs" failure).
    """
    free_tools = free_tools or frozenset()
    summaries: list[dict[str, Any]] = []
    if not getattr(response, "content", None):
        return summaries
    for block in response.content:
        if getattr(block, "type", "") != "tool_use":
            continue
        name = getattr(block, "name", "")
        tool_input = getattr(block, "input", {}) or {}
        is_free = name in free_tools
        if budget_remaining is not None and budget_remaining <= 0 and not is_free:
            # Refuse without executing. Surfaces in the next user-turn
            # tool_result so the model knows further calls are blocked.
            # Free (read-only) tools are never refused on budget grounds.
            summaries.append({
                "tool": name,
                "input_preview": str(tool_input)[:120],
                "result_preview": "over budget — skipped",
                "ok": False,
            })
            continue
        handler = handlers.get(name)
        if handler is None:
            summaries.append({
                "tool": name,
                "input_preview": str(tool_input)[:120],
                "result_preview": "unknown tool — skipped",
                "ok": False,
            })
            continue
        try:
            async with _tool_company_guard(origin_company_context, name):
                result = await handler(tool_input)
        except StaleCompanyContextError:
            # The active company changed since this run started, so this call
            # would resolve the NEW client's channels / DB. Skip it: nothing is
            # sent, nothing is written, and it is NOT retried under the new
            # company. A summary entry is still appended — the loop hands every
            # tool_use block a result, and the model must see that the action
            # did not happen rather than assume it did. `ok=False` so it does
            # not count toward the routing budget as a success.
            logger.warning(
                "synthesis: skipped tool %s — the company this run started "
                "under is no longer active",
                name,
            )
            summaries.append({
                "tool": name,
                "input_preview": str(tool_input)[:120],
                "result_preview": "not executed — active company changed",
                "ok": False,
            })
            if budget_remaining is not None and not is_free:
                budget_remaining -= 1
            continue
        except Exception as exc:
            logger.exception("synthesis: tool %s raised", name)
            summaries.append({
                "tool": name,
                "input_preview": str(tool_input)[:120],
                "result_preview": f"raised: {exc}"[:160],
                "ok": False,
            })
            if budget_remaining is not None and not is_free:
                budget_remaining -= 1
            continue
        ok = True
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            if isinstance(parsed, dict) and "error" in parsed:
                ok = False
        except (ValueError, TypeError):
            # Non-JSON result — treat as success since we have no
            # signal otherwise (same as action_chips).
            pass
        summaries.append({
            "tool": name,
            "input_preview": str(tool_input)[:120],
            "result_preview": str(result)[:160],
            "ok": ok,
        })
        if budget_remaining is not None and not is_free:
            budget_remaining -= 1
    return summaries


def extract_artifact_from_response(response: Any) -> str:
    """Pull the final text summary from an Anthropic response.

    Returns the concatenated text of all text blocks. Empty string when
    the model only emitted tool_use blocks — callers fall back to a
    synthesized "no summary" artifact.
    """
    if not getattr(response, "content", None):
        return ""
    chunks: list[str] = []
    for block in response.content:
        if getattr(block, "type", "") == "text":
            text = getattr(block, "text", "")
            if text:
                chunks.append(text)
    return "\n\n".join(chunks).strip()


__all__ = ["execute_tool_calls", "extract_artifact_from_response"]
