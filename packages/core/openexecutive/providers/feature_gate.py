"""Spec-driven stripping of Anthropic-only request features.

When a non-Claude model is selected (or a Claude model is routed via a
backend that doesn't honor Anthropic-only features), we strip the
unsupported fields before the request leaves our process. The gate is
applied inside the OpenRouter provider so any future re-routing keeps
the same guarantees — a caller cannot accidentally ship a request with
``cache_control`` to a model that would 400 on it.
"""
from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


class UnsupportedFeatureError(RuntimeError):
    """A request reached a backend that cannot represent one of its features.

    Raised as an invariant guard, not as a routine control-flow path: by
    contract ``apply_feature_gates`` strips every unsupported feature before
    the request is translated, so reaching this means a caller bypassed the
    gate. Failing loudly is the point — the alternative is the request going
    out silently missing a capability the caller asked for.
    """

    def __init__(self, model: str, features: list[str]) -> None:
        self.model = model
        self.features = features
        super().__init__(
            f"Model {model!r} cannot represent requested feature(s) "
            f"{', '.join(features)}; they must be removed by "
            f"apply_feature_gates() before translation."
        )


@dataclass(frozen=True)
class FeatureSpec:
    """What a given model is allowed to receive.

    All four flags default to True for Claude family; the registry's
    ``MODEL_SPECS`` table flips them off for non-Claude models so a
    misconfigured caller can't bypass the gate.
    """

    supports_cache_control: bool = True
    supports_thinking: bool = True
    supports_web_search: bool = True
    supports_tool_use: bool = True


def apply_feature_gates(spec: FeatureSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a new kwargs dict with unsupported features removed.

    Never mutates the input — the caller's dict is reused on retry paths
    and elsewhere, and an in-place strip would corrupt those.
    """
    if (
        spec.supports_cache_control
        and spec.supports_thinking
        and spec.supports_web_search
        and spec.supports_tool_use
    ):
        # Fast path: nothing to strip.
        return kwargs

    out = deepcopy(kwargs)

    if not spec.supports_cache_control:
        _strip_cache_control(out)

    if not spec.supports_thinking:
        out.pop("thinking", None)
        out.pop("output_config", None)

    if not spec.supports_web_search:
        _strip_web_search_tools(out)

    if not spec.supports_tool_use:
        out.pop("tools", None)
        out.pop("tool_choice", None)

    return out


def unsupported_requested(spec: FeatureSpec, kwargs: dict[str, Any]) -> list[str]:
    """Names of the capabilities this request ASKS FOR that ``spec`` forbids.

    The companion to ``apply_feature_gates``: the gate removes those fields,
    this reports which ones were actually present so the caller can surface
    the loss instead of dropping it silently. Pure and side-effect free —
    call it on the pre-gate kwargs.

    Returns a stable, sorted list of capability names matching the
    ``FeatureSpec`` flags (``cache_control``, ``thinking``, ``tool_use``,
    ``web_search``). Empty when the request asks for nothing the model
    cannot do — which is the common case and the fast path.
    """
    found: list[str] = []

    if not spec.supports_cache_control and _has_cache_control(kwargs):
        found.append("cache_control")

    if not spec.supports_thinking and (
        kwargs.get("thinking") is not None or kwargs.get("output_config") is not None
    ):
        found.append("thinking")

    # ``tool_choice`` counts on its own: apply_feature_gates pops BOTH keys,
    # so a request carrying only tool_choice would otherwise be stripped with
    # nothing reported — the exact silent-loss pattern this module prevents.
    if not spec.supports_tool_use and (
        kwargs.get("tools") or kwargs.get("tool_choice") is not None
    ):
        found.append("tool_use")

    if not spec.supports_web_search and _has_web_search_tool(kwargs):
        found.append("web_search")

    return sorted(found)


def _has_cache_control(kwargs: dict[str, Any]) -> bool:
    """True iff any system block, tool entry, or message content block carries
    ``cache_control``.

    Shares ``_iter_cache_control_carriers`` with ``_strip_cache_control`` so
    the reporter and the stripper cannot drift apart — a mismatch between them
    is precisely how a stripped capability would go unreported."""
    for block in _iter_cache_control_carriers(kwargs):
        if block.get("cache_control") is not None:
            return True
    return False


def _iter_cache_control_carriers(kwargs: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield every dict that may carry a ``cache_control`` marker.

    Single source of truth for "where can cache_control appear": both
    ``_has_cache_control`` (reports) and ``_strip_cache_control`` (removes)
    iterate through here, so adding a new carrier location updates both."""
    for key in ("system", "tools"):
        entries = kwargs.get(key)
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    yield entry

    messages = kwargs.get("messages")
    if isinstance(messages, list):
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        yield block


def _has_web_search_tool(kwargs: dict[str, Any]) -> bool:
    """True iff ``tools`` contains an Anthropic server-side search/fetch tool.
    Mirrors the predicate in ``_strip_web_search_tools``."""
    tools = kwargs.get("tools")
    if not isinstance(tools, list):
        return False
    return any(
        isinstance(t, dict)
        and isinstance(t.get("type"), str)
        and t["type"].startswith(("web_search_", "web_fetch_"))
        for t in tools
    )


def _strip_cache_control(kwargs: dict[str, Any]) -> None:
    """Remove ``cache_control`` from system blocks and tool entries.

    Non-Claude OpenRouter models 400 on unknown fields, and even where
    they tolerate them, the field has no semantic effect — so dropping
    it is the right call. Anthropic-routed callers never hit this path.

    Walks the same carriers ``_has_cache_control`` inspects (system blocks,
    tool entries, and user-turn content blocks for the rolling cache), so
    what gets reported and what gets removed stay in lockstep by
    construction rather than by matching comments.
    """
    for block in _iter_cache_control_carriers(kwargs):
        block.pop("cache_control", None)


def _strip_web_search_tools(kwargs: dict[str, Any]) -> None:
    """Remove Anthropic server-side web_search / web_fetch tool entries.

    These use a ``type: "web_search_…"`` field instead of ``input_schema``;
    OpenRouter has no equivalent server tool, so they must be dropped.
    """
    tools = kwargs.get("tools")
    if not isinstance(tools, list):
        return
    kwargs["tools"] = [
        t
        for t in tools
        if not (
            isinstance(t, dict)
            and isinstance(t.get("type"), str)
            and t["type"].startswith(("web_search_", "web_fetch_"))
        )
    ]
    if not kwargs["tools"]:
        kwargs.pop("tools", None)
