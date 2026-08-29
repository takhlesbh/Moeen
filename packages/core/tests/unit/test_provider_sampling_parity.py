"""Sampling + reasoning-control parity for the OpenAI-compatible path.

Covers the seam added so a self-hosted backend can carry its own request
profile without any call site knowing which model is behind the endpoint:

* ``to_openai_request`` forwards sampling fields only when explicitly supplied.
* ``OpenAICompatibleProvider.default_params`` fills gaps, never overrides.
* ``FeatureSpec.supports_reasoning_effort`` gates ``reasoning_effort`` with the
  same strip-and-report contract the other capabilities use.

The invariant under test throughout: a caller that supplies nothing must get
byte-identical requests to the ones this app sent before any of this existed.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from openexecutive.providers.feature_gate import (
    FeatureSpec,
    apply_feature_gates,
    unsupported_requested,
)
from openexecutive.providers.openai_compatible import OpenAICompatibleProvider
from openexecutive.providers.translator import to_openai_request

_MSGS = [{"role": "user", "content": "hi"}]

_LOCAL_SPEC = FeatureSpec(
    supports_cache_control=False,
    supports_thinking=False,
    supports_web_search=False,
    supports_tool_use=True,
    supports_reasoning_effort=True,
)

_OPENROUTER_SPEC = FeatureSpec(
    supports_cache_control=False,
    supports_thinking=False,
    supports_web_search=False,
    supports_tool_use=True,
)


# ---------------------------------------------------------------------------
# translator: explicit-only forwarding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.3),
        ("top_p", 0.95),
        ("presence_penalty", 0),
        ("frequency_penalty", 0.5),
    ],
)
def test_sampling_field_survives_translation(field: str, value: Any) -> None:
    body = to_openai_request("m", {"messages": _MSGS, field: value})
    assert body[field] == value


@pytest.mark.parametrize(
    "field",
    ["temperature", "top_p", "presence_penalty", "frequency_penalty", "stop",
     "reasoning_effort"],
)
def test_absent_sampling_field_stays_absent(field: str) -> None:
    """No invented defaults. The translator is a translator, not a policy."""
    body = to_openai_request("m", {"messages": _MSGS})
    assert field not in body


@pytest.mark.parametrize(
    ("field", "zero"),
    [("temperature", 0), ("temperature", 0.0), ("top_p", 0.0),
     ("presence_penalty", 0), ("frequency_penalty", 0.0)],
)
def test_explicit_zero_survives(field: str, zero: Any) -> None:
    """0 is a meaningful value, not "unset" — a falsiness check would eat it."""
    body = to_openai_request("m", {"messages": _MSGS, field: zero})
    assert field in body
    assert body[field] == zero


def test_explicit_none_is_not_emitted() -> None:
    kwargs = {"messages": _MSGS, "temperature": None, "top_p": None}
    body = to_openai_request("m", kwargs)
    assert "temperature" not in body
    assert "top_p" not in body


def test_caller_values_preserved_exactly() -> None:
    """No rounding, clamping, or coercion between caller and wire."""
    kwargs = {
        "messages": _MSGS,
        "temperature": 0.3,
        "top_p": 0.95,
        "presence_penalty": 0,
        "frequency_penalty": 1.25,
    }
    body = to_openai_request("m", kwargs)
    for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        assert body[key] == kwargs[key]
        assert type(body[key]) is type(kwargs[key])


def test_anthropic_stop_sequences_translates_to_openai_stop() -> None:
    body = to_openai_request("m", {"messages": _MSGS, "stop_sequences": ["END"]})
    assert body["stop"] == ["END"]
    assert "stop_sequences" not in body


def test_openai_stop_alias_is_accepted() -> None:
    body = to_openai_request("m", {"messages": _MSGS, "stop": ["END"]})
    assert body["stop"] == ["END"]


def test_explicit_stop_wins_over_stop_sequences_alias() -> None:
    body = to_openai_request(
        "m", {"messages": _MSGS, "stop": ["A"], "stop_sequences": ["B"]}
    )
    assert body["stop"] == ["A"]


def test_sampling_parity_does_not_disturb_existing_body_shape() -> None:
    """A request with no sampling fields is exactly what it was before."""
    body = to_openai_request("m", {"messages": _MSGS, "max_tokens": 100})
    assert body == {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "usage": {"include": True},
    }


# ---------------------------------------------------------------------------
# feature gate: reasoning_effort is gated, not globally unrepresentable
# ---------------------------------------------------------------------------


def test_reasoning_effort_survives_gate_when_supported() -> None:
    kwargs = {"messages": _MSGS, "reasoning_effort": "none"}
    gated = apply_feature_gates(_LOCAL_SPEC, kwargs)
    assert gated["reasoning_effort"] == "none"
    assert unsupported_requested(_LOCAL_SPEC, kwargs) == []


def test_reasoning_effort_stripped_and_reported_when_unsupported() -> None:
    kwargs = {"messages": _MSGS, "reasoning_effort": "none"}
    assert unsupported_requested(_OPENROUTER_SPEC, kwargs) == ["reasoning_effort"]
    gated = apply_feature_gates(_OPENROUTER_SPEC, kwargs)
    assert "reasoning_effort" not in gated
    # The caller's dict is never mutated — the retry path reuses it.
    assert kwargs["reasoning_effort"] == "none"


def test_reasoning_effort_defaults_to_unsupported() -> None:
    """Additive: a spec written before this flag existed keeps its behaviour."""
    assert FeatureSpec().supports_reasoning_effort is False
    kwargs = {"messages": _MSGS, "reasoning_effort": "none"}
    assert "reasoning_effort" not in apply_feature_gates(FeatureSpec(), kwargs)


def test_gate_fast_path_untouched_for_fully_capable_spec() -> None:
    spec = FeatureSpec(supports_reasoning_effort=True)
    kwargs = {"messages": _MSGS}
    assert apply_feature_gates(spec, kwargs) is kwargs


def test_anthropic_shaped_request_keeps_the_allocation_free_fast_path() -> None:
    """The new flag must not cost Anthropic-family calls their fast path.

    ``supports_reasoning_effort`` is the one flag that defaults off, so gating
    the fast path on the flag alone would make every default-spec call deepcopy.
    """
    kwargs = {"messages": _MSGS, "thinking": {"type": "adaptive"}}
    assert apply_feature_gates(FeatureSpec(), kwargs) is kwargs


def test_fast_path_still_strips_reasoning_effort_when_present() -> None:
    """…but presence of the field must defeat the fast path, not slip through."""
    kwargs = {"messages": _MSGS, "reasoning_effort": "none"}
    out = apply_feature_gates(FeatureSpec(), kwargs)
    assert out is not kwargs
    assert "reasoning_effort" not in out


def test_reasoning_effort_is_not_declared_globally_unrepresentable() -> None:
    """It IS expressible in the OpenAI wire format — unlike ``thinking``.

    So it must be gated per-backend, never raise UnsupportedFeatureError the
    way a field with no wire representation at all does.
    """
    body = to_openai_request("m", {"messages": _MSGS, "reasoning_effort": "none"})
    assert body["reasoning_effort"] == "none"


# ---------------------------------------------------------------------------
# provider: per-backend defaults
# ---------------------------------------------------------------------------


def _capture(provider: OpenAICompatibleProvider) -> list[dict[str, Any]]:
    """Swap in a transport that records request bodies instead of sending them."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": "m",
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    provider._client = httpx.AsyncClient(
        base_url="http://local", transport=httpx.MockTransport(handler)
    )
    return seen


def _local_provider(**kw: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://local/v1", spec_lookup={"qwen": _LOCAL_SPEC}, **kw
    )


@pytest.mark.asyncio
async def test_backend_defaults_applied_when_caller_silent() -> None:
    provider = _local_provider(
        default_params={
            "temperature": 0.3,
            "top_p": 0.95,
            "presence_penalty": 0,
            "reasoning_effort": "none",
        }
    )
    seen = _capture(provider)
    await provider.messages_create(model="qwen", messages=_MSGS, max_tokens=10)

    body = seen[0]
    assert body["temperature"] == 0.3
    assert body["top_p"] == 0.95
    assert body["presence_penalty"] == 0
    assert body["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_caller_value_overrides_backend_default() -> None:
    provider = _local_provider(default_params={"temperature": 0.3})
    seen = _capture(provider)
    await provider.messages_create(
        model="qwen", messages=_MSGS, max_tokens=10, temperature=0.9
    )
    assert seen[0]["temperature"] == 0.9


@pytest.mark.asyncio
async def test_caller_explicit_zero_overrides_backend_default() -> None:
    """The override test that a falsiness bug would pass silently."""
    provider = _local_provider(default_params={"temperature": 0.3})
    seen = _capture(provider)
    await provider.messages_create(
        model="qwen", messages=_MSGS, max_tokens=10, temperature=0
    )
    assert seen[0]["temperature"] == 0


@pytest.mark.asyncio
async def test_no_defaults_configured_means_no_sampling_fields() -> None:
    provider = _local_provider()
    seen = _capture(provider)
    await provider.messages_create(model="qwen", messages=_MSGS, max_tokens=10)
    for key in ("temperature", "top_p", "presence_penalty", "reasoning_effort"):
        assert key not in seen[0]


@pytest.mark.asyncio
async def test_defaults_do_not_mutate_the_callers_kwargs() -> None:
    """The retry path re-sends the same dict; mutation would corrupt attempt 2."""
    provider = _local_provider(default_params={"temperature": 0.3})
    _capture(provider)
    kwargs: dict[str, Any] = {"model": "qwen", "messages": _MSGS, "max_tokens": 10}
    snapshot = dict(kwargs)
    await provider.messages_create(**kwargs)
    assert kwargs == snapshot


@pytest.mark.asyncio
async def test_unsupported_default_is_gated_not_smuggled() -> None:
    """A misconfigured default cannot bypass the gate the way a caller can't.

    Defaults are applied BEFORE gating precisely so this holds.
    """
    provider = OpenAICompatibleProvider(
        base_url="http://local/v1",
        spec_lookup={"plain": _OPENROUTER_SPEC},
        default_params={"reasoning_effort": "none", "temperature": 0.3},
    )
    seen = _capture(provider)
    await provider.messages_create(model="plain", messages=_MSGS, max_tokens=10)
    assert "reasoning_effort" not in seen[0]
    assert seen[0]["temperature"] == 0.3


@pytest.mark.asyncio
async def test_backend_defaults_apply_on_the_streaming_path_too() -> None:
    provider = _local_provider(default_params={"temperature": 0.3, "reasoning_effort": "none"})
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, text="data: [DONE]\n\n")

    provider._client = httpx.AsyncClient(
        base_url="http://local", transport=httpx.MockTransport(handler)
    )
    async with provider.messages_stream(model="qwen", messages=_MSGS, max_tokens=10) as s:
        async for _ in s:
            pass

    assert seen[0]["temperature"] == 0.3
    assert seen[0]["reasoning_effort"] == "none"
    assert seen[0]["stream"] is True
