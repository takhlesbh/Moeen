"""Registry wiring for the local backend's request defaults.

Guards the blast radius of the sampling/reasoning seam: the local profile must
reach the local provider and nothing else. Anthropic-direct never runs the
translator or the gate at all, and OpenRouter must not inherit a neighbour
backend's tuning.
"""
from __future__ import annotations

import pytest

from openexecutive.providers import registry
from openexecutive.providers.anthropic_provider import AnthropicProvider
from openexecutive.providers.openai_compatible import OpenAICompatibleProvider
from openexecutive.providers.openrouter_provider import OpenRouterProvider


@pytest.fixture(autouse=True)
def _reset_providers():
    registry._reset_for_tests()
    yield
    registry._reset_for_tests()


# Settings the registry branches on. Cleared before each case so a developer's
# real .env cannot decide which backend a test resolves to.
_ROUTING_ENV = (
    "OPENROUTER_ENABLED",
    "OPENROUTER_API_KEY",
    "LOCAL_MODELS_ENABLED",
    "LOCAL_BASE_URL",
    "LOCAL_MODELS",
    "LOCAL_API_KEY",
    "LOCAL_TEMPERATURE",
    "LOCAL_TOP_P",
    "LOCAL_PRESENCE_PENALTY",
    "LOCAL_REASONING_EFFORT",
)


def _env(monkeypatch: pytest.MonkeyPatch, **pairs: str) -> None:
    # get_settings() builds a fresh Settings() per call, so there is no cache to
    # clear — but the env must be scrubbed first or an ambient value survives.
    for key in _ROUTING_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in pairs.items():
        monkeypatch.setenv(key, value)


def _local_env(monkeypatch: pytest.MonkeyPatch, **extra: str) -> None:
    pairs = {
        "ANTHROPIC_API_KEY": "k",
        "LOCAL_MODELS_ENABLED": "true",
        "LOCAL_BASE_URL": "http://localhost:11434/v1",
        "LOCAL_MODELS": "qwen3.5:latest",
    }
    pairs.update(extra)
    _env(monkeypatch, **pairs)


def test_local_defaults_reach_the_local_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_env(
        monkeypatch,
        LOCAL_TEMPERATURE="0.3",
        LOCAL_TOP_P="0.95",
        LOCAL_PRESENCE_PENALTY="0",
        LOCAL_REASONING_EFFORT="none",
    )
    provider = registry.get_provider("qwen3.5:latest")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._default_params == {
        "temperature": 0.3,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "reasoning_effort": "none",
    }


def test_unset_local_defaults_send_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh checkout must keep the exact request shape it had before."""
    _local_env(monkeypatch)
    provider = registry.get_provider("qwen3.5:latest")
    assert provider._default_params == {}


def test_local_presence_penalty_zero_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """0 is the whole point here — Qwen3.5 ships a non-zero built-in default."""
    _local_env(monkeypatch, LOCAL_PRESENCE_PENALTY="0")
    provider = registry.get_provider("qwen3.5:latest")
    assert provider._default_params == {"presence_penalty": 0.0}


def test_local_spec_declares_reasoning_effort_support(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_env(monkeypatch)
    provider = registry.get_provider("qwen3.5:latest")
    spec = provider._spec_lookup["qwen3.5:latest"]
    assert spec.supports_reasoning_effort is True
    # …and still no Anthropic-only features.
    assert spec.supports_cache_control is False
    assert spec.supports_thinking is False
    assert spec.supports_web_search is False
    assert spec.supports_tool_use is True


def test_openrouter_does_not_inherit_local_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_env(
        monkeypatch,
        OPENROUTER_ENABLED="true",
        OPENROUTER_API_KEY="or-key",
        LOCAL_TEMPERATURE="0.3",
        LOCAL_REASONING_EFFORT="none",
    )
    provider = registry.get_provider("openai/gpt-5")
    assert isinstance(provider, OpenRouterProvider)
    assert provider._default_params == {}


def test_openrouter_specs_do_not_gain_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_env(
        monkeypatch,
        OPENROUTER_ENABLED="true",
        OPENROUTER_API_KEY="or-key",
        LOCAL_REASONING_EFFORT="none",
    )
    provider = registry.get_provider("openai/gpt-5")
    for slug, spec in provider._spec_lookup.items():
        assert spec.supports_reasoning_effort is False, slug


def test_anthropic_routing_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude slugs still go direct — no gate, no translator, no defaults."""
    _local_env(
        monkeypatch,
        LOCAL_TEMPERATURE="0.3",
        LOCAL_REASONING_EFFORT="none",
    )
    provider = registry.get_provider("claude-sonnet-4-6")
    assert isinstance(provider, AnthropicProvider)
    assert not hasattr(provider, "_default_params")


def test_claude_slug_listed_in_local_models_still_routes_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defence in depth: the Claude check precedes the local branch."""
    _local_env(monkeypatch, LOCAL_MODELS="claude-sonnet-4-6,qwen3.5:latest")
    assert isinstance(registry.get_provider("claude-sonnet-4-6"), AnthropicProvider)


# ---------------------------------------------------------------------------
# blank-value handling — .env.example ships these keys present-but-empty
# ---------------------------------------------------------------------------


def test_blank_local_defaults_boot_and_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """`cp .env.example .env` must still boot.

    The float fields reject "" outright, so shipping the keys blank without
    this coercion breaks the documented first-run path with a ValidationError
    before anything else happens.
    """
    from openexecutive.config import Settings

    _local_env(
        monkeypatch,
        LOCAL_TEMPERATURE="",
        LOCAL_TOP_P="",
        LOCAL_PRESENCE_PENALTY="",
        LOCAL_REASONING_EFFORT="",
    )
    settings = Settings()  # type: ignore[call-arg]
    assert settings.local_temperature is None
    assert settings.local_top_p is None
    assert settings.local_presence_penalty is None
    assert settings.local_reasoning_effort is None


def test_blank_values_send_nothing_on_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank must mean absent, not ``reasoning_effort: ""``.

    A blank string is not None, so without coercion it survives every
    ``is not None`` guard from settings through to the request body and ships
    an out-of-enum value on every local call.
    """
    _local_env(
        monkeypatch,
        LOCAL_TEMPERATURE="",
        LOCAL_REASONING_EFFORT="",
    )
    provider = registry.get_provider("qwen3.5:latest")
    assert provider._default_params == {}


def test_whitespace_is_trimmed_not_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    _local_env(monkeypatch, LOCAL_REASONING_EFFORT="  none  ")
    provider = registry.get_provider("qwen3.5:latest")
    assert provider._default_params == {"reasoning_effort": "none"}


def test_out_of_range_top_p_is_rejected_at_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    """top_p is probability mass — outside [0, 1] is a typo, not a preference.

    Failing here beats failing on every request.
    """
    import pydantic

    from openexecutive.config import Settings

    _local_env(monkeypatch, LOCAL_TOP_P="1.5")
    with pytest.raises(pydantic.ValidationError):
        Settings()  # type: ignore[call-arg]
