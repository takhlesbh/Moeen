"""Generic HTTP-backed LLMProvider for any OpenAI-compatible chat API.

This is the shared engine behind both ``OpenRouterProvider`` and local /
self-hosted backends (Ollama, LM Studio, vLLM, llama.cpp). It speaks the
OpenAI ``/chat/completions`` wire format; the translator package converts
request/response/stream shapes so call sites stay Anthropic-native.

What's intentionally configurable so a single class serves every
OpenAI-compatible endpoint:

* ``base_url`` — ``https://openrouter.ai/api/v1`` for OpenRouter,
  ``http://localhost:11434/v1`` for Ollama, etc.
* ``api_key`` — optional. Many local servers need no auth, so when it's
  ``None`` we omit the ``Authorization`` header entirely rather than send
  ``Bearer None``.
* ``default_headers`` — provider-specific attribution headers (OpenRouter
  sets ``HTTP-Referer`` / ``X-Title``); local backends pass nothing.

Streaming notes:

* SSE chunks arrive as JSON objects in ``data: ...`` lines.
* The Executive's loop iterates ``async for event in stream`` looking for
  ``content_block_delta`` with ``delta.type == "text_delta"`` (the streaming
  text path) and then awaits ``stream.get_final_message()`` to read the
  fully assembled message (with tool_use blocks). We yield text deltas
  inline and accumulate tool_call fragments into the final message.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, TypeVar

import httpx

from openexecutive.providers.feature_gate import (
    FeatureSpec,
    apply_feature_gates,
    unsupported_requested,
)
from openexecutive.providers.translator import (
    StreamAccumulator,
    from_openai_response,
    to_openai_request,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Bounded retries, matching the ``anthropic.AsyncAnthropic`` default of
# ``DEFAULT_MAX_RETRIES = 2`` (3 attempts total) so reliability stops
# depending on which backend a model happens to route to.
#
# This is deliberately NOT full parity with that SDK: it sends no
# Idempotency-Key, so we cannot let a retry double-charge for work the
# backend already did. ``/chat/completions`` is billable and generation
# starts as soon as the request lands, which drives the retry policy below.
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_RETRY_BACKOFF_S = 0.5
# Cap a single backoff sleep. With 300s local timeouts, an unbounded
# exponential would let one call sit far past any upstream proxy deadline.
_MAX_RETRY_BACKOFF_S = 8.0
# Honour a server's own pacing, but never park a coroutine indefinitely
# because a backend sent an outlandish Retry-After.
_MAX_RETRY_AFTER_S = 30.0
# Chars of a backend error body to log. Bodies can echo request fragments,
# so we log one truncated copy on the final failure only.
_ERROR_BODY_LOG_CHARS = 500

# Server rejected the request without generating: safe and free to retry.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

# Transport failures where the request provably never reached the model, so a
# retry cannot double-bill. Read-side failures (ReadTimeout, ReadError,
# RemoteProtocolError) are deliberately EXCLUDED: the request landed and the
# completion is already being generated and charged, so retrying would pay
# for the same call up to three times and multiply a slow generation's
# latency by three. Those propagate on the first attempt instead.
_RETRYABLE_TRANSPORT = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)


def _is_retryable(exc: BaseException) -> bool:
    """True only for failures that are both transient AND provably unbilled."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, _RETRYABLE_TRANSPORT)


def _retry_after_seconds(exc: BaseException) -> float | None:
    """The server's requested delay from a ``Retry-After`` header, if any.

    Only the delta-seconds form is honoured; the HTTP-date form is rare on
    these APIs and parsing it would add clock-skew failure modes for no real
    benefit. Clamped so a hostile or buggy value cannot hang the caller.

    The header is backend-controlled, so non-finite values are rejected
    explicitly: ``float("nan")`` passes a bare ``< 0`` test (every NaN
    comparison is False) and ``min(nan, cap)`` returns NaN, which would reach
    ``asyncio.sleep`` — collapsing the backoff to ~0 on Python 3.11/3.12 and
    raising an uncaught ValueError on 3.13+. Either way one header would
    defeat the rate-limit backoff entirely.
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    raw = exc.response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, _MAX_RETRY_AFTER_S)


def _retry_delay(attempt: int, exc: BaseException, base_backoff_s: float) -> float:
    """Seconds to wait before ``attempt`` (1-based), for one failure.

    Prefers the server's own ``Retry-After`` over our guess — retrying faster
    than a rate limiter asked only deepens the rate limit. Otherwise capped
    exponential backoff with jitter: the orchestrator fans specialists out in
    parallel, so without jitter every one of them would retry in lockstep and
    re-create the burst that caused the 429.
    """
    requested = _retry_after_seconds(exc)
    if requested is not None:
        # Jitter the server's value too. A parallel specialist fan-out that
        # all get "429, Retry-After: 5" would otherwise wake at the identical
        # instant — the exact thundering herd jitter exists to break. Stay in
        # [0.5x, 1.0x] so we never retry sooner than half what was asked.
        return random.uniform(requested * 0.5, requested)
    capped = min(base_backoff_s * (2 ** (attempt - 1)), _MAX_RETRY_BACKOFF_S)
    # Full jitter over [0, capped]; spreads a synchronized fan-out.
    return random.uniform(0, capped)


def _without_thinking(spec: FeatureSpec) -> FeatureSpec:
    """``spec`` with ``supports_thinking`` forced off (no-op if already off)."""
    if not spec.supports_thinking:
        return spec
    return replace(spec, supports_thinking=False)


def _describe(exc: BaseException) -> str:
    """Short, body-free description of a failure for intermediate retry logs."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def _sanitize_for_log(text: str) -> str:
    """Collapse control characters in backend-controlled text.

    An error body is written by the backend, so newlines in it would let a
    hostile or compromised endpoint forge additional log lines (log
    injection / audit forgery). Replacing every control char keeps the body
    on exactly one line.
    """
    return "".join(" " if ch < " " or ch == "\x7f" else ch for ch in text)


def _log_terminal_failure(label: str, exc: BaseException) -> None:
    """The single detailed record of a failure we are done retrying.

    Intermediate attempts stay terse: an error body can echo request
    fragments, so it is logged once at the end rather than once per attempt
    (and never at all when a later attempt succeeds).
    """
    if isinstance(exc, httpx.HTTPStatusError):
        logger.error(
            "%s returned %s: %s",
            label,
            exc.response.status_code,
            _sanitize_for_log(exc.response.text[:_ERROR_BODY_LOG_CHARS]),
        )
    else:
        # Transport failures carry no body — log the cause so a non-retried
        # (or retry-exhausted) failure is never silent.
        logger.error("%s transport failure: %s", label, _describe(exc))


async def _with_retry(
    operation: Callable[[], Awaitable[_T]],
    *,
    label: str,
    max_retries: int,
    backoff_s: float,
    on_retry: Callable[[], Awaitable[None]] | None = None,
) -> _T:
    """Run ``operation`` with bounded retries on transient, unbilled failures.

    Module-level so the non-streaming POST and the stream open share ONE
    retry driver — the bound, the backoff policy and the logging cannot drift
    apart between them. ``on_retry`` releases or resets whatever the failed
    attempt left behind before the next try.
    """
    attempt = 0
    while True:
        try:
            return await operation()
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            if not _is_retryable(exc) or attempt >= max_retries:
                _log_terminal_failure(label, exc)
                raise
            attempt += 1
            delay = _retry_delay(attempt, exc, backoff_s)
            logger.warning(
                "%s failed (%s); retry %d of %d in %.2fs",
                label,
                _describe(exc),
                attempt,
                max_retries,
                delay,
            )
            if on_retry is not None:
                await on_retry()
            await asyncio.sleep(delay)


class OpenAICompatibleProvider:
    """LLMProvider implementation backed by any OpenAI-compatible endpoint.

    Holds one ``httpx.AsyncClient`` for the lifetime of the process; the
    client pools its own TCP connections and is async-safe.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout_s: float = 180.0,
        slug_lookup: dict[str, str] | None = None,
        spec_lookup: dict[str, FeatureSpec] | None = None,
        default_params: dict[str, Any] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff_s: float = _DEFAULT_RETRY_BACKOFF_S,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=default_headers or {},
            timeout=timeout_s,
        )
        # Bounded: never negative, so a misconfigured 0 means "one attempt,
        # no retries" rather than an unbounded loop.
        self._max_retries = max(0, max_retries)
        self._retry_backoff_s = max(0.0, retry_backoff_s)
        # internal_name → backend slug. Populated by the registry so call
        # sites can use the same model names everywhere. Unknown models pass
        # through unchanged (the common case for local model names).
        self._slug_lookup = slug_lookup or {}
        # Thinking is unrepresentable in the OpenAI wire format regardless of
        # which model is behind it, so normalize every incoming spec rather
        # than trusting callers to remember. FeatureSpec() defaults
        # supports_thinking=True; a default-constructed spec reaching here
        # would otherwise skip the gate and make to_openai_request raise
        # mid-turn. Normalizing keeps that raise a true invariant guard.
        self._spec_lookup = {
            model: _without_thinking(spec) for model, spec in (spec_lookup or {}).items()
        }
        # Per-backend request defaults, in Anthropic-kwargs shape, applied only
        # where the caller said nothing (see ``_apply_defaults``). This is the
        # seam that lets a self-hosted backend carry its own sampling /
        # reasoning profile without any call site — Executive, specialists,
        # workflows — knowing which model is behind the endpoint. Empty for
        # OpenRouter, so its requests are byte-identical to before.
        self._default_params = dict(default_params or {})

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, anthropic_model: str) -> tuple[str, FeatureSpec]:
        slug = self._slug_lookup.get(anthropic_model, anthropic_model)
        spec = self._spec_lookup.get(
            anthropic_model,
            # Default for unknown models: assume non-Claude, no Anthropic-only
            # features. Safer than the optimistic default — a misconfigured
            # slug then quietly gets the right gating.
            FeatureSpec(
                supports_cache_control=False,
                supports_thinking=False,
                supports_web_search=False,
                supports_tool_use=True,
            ),
        )
        return slug, spec

    def _apply_defaults(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Fill in this backend's defaults for keys the caller left unset.

        An explicit caller value always wins — including an explicit ``0`` or
        ``0.0``, which is why the test is ``is None`` and not falsiness: a
        caller asking for ``temperature=0`` means deterministic sampling, and a
        truthiness test would silently replace it with the backend default.

        Runs BEFORE the feature gate, so a default this backend turns out not
        to support is stripped and reported by the same machinery that polices
        caller-supplied features — a misconfigured default cannot smuggle a
        capability past the gate.

        Never mutates the caller's dict: it is reused across the retry path,
        and the copy is made only when a default actually applies so the
        common (no-defaults) case stays allocation-free.
        """
        if not self._default_params:
            return kwargs
        out: dict[str, Any] | None = None
        for key, value in self._default_params.items():
            if kwargs.get(key) is None:
                if out is None:
                    out = dict(kwargs)
                out[key] = value
        return out if out is not None else kwargs

    def _gate(self, slug: str, spec: FeatureSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Apply the feature gate, reporting anything it had to remove.

        The gate on its own is silent, which is how a caller could ask for
        deep reasoning and never learn it did not happen. Logging the removed
        capabilities here — once per request, with the resolved slug — is the
        difference between a gated capability and a lost one.
        """
        dropped = unsupported_requested(spec, kwargs)
        if dropped:
            logger.warning(
                "model %s does not support %s; removing from request "
                "(requested but not delivered)",
                slug,
                ", ".join(dropped),
            )
        return apply_feature_gates(spec, kwargs)

    async def _post_with_retry(self, body: dict[str, Any], timeout: Any) -> httpx.Response:
        """POST /chat/completions with bounded retries on transient failures."""

        async def _once() -> httpx.Response:
            resp = await self._client.post(
                "/chat/completions",
                json=body,
                headers=self._auth_headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp

        return await _with_retry(
            _once,
            label=f"backend {body.get('model')}",
            max_retries=self._max_retries,
            backoff_s=self._retry_backoff_s,
        )

    def _auth_headers(self) -> dict[str, str]:
        # Local backends (Ollama, LM Studio) typically need no auth — omit
        # the header entirely rather than send a bogus ``Bearer None``.
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    # ------------------------------------------------------------------
    # LLMProvider surface
    # ------------------------------------------------------------------

    def messages_create(self, **kwargs: Any) -> Awaitable[Any]:
        return self._messages_create(kwargs)

    async def _messages_create(self, kwargs: dict[str, Any]) -> Any:
        # Strip the SDK's own ``timeout`` kwarg — httpx already has it from
        # the client; passing it into the body would break the request.
        request_timeout = kwargs.pop("timeout", None)
        model = kwargs.pop("model", "")
        slug, spec = self._resolve(model)
        gated = self._gate(slug, spec, self._apply_defaults(kwargs))
        body = to_openai_request(slug, gated)

        resp = await self._post_with_retry(
            body,
            request_timeout if request_timeout is not None else httpx.USE_CLIENT_DEFAULT,
        )
        return from_openai_response(resp.json())

    def messages_stream(self, **kwargs: Any) -> AbstractAsyncContextManager[Any]:
        request_timeout = kwargs.pop("timeout", None)
        model = kwargs.pop("model", "")
        slug, spec = self._resolve(model)
        gated = self._gate(slug, spec, self._apply_defaults(kwargs))
        body = to_openai_request(slug, gated)
        body["stream"] = True
        return _OpenAICompatibleStream(
            client=self._client,
            body=body,
            headers=self._auth_headers(),
            timeout=request_timeout,
            max_retries=self._max_retries,
            retry_backoff_s=self._retry_backoff_s,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class _OpenAICompatibleStream:
    """Async context manager that wraps an OpenAI-compatible SSE response so
    it quacks like ``anthropic.AsyncMessageStreamManager``.

    Consumers do:

        async with provider.messages_stream(...) as stream:
            async for event in stream:
                ...  # event has Anthropic shape
            final = await stream.get_final_message()
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout: float | None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff_s: float = _DEFAULT_RETRY_BACKOFF_S,
    ) -> None:
        self._client = client
        self._body = body
        self._headers = headers
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._retry_backoff_s = max(0.0, retry_backoff_s)
        self._response_cm: contextlib.AbstractAsyncContextManager[httpx.Response] | None = None
        self._response: httpx.Response | None = None
        self._accumulator = StreamAccumulator()
        self._finalized: SimpleNamespace | None = None

    async def __aenter__(self) -> _OpenAICompatibleStream:
        """Open the SSE response, retrying transient failures.

        Retrying here is safe because it happens strictly before any event is
        yielded to the consumer — a failed attempt produced no partial output.
        Once iteration starts we never retry, since half a reply cannot be
        replayed. Matches the Anthropic SDK, which also retries only the
        request that establishes the stream.
        """
        await _with_retry(
            self._open_once,
            label=f"stream {self._body.get('model')}",
            max_retries=self._max_retries,
            backoff_s=self._retry_backoff_s,
            on_retry=self._reset_for_retry,
        )
        return self

    async def _reset_for_retry(self) -> None:
        """Release the failed attempt's connection and drop its partial state.

        Closing first is what keeps a retry from orphaning the previous
        response context — ``_open_once`` would otherwise overwrite
        ``_response_cm`` and the old one would never be exited.
        """
        await self._close_response_cm()
        self._accumulator = StreamAccumulator()

    async def _close_response_cm(self) -> None:
        """Exit and forget the current response context, if one is open.

        Every failure path must run this before the next attempt: otherwise
        ``_open_once`` overwrites ``_response_cm`` with a fresh context and the
        previous one is never exited, orphaning its connection. Enough of those
        exhausts the httpx pool and every later call blocks on PoolTimeout.
        """
        if self._response_cm is None:
            return
        cm, self._response_cm, self._response = self._response_cm, None, None
        try:
            await cm.__aexit__(None, None, None)
        except Exception as exc:
            # Must not propagate: this runs from a ``finally`` while an
            # HTTPStatusError is in flight, and replacing that exception would
            # destroy the retry classification. But a failure here means the
            # connection was NOT returned to the pool — the very thing this
            # method exists to guarantee — so it is logged rather than
            # swallowed silently. CancelledError is a BaseException and still
            # propagates, which is correct.
            logger.warning("failed to close response context: %s", _describe(exc))

    async def _open_once(self) -> None:
        # ``client.stream(...)`` is a context manager; we open it here and
        # close it in ``__aexit__`` so callers get the same scoping as
        # Anthropic's manager.
        self._response_cm = self._client.stream(
            "POST",
            "/chat/completions",
            json=self._body,
            headers=self._headers,
            timeout=self._timeout if self._timeout is not None else httpx.USE_CLIENT_DEFAULT,
        )
        try:
            self._response = await self._response_cm.__aenter__()
        except BaseException:
            # The context was never entered, so __aexit__ must not run on it.
            # Clearing here keeps a retry (or the caller's __aexit__) from
            # exiting a context that was never opened.
            self._response_cm = None
            self._response = None
            raise
        # If the backend returned a non-2xx we must close the response context
        # ourselves — the outer ``async with`` will NOT call __aexit__ when
        # __aenter__ raises, so a naive ``raise_for_status`` would leak the
        # connection back to the pool half-read.
        #
        # The whole block is guarded, not just raise_for_status: reading the
        # error body can itself fail mid-read (ReadError / RemoteProtocolError),
        # and an unguarded read would leave the context open for a retry to
        # overwrite.
        if self._response.status_code >= 400:
            try:
                # Buffer the body before closing so the terminal-failure log
                # (via _log_terminal_failure) can still read exc.response.text.
                await self._response.aread()
                self._response.raise_for_status()
            finally:
                await self._close_response_cm()

    async def __aexit__(self, *exc_info: Any) -> None:
        cm = self._response_cm
        # Null FIRST so the "every path leaves no live context behind"
        # invariant holds even if __aexit__ itself raises — otherwise the
        # fields would keep pointing at a dead object.
        self._response_cm = None
        self._response = None
        if cm is not None:
            await cm.__aexit__(*exc_info)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[Any]:
        if self._response is None:
            return
        async for raw_line in self._response.aiter_lines():
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                logger.debug("OpenAI-compatible: undecodable SSE payload %s", payload[:120])
                continue
            for event in self._accumulator.feed(chunk):
                yield event
        # Cache the final message so a later get_final_message() call returns
        # immediately without re-reading the stream.
        self._finalized = self._accumulator.finalize()

    async def get_final_message(self) -> Any:
        if self._finalized is None:
            # The consumer didn't iterate the stream; drive it to completion.
            async for _ in self._iter_events():
                pass
        # _iter_events sets _finalized in finalize(); if for some reason we
        # got here without it being set, fall back to a fresh finalize().
        if self._finalized is None:
            self._finalized = self._accumulator.finalize()
        return self._finalized
