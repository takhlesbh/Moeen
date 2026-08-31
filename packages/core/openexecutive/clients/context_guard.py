"""Company-bound mutation guard — keeps a write inside the company it started in.

The problem this exists for, proven with a deterministic barrier test: an upload
that begins while client A is active can *finish* after the operator (or the
nightly rotation) switches to client B, and its chunks and its file both land in
B. Nothing catches it afterwards — the leaked chunk carries a valid
``company_doc::`` id and a real filename, so it is indistinguishable from B's own
documents.

The mechanism is small on purpose:

1. **Capture** the active company context *before* the operation can cross a
   switch — at request entry, or at the moment a background task is scheduled.
2. Do the slow work (network read, extraction) **outside** any lock.
3. **Guard** the commit: take the same lock every slot/fixture mutation takes,
   re-read the context *while holding it*, and refuse if it moved.

Re-reading inside the lock is what makes the check sound. Verifying first and
locking afterwards would leave a window in which a switch lands between the two,
which is exactly the race being closed.

SCOPE — read this before relying on it: the lock is an :class:`asyncio.Lock`,
so this is a **single-process guarantee only**. It does not protect against a
second uvicorn worker, a CLI run, or any other process sharing
``vector_store_path``. That residual risk is recorded in
``architecture-facts.yaml`` and is deliberately not addressed here.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class StaleCompanyContextError(RuntimeError):
    """The active company changed between capture and commit.

    Deliberately distinct from an ingestion failure: nothing went wrong with the
    document, and retrying under the now-active company is a decision for the
    caller (or the user), never something to do silently. Callers map this to a
    409 or to an abort-and-log; they must not treat it as a generic error.
    """


@dataclass(frozen=True, slots=True)
class CompanyContext:
    """Which company's data a write belongs to.

    Two fields, because either can move independently and either changes whose
    company ``company/docs/`` and the ``company_docs`` collection represent:

    * ``client`` — the active client slot slug (``.active_client`` sentinel), or
      ``None`` in single-company mode.
    * ``fixture`` — the active demo fixture (``.fixture_active`` sentinel), or
      ``None``.

    Tracking ``client`` alone would miss a real switch: loading a fixture calls
    ``park_active_client``, which *unlinks* the client sentinel, so a
    fixture→fixture swap reads as ``None → None`` and would slip through.

    This is a LOCKING identity, not a persisted tenant field. It is compared
    in-process and never written into chunk metadata; no tenant model exists.
    """

    client: str | None
    fixture: str | None
    # None for a context that was read successfully. When the context could NOT
    # be read this holds a value unique to that read, which makes the dataclass
    # unequal to every other context — including another unreadable one. That is
    # what makes an unreadable sentinel fail closed instead of comparing
    # `(None, None) == (None, None)` and waving the commit through.
    unreadable_id: int | None = None


_unreadable_counter = itertools.count()


def _unknown_context() -> CompanyContext:
    """A context that cannot equal anything, so the guard always rejects."""
    return CompanyContext(
        client=None, fixture=None, unreadable_id=next(_unreadable_counter)
    )


def _settings(settings: Any | None) -> Any:
    if settings is not None:
        return settings
    from openexecutive.config import get_settings

    return get_settings()


def capture_company_context(settings: Any | None = None) -> CompanyContext:
    """Read the active company context now.

    Call this at the point the operation *originates* — for an HTTP upload that
    is the ASGI middleware, because FastAPI streams and spools the whole
    multipart body during dependency resolution, before the handler is entered
    (verified by instrumenting ``MultiPartParser.parse``). Capturing in the
    handler would sit *after* the slow transfer, which is the very window a
    switch lands in. For a background task it is schedule time, never inside the
    delayed task.

    Read failure is treated as UNKNOWN, not as "no company". Both underlying
    readers already swallow their own errors and return ``None``, so an
    unreadable sentinel — a permissions accident on ``_client_slots/``, EIO, a
    slug that fails the safe-name pattern — would otherwise make capture and
    verify BOTH read ``CompanyContext(None, None)``, compare equal, and silently
    degrade this control to an unconditional pass while the system is genuinely
    switching clients. A security control must fail closed, so an unreadable
    sentinel yields a value that never compares equal to anything, including
    itself.
    """
    resolved = _settings(settings)

    client: str | None
    fixture: str | None
    try:
        from openexecutive.clients.slots import _active_client_sentinel, get_active_client

        client = get_active_client(resolved)
        # `get_active_client` returns None both for "no client active" and for
        # "sentinel exists but is unreadable/garbage". Only the first is a real
        # context; the second must not be mistaken for it.
        if client is None and _active_client_sentinel(resolved).exists():
            logger.warning(
                "company-guard: active-client sentinel exists but is unreadable; "
                "treating company context as UNKNOWN (fail closed)"
            )
            return _unknown_context()
    except Exception:
        logger.exception("company-guard: could not read active client")
        return _unknown_context()
    try:
        from openexecutive.cli.fixture_loader import get_fixture_status

        status = get_fixture_status(resolved)
        value = status.get("active_fixture") if isinstance(status, dict) else None
        fixture = value if isinstance(value, str) else None
    except Exception:
        logger.exception("company-guard: could not read active fixture")
        return _unknown_context()

    return CompanyContext(client=client, fixture=fixture)


# Where the middleware parks the captured context for the handler to read.
REQUEST_CONTEXT_ATTR = "company_context"


def context_from_request(request: Any) -> CompanyContext:
    """The context captured by the middleware for this request.

    Falls back to reading it now when the middleware did not run — a router
    mounted on a bare ``FastAPI()`` in a test, or any app built without
    ``install_company_context_middleware``. The fallback is strictly weaker
    (it reads after the body was spooled), so it is a compatibility shim, not
    the intended path.
    """
    captured = getattr(getattr(request, "state", None), REQUEST_CONTEXT_ATTR, None)
    if isinstance(captured, CompanyContext):
        return captured
    return capture_company_context()


def install_company_context_middleware(app: Any) -> None:
    """Capture the company context before the request body is parsed.

    This has to be middleware. FastAPI resolves ``UploadFile``/``Form``
    dependencies *before* entering the handler, which means the entire
    multipart body — up to 50 MB from a possibly slow client — is streamed and
    spooled first. Instrumenting ``MultiPartParser.parse`` shows the real order:

        middleware-entry → multipart-parse-START → multipart-parse-DONE
        → handler-entry → await file.read()

    So a capture in the handler happens *after* the transfer, and a switch that
    lands during the upload would be captured as the NEW company, compare equal
    to itself, and let the write through — the original defect, unmitigated.
    Middleware is the first point that runs before the body is touched.
    """

    @app.middleware("http")
    async def _capture_company_context(request: Any, call_next: Any) -> Any:
        request.state.company_context = capture_company_context()
        return await call_next(request)


def _company_state_lock() -> asyncio.Lock:
    """THE company-state lock — the one every slot/fixture mutation already takes.

    Imported lazily and by reference so this is the *same object*, not a second
    lock that would serialise nothing. ``clients`` and ``cli.fixture_loader``
    import each other at function level, so a module-level import here would
    risk a cycle.

    Never acquire this inside ``knowledge/loader.py``: slot restore already holds
    it and calls ``ingest_file``, and :class:`asyncio.Lock` is not reentrant, so
    a guard down there would deadlock the restore path.
    """
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    return _FIXTURE_OP_LOCK


def verify_company_context_unlocked(
    expected: CompanyContext, *, settings: Any | None = None, operation: str = "commit"
) -> None:
    """Compare the active context against ``expected`` WITHOUT taking the lock.

    Raises :class:`StaleCompanyContextError` on mismatch, same as the guard.

    Strictly weaker than :func:`company_mutation_guard` and only for callers
    that genuinely cannot use it — specifically work running on a *different
    event loop* (a ``threading.Thread`` + ``asyncio.run`` fallback). An
    :class:`asyncio.Lock` cannot be shared across loops: an uncontended acquire
    silently succeeds without serialising anything, and a contended one raises
    ``RuntimeError: bound to a different event loop``. Pretending to lock there
    would be worse than not locking, because it would read as protection.

    What this still catches: a switch that has already COMPLETED by the time the
    commit runs — the dominant window, since triage takes seconds and a switch
    is fast. What it cannot catch: a switch landing *during* the commit itself.
    Callers must document that distinction rather than claim serialisation.
    """
    current = capture_company_context(settings)
    if current != expected:
        logger.warning(
            "company-guard: rejecting stale %s (unserialised check) — active "
            "company changed between capture and commit",
            operation,
        )
        raise StaleCompanyContextError("Active company changed during the operation.")


@asynccontextmanager
async def company_mutation_guard(
    expected: CompanyContext, *, settings: Any | None = None, operation: str = "ingest"
) -> AsyncIterator[None]:
    """Hold the company-state lock for a commit that belongs to ``expected``.

    Raises :class:`StaleCompanyContextError` — before yielding, so before any
    caller-side mutation — when the active company has moved since capture.

    Comparison is dataclass equality, which means ``None`` participates like any
    other value: ``None → None`` passes, and ``A → None`` / ``None → B`` are both
    rejected. A truthiness shortcut (``if expected and current != expected``)
    would silently allow every transition out of single-company mode.

    ``async with`` releases the lock on exception and on cancellation, so a
    failed or cancelled ingest cannot strand company mutations.

    The caller must do its *entire* company-bound commit inside the block —
    vector write and file write both. Committing anything after the block
    reopens the window this closes.
    """
    lock = _company_state_lock()
    async with lock:
        current = capture_company_context(settings)
        if current != expected:
            # The message carries no slugs or paths: it reaches an HTTP client
            # and a log line, and neither should learn another client's name.
            logger.warning(
                "company-guard: rejecting stale %s — active company changed "
                "between capture and commit",
                operation,
            )
            raise StaleCompanyContextError(
                "Active company changed during the operation."
            )
        yield
