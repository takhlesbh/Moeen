from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from openexecutive.alerts import dispatcher, preferences, store
from openexecutive.alerts.models import (
    AlertEvent,
    TriageDecision,
)
from openexecutive.clients.context_guard import (
    CompanyContext,
    StaleCompanyContextError,
    capture_company_context,
    company_mutation_guard,
    verify_company_context_unlocked,
)

logger = logging.getLogger(__name__)

# Hold strong references to in-flight background tasks so GC cannot cancel them
# mid-flight (mirrors memory.episodic._background_tasks).
_background_tasks: set[asyncio.Task[Any]] = set()

# Simple rolling-window rate limiter on event evaluations. Cost guard against
# runaway cost from a misbehaving integration. Single-process only.
_RATE_LIMIT_PER_MIN = 60
_recent_event_ts: deque[float] = deque(maxlen=_RATE_LIMIT_PER_MIN)
_rate_lock = threading.Lock()


def _rate_limited() -> bool:
    """Returns True when the per-minute cap has been hit."""
    now = time.monotonic()
    with _rate_lock:
        cutoff = now - 60.0
        while _recent_event_ts and _recent_event_ts[0] < cutoff:
            _recent_event_ts.popleft()
        if len(_recent_event_ts) >= _RATE_LIMIT_PER_MIN:
            return True
        _recent_event_ts.append(now)
        return False


@asynccontextmanager
async def _alert_commit_guard(
    origin: CompanyContext | None, *, serialised: bool = True
) -> AsyncIterator[None]:
    """Guard the alert commit, or pass straight through when no origin was given.

    A nullcontext-style passthrough rather than a branch at the call site, so
    the guarded and unguarded paths run the *same* statements in the same order
    and cannot drift apart.

    ``serialised=False`` verifies the context but does NOT take the lock. It is
    for the ``threading.Thread`` + ``asyncio.run`` fallback below, which runs on
    a different event loop: an ``asyncio.Lock`` cannot cross loops — an
    uncontended acquire succeeds while serialising nothing, and a contended one
    raises. Verification without serialisation is the honest option there, and
    it still rejects the dominant case (a switch that completed during triage).
    """
    if origin is None:
        yield
        return
    if not serialised:
        verify_company_context_unlocked(origin, operation="alert dispatch")
        yield
        return
    async with company_mutation_guard(origin, operation="alert dispatch"):
        yield


async def evaluate_and_dispatch(
    event: AlertEvent,
    db_path: Path | None = None,
    *,
    origin: CompanyContext | None = None,
    serialised: bool = True,
) -> tuple[TriageDecision, int | None]:
    """Run triage on a single event and dispatch the alert.

    Returns (decision, alert_id). alert_id is None when the alert is
    suppressed, a duplicate, or dropped because the company changed.

    ``origin`` is the company context captured when this evaluation was
    SCHEDULED, and it must be passed in rather than read here: triage is an LLM
    call taking seconds, and a client switch during it would otherwise be
    observed as the origin and compare equal to itself.

    Why it matters: ``insert_alert`` and ``dispatch_all`` both resolve against
    ``EPISODIC_DB_PATH`` — a fixed path whose FILE the slot switch swaps
    (``_save_slot_state`` / ``_restore_slot_state`` copy ``state.db`` in and
    out). Resolving the path early therefore protects nothing: after a switch,
    ``dispatcher`` → ``handle_send_department_message`` → ``get_department``
    reads the NEXT client's channel id, so an alert whose headline was derived
    from THIS client's document would be posted into ANOTHER client's team
    room. That is a disclosure to the wrong humans, not a stale row.

    ``origin=None`` skips the check — the pre-existing behaviour, kept for
    direct callers (tests, one-off scripts) that never cross a switch.
    """
    from openexecutive.agents.triage import TriageAgent
    from openexecutive.memory.episodic import get_active_initiatives

    path = db_path or store.DB_PATH

    if _rate_limited():
        logger.warning(
            "alerts.pipeline rate-limited: dropping event source=%s ext_id=%s",
            event.source,
            event.external_id,
        )
        return (
            TriageDecision(
                alert=False,
                dedup_key=f"rate-limited-{event.source}-{event.external_id}",
                reason_if_suppressed="rate_limited",
            ),
            None,
        )

    # Cheap pre-checks: load context for the triage prompt + post-decision mute.
    prefs = preferences.get_preferences(db_path=path)
    mutes = [m.pattern for m in store.list_mutes(db_path=path)]
    recent = [
        {
            "headline": a.headline,
            "severity": a.severity,
            "dedup_key": a.dedup_key,
            "topic_tags": a.topic_tags,
        }
        for a in store.recent_alerts(limit=20, db_path=path)
    ]
    try:
        initiatives = get_active_initiatives()
    except Exception:
        initiatives = []

    agent = TriageAgent()
    decision = await agent.triage(
        event,
        recent_alerts=recent,
        mute_patterns=mutes,
        active_initiatives=initiatives,
    )

    # Post-decision mute check: triage may have produced a tag we mute.
    if decision.alert and preferences.matches_mute(decision.topic_tags, mutes):
        logger.info(
            "alerts.pipeline post-mute suppressing event source=%s ext_id=%s tags=%s",
            event.source,
            event.external_id,
            decision.topic_tags,
        )
        decision = decision.model_copy(
            update={"alert": False, "reason_if_suppressed": "muted_post_triage"}
        )

    if not decision.alert:
        # We still persist a low-severity record so the UI can show "23 events
        # triaged, 1 surfaced" if we ever want it. For now keep it simple:
        # only persist when the decision says alert=true.
        return decision, None

    effective_channels = preferences.resolve_channels(
        decision.channels, decision.severity, prefs
    )

    # Everything company-bound happens under ONE guard: the row write, the
    # read-back, and the outbound dispatch. Splitting them — verify, release,
    # then dispatch — would leave a TOCTOU window in which a switch lands
    # between the insert and the dispatch, which is the disclosure itself.
    # Triage above ran OUTSIDE the lock; only this commit is serialised against
    # slot switches.
    try:
        async with _alert_commit_guard(origin, serialised=serialised):
            alert_id = store.insert_alert(
                source=event.source,
                external_id=event.external_id,
                severity=decision.severity.value,
                headline=decision.headline
                or (event.subject or event.title or event.source),
                body=decision.body,
                suggested_action=decision.suggested_action,
                topic_tags=decision.topic_tags,
                dedup_key=decision.dedup_key,
                db_path=path,
            )
            if alert_id is None:
                logger.info(
                    "alerts.pipeline duplicate suppressed source=%s ext_id=%s",
                    event.source,
                    event.external_id,
                )
                return decision, None

            alert = store.get_alert(alert_id, db_path=path)
            if alert is None:
                logger.error(
                    "alerts.pipeline could not re-read alert id=%s", alert_id
                )
                return decision, alert_id

            await dispatcher.dispatch_all(
                alert,
                effective_channels,
                db_path=path,
                # Shift 3: thread the broadcast-routing context through so the
                # DEPARTMENT_CHANNEL / COMPANY_BROADCAST dispatchers can resolve
                # the right room. Empty strings when triage picked per-person
                # channels only — those dispatchers ignore the kwargs.
                department_slug=decision.department_slug,
                broadcast_integration=decision.broadcast_integration,
            )
    except StaleCompanyContextError:
        # Nothing was written and nothing was sent: the guard raises before
        # yielding. The alert is dropped rather than re-aimed at whoever is now
        # live — re-triaging it for the new company would be inventing an alert
        # that company never had an event for.
        logger.warning(
            "alerts.pipeline dropping alert — the company it was triaged for is "
            "no longer active (source=%s ext_id=%s)",
            event.source,
            event.external_id,
        )
        return decision, None
    return decision, alert_id


def schedule_evaluation(event: AlertEvent) -> None:
    """Fire-and-forget evaluation. Safe to call from sync or async context.

    Mirrors memory.episodic.schedule_extraction's GC-safe pattern.

    The company context is captured HERE, synchronously, and handed to the
    detached task. That placement is the whole protection: the task then awaits
    an LLM triage call for seconds before it writes or dispatches anything, so a
    context read inside it would observe whichever client is active by then —
    the switch itself — and wave the commit through. Captured here, the task
    stays bound to the company the event actually belongs to.

    Capturing for every caller keeps all six of them (documents, monitoring,
    the four chat integrations, the alert tool) unchanged.
    """
    origin = capture_company_context()
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(evaluate_and_dispatch(event, origin=origin))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except RuntimeError:
        threading.Thread(
            # Different loop: verify without the lock. See _alert_commit_guard.
            target=lambda: asyncio.run(
                evaluate_and_dispatch(event, origin=origin, serialised=False)
            ),
            daemon=True,
        ).start()
