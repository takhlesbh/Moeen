"""Company-bound ingest guard — a write must not commit into another company.

The defect, proven with barriers before this guard existed: an upload that
begins while client A is active can finish *after* a switch to client B, and its
chunks and its file both land in B. The leaked chunk carries a valid
``company_doc::`` id and a real filename, so nothing downstream can tell it apart
from B's own documents.

Every race test here is driven by :class:`asyncio.Event`, never by ``sleep`` —
a sleep-timed race test passes for the wrong reason on a loaded machine.

Two mutation tests at the end are the real proof of worth: they disable the
verification, and separately swap in an independent lock, and assert the race
test fails. A guard whose removal changes nothing is not a guard.

Scope under test is a SINGLE-PROCESS guarantee. Nothing here claims protection
against a second worker or a CLI process.
"""
from __future__ import annotations

import asyncio
import io
import logging
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openexecutive.clients.context_guard import (
    CompanyContext,
    StaleCompanyContextError,
    capture_company_context,
    company_mutation_guard,
    install_company_context_middleware,
)
from openexecutive.knowledge.store import ChromaDBStore

COMPANY = ChromaDBStore.COMPANY_COLLECTION


# ---------------------------------------------------------------------------
# Context fixtures — drive the two real sentinels, not a mock of them.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _unbind_company_lock() -> Iterator[None]:
    """Detach the module-global lock from the previous test's event loop.

    ``_FIXTURE_OP_LOCK`` is created at import time and binds lazily to the first
    loop that awaits it; using it from a second loop then raises "bound to a
    different event loop". Each ``asyncio.run`` / ``TestClient`` here makes its
    own loop, so without this the tests interfere with one another.

    Test hygiene only, and deliberately NOT a production concern: uvicorn runs a
    single loop for the process lifetime, which is the same reason the existing
    slot/fixture tests have never needed it. Resetting the binding rather than
    replacing the object keeps the same-lock-identity assertions meaningful.
    """
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    _FIXTURE_OP_LOCK._loop = None  # type: ignore[attr-defined]
    yield
    _FIXTURE_OP_LOCK._loop = None  # type: ignore[attr-defined]


@pytest.fixture
def company_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated company dir. Never touches the real `company/`."""
    root = tmp_path / "company"
    (root / "_client_slots").mkdir(parents=True)
    (root / "_user_backup").mkdir(parents=True)
    monkeypatch.setenv("COMPANY_PROFILE_PATH", str(root / "profile.yaml"))
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "chroma"))
    return root


def set_active_client(company_root: Path, slug: str | None) -> None:
    sentinel = company_root / "_client_slots" / ".active_client"
    if slug is None:
        sentinel.unlink(missing_ok=True)
    else:
        sentinel.write_text(slug)


def set_active_fixture(company_root: Path, name: str | None) -> None:
    sentinel = company_root / "_user_backup" / ".fixture_active"
    if name is None:
        sentinel.unlink(missing_ok=True)
    else:
        sentinel.write_text(name)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ChromaDBStore]:
    persist = tmp_path / "chroma"
    yield ChromaDBStore(persist_directory=persist)
    shutil.rmtree(persist, ignore_errors=True)


def _rows(store: ChromaDBStore) -> list[dict[str, Any]]:
    col = store._get_or_create_collection(COMPANY)
    got = col.get(include=["metadatas"])
    return list(got["metadatas"])


def _filenames(store: ChromaDBStore) -> set[str]:
    return {m.get("filename") for m in _rows(store)}


@contextmanager
def _captured_warnings(logger_name: str) -> Iterator[list[str]]:
    """Collect WARNING messages from one logger.

    Deliberately not pytest's ``caplog``: ``api.main._configure_logging`` sets
    ``propagate = False`` on the ``openexecutive`` logger, so once any test in
    the session imports it, records never reach the root handler caplog
    installs — the assertion then passes on an empty list in isolation and fails
    only in the full suite. Same pattern as
    ``test_provider_capability_parity.capture_warnings``.
    """
    messages: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Collector(level=logging.WARNING)
    logger = logging.getLogger(logger_name)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


# ---------------------------------------------------------------------------
# Lock authority — one object, shared with every slot/fixture mutation.
# ---------------------------------------------------------------------------


def test_guard_uses_the_same_lock_object_as_slot_mutations() -> None:
    """A second lock would serialise nothing and silently protect nothing."""
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK
    from openexecutive.clients.context_guard import _company_state_lock

    assert _company_state_lock() is _FIXTURE_OP_LOCK
    # And the slot module operates on that identical object, so a switch and an
    # ingest genuinely contend.
    from openexecutive.clients import slots

    assert slots._FIXTURE_OP_LOCK is _FIXTURE_OP_LOCK


def test_guard_actually_holds_the_lock_while_inside() -> None:
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    async def run() -> None:
        assert not _FIXTURE_OP_LOCK.locked()
        async with company_mutation_guard(CompanyContext(None, None)):
            assert _FIXTURE_OP_LOCK.locked()
        assert not _FIXTURE_OP_LOCK.locked()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# None semantics — equality, never truthiness.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start,switch,allowed",
    [
        (None, None, True),      # single-company mode, nothing moved
        ("acme", "acme", True),  # same client
        ("acme", "globex", False),
        ("acme", None, False),   # a truthiness check would allow this
        (None, "globex", False),  # ...and this
    ],
)
def test_client_transition_equality(
    company_root: Path, start: str | None, switch: str | None, allowed: bool
) -> None:
    set_active_client(company_root, start)
    origin = capture_company_context()
    set_active_client(company_root, switch)

    async def run() -> None:
        async with company_mutation_guard(origin):
            pass

    if allowed:
        asyncio.run(run())
    else:
        with pytest.raises(StaleCompanyContextError):
            asyncio.run(run())


def test_fixture_swap_is_detected_even_though_client_stays_none(
    company_root: Path,
) -> None:
    """Loading a fixture unlinks the client sentinel, so client alone reads
    None → None. Tracking the fixture too is what catches a fixture swap."""
    set_active_client(company_root, None)
    set_active_fixture(company_root, "acme_demo")
    origin = capture_company_context()
    assert origin == CompanyContext(client=None, fixture="acme_demo")

    set_active_fixture(company_root, "globex_demo")

    async def run() -> None:
        async with company_mutation_guard(origin):
            pass

    with pytest.raises(StaleCompanyContextError):
        asyncio.run(run())


def test_client_to_fixture_transition_is_detected(company_root: Path) -> None:
    set_active_client(company_root, "acme")
    origin = capture_company_context()
    # What a fixture load does: park the client, then mark the fixture.
    set_active_client(company_root, None)
    set_active_fixture(company_root, "demo")

    async def run() -> None:
        async with company_mutation_guard(origin):
            pass

    with pytest.raises(StaleCompanyContextError):
        asyncio.run(run())


def test_context_is_frozen() -> None:
    ctx = CompanyContext(client="a", fixture=None)
    with pytest.raises(AttributeError):
        ctx.client = "b"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The race — barrier-driven, through the real HTTP route.
# ---------------------------------------------------------------------------


def _app(store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    from openexecutive.api.routes import documents as documents_route

    monkeypatch.setattr(
        "openexecutive.alerts.pipeline.schedule_evaluation", lambda *a, **k: None
    )
    app = FastAPI()
    install_company_context_middleware(app)
    app.include_router(documents_route.router)
    app.state.store = store
    return app


def _upload(client: TestClient, name: str, body: bytes) -> Any:
    return client.post("/documents", files={"file": (name, io.BytesIO(body), "text/markdown")})


BODY = ("# Doc\n" + " ".join(f"w{i}" for i in range(600))).encode()


def test_upload_started_under_a_cannot_land_in_b(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline race: 409, and NOTHING of A's in B — vectors or file.

    The switch is injected exactly at the route's `await file.read()` — the real
    yield point — by making the read itself await a barrier.
    """
    set_active_client(company_root, "acme")
    app = _app(store, monkeypatch)

    switched = asyncio.Event()

    from starlette.datastructures import UploadFile as StarletteUploadFile

    original_read = StarletteUploadFile.read

    async def read_then_switch(self: Any, size: int = -1) -> bytes:
        data = await original_read(self, size)
        if not switched.is_set():
            # The slot switch lands here, mid-upload.
            set_active_client(company_root, "globex")
            switched.set()
        return data

    monkeypatch.setattr(StarletteUploadFile, "read", read_then_switch)

    with TestClient(app) as client:
        resp = _upload(client, "A-secret.md", BODY)

    assert resp.status_code == 409, resp.text
    assert "retry" in resp.json()["detail"].lower()
    assert switched.is_set(), "the race never actually happened"

    # No vector leak.
    assert _filenames(store) == set()
    # No file leak into the now-active company's docs dir.
    docs = company_root / "docs"
    assert not docs.exists() or list(docs.iterdir()) == []


def test_upload_succeeds_when_company_does_not_change(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must not break the ordinary path."""
    set_active_client(company_root, "acme")
    app = _app(store, monkeypatch)

    with TestClient(app) as client:
        resp = _upload(client, "plan.md", BODY)

    assert resp.status_code == 200, resp.text
    assert _filenames(store) == {"plan.md"}
    assert (company_root / "docs" / "plan.md").exists()


def test_two_uploads_under_the_same_client_both_succeed(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_active_client(company_root, "acme")
    app = _app(store, monkeypatch)

    with TestClient(app) as client:
        assert _upload(client, "one.md", BODY).status_code == 200
        assert _upload(client, "two.md", BODY).status_code == 200

    assert _filenames(store) == {"one.md", "two.md"}


def test_stale_upload_cleans_up_its_temp_file(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 409 must not leak the staged temp file."""
    import tempfile as _tempfile

    set_active_client(company_root, "acme")
    app = _app(store, monkeypatch)

    created: list[Path] = []
    real_named = _tempfile.NamedTemporaryFile

    def spy(*a: Any, **k: Any) -> Any:
        handle = real_named(*a, **k)
        created.append(Path(handle.name))
        return handle

    monkeypatch.setattr(
        "openexecutive.api.routes.documents.tempfile.NamedTemporaryFile", spy
    )

    from starlette.datastructures import UploadFile as StarletteUploadFile

    original_read = StarletteUploadFile.read

    async def read_then_switch(self: Any, size: int = -1) -> bytes:
        data = await original_read(self, size)
        set_active_client(company_root, "globex")
        return data

    monkeypatch.setattr(StarletteUploadFile, "read", read_then_switch)

    with TestClient(app) as client:
        assert _upload(client, "x.md", BODY).status_code == 409

    assert created, "the route did not stage a temp file"
    assert all(not p.exists() for p in created), "temp file leaked on the 409 path"


# ---------------------------------------------------------------------------
# Attachments — captured at schedule time, verified at commit time.
# ---------------------------------------------------------------------------


def test_attachment_scheduled_under_a_does_not_write_into_b(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The detached task is the more exposed path: nothing awaits or cancels it."""
    from openexecutive.integrations import attachments as att

    set_active_client(company_root, "acme")
    monkeypatch.setattr(
        "openexecutive.knowledge.store.ChromaDBStore", lambda *a, **k: store
    )

    async def run() -> None:
        # Scheduled while acme is active...
        att._schedule_ingest(BODY, "A-notes.md")
        # ...the switch happens before the task gets to its commit.
        set_active_client(company_root, "globex")
        await asyncio.gather(*list(att._ingest_tasks))

    with _captured_warnings("openexecutive.integrations.attachments") as messages:
        asyncio.run(run())

    assert _filenames(store) == set(), "attachment leaked into the new company"
    assert any("no longer active" in m for m in messages), (
        f"the stale-company rejection was not logged; saw {messages}"
    )


def test_attachment_ingests_normally_when_company_is_unchanged(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.integrations import attachments as att

    set_active_client(company_root, "acme")
    monkeypatch.setattr(
        "openexecutive.knowledge.store.ChromaDBStore", lambda *a, **k: store
    )

    async def run() -> None:
        att._schedule_ingest(BODY, "notes.md")
        await asyncio.gather(*list(att._ingest_tasks))

    asyncio.run(run())
    assert _filenames(store) == {"notes.md"}


def test_stale_attachment_does_not_raise_out_of_the_background_task(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unhandled exception in a detached task is a silent process-level noise
    source; the rejection must be handled, not propagated."""
    from openexecutive.integrations import attachments as att

    set_active_client(company_root, "acme")
    monkeypatch.setattr(
        "openexecutive.knowledge.store.ChromaDBStore", lambda *a, **k: store
    )

    async def run() -> list[Any]:
        att._schedule_ingest(BODY, "n.md")
        set_active_client(company_root, "globex")
        return await asyncio.gather(*list(att._ingest_tasks), return_exceptions=True)

    results = asyncio.run(run())
    assert all(not isinstance(r, BaseException) for r in results), results


def test_attachment_captures_origin_at_schedule_time_not_run_time(
    company_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deriving the origin inside the task would just observe B and pass."""
    from openexecutive.integrations import attachments as att

    set_active_client(company_root, "acme")
    seen: list[CompanyContext] = []

    real_capture = capture_company_context

    def spy(*a: Any, **k: Any) -> CompanyContext:
        ctx = real_capture(*a, **k)
        seen.append(ctx)
        return ctx

    monkeypatch.setattr(
        "openexecutive.clients.context_guard.capture_company_context", spy
    )

    async def run() -> None:
        att._schedule_ingest(BODY, "n.md")
        set_active_client(company_root, "globex")
        await asyncio.gather(*list(att._ingest_tasks), return_exceptions=True)

    asyncio.run(run())
    assert seen, "capture never ran"
    assert seen[0].client == "acme", "origin was captured after the switch"


# ---------------------------------------------------------------------------
# Lock hygiene — exceptions, cancellation, reentrancy.
# ---------------------------------------------------------------------------


def test_exception_inside_the_guard_releases_the_lock() -> None:
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    async def run() -> None:
        with pytest.raises(RuntimeError):
            async with company_mutation_guard(CompanyContext(None, None)):
                raise RuntimeError("ingest blew up")
        assert not _FIXTURE_OP_LOCK.locked()
        # And the lock is still usable afterwards.
        async with company_mutation_guard(CompanyContext(None, None)):
            pass

    asyncio.run(run())


def test_stale_rejection_releases_the_lock(company_root: Path) -> None:
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    set_active_client(company_root, "acme")
    origin = capture_company_context()
    set_active_client(company_root, "globex")

    async def run() -> None:
        with pytest.raises(StaleCompanyContextError):
            async with company_mutation_guard(origin):
                pass
        assert not _FIXTURE_OP_LOCK.locked()

    asyncio.run(run())


def test_cancellation_releases_the_lock() -> None:
    """A cancelled ingest must not strand company mutations forever."""
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    inside = asyncio.Event()

    async def holder() -> None:
        async with company_mutation_guard(CompanyContext(None, None)):
            inside.set()
            await asyncio.Event().wait()  # blocks until cancelled

    async def run() -> None:
        task = asyncio.create_task(holder())
        await inside.wait()
        assert _FIXTURE_OP_LOCK.locked()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not _FIXTURE_OP_LOCK.locked()

    asyncio.run(run())


def test_guard_is_not_reentrant_so_it_must_never_wrap_loader_ingest() -> None:
    """Pins the reason `ingest_file` must stay guard-free.

    Slot restore already holds this lock and calls `ingest_file`. If the guard
    lived down there, restore would await a lock it already owns — a permanent
    deadlock. This asserts the hazard is real rather than trusting a comment.
    """
    async def run() -> None:
        async with company_mutation_guard(CompanyContext(None, None)):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    company_mutation_guard(CompanyContext(None, None)).__aenter__(),
                    timeout=0.1,
                )

    asyncio.run(run())


def test_loader_does_not_import_or_take_the_company_lock() -> None:
    """Structural guard against re-introducing the deadlock."""
    source = (
        Path(__file__).resolve().parents[2]
        / "openexecutive" / "knowledge" / "loader.py"
    ).read_text()
    assert "company_mutation_guard" not in source
    assert "_FIXTURE_OP_LOCK" not in source
    assert "context_guard" not in source


def test_slot_restore_style_nesting_does_not_deadlock(
    company_root: Path, store: ChromaDBStore, tmp_path: Path
) -> None:
    """Exactly what activate_client_slot does: hold the lock, then ingest."""
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK
    from openexecutive.knowledge.loader import company_document_id, ingest_file

    doc = tmp_path / "restored.md"
    doc.write_text(" ".join(f"r{i}" for i in range(600)))

    async def run() -> None:
        async with _FIXTURE_OP_LOCK:  # the slot-restore critical section
            n = await ingest_file(
                path=doc,
                store=store,
                collection=COMPANY,
                display_filename=doc.name,
                document_id=company_document_id(doc.name),
            )
            assert n > 0

    asyncio.run(asyncio.wait_for(run(), timeout=5))
    assert _filenames(store) == {"restored.md"}


# ---------------------------------------------------------------------------
# Mutation tests — the guard must be load-bearing.
# ---------------------------------------------------------------------------


def test_mutation_disabling_verification_reopens_the_race(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the comparison removed, A's document lands in B — the original bug."""
    from openexecutive.knowledge.loader import company_document_id, ingest_file

    set_active_client(company_root, "acme")
    origin = capture_company_context()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def no_verify(expected: Any, **_k: Any) -> Any:
        from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

        async with _FIXTURE_OP_LOCK:
            yield  # MUTATION: context is never compared

    doc = company_root / "A.md"
    doc.write_text(" ".join(f"a{i}" for i in range(600)))

    async def run() -> None:
        set_active_client(company_root, "globex")
        async with no_verify(origin):
            await ingest_file(
                path=doc, store=store, collection=COMPANY,
                display_filename="A-secret.md",
                document_id=company_document_id("A-secret.md"),
            )

    asyncio.run(run())
    # The leak the real guard prevents.
    assert "A-secret.md" in _filenames(store)


def test_mutation_independent_lock_would_not_serialise_with_slot_switches(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private lock is not a second-best guard — it protects nothing.

    Asserted directly: with an independent lock, a slot switch can hold the real
    lock while the guard sails through, which is precisely the contention the
    design depends on.
    """
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    independent = asyncio.Lock()
    monkeypatch.setattr(
        "openexecutive.clients.context_guard._company_state_lock",
        lambda: independent,
    )

    async def run() -> None:
        # Nested on purpose: the outer lock stands in for an in-progress slot
        # switch, and the inner guard must NOT be able to enter while it is
        # held. With the mutated (independent) lock it enters anyway.
        await _FIXTURE_OP_LOCK.acquire()
        try:
            async with company_mutation_guard(CompanyContext(None, None)):
                assert independent.locked()
                assert _FIXTURE_OP_LOCK.locked()  # held by the "switch", not us
        finally:
            _FIXTURE_OP_LOCK.release()

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_real_guard_does_serialise_against_a_held_lock() -> None:
    """The inverse of the mutation above — the guard genuinely waits."""
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    entered = asyncio.Event()

    async def run() -> None:
        async with _FIXTURE_OP_LOCK:  # stand-in for a slot switch
            async def try_guard() -> None:
                async with company_mutation_guard(CompanyContext(None, None)):
                    entered.set()

            task = asyncio.create_task(try_guard())
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
            assert not entered.is_set(), "guard did not wait for the switch"
            task.cancel()
        # Released — now it can proceed.
        async with company_mutation_guard(CompanyContext(None, None)):
            pass

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Existing invariants must not move.
# ---------------------------------------------------------------------------


def test_document_identity_semantics_unchanged(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.knowledge.loader import company_document_id

    set_active_client(company_root, "acme")
    app = _app(store, monkeypatch)
    with TestClient(app) as client:
        assert _upload(client, "plan.md", BODY).status_code == 200

    ids = {m.get("document_id") for m in _rows(store)}
    assert ids == {company_document_id("plan.md")}
    assert ids == {"company_doc::plan.md"}
    # No tenant field was invented.
    for m in _rows(store):
        assert "client" not in m and "tenant" not in m and "company_id" not in m


def test_attachment_identity_still_fresh_per_ingest(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openexecutive.integrations import attachments as att

    set_active_client(company_root, "acme")
    monkeypatch.setattr(
        "openexecutive.knowledge.store.ChromaDBStore", lambda *a, **k: store
    )

    async def run() -> None:
        att._schedule_ingest(BODY, "n.md")
        await asyncio.gather(*list(att._ingest_tasks))
        att._schedule_ingest(BODY, "n.md")
        await asyncio.gather(*list(att._ingest_tasks))

    asyncio.run(run())
    doc_ids = {m.get("document_id") for m in _rows(store)}
    assert len(doc_ids) == 2
    assert all(str(d).startswith("attachment::") for d in doc_ids)


def test_stale_upload_does_not_fire_the_alerts_pipeline(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 409 means nothing was ingested, so nothing should be announced.

    The alerts block sits after the guard in the same outer `try`. Pinning this
    guards the control flow: if the guard block were ever reordered or the
    rejection downgraded to a soft skip, an alert would fire for a document that
    does not exist in this company.
    """
    fired: list[Any] = []
    monkeypatch.setattr(
        "openexecutive.alerts.pipeline.schedule_evaluation",
        lambda event: fired.append(event),
    )
    set_active_client(company_root, "acme")

    from openexecutive.api.routes import documents as documents_route

    app = FastAPI()
    install_company_context_middleware(app)
    app.include_router(documents_route.router)
    app.state.store = store

    from starlette.datastructures import UploadFile as StarletteUploadFile

    original_read = StarletteUploadFile.read

    async def read_then_switch(self: Any, size: int = -1) -> bytes:
        data = await original_read(self, size)
        set_active_client(company_root, "globex")
        return data

    monkeypatch.setattr(StarletteUploadFile, "read", read_then_switch)

    with TestClient(app) as client:
        assert _upload(client, "x.md", BODY).status_code == 409

    assert fired == [], "an alert was scheduled for a document that was never ingested"


def test_successful_upload_still_fires_the_alerts_pipeline(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inverse — the guard must not have broken the normal announcement."""
    fired: list[Any] = []
    monkeypatch.setattr(
        "openexecutive.alerts.pipeline.schedule_evaluation",
        lambda event: fired.append(event),
    )
    set_active_client(company_root, "acme")

    from openexecutive.api.routes import documents as documents_route

    app = FastAPI()
    install_company_context_middleware(app)
    app.include_router(documents_route.router)
    app.state.store = store

    with TestClient(app) as client:
        assert _upload(client, "plan.md", BODY).status_code == 200

    assert len(fired) == 1


# ---------------------------------------------------------------------------
# Capture must beat the body transfer, not follow it.
# ---------------------------------------------------------------------------


def test_capture_runs_before_the_multipart_body_is_parsed(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordering the whole guard depends on.

    FastAPI spools the entire multipart body during dependency resolution, so a
    handler-level capture happens AFTER the transfer — and a switch during a
    slow 50MB upload would then be captured as the origin, compare equal to
    itself, and let the write through. Only middleware runs early enough.
    """
    import starlette.formparsers as fp

    order: list[str] = []
    set_active_client(company_root, "acme")

    real_capture = capture_company_context

    def spy_capture(*a: Any, **k: Any) -> CompanyContext:
        order.append("capture")
        return real_capture(*a, **k)

    monkeypatch.setattr(
        "openexecutive.clients.context_guard.capture_company_context", spy_capture
    )

    real_parse = fp.MultiPartParser.parse

    async def spy_parse(self: Any) -> Any:
        order.append("body-parse")
        return await real_parse(self)

    monkeypatch.setattr(fp.MultiPartParser, "parse", spy_parse)

    app = _app(store, monkeypatch)
    with TestClient(app) as client:
        assert _upload(client, "plan.md", BODY).status_code == 200

    assert "capture" in order and "body-parse" in order
    assert order.index("capture") < order.index("body-parse"), (
        f"capture ran after the body was parsed: {order}"
    )


def test_switch_during_body_transfer_is_rejected(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real-world window: a slow client uploading while the rotation switches.

    The switch is injected inside multipart parsing — where the transfer
    actually happens — not at `file.read()`, which by then reads a local spool.
    """
    import starlette.formparsers as fp

    set_active_client(company_root, "acme")
    real_parse = fp.MultiPartParser.parse
    switched: list[bool] = []

    async def parse_then_switch(self: Any) -> Any:
        result = await real_parse(self)
        set_active_client(company_root, "globex")  # rotation lands mid-transfer
        switched.append(True)
        return result

    monkeypatch.setattr(fp.MultiPartParser, "parse", parse_then_switch)

    app = _app(store, monkeypatch)
    with TestClient(app) as client:
        resp = _upload(client, "A-secret.md", BODY)

    assert switched, "the switch never fired"
    assert resp.status_code == 409, resp.text
    assert _filenames(store) == set(), "document leaked into the new company"
    docs = company_root / "docs"
    assert not docs.exists() or list(docs.iterdir()) == []


# ---------------------------------------------------------------------------
# Fail closed when the context cannot be read.
# ---------------------------------------------------------------------------


def test_unreadable_sentinel_never_compares_equal(company_root: Path) -> None:
    """Two unreadable reads must NOT look like a matching (None, None) pair."""
    sentinel = company_root / "_client_slots" / ".active_client"
    sentinel.write_text("Not A Valid Slug!!")  # exists, fails the safe-name check

    a = capture_company_context()
    b = capture_company_context()
    assert a != b, "unreadable contexts compared equal — the guard would pass"
    assert a != CompanyContext(None, None)
    assert a.unreadable_id is not None


def test_unreadable_sentinel_rejects_the_commit(company_root: Path) -> None:
    sentinel = company_root / "_client_slots" / ".active_client"
    sentinel.write_text("!!bad!!")
    origin = capture_company_context()

    async def run() -> None:
        async with company_mutation_guard(origin):
            pass

    with pytest.raises(StaleCompanyContextError):
        asyncio.run(run())


def test_genuine_single_company_mode_still_passes(company_root: Path) -> None:
    """No sentinel at all is a real context, not an unreadable one."""
    set_active_client(company_root, None)
    origin = capture_company_context()
    assert origin == CompanyContext(None, None)
    assert origin.unreadable_id is None

    async def run() -> None:
        async with company_mutation_guard(origin):
            pass

    asyncio.run(run())


# ---------------------------------------------------------------------------
# The alert must be scheduled under the owning company.
# ---------------------------------------------------------------------------


def test_alert_scheduling_happens_inside_the_guard(
    company_root: Path, store: ChromaDBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The alert is LLM-triaged then DISPATCHED to the live client's channels.

    Scheduled outside the guard, a switch during triage would push a headline
    derived from this company's confidential document to another company's
    humans. Asserted structurally: the lock is held at scheduling time.
    """
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    held: list[bool] = []
    set_active_client(company_root, "acme")

    app = _app(store, monkeypatch)
    # AFTER _app, which installs its own no-op patch on the same target.
    monkeypatch.setattr(
        "openexecutive.alerts.pipeline.schedule_evaluation",
        lambda event: held.append(_FIXTURE_OP_LOCK.locked()),
    )
    with TestClient(app) as client:
        assert _upload(client, "plan.md", BODY).status_code == 200

    assert held == [True], "alert was scheduled outside the company guard"


# ---------------------------------------------------------------------------
# executive_research — the second company-bound writer in this class.
# ---------------------------------------------------------------------------


class _ResearchStore:
    """Records the research collection's writes and deletes."""

    def __init__(self) -> None:
        self.added: list[dict[str, Any]] = []
        self.deletes: list[tuple[str, dict[str, Any]]] = []

    def add_documents(
        self, texts: list[str], metadatas: list[dict[str, Any]], ids: list[str],
        collection: str,
    ) -> None:
        self.added.extend(metadatas)

    def delete_documents(
        self, collection: str, where: dict[str, Any], *, strict: bool = False
    ) -> None:
        self.deletes.append((collection, where))


async def _persist_research(
    origin: CompanyContext, store: _ResearchStore, artifact: str = "# Research\nbody"
) -> str:
    """The exact guarded commit from `ExecutiveResearchWorkflow.run`.

    Mirrored rather than driven through the whole workflow because the real run
    needs a 7-specialist LLM fan-out; the property under test is the guard
    around the trailing persistence, which is reproduced here statement for
    statement (delete THEN ingest, both inside one guard).
    """
    from datetime import UTC, datetime

    from openexecutive.clients.context_guard import company_mutation_guard
    from openexecutive.knowledge.loader import ingest_text
    from openexecutive.knowledge.store import ChromaDBStore

    now = datetime(2026, 8, 31, tzinfo=UTC)
    try:
        async with company_mutation_guard(origin, operation="executive research"):
            store.delete_documents(
                ChromaDBStore.RESEARCH_COLLECTION, where={"type": "recent_research"}
            )
            await ingest_text(
                artifact, store,
                source_name=f"recent_research_{now.date().isoformat()}",
                collection=ChromaDBStore.RESEARCH_COLLECTION,
                extra_metadata={"type": "recent_research", "created_at": now.isoformat()},
            )
    except StaleCompanyContextError:
        return "stale"
    return "persisted"


def test_research_started_under_a_does_not_persist_into_b(
    company_root: Path,
) -> None:
    """A: capture under A, switch to B mid-run, resume — nothing lands in B."""
    set_active_client(company_root, "acme")
    origin = capture_company_context()          # origin, before the long awaits
    store = _ResearchStore()

    set_active_client(company_root, "globex")   # rotation switches mid-research

    assert asyncio.run(_persist_research(origin, store)) == "stale"
    assert store.added == [], "A's research artifact landed in B"
    assert store.deletes == [], "A's stale run wiped B's research"


def test_research_under_unchanged_company_persists_as_before(
    company_root: Path,
) -> None:
    """B: baseline behaviour is untouched when nothing switches."""
    set_active_client(company_root, "acme")
    origin = capture_company_context()
    store = _ResearchStore()

    assert asyncio.run(_persist_research(origin, store)) == "persisted"
    assert store.added, "research was not persisted under an unchanged company"
    assert all(m["type"] == "recent_research" for m in store.added)
    assert store.deletes == [
        (ChromaDBStore.RESEARCH_COLLECTION, {"type": "recent_research"})
    ]


def test_research_a_to_none_does_not_persist(company_root: Path) -> None:
    """C."""
    set_active_client(company_root, "acme")
    origin = capture_company_context()
    set_active_client(company_root, None)
    store = _ResearchStore()

    assert asyncio.run(_persist_research(origin, store)) == "stale"
    assert store.added == [] and store.deletes == []


def test_research_none_to_b_does_not_persist(company_root: Path) -> None:
    """D."""
    set_active_client(company_root, None)
    origin = capture_company_context()
    set_active_client(company_root, "globex")
    store = _ResearchStore()

    assert asyncio.run(_persist_research(origin, store)) == "stale"
    assert store.added == [] and store.deletes == []


def test_research_persistence_exception_releases_the_guard(
    company_root: Path,
) -> None:
    """E: a failing write must not strand the company-state lock."""
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    set_active_client(company_root, "acme")
    origin = capture_company_context()

    class Boom(_ResearchStore):
        def add_documents(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("chroma down")

    async def run() -> None:
        with pytest.raises(RuntimeError):
            await _persist_research(origin, Boom())
        assert not _FIXTURE_OP_LOCK.locked()

    asyncio.run(run())


def test_research_stale_rejection_releases_the_guard(company_root: Path) -> None:
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    set_active_client(company_root, "acme")
    origin = capture_company_context()
    set_active_client(company_root, "globex")

    async def run() -> None:
        assert await _persist_research(origin, _ResearchStore()) == "stale"
        assert not _FIXTURE_OP_LOCK.locked()

    asyncio.run(run())


def test_research_commit_waits_for_an_in_progress_slot_switch(
    company_root: Path,
) -> None:
    """F/G: the nightly rotation path, deterministically.

    The rotation holds `_FIXTURE_OP_LOCK` for the whole switch. A research
    commit that arrives mid-switch must WAIT (not interleave), and then be
    rejected once it sees the new company — proving the shared lock is what
    orders the two.
    """
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    set_active_client(company_root, "acme")
    origin = capture_company_context()
    store = _ResearchStore()

    switch_started = asyncio.Event()
    commit_attempted = asyncio.Event()

    async def rotation() -> None:
        async with _FIXTURE_OP_LOCK:
            switch_started.set()
            await commit_attempted.wait()       # research is queued on the lock
            set_active_client(company_root, "globex")

    async def research() -> str:
        await switch_started.wait()
        commit_attempted.set()
        return await _persist_research(origin, store)

    async def both() -> Any:
        return await asyncio.gather(rotation(), research())

    _, outcome = asyncio.run(both())
    assert outcome == "stale"
    assert store.added == [] and store.deletes == []


def test_research_guard_removal_reopens_the_race(company_root: Path) -> None:
    """G (mutation): without the guard, A's research lands in B."""
    from datetime import UTC, datetime

    from openexecutive.knowledge.loader import ingest_text
    from openexecutive.knowledge.store import ChromaDBStore

    set_active_client(company_root, "acme")
    capture_company_context()
    set_active_client(company_root, "globex")
    store = _ResearchStore()

    async def unguarded() -> None:  # MUTATION: no company_mutation_guard
        now = datetime(2026, 8, 31, tzinfo=UTC)
        store.delete_documents(
            ChromaDBStore.RESEARCH_COLLECTION, where={"type": "recent_research"}
        )
        await ingest_text(
            "# Research\nbody", store,
            source_name=f"recent_research_{now.date().isoformat()}",
            collection=ChromaDBStore.RESEARCH_COLLECTION,
            extra_metadata={"type": "recent_research", "created_at": now.isoformat()},
        )

    asyncio.run(unguarded())
    assert store.added, "the unguarded path should have leaked — mutation is inert"
    assert store.deletes, "the unguarded delete should have wiped B's research"


def test_research_workflow_captures_origin_before_its_awaits() -> None:
    """Structural: capture must precede the specialist fan-out in the real file."""
    source = (
        Path(__file__).resolve().parents[2]
        / "openexecutive" / "workflows" / "executive_research.py"
    ).read_text()
    capture_at = source.index("origin = capture_company_context()")
    first_await = source.index("await research_one_specialist")
    guard_at = source.index('operation="executive research"')
    assert capture_at < first_await, "origin captured after the long awaits"
    assert first_await < guard_at, "guard does not wrap the trailing commit"
    # And the guard must not wrap the LLM loops.
    assert "async with company_mutation_guard" not in source[:first_await]


# ---------------------------------------------------------------------------
# Alert triage — the proven cross-company disclosure to another client's humans.
# ---------------------------------------------------------------------------


def _alert_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Wire the alert pipeline to isolated state and record every side effect."""
    from openexecutive.alerts import pipeline as ap
    from openexecutive.alerts.models import TriageDecision

    inserted: list[dict[str, Any]] = []
    dispatched: list[Any] = []

    monkeypatch.setattr(ap, "_rate_limited", lambda: False)
    monkeypatch.setattr(ap.preferences, "get_preferences", lambda **k: object())
    monkeypatch.setattr(ap.preferences, "resolve_channels", lambda *a, **k: ["slack"])
    monkeypatch.setattr(ap.preferences, "matches_mute", lambda *a, **k: False)
    monkeypatch.setattr(ap.store, "list_mutes", lambda **k: [])
    monkeypatch.setattr(ap.store, "recent_alerts", lambda **k: [])
    monkeypatch.setattr(
        "openexecutive.memory.episodic.get_active_initiatives", lambda: []
    )

    def fake_insert(**kwargs: Any) -> int:
        inserted.append(kwargs)
        return 1

    monkeypatch.setattr(ap.store, "insert_alert", fake_insert)
    monkeypatch.setattr(ap.store, "get_alert", lambda aid, **k: object())

    async def fake_dispatch(alert: Any, channels: Any, **k: Any) -> None:
        dispatched.append(channels)

    monkeypatch.setattr(ap.dispatcher, "dispatch_all", fake_dispatch)

    decision = TriageDecision(
        alert=True, dedup_key="d1", headline="A-CONFIDENTIAL", body="from A's document"
    )
    return {"inserted": inserted, "dispatched": dispatched, "decision": decision}


def _triage_gate(
    monkeypatch: pytest.MonkeyPatch, decision: Any, on_triage: Any = None
) -> None:
    """Replace the LLM triage await with a barrier-controlled coroutine."""
    from openexecutive.agents import triage as triage_mod

    async def fake_triage(self: Any, event: Any, **kwargs: Any) -> Any:
        if on_triage is not None:
            await on_triage()
        return decision

    monkeypatch.setattr(triage_mod.TriageAgent, "triage", fake_triage)


def _event() -> Any:
    from openexecutive.alerts.models import AlertEvent

    return AlertEvent(source="document", external_id="A-secret.md",
                      title="A-secret.md", body="A confidential body")


def test_alert_scheduled_under_a_is_not_inserted_or_dispatched_after_switch(
    company_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A: the disclosure. Switch lands during triage; nothing reaches B."""
    from openexecutive.alerts import pipeline as ap

    env = _alert_env(tmp_path, monkeypatch)
    set_active_client(company_root, "acme")
    origin = capture_company_context()

    async def switch_mid_triage() -> None:
        set_active_client(company_root, "globex")  # rotation completes

    _triage_gate(monkeypatch, env["decision"], on_triage=switch_mid_triage)

    decision, alert_id = asyncio.run(
        ap.evaluate_and_dispatch(_event(), db_path=tmp_path / "x.db", origin=origin)
    )

    assert alert_id is None
    assert env["inserted"] == [], "A-derived alert row was written into B's DB"
    assert env["dispatched"] == [], "A-derived alert was dispatched to B's channels"


def test_alert_under_unchanged_company_behaves_exactly_as_before(
    company_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B: no switch — insert and dispatch happen as they always did."""
    from openexecutive.alerts import pipeline as ap

    env = _alert_env(tmp_path, monkeypatch)
    set_active_client(company_root, "acme")
    origin = capture_company_context()
    _triage_gate(monkeypatch, env["decision"])

    decision, alert_id = asyncio.run(
        ap.evaluate_and_dispatch(_event(), db_path=tmp_path / "x.db", origin=origin)
    )

    assert alert_id == 1
    assert len(env["inserted"]) == 1
    assert env["dispatched"] == [["slack"]]
    assert env["inserted"][0]["headline"] == "A-CONFIDENTIAL"


@pytest.mark.parametrize("start,switch", [("acme", None), (None, "globex")])
def test_alert_none_transitions_reject(
    company_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    start: str | None, switch: str | None,
) -> None:
    """C and D."""
    from openexecutive.alerts import pipeline as ap

    env = _alert_env(tmp_path, monkeypatch)
    set_active_client(company_root, start)
    origin = capture_company_context()

    async def do_switch() -> None:
        set_active_client(company_root, switch)

    _triage_gate(monkeypatch, env["decision"], on_triage=do_switch)

    _decision, alert_id = asyncio.run(
        ap.evaluate_and_dispatch(_event(), db_path=tmp_path / "x.db", origin=origin)
    )
    assert alert_id is None
    assert env["inserted"] == [] and env["dispatched"] == []


def test_alert_dispatch_exception_releases_the_shared_lock(
    company_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E."""
    from openexecutive.alerts import pipeline as ap
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    env = _alert_env(tmp_path, monkeypatch)
    set_active_client(company_root, "acme")
    origin = capture_company_context()
    _triage_gate(monkeypatch, env["decision"])

    async def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("slack down")

    monkeypatch.setattr(ap.dispatcher, "dispatch_all", boom)

    async def run() -> None:
        with pytest.raises(RuntimeError):
            await ap.evaluate_and_dispatch(
                _event(), db_path=tmp_path / "x.db", origin=origin
            )
        assert not _FIXTURE_OP_LOCK.locked()

    asyncio.run(run())


def test_alert_cancellation_releases_the_shared_lock(
    company_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F."""
    from openexecutive.alerts import pipeline as ap
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    env = _alert_env(tmp_path, monkeypatch)
    set_active_client(company_root, "acme")
    origin = capture_company_context()
    _triage_gate(monkeypatch, env["decision"])

    inside = asyncio.Event()

    async def block(*a: Any, **k: Any) -> None:
        inside.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(ap.dispatcher, "dispatch_all", block)

    async def run() -> None:
        task = asyncio.create_task(
            ap.evaluate_and_dispatch(_event(), db_path=tmp_path / "x.db", origin=origin)
        )
        await inside.wait()
        assert _FIXTURE_OP_LOCK.locked()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not _FIXTURE_OP_LOCK.locked()

    asyncio.run(run())


def test_switch_cannot_land_between_insert_and_dispatch(
    company_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G: no TOCTOU window — a switch attempted mid-commit must WAIT."""
    from openexecutive.alerts import pipeline as ap
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK

    env = _alert_env(tmp_path, monkeypatch)
    set_active_client(company_root, "acme")
    origin = capture_company_context()
    _triage_gate(monkeypatch, env["decision"])

    at_insert = asyncio.Event()
    switch_tried = asyncio.Event()
    observed: dict[str, Any] = {}

    def insert_then_pause(**kwargs: Any) -> int:
        env["inserted"].append(kwargs)
        at_insert.set()
        return 1

    monkeypatch.setattr(ap.store, "insert_alert", insert_then_pause)

    async def dispatch_after_switch_attempt(alert: Any, channels: Any, **k: Any) -> None:
        await switch_tried.wait()
        observed["switch_got_lock_before_dispatch"] = not _FIXTURE_OP_LOCK.locked()
        env["dispatched"].append(channels)

    monkeypatch.setattr(ap.dispatcher, "dispatch_all", dispatch_after_switch_attempt)

    async def rotation() -> None:
        await at_insert.wait()
        switch_tried.set()
        # The switch must queue on the lock the commit is holding.
        async with _FIXTURE_OP_LOCK:
            observed["switch_ran_after_dispatch"] = bool(env["dispatched"])
            set_active_client(company_root, "globex")

    async def both() -> Any:
        return await asyncio.gather(
            ap.evaluate_and_dispatch(_event(), db_path=tmp_path / "x.db", origin=origin),
            rotation(),
        )

    asyncio.run(asyncio.wait_for(both(), timeout=5))
    assert env["dispatched"], "dispatch did not run"
    assert observed["switch_ran_after_dispatch"] is True, (
        "the switch interleaved between insert and dispatch"
    )


def test_mutation_removing_the_alert_guard_reopens_the_disclosure(
    company_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H: origin=None is the pre-guard behaviour — the leak returns."""
    from openexecutive.alerts import pipeline as ap

    env = _alert_env(tmp_path, monkeypatch)
    set_active_client(company_root, "acme")

    async def switch_mid_triage() -> None:
        set_active_client(company_root, "globex")

    _triage_gate(monkeypatch, env["decision"], on_triage=switch_mid_triage)

    _d, alert_id = asyncio.run(
        ap.evaluate_and_dispatch(_event(), db_path=tmp_path / "x.db", origin=None)
    )
    assert alert_id == 1
    assert env["inserted"] and env["dispatched"], (
        "unguarded path should have leaked — the mutation is inert"
    )


def test_mutation_capturing_inside_the_task_defeats_the_guard(
    company_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I: capturing after triage observes B and passes — why capture is at schedule."""
    from openexecutive.alerts import pipeline as ap

    env = _alert_env(tmp_path, monkeypatch)
    set_active_client(company_root, "acme")

    async def switch_mid_triage() -> None:
        set_active_client(company_root, "globex")

    _triage_gate(monkeypatch, env["decision"], on_triage=switch_mid_triage)

    async def run() -> Any:
        # MUTATION: origin read INSIDE the task, i.e. after the LLM await has
        # already let the switch complete — exactly what a naive
        # `origin = capture_company_context()` inside evaluate_and_dispatch does.
        await switch_mid_triage()
        late_origin = capture_company_context()   # reads globex, not acme
        return await ap.evaluate_and_dispatch(
            _event(), db_path=tmp_path / "x.db", origin=late_origin
        )

    _d, alert_id = asyncio.run(run())
    assert alert_id == 1, "late capture should pass — proving schedule-time matters"
    assert env["dispatched"], "late capture should have allowed the dispatch"


def test_schedule_evaluation_captures_origin_and_passes_it_to_the_task(
    company_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Propagation: the detached task receives the schedule-time context."""
    from openexecutive.alerts import pipeline as ap

    set_active_client(company_root, "acme")
    seen: dict[str, Any] = {}

    async def fake_eval(event: Any, db_path: Any = None, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return None, None

    monkeypatch.setattr(ap, "evaluate_and_dispatch", fake_eval)

    async def run() -> None:
        ap.schedule_evaluation(_event())
        set_active_client(company_root, "globex")  # switch AFTER scheduling
        await asyncio.gather(*list(ap._background_tasks))

    asyncio.run(run())
    assert seen["origin"].client == "acme", "origin was not captured at schedule time"


def test_alert_callers_still_call_schedule_evaluation_unchanged() -> None:
    """All six call sites keep a single-argument signature."""
    import inspect

    from openexecutive.alerts import pipeline as ap

    params = inspect.signature(ap.schedule_evaluation).parameters
    assert list(params) == ["event"], "schedule_evaluation's signature changed"


# ---------------------------------------------------------------------------
# Per-call tool guard — outbound tools run BETWEEN provider awaits.
# ---------------------------------------------------------------------------


class _Block:
    def __init__(self, name: str, tool_input: dict[str, Any] | None = None) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = tool_input or {}


class _Response:
    def __init__(self, *blocks: Any) -> None:
        self.content = list(blocks)
        self.stop_reason = "tool_use"


def _mk(name: str, calls: list[str]) -> Any:
    """A handler that records that it ran — standing in for a real send/write."""
    async def handler(tool_input: dict[str, Any]) -> str:
        calls.append(name)
        return '{"ok": true}'

    return handler


def _handlers(calls: list[str]) -> dict[str, Any]:
    return {
        name: _mk(name, calls)
        for name in (
            "send_department_message", "create_alert", "add_watchlist_entry",
            "schedule_followup", "lookup_person", "list_people",
        )
    }


async def _run_tools(
    origin: CompanyContext | None, tools: list[str], calls: list[str]
) -> list[dict[str, Any]]:
    from openexecutive.workflows._synthesis import execute_tool_calls

    return await execute_tool_calls(
        _Response(*[_Block(t) for t in tools]),
        _handlers(calls),
        origin_company_context=origin,
    )


def test_tool_call_after_switch_sends_nothing_to_b(company_root: Path) -> None:
    """A: the disclosure — a department message must not reach B."""
    set_active_client(company_root, "acme")
    origin = capture_company_context()
    set_active_client(company_root, "globex")      # switch during a provider await

    calls: list[str] = []
    summaries = asyncio.run(_run_tools(origin, ["send_department_message"], calls))

    assert calls == [], "a message was sent into the new company"
    assert len(summaries) == 1
    assert summaries[0]["ok"] is False
    assert "active company changed" in summaries[0]["result_preview"]


def test_tool_call_under_same_company_is_unchanged(company_root: Path) -> None:
    """B."""
    set_active_client(company_root, "acme")
    origin = capture_company_context()

    calls: list[str] = []
    summaries = asyncio.run(_run_tools(origin, ["send_department_message"], calls))

    assert calls == ["send_department_message"]
    assert summaries[0]["ok"] is True


@pytest.mark.parametrize("start,switch", [("acme", None), (None, "globex")])
def test_tool_call_none_transitions_are_skipped(
    company_root: Path, start: str | None, switch: str | None
) -> None:
    """C and D."""
    set_active_client(company_root, start)
    origin = capture_company_context()
    set_active_client(company_root, switch)

    calls: list[str] = []
    asyncio.run(_run_tools(origin, ["send_department_message"], calls))
    assert calls == []


@pytest.mark.parametrize(
    "tool", ["create_alert", "add_watchlist_entry", "schedule_followup"]
)
def test_company_bound_writes_are_blocked_after_switch(
    company_root: Path, tool: str
) -> None:
    """E and F: alert creation, watchlist and schedule mutations."""
    set_active_client(company_root, "acme")
    origin = capture_company_context()
    set_active_client(company_root, "globex")

    calls: list[str] = []
    asyncio.run(_run_tools(origin, [tool], calls))
    assert calls == [], f"{tool} executed under the new company"


@pytest.mark.parametrize("tool", ["lookup_person", "list_people"])
def test_read_only_tools_still_run_after_a_switch(
    company_root: Path, tool: str
) -> None:
    """Read-only lookups are not company-bound writes and must not be blocked
    (nor made to queue behind a slot switch)."""
    set_active_client(company_root, "acme")
    origin = capture_company_context()
    set_active_client(company_root, "globex")

    calls: list[str] = []
    asyncio.run(_run_tools(origin, [tool], calls))
    assert calls == [tool]


def test_rotation_waits_for_an_in_flight_protected_handler(
    company_root: Path,
) -> None:
    """G: a switch attempted DURING a protected handler queues on the lock."""
    from openexecutive.cli.fixture_loader import _FIXTURE_OP_LOCK
    from openexecutive.workflows._synthesis import execute_tool_calls

    set_active_client(company_root, "acme")
    origin = capture_company_context()

    in_handler = asyncio.Event()
    rotation_tried = asyncio.Event()
    order: list[str] = []

    async def slow_send(tool_input: dict[str, Any]) -> str:
        in_handler.set()
        await rotation_tried.wait()
        order.append("handler-done")
        return '{"ok": true}'

    async def rotation() -> None:
        await in_handler.wait()
        rotation_tried.set()
        async with _FIXTURE_OP_LOCK:            # must wait for the handler
            order.append("switch")
            set_active_client(company_root, "globex")

    async def both() -> Any:
        return await asyncio.gather(
            execute_tool_calls(
                _Response(_Block("send_department_message")),
                {"send_department_message": slow_send},
                origin_company_context=origin,
            ),
            rotation(),
        )

    asyncio.run(asyncio.wait_for(both(), timeout=5))
    assert order == ["handler-done", "switch"], f"switch interleaved: {order}"


def test_mutation_no_origin_reproduces_the_disclosure(company_root: Path) -> None:
    """H: without the origin (pre-guard behaviour) the send goes through."""
    set_active_client(company_root, "acme")
    capture_company_context()
    set_active_client(company_root, "globex")

    calls: list[str] = []
    asyncio.run(_run_tools(None, ["send_department_message"], calls))
    assert calls == ["send_department_message"], (
        "unguarded path should have leaked — the mutation is inert"
    )


def test_mutation_recapturing_at_handler_time_defeats_the_guard(
    company_root: Path,
) -> None:
    """I: context read at handler time observes B and passes."""
    set_active_client(company_root, "acme")
    set_active_client(company_root, "globex")     # switch already happened
    late_origin = capture_company_context()       # MUTATION: recaptured here

    calls: list[str] = []
    asyncio.run(_run_tools(late_origin, ["send_department_message"], calls))
    assert calls == ["send_department_message"], (
        "late capture should pass — proving run()-time capture is what protects"
    )


def test_read_only_allowlist_fails_closed_for_unknown_tools(
    company_root: Path,
) -> None:
    """A handler added later must be guarded by DEFAULT, not silently exempt."""
    from openexecutive.workflows._synthesis import READ_ONLY_TOOLS

    set_active_client(company_root, "acme")
    origin = capture_company_context()
    set_active_client(company_root, "globex")

    calls: list[str] = []

    async def newly_added(tool_input: dict[str, Any]) -> str:
        calls.append("a_future_handler")
        return '{"ok": true}'

    from openexecutive.workflows._synthesis import execute_tool_calls

    asyncio.run(execute_tool_calls(
        _Response(_Block("a_future_handler")),
        {"a_future_handler": newly_added},
        origin_company_context=origin,
    ))
    assert "a_future_handler" not in READ_ONLY_TOOLS
    assert calls == [], "an unclassified handler ran under the wrong company"


def test_read_only_allowlist_contains_no_writers() -> None:
    """Every allowlisted tool must genuinely be read-only."""
    import inspect
    import re

    from openexecutive.orchestrator.executive import _ALL_SKILL_HANDLERS
    from openexecutive.workflows._synthesis import READ_ONLY_TOOLS

    writer = re.compile(
        r"insert_alert|update_\w+\(|delete_\w+\(|upsert_|write_text|write_bytes"
        r"|dispatch_all|schedule_evaluation|send_message|\.commit\(",
        re.I,
    )
    for name in sorted(READ_ONLY_TOOLS):
        handler = _ALL_SKILL_HANDLERS.get(name)
        if handler is None:
            continue
        src = inspect.getsource(handler)
        hits = sorted({m.group(0) for m in writer.finditer(src)})
        assert not hits, f"{name} is allowlisted read-only but writes: {hits}"


def test_execute_tool_calls_signature_is_additive() -> None:
    """Existing callers (executive_reflection) must be unaffected."""
    import inspect

    from openexecutive.workflows._synthesis import execute_tool_calls

    params = inspect.signature(execute_tool_calls).parameters
    assert params["origin_company_context"].default is None
    assert params["origin_company_context"].kind is inspect.Parameter.KEYWORD_ONLY


def test_reflection_caller_passes_no_origin_and_is_unguarded() -> None:
    """executive_reflection did not opt in; company semantics must not apply."""
    source = (
        Path(__file__).resolve().parents[2]
        / "openexecutive" / "workflows" / "executive_reflection.py"
    ).read_text()
    assert "origin_company_context" not in source
