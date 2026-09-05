"""Scope isolation, the ownership foreign key, concurrency, and slot lifecycle.

The centre of this file is one claim: a document version can never be attached
to a logical source owned by a different scope -- not through the repository,
and not through raw SQL either.
"""
from __future__ import annotations

import ast
import sqlite3
import threading
from pathlib import Path

import pytest

from openexecutive.evidence.registry import (
    REGISTRY_TABLES,
    EvidenceRegistry,
    RegistryError,
    ScopeBindingKind,
    initialize_evidence_registry,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "openexecutive"


class ProxyConnection:
    """Delegating wrapper around a real connection.

    ``sqlite3.Connection`` is an immutable type, so its methods cannot be
    patched. Injecting this through ``registry.sqlite3.connect`` is the only way
    to observe or perturb the statements the registry issues.
    """

    def __init__(self, conn, on_execute=None):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_on_execute", on_execute)

    def execute(self, sql, *args, **kwargs):
        hook = object.__getattribute__(self, "_on_execute")
        if hook is not None:
            replacement = hook(sql)
            if replacement is not None:
                return replacement
        return object.__getattribute__(self, "_conn").execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_conn"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_conn"), name, value)


class Rows:
    """Minimal stand-in for a cursor returning one fixed row."""

    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


def raw(db_path, *, foreign_keys=True):
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    if foreign_keys:
        conn.execute("PRAGMA foreign_keys = ON")
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    return conn


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "registry.db"
    initialize_evidence_registry(path)
    return path


@pytest.fixture
def reg(db):
    return EvidenceRegistry(db)


@pytest.fixture
def two_scopes(reg):
    """Scope A and scope B, each owning exactly one logical source."""
    a, _ = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key="alpha", display_name="Alpha"
    )
    b, _ = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key="bravo", display_name="Bravo"
    )
    ls_a = reg.create_logical_source(scope_id=a.scope_id, display_label="A doc")
    ls_b = reg.create_logical_source(scope_id=b.scope_id, display_label="B doc")
    return a, b, ls_a, ls_b


# ── the ownership invariant ─────────────────────────────────────────────


def test_raw_cross_scope_insert_is_rejected_by_the_database(db, two_scopes):
    """Scope A + a logical source owned by B. Two independent foreign keys would
    both be satisfied; the composite one makes the row unrepresentable."""
    a, _b, _ls_a, ls_b = two_scopes
    conn = raw(db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO evidence_document_versions VALUES (?,?,?,?,?,?,NULL)",
            ("forged", a.scope_id, ls_b.logical_source_id, "a" * 64, 1, "2026-01-01"),
        )
    assert conn.execute("SELECT count(*) FROM evidence_document_versions").fetchone()[0] == 0


def test_the_matching_in_scope_insert_is_still_accepted(db, two_scopes):
    """Rejection above must be specific to the mismatch, not a blanket refusal."""
    a, _b, ls_a, _ls_b = two_scopes
    conn = raw(db)
    conn.execute(
        "INSERT INTO evidence_document_versions VALUES (?,?,?,?,?,?,NULL)",
        ("ok", a.scope_id, ls_a.logical_source_id, "a" * 64, 1, "2026-01-01"),
    )
    assert conn.execute("SELECT count(*) FROM evidence_document_versions").fetchone()[0] == 1


def test_typed_scope_mismatch_is_raised_before_any_insert(db, two_scopes, monkeypatch):
    """The database constraint is the backstop; the typed error is the contract.
    Nothing may reach a DML statement on the versions table on this path."""
    from openexecutive.evidence import registry

    a, _b, _ls_a, ls_b = two_scopes
    reg = EvidenceRegistry(db)
    statements: list[str] = []
    original = sqlite3.connect

    def spy(*args, **kwargs):
        return ProxyConnection(original(*args, **kwargs), on_execute=statements.append)

    monkeypatch.setattr(registry.sqlite3, "connect", spy)
    with pytest.raises(RegistryError) as exc:
        reg.register_document_version(
            scope_id=a.scope_id, logical_source_id=ls_b.logical_source_id, raw_bytes=b"x"
        )

    assert exc.value.check == "scope_mismatch"
    assert statements, "the spy saw nothing -- the test would pass vacuously"
    assert not [
        sql for sql in statements
        if "INSERT" in sql.upper() and "evidence_document_versions" in sql
    ]
    assert not [sql for sql in statements if sql.strip() == "BEGIN IMMEDIATE"]


def test_the_composite_foreign_key_is_present_in_the_live_schema(db):
    conn = raw(db)
    fks = conn.execute("PRAGMA foreign_key_list(evidence_document_versions)").fetchall()
    by_id: dict[int, list[sqlite3.Row]] = {}
    for row in fks:
        by_id.setdefault(row["id"], []).append(row)

    composite = [
        rows for rows in by_id.values()
        if len(rows) == 2
        and {r["from"] for r in rows} == {"logical_source_id", "scope_id"}
        and {r["to"] for r in rows} == {"logical_source_id", "scope_id"}
        and rows[0]["table"] == "evidence_logical_sources"
    ]
    assert len(composite) == 1, "composite ownership FK missing"

    singles = [rows for rows in by_id.values() if len(rows) == 1]
    assert [r[0]["from"] for r in singles] == ["scope_id"]
    assert singles[0][0]["table"] == "evidence_scopes"
    # The individual logical_source_id FK is deliberately absent: implied by the composite.
    assert not [r for r in singles if r[0]["from"] == "logical_source_id"]


def test_the_unique_parent_key_is_required_not_an_optimisation(tmp_path):
    """Without the parent index SQLite refuses EVERY insert, not just a bad one."""
    from openexecutive.evidence import registry

    path = tmp_path / "noparent.db"
    schema = registry._SCHEMA.replace(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_ls_id_scope\n"
        "    ON evidence_logical_sources(logical_source_id, scope_id);",
        "",
    )
    assert "idx_evidence_ls_id_scope" not in schema
    conn = raw(path)
    conn.executescript(schema)
    conn.execute(
        "INSERT INTO evidence_scopes VALUES ('s','client_slot','k','K','t',NULL)"
    )
    conn.execute(
        "INSERT INTO evidence_logical_sources VALUES ('ls','s','key','L','t',NULL)"
    )
    with pytest.raises(sqlite3.OperationalError, match="foreign key mismatch"):
        conn.execute(
            "INSERT INTO evidence_document_versions VALUES (?,?,?,?,?,?,NULL)",
            ("dv", "s", "ls", "a" * 64, 1, "t"),
        )


def test_an_orphaned_logical_source_cannot_propagate_its_ghost_scope(tmp_path):
    """Why the individual scope_id FK is kept beside the composite one."""
    path = tmp_path / "orphan.db"
    initialize_evidence_registry(path)
    conn = raw(path, foreign_keys=False)
    conn.execute(
        "INSERT INTO evidence_logical_sources VALUES ('ls','ghost','key','L','t',NULL)"
    )
    conn.close()

    conn = raw(path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO evidence_document_versions VALUES (?,?,?,?,?,?,NULL)",
            ("dv", "ghost", "ls", "a" * 64, 1, "t"),
        )
    assert conn.execute("PRAGMA foreign_key_check").fetchall()


# ── enforcement is actually on ──────────────────────────────────────────


def test_every_registry_connection_has_foreign_keys_enabled(db, monkeypatch):
    from openexecutive.evidence import registry

    seen: list[int] = []
    original = sqlite3.connect

    def spy(*args, **kwargs):
        conn = original(*args, **kwargs)
        seen.append(id(conn))
        return conn

    monkeypatch.setattr(registry.sqlite3, "connect", spy)
    with registry._connect(db) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.isolation_level is None
    assert seen


def test_registry_fails_closed_when_enforcement_cannot_be_enabled(db, monkeypatch):
    """If foreign keys cannot be turned on, the ownership invariant is not being
    enforced -- so the registry refuses to hand out the connection at all."""
    from openexecutive.evidence import registry

    original = sqlite3.connect

    def refusing(sql):
        if sql.strip() == "PRAGMA foreign_keys":
            return Rows((0,))
        return None

    monkeypatch.setattr(
        registry.sqlite3, "connect",
        lambda *a, **k: ProxyConnection(original(*a, **k), on_execute=refusing),
    )
    with pytest.raises(RegistryError) as exc, registry._connect(db):
        pass

    assert exc.value.check == "storage_unavailable"
    assert str(db) not in str(exc.value)


def test_pragma_is_a_noop_inside_a_transaction(tmp_path):
    """The trap the read-back assertion exists to catch. Documented, not theoretical:
    in the driver's default mode the first DML opens a transaction, and a later
    PRAGMA is ignored silently -- no error, enforcement simply off."""
    path = tmp_path / "trap.db"
    conn = sqlite3.connect(str(path))  # legacy isolation_level
    conn.execute("CREATE TABLE t (a TEXT)")
    conn.execute("INSERT INTO t VALUES ('x')")
    conn.execute("PRAGMA foreign_keys = ON")
    assert conn.in_transaction
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0


# ── scope isolation ─────────────────────────────────────────────────────


def test_identical_bytes_in_two_scopes_get_independent_identities(reg, two_scopes):
    a, b, ls_a, ls_b = two_scopes
    va, _ = reg.register_document_version(
        scope_id=a.scope_id, logical_source_id=ls_a.logical_source_id, raw_bytes=b"same"
    )
    vb, created = reg.register_document_version(
        scope_id=b.scope_id, logical_source_id=ls_b.logical_source_id, raw_bytes=b"same"
    )
    assert created, "no cross-scope dedup: scope B must get its own row"
    assert va.document_version_id != vb.document_version_id
    assert va.content_sha256 == vb.content_sha256


def test_reads_never_cross_a_scope_boundary(reg, two_scopes):
    a, b, ls_a, ls_b = two_scopes
    assert [s.logical_source_id for s in reg.list_logical_sources(scope_id=a.scope_id)] == [
        ls_a.logical_source_id
    ]
    assert [s.logical_source_id for s in reg.list_logical_sources(scope_id=b.scope_id)] == [
        ls_b.logical_source_id
    ]
    assert reg.list_logical_sources(scope_id="nonexistent") == ()


# ── concurrency and rollback ────────────────────────────────────────────


def test_concurrent_identical_registration_creates_exactly_one_row(db, reg):
    scope, _ = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key="race", display_name="Race"
    )
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    barrier = threading.Barrier(6)
    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker():
        local = EvidenceRegistry(db)
        try:
            barrier.wait(timeout=10)
            record, created = local.register_document_version(
                scope_id=scope.scope_id,
                logical_source_id=source.logical_source_id,
                raw_bytes=b"contended",
            )
            with lock:
                results.append((record.document_version_id, created))
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"concurrent registration raised: {errors}"
    assert len(results) == 6
    assert len({r[0] for r in results}) == 1
    assert sum(1 for _, created in results if created) == 1
    assert len(reg.list_document_versions(logical_source_id=source.logical_source_id)) == 1


def test_concurrent_get_or_create_scope_yields_one_scope(db):
    barrier = threading.Barrier(6)
    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker():
        local = EvidenceRegistry(db)
        try:
            barrier.wait(timeout=10)
            record, created = local.get_or_create_scope(
                binding_kind=ScopeBindingKind.CLIENT_SLOT,
                binding_key="contended",
                display_name="C",
            )
            with lock:
                results.append((record.scope_id, created))
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"concurrent scope creation raised: {errors}"
    assert len({r[0] for r in results}) == 1
    assert sum(1 for _, created in results if created) == 1


def test_concurrent_create_logical_source_is_deliberately_not_idempotent(db, reg):
    """Two calls are two sources. Creation is an explicit semantic act."""
    scope, _ = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key="ls", display_name="L"
    )
    barrier = threading.Barrier(4)
    ids: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker():
        local = EvidenceRegistry(db)
        try:
            barrier.wait(timeout=10)
            record = local.create_logical_source(
                scope_id=scope.scope_id, display_label="Same Label"
            )
            with lock:
                ids.append(record.logical_source_id)
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"concurrent creation raised: {errors}"
    assert len(set(ids)) == 4


def test_a_failed_registration_leaves_no_partial_row(db, reg, monkeypatch):
    from openexecutive.evidence import registry

    scope, _ = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key="rb", display_name="RB"
    )
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    before = raw(db).execute(
        "SELECT count(*) FROM evidence_document_versions"
    ).fetchone()[0]

    original = sqlite3.connect

    def explode(sql):
        if "INSERT INTO evidence_document_versions" in sql:
            raise sqlite3.OperationalError("disk I/O error")
        return None

    monkeypatch.setattr(
        registry.sqlite3, "connect",
        lambda *a, **k: ProxyConnection(original(*a, **k), on_execute=explode),
    )
    with pytest.raises(sqlite3.OperationalError):
        reg.register_document_version(
            scope_id=scope.scope_id,
            logical_source_id=source.logical_source_id,
            raw_bytes=b"x",
        )
    monkeypatch.undo()

    assert raw(db).execute(
        "SELECT count(*) FROM evidence_document_versions"
    ).fetchone()[0] == before
    # The rollback released the write lock: a fresh transaction still succeeds.
    reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=source.logical_source_id, raw_bytes=b"y"
    )


# ── durability across the slot save/restore mechanisms ──────────────────


@pytest.mark.parametrize("mechanism", ["vacuum_into", "backup"])
def test_schema_and_constraints_survive_slot_save_and_restore(db, two_scopes, tmp_path, mechanism):
    a, _b, ls_a, ls_b = two_scopes
    reg = EvidenceRegistry(db)
    for payload in (b"1", b"2", b"3"):
        reg.register_document_version(
            scope_id=a.scope_id, logical_source_id=ls_a.logical_source_id, raw_bytes=payload
        )
    expected = [
        r.document_version_id
        for r in reg.list_document_versions(logical_source_id=ls_a.logical_source_id)
    ]

    copy = tmp_path / f"{mechanism}.db"
    src = raw(db)
    if mechanism == "vacuum_into":
        src.execute("VACUUM INTO ?", (str(copy),))
    else:
        dst = sqlite3.connect(str(copy))
        src.backup(dst)
        dst.close()

    restored = EvidenceRegistry(copy)
    assert [
        r.document_version_id
        for r in restored.list_document_versions(logical_source_id=ls_a.logical_source_id)
    ] == expected

    conn = raw(copy)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert "idx_evidence_ls_id_scope" in names
    assert {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'"
    )} >= {"trg_evidence_document_versions_immutable"}

    # The ownership constraint and the triggers must still bite in the copy.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO evidence_document_versions VALUES (?,?,?,?,?,?,NULL)",
            ("forged", a.scope_id, ls_b.logical_source_id, "a" * 64, 1, "t"),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE evidence_document_versions SET content_sha256 = 'b'",
        )


# ── client-slot lifecycle ───────────────────────────────────────────────


def test_blank_slot_wipe_removes_every_registry_row(db, two_scopes):
    from openexecutive.clients.slots import _BLANK_WIPE_TABLES

    for table in REGISTRY_TABLES:
        assert table in _BLANK_WIPE_TABLES, f"{table} missing from the blank-wipe list"
    positions = [_BLANK_WIPE_TABLES.index(t) for t in REGISTRY_TABLES]
    assert positions == sorted(positions), "children must be wiped before parents"

    conn = raw(db)
    for table in REGISTRY_TABLES:
        conn.execute(f"DELETE FROM {table}")  # noqa: S608 -- fixed allowlist
    for table in REGISTRY_TABLES:
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0  # noqa: S608


def test_wiping_parents_before_children_would_fail(db, two_scopes):
    """Proves the ordering above is load-bearing, not incidental."""
    conn = raw(db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM evidence_scopes")


def test_restoring_an_old_slot_initializes_the_missing_schema(tmp_path):
    """A slot saved before the registry shipped has none of its tables."""
    old_slot = tmp_path / "old_state.db"
    conn = sqlite3.connect(str(old_slot), isolation_level=None)
    conn.execute("CREATE TABLE decisions (id INTEGER PRIMARY KEY)")
    conn.close()
    assert not _tables(old_slot) & set(REGISTRY_TABLES)

    initialize_evidence_registry(old_slot)
    assert set(REGISTRY_TABLES) <= _tables(old_slot)
    initialize_evidence_registry(old_slot)  # idempotent on the restore path


def _tables(path):
    conn = sqlite3.connect(str(path))
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


# ── wiring and boundaries ───────────────────────────────────────────────


def test_startup_initializer_is_not_wrapped_in_try_except():
    """Registry init must abort startup on failure, like every initializer
    beside it -- serving without the ownership constraints is worse than not
    serving."""
    from openexecutive.api import main

    tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "initialize_evidence_registry"
    ]
    assert len(calls) == 1

    guarded = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "initialize_evidence_registry"
            for inner in ast.walk(node)
        )
    ]
    assert not guarded, "initialize_evidence_registry must not swallow failures"


def test_startup_initialization_failure_propagates(monkeypatch, tmp_path):
    from openexecutive.evidence import registry

    def boom(_path):
        raise RegistryError("storage_unavailable")

    monkeypatch.setattr(registry, "initialize_evidence_registry", boom)
    with pytest.raises(RegistryError):
        registry.initialize_evidence_registry(tmp_path / "x.db")


def test_no_model_facing_module_imports_the_registry():
    """Identity minting must stay unreachable from anything the model drives."""
    package = PACKAGE_ROOT
    offenders = []
    for area in ("agents", "specialists", "orchestrator", "prompts", "personas"):
        for path in (package / area).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules.append(node.module)
                elif isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                if any(m.startswith("openexecutive.evidence.registry") for m in modules):
                    offenders.append(str(path.relative_to(package)))
    assert offenders == []


def test_the_registry_never_enters_trusted_construction():
    """It persists canonical records; only the factory may mint them."""
    from openexecutive.evidence import registry

    source = Path(registry.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        name = getattr(node, "id", None) or getattr(node, "attr", None)
        assert name != "trusted_construction"


def test_evidence_public_api_was_not_widened():
    """4A2b adds no exported symbol: callers import the registry directly, as
    every other store in this codebase is imported."""
    import openexecutive.evidence as pkg

    assert "registry" not in pkg.__all__
    assert "rehydrate_document_version" not in pkg.__all__
    assert "chunk_with_offsets" not in pkg.__all__
