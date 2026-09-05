"""The durable registry: scopes, logical sources and document versions.

**The database is authoritative.** Chroma is a replaceable retrieval index and
holds no identity this module recognises; a rebuilt collection loses nothing a
registry row asserts. Every id here is either opaque and application-minted, or
derived by :mod:`openexecutive.evidence.factory` — never by a model, a filename,
a path or a display label.

A leaf module, like the rest of ``evidence``: standard library plus its own
siblings. It takes an explicit ``db_path`` and never imports
:mod:`openexecutive.config`, so a test can point it at a temp file and the
application decides where the database lives.

**What this module wires, stated plainly.** Schema initialization and client
isolation, nothing else. No production caller writes a registry row yet: document
ingestion, extraction, chunking, Chroma and retrieval are untouched, and upload
observations — filename, media type, uploader, time, origin — are deliberately
absent (they belong to a later append-only observation record, not to immutable
content identity).

**The ownership invariant.** Two independent foreign keys would prove that a
scope and a logical source both exist, but *not* that the logical source belongs
to that scope — a raw insert pairing scope A with scope B's logical source
satisfies both. This module makes that state unrepresentable with a composite
foreign key against an exact unique parent key. See :data:`_SCHEMA`.
"""
from __future__ import annotations

import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from openexecutive.evidence import identity
from openexecutive.evidence.contracts import (
    DocumentVersion,
    reject_unsafe_characters,
)
from openexecutive.evidence.factory import MAX_RAW_BYTES, rehydrate_document_version

MAX_DISPLAY_NAME = 200
MAX_DISPLAY_LABEL = 200

MAX_BINDING_KEY = 255
"""The OS filesystem's path-component ceiling (POSIX ``NAME_MAX``), not an
arbitrary round number. Both binding kinds this phase supports name an
on-disk directory keyed by exactly this string — ``clients.slots._slot_dir``
for ``client_slot``, ``FIXTURES_ROOT / fixture_name`` for ``fixture`` — so a
slug longer than this was never actually persisted by the product regardless
of what the registry accepts. Verified empirically: ``mkdir`` succeeds at 255
bytes and raises ``OSError: File name too long`` at 256 on this filesystem,
matching the standard POSIX limit. This is also the exact ceiling
``evidence.contracts.MAX_FILENAME_BYTES`` already uses for the same reason.

64 was too narrow: ``POST /clients`` bounds an *explicit* ``slug`` at 64
(``CreateClientRequest.slug``), but a *derived* one comes from
``derive_client_slug(display_name)`` where ``display_name`` is bounded at 200
(``CreateClientRequest.display_name``), and ``derive_slug``'s collision
suffix (``_2``, ``_3``, …) can add a few more characters on top. 200 alone
would also have been too narrow the moment a real collision suffix landed;
255 is the true ceiling, not the next guess.
"""

_BINDING_KEY_RE = re.compile(rf"^[a-z0-9_-]{{1,{MAX_BINDING_KEY}}}$")
"""Slot slugs and fixture names. Same character class as
``cli.fixture_loader._SAFE_NAME_RE``; the length bound is ``MAX_BINDING_KEY``,
not restated here, so the two can never drift apart. Not imported from
``cli.fixture_loader`` directly: ``evidence`` is a leaf package."""


class ScopeBindingKind(StrEnum):
    """The closed set of things a scope may be bound to.

    Closed on purpose, and closed **three times over** — this enum, the
    per-kind key validator, and a SQL ``CHECK`` constraint on the column. A
    future binding (an analysis workspace, say) therefore cannot become valid by
    someone writing a new string: SQLite cannot ``ALTER`` a ``CHECK``, so adding
    a member requires a deliberate table rebuild alongside the code change.
    """

    CLIENT_SLOT = "client_slot"
    FIXTURE = "fixture"
    SINGLE_COMPANY = "single_company"


class RegistryError(ValueError):
    """A registry operation was refused.

    ``check`` is a stable lowercase literal naming the failed check, matching the
    convention in ``factory.SpanVerificationError`` and
    ``extraction_text.ChunkingError``. ``detail`` carries only bounds and counts:
    these messages reach logs and callers, and the inputs are document bytes,
    client names and source labels, so **no caller value is ever echoed**.
    """

    def __init__(self, check: str, detail: str = "") -> None:
        self.check = check
        self.detail = detail
        super().__init__(f"registry rejected: {check}" + (f" ({detail})" if detail else ""))


@dataclass(frozen=True, slots=True)
class ScopeRecord:
    """One isolation boundary. ``scope_id`` is opaque and immutable; the
    binding is how it is *found*, and ``display_name`` is how it is *shown*."""

    scope_id: str
    binding_kind: ScopeBindingKind
    binding_key: str
    display_name: str
    created_at: str
    retired_at: str | None


@dataclass(frozen=True, slots=True)
class LogicalSourceRecord:
    """One continuing source identity within one scope.

    ``logical_source_key`` is deliberately **absent**: it is a minting input, not
    a label, and exposing it would invite a caller to treat it as one. The
    repository reads it from the row when it needs to derive an id.
    """

    logical_source_id: str
    scope_id: str
    display_label: str
    created_at: str
    retired_at: str | None


@dataclass(frozen=True, slots=True)
class DocumentVersionRecord:
    """One immutable byte-content of one logical source.

    ``registered_at`` is a *registry observation* — when this registry first saw
    these bytes. It is not a document date, not a publication date and not an
    upload time, and it is part of no identity.
    """

    document_version_id: str
    scope_id: str
    logical_source_id: str
    content_sha256: str
    byte_size: int
    registered_at: str
    retired_at: str | None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_scopes (
    scope_id     TEXT NOT NULL PRIMARY KEY,
    binding_kind TEXT NOT NULL
        CHECK (binding_kind IN ('client_slot', 'fixture', 'single_company')),
    binding_key  TEXT NOT NULL,
    display_name TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    retired_at   TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_scope_binding_live
    ON evidence_scopes(binding_kind, binding_key) WHERE retired_at IS NULL;

CREATE TRIGGER IF NOT EXISTS trg_evidence_scopes_immutable
BEFORE UPDATE ON evidence_scopes FOR EACH ROW
WHEN OLD.scope_id     IS NOT NEW.scope_id
  OR OLD.binding_kind IS NOT NEW.binding_kind
  OR OLD.binding_key  IS NOT NEW.binding_key
  OR OLD.created_at   IS NOT NEW.created_at
  OR (OLD.retired_at IS NOT NULL AND OLD.retired_at IS NOT NEW.retired_at)
BEGIN
    SELECT RAISE(ABORT, 'evidence_scopes: immutable column');
END;

CREATE TABLE IF NOT EXISTS evidence_logical_sources (
    logical_source_id  TEXT NOT NULL PRIMARY KEY,
    scope_id           TEXT NOT NULL REFERENCES evidence_scopes(scope_id),
    logical_source_key TEXT NOT NULL,
    display_label      TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    retired_at         TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_ls_scope_key
    ON evidence_logical_sources(scope_id, logical_source_key);

CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_ls_id_scope
    ON evidence_logical_sources(logical_source_id, scope_id);

CREATE INDEX IF NOT EXISTS idx_evidence_ls_scope_live
    ON evidence_logical_sources(scope_id, retired_at);

CREATE TRIGGER IF NOT EXISTS trg_evidence_logical_sources_immutable
BEFORE UPDATE ON evidence_logical_sources FOR EACH ROW
WHEN OLD.logical_source_id  IS NOT NEW.logical_source_id
  OR OLD.scope_id           IS NOT NEW.scope_id
  OR OLD.logical_source_key IS NOT NEW.logical_source_key
  OR OLD.created_at         IS NOT NEW.created_at
  OR (OLD.retired_at IS NOT NULL AND OLD.retired_at IS NOT NEW.retired_at)
BEGIN
    SELECT RAISE(ABORT, 'evidence_logical_sources: immutable column');
END;

CREATE TABLE IF NOT EXISTS evidence_document_versions (
    document_version_id TEXT    NOT NULL PRIMARY KEY,
    scope_id            TEXT    NOT NULL,
    logical_source_id   TEXT    NOT NULL,
    content_sha256      TEXT    NOT NULL,
    byte_size           INTEGER NOT NULL,
    registered_at       TEXT    NOT NULL,
    retired_at          TEXT,

    FOREIGN KEY (scope_id) REFERENCES evidence_scopes(scope_id),

    FOREIGN KEY (logical_source_id, scope_id)
        REFERENCES evidence_logical_sources(logical_source_id, scope_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_dv_source_content
    ON evidence_document_versions(logical_source_id, content_sha256);

CREATE INDEX IF NOT EXISTS idx_evidence_dv_scope_content
    ON evidence_document_versions(scope_id, content_sha256);

CREATE INDEX IF NOT EXISTS idx_evidence_dv_source_registered
    ON evidence_document_versions(logical_source_id, registered_at, document_version_id);

CREATE TRIGGER IF NOT EXISTS trg_evidence_document_versions_immutable
BEFORE UPDATE ON evidence_document_versions FOR EACH ROW
WHEN OLD.document_version_id IS NOT NEW.document_version_id
  OR OLD.scope_id            IS NOT NEW.scope_id
  OR OLD.logical_source_id   IS NOT NEW.logical_source_id
  OR OLD.content_sha256      IS NOT NEW.content_sha256
  OR OLD.byte_size           IS NOT NEW.byte_size
  OR OLD.registered_at       IS NOT NEW.registered_at
  OR (OLD.retired_at IS NOT NULL AND OLD.retired_at IS NOT NEW.retired_at)
BEGIN
    SELECT RAISE(ABORT, 'evidence_document_versions: immutable column');
END;
"""
"""The whole schema, idempotent throughout.

Three notes on things that are load-bearing and do not look it:

* ``idx_evidence_ls_id_scope`` is **required**, not an optimisation. SQLite
  demands a unique index over a composite foreign key's parent columns; without
  it *every* insert into ``evidence_document_versions`` fails with
  ``OperationalError: foreign key mismatch``, not merely the cross-scope one.
* The individual ``scope_id`` foreign key is **not** redundant beside the
  composite. The composite proves the version's scope matches its logical
  source's; that the value is a *real* scope is then only transitive, through
  ``evidence_logical_sources.scope_id``, and that transitivity holds only if
  enforcement was on when the logical-source row was written. Keep it, and an
  already-orphaned logical source cannot propagate its ghost scope here.
* The individual ``logical_source_id`` foreign key is deliberately **absent**:
  it is strictly implied by the composite (same parent table, and the id is the
  prefix of the composite parent key), so it would buy no guarantee and cost an
  index lookup on every insert.

``IS NOT`` is SQLite's NULL-safe inequality, so NULL-to-NULL never fires a
trigger spuriously, and ``BEFORE UPDATE`` triggers do not fire on ``DELETE``, so
the client-slot blank wipe is unaffected.
"""

REGISTRY_TABLES: tuple[str, ...] = (
    "evidence_document_versions",
    "evidence_logical_sources",
    "evidence_scopes",
)
"""Children before parents — the order a delete must use under
``PRAGMA foreign_keys = ON``. ``clients.slots`` consumes this rather than
restating the names, so the two can never drift apart."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def _connect(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    """Open a registry connection with foreign keys **verified** on.

    ``isolation_level=None`` is load-bearing twice. It keeps DDL out of an
    implicit transaction, so the all-or-nothing guarantee below is real; and it
    keeps ``PRAGMA foreign_keys = ON`` effective, because that pragma is a
    **silent no-op while a transaction is open** — in the driver's default mode
    the first DML statement opens one, and a later pragma is ignored without
    raising. A connection that quietly lost foreign keys would quietly lose the
    ownership invariant, so the value is read back and asserted rather than
    assumed.
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    except sqlite3.Error as exc:
        # The driver message can carry the database path; the check code cannot.
        raise RegistryError("storage_unavailable") from exc
    try:
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            enforced = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        except sqlite3.Error as exc:
            raise RegistryError("storage_unavailable") from exc
        if enforced != 1:
            raise RegistryError("storage_unavailable", "foreign keys could not be enabled")
        # The yield is deliberately OUTSIDE the setup guards: wrapping it would
        # relabel every error raised by the caller's own work -- an IntegrityError
        # from the ownership foreign key, a trigger ABORT -- as a storage fault,
        # destroying exactly the distinctions the failure contract exists to make.
        yield conn
    finally:
        conn.close()


@contextmanager
def _write_txn(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` … ``COMMIT``, rolling back on any exception.

    Immediate rather than deferred: it takes the RESERVED lock up front, so a
    concurrent duplicate blocks and then *reads the winner's row*, instead of
    building on a stale snapshot and failing at commit.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def initialize_evidence_registry(db_path: Path | str) -> None:
    """Create the registry schema. Idempotent, additive, safe on every boot.

    Mirrors every other store in this codebase: ``CREATE … IF NOT EXISTS``
    throughout, no ``ALTER``, no migration framework, no backfill. Safe to run
    against an empty database, against one that already holds the other
    subsystems' tables, and against a client slot saved before this shipped.
    """
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def _bounded(value: str, label: str, limit: int, check: str) -> str:
    if not isinstance(value, str):
        raise RegistryError(check, f"{label} must be str")
    if not 1 <= len(value) <= limit:
        raise RegistryError(check, f"{label} must be 1..{limit} code points")
    try:
        reject_unsafe_characters(value, label)
    except ValueError:
        # The offending value never reaches the message.
        raise RegistryError(check, f"{label} contains a disallowed character") from None
    return value


def _coerce_binding(kind: ScopeBindingKind | str, key: str) -> tuple[ScopeBindingKind, str]:
    """Validate the closed binding domain and the per-kind key shape."""
    try:
        resolved = ScopeBindingKind(kind)
    except ValueError:
        raise RegistryError("scope_binding_kind_unknown") from None
    if not isinstance(key, str):
        raise RegistryError("scope_binding_key_invalid", "must be str")
    if resolved is ScopeBindingKind.SINGLE_COMPANY:
        # Exactly one single-company scope can exist, so its key carries nothing.
        if key != "":
            raise RegistryError("scope_binding_key_invalid", "single_company key must be empty")
    elif not _BINDING_KEY_RE.match(key):
        raise RegistryError("scope_binding_key_invalid", f"must match 1..{MAX_BINDING_KEY} [a-z0-9_-]")
    return resolved, key


def _scope_row(row: sqlite3.Row) -> ScopeRecord:
    return ScopeRecord(
        scope_id=row["scope_id"],
        binding_kind=ScopeBindingKind(row["binding_kind"]),
        binding_key=row["binding_key"],
        display_name=row["display_name"],
        created_at=row["created_at"],
        retired_at=row["retired_at"],
    )


def _ls_row(row: sqlite3.Row) -> LogicalSourceRecord:
    return LogicalSourceRecord(
        logical_source_id=row["logical_source_id"],
        scope_id=row["scope_id"],
        display_label=row["display_label"],
        created_at=row["created_at"],
        retired_at=row["retired_at"],
    )


def _dv_row(row: sqlite3.Row) -> DocumentVersionRecord:
    return DocumentVersionRecord(
        document_version_id=row["document_version_id"],
        scope_id=row["scope_id"],
        logical_source_id=row["logical_source_id"],
        content_sha256=row["content_sha256"],
        byte_size=row["byte_size"],
        registered_at=row["registered_at"],
        retired_at=row["retired_at"],
    )


class EvidenceRegistry:
    """Transactional access to the registry. One instance per database path."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = str(db_path)

    # ── scopes ──────────────────────────────────────────────────────────

    def get_or_create_scope(
        self,
        *,
        binding_kind: ScopeBindingKind | str,
        binding_key: str,
        display_name: str,
    ) -> tuple[ScopeRecord, bool]:
        """Find the live scope for a binding, or mint one. Returns ``(scope, created)``.

        ``scope_id`` is a fresh ``uuid4().hex``: opaque, encoding no name, path or
        time. That is what lets a client slot be renamed later without re-keying
        a single document — only the binding would move, and every id is derived
        from ``scope_id``, never from ``binding_key``.

        ``display_name`` is **not** updated on an existing scope. Renaming is a
        deliberate act, not a side effect of looking a scope up.
        """
        kind, key = _coerce_binding(binding_kind, binding_key)
        display_name = _bounded(
            display_name, "display_name", MAX_DISPLAY_NAME, "scope_display_name_invalid"
        )
        with _connect(self._db_path) as conn, _write_txn(conn):
            row = conn.execute(
                "SELECT * FROM evidence_scopes "
                "WHERE binding_kind = ? AND binding_key = ? AND retired_at IS NULL",
                (kind.value, key),
            ).fetchone()
            if row is not None:
                return _scope_row(row), False
            scope = ScopeRecord(
                scope_id=uuid.uuid4().hex,
                binding_kind=kind,
                binding_key=key,
                display_name=display_name,
                created_at=_utc_now(),
                retired_at=None,
            )
            conn.execute(
                "INSERT INTO evidence_scopes "
                "(scope_id, binding_kind, binding_key, display_name, created_at, retired_at) "
                "VALUES (?, ?, ?, ?, ?, NULL)",
                (
                    scope.scope_id,
                    scope.binding_kind.value,
                    scope.binding_key,
                    scope.display_name,
                    scope.created_at,
                ),
            )
            return scope, True

    def get_scope(self, scope_id: str) -> ScopeRecord | None:
        with _connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM evidence_scopes WHERE scope_id = ?", (scope_id,)
            ).fetchone()
        return _scope_row(row) if row is not None else None

    def rename_scope(self, *, scope_id: str, display_name: str) -> ScopeRecord:
        """Change only the display name. No id is re-derived and nothing moves."""
        display_name = _bounded(
            display_name, "display_name", MAX_DISPLAY_NAME, "scope_display_name_invalid"
        )
        with _connect(self._db_path) as conn, _write_txn(conn):
            self._require_live_scope(conn, scope_id)
            conn.execute(
                "UPDATE evidence_scopes SET display_name = ? WHERE scope_id = ?",
                (display_name, scope_id),
            )
            row = conn.execute(
                "SELECT * FROM evidence_scopes WHERE scope_id = ?", (scope_id,)
            ).fetchone()
            return _scope_row(row)

    def retire_scope(self, scope_id: str) -> ScopeRecord:
        """Retire a scope, one way. Its rows and history all survive.

        Retirement releases the binding — the live-only unique index stops
        applying — so a reused client slug later mints a *new* scope rather than
        colliding with this one.
        """
        with _connect(self._db_path) as conn, _write_txn(conn):
            self._require_live_scope(conn, scope_id)
            conn.execute(
                "UPDATE evidence_scopes SET retired_at = ? WHERE scope_id = ?",
                (_utc_now(), scope_id),
            )
            row = conn.execute(
                "SELECT * FROM evidence_scopes WHERE scope_id = ?", (scope_id,)
            ).fetchone()
            return _scope_row(row)

    # ── logical sources ─────────────────────────────────────────────────

    def create_logical_source(
        self, *, scope_id: str, display_label: str
    ) -> LogicalSourceRecord:
        """Create a new logical source. **Always** creates; never idempotent.

        ``logical_source_key`` is a fresh ``uuid4().hex`` minted here and nowhere
        else — never a filename, a path, a display label or model output. Two
        calls with the same label produce two distinct sources, deliberately:
        deciding that two documents are "the same source" is a human judgement,
        not a string comparison, and there is no token by which a caller could
        attach a second document to an existing source by mistake.

        Retry safety for ingestion belongs to the upload transaction that does
        not exist yet, not to a token here.
        """
        display_label = _bounded(
            display_label, "display_label", MAX_DISPLAY_LABEL, "display_label_invalid"
        )
        with _connect(self._db_path) as conn, _write_txn(conn):
            self._require_live_scope(conn, scope_id)
            key = uuid.uuid4().hex
            record = LogicalSourceRecord(
                logical_source_id=identity.mint_id(
                    identity.TAG_LOGICAL_SOURCE, scope_id, key
                ),
                scope_id=scope_id,
                display_label=display_label,
                created_at=_utc_now(),
                retired_at=None,
            )
            try:
                # No ON CONFLICT: a fresh uuid4 cannot collide, so a violation
                # here is a minting defect or a corrupted row and must surface.
                conn.execute(
                    "INSERT INTO evidence_logical_sources "
                    "(logical_source_id, scope_id, logical_source_key, display_label, "
                    " created_at, retired_at) VALUES (?, ?, ?, ?, ?, NULL)",
                    (
                        record.logical_source_id,
                        scope_id,
                        key,
                        display_label,
                        record.created_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RegistryError("logical_source_conflict") from exc
            return record

    def get_logical_source(self, logical_source_id: str) -> LogicalSourceRecord | None:
        with _connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM evidence_logical_sources WHERE logical_source_id = ?",
                (logical_source_id,),
            ).fetchone()
        return _ls_row(row) if row is not None else None

    def list_logical_sources(
        self, *, scope_id: str, include_retired: bool = False
    ) -> tuple[LogicalSourceRecord, ...]:
        """Every logical source in one scope. Always filtered by ``scope_id``."""
        sql = "SELECT * FROM evidence_logical_sources WHERE scope_id = ?"
        if not include_retired:
            sql += " AND retired_at IS NULL"
        sql += " ORDER BY created_at, logical_source_id"
        with _connect(self._db_path) as conn:
            rows = conn.execute(sql, (scope_id,)).fetchall()
        return tuple(_ls_row(r) for r in rows)

    def relabel_logical_source(
        self, *, logical_source_id: str, display_label: str
    ) -> LogicalSourceRecord:
        """Change only the display label. Identity is untouched."""
        display_label = _bounded(
            display_label, "display_label", MAX_DISPLAY_LABEL, "display_label_invalid"
        )
        with _connect(self._db_path) as conn, _write_txn(conn):
            self._require_live_logical_source(conn, logical_source_id)
            conn.execute(
                "UPDATE evidence_logical_sources SET display_label = ? "
                "WHERE logical_source_id = ?",
                (display_label, logical_source_id),
            )
            row = conn.execute(
                "SELECT * FROM evidence_logical_sources WHERE logical_source_id = ?",
                (logical_source_id,),
            ).fetchone()
            return _ls_row(row)

    def retire_logical_source(self, logical_source_id: str) -> LogicalSourceRecord:
        """Retire a logical source, one way. Its versions all survive."""
        with _connect(self._db_path) as conn, _write_txn(conn):
            self._require_live_logical_source(conn, logical_source_id)
            conn.execute(
                "UPDATE evidence_logical_sources SET retired_at = ? "
                "WHERE logical_source_id = ?",
                (_utc_now(), logical_source_id),
            )
            row = conn.execute(
                "SELECT * FROM evidence_logical_sources WHERE logical_source_id = ?",
                (logical_source_id,),
            ).fetchone()
            return _ls_row(row)

    # ── document versions ───────────────────────────────────────────────

    def register_document_version(
        self, *, scope_id: str, logical_source_id: str, raw_bytes: bytes
    ) -> tuple[DocumentVersionRecord, bool]:
        """Register one immutable byte-content. Returns ``(record, created)``.

        Idempotent by construction rather than by mechanism:
        ``document_version_id`` is a pure function of
        ``(scope_id, logical_source_key, bytes)``, so re-registering identical
        bytes under the same source recomputes the same primary key and the
        insert collapses to a no-op.

        The expensive work — hashing up to 64 MiB — happens **outside** the
        transaction, so a large upload never holds a write lock.
        """
        if not isinstance(raw_bytes, bytes):
            raise RegistryError("bytes_invalid", "raw_bytes must be bytes")
        if len(raw_bytes) > MAX_RAW_BYTES:
            raise RegistryError("bytes_invalid", f"exceeds {MAX_RAW_BYTES} bytes")

        with _connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM evidence_logical_sources WHERE logical_source_id = ?",
                (logical_source_id,),
            ).fetchone()
            if row is None:
                raise RegistryError("logical_source_not_found")
            # Ownership is checked here so the caller gets a typed failure, not a
            # raw IntegrityError. The composite FK below is the backstop that
            # makes the same state unrepresentable even through raw SQL.
            if row["scope_id"] != scope_id:
                raise RegistryError("scope_mismatch")
            if row["retired_at"] is not None:
                raise RegistryError("logical_source_retired")
            source_key = row["logical_source_key"]

        version: DocumentVersion = rehydrate_document_version(
            scope_id=scope_id,
            logical_source_key=source_key,
            content_sha256=identity.content_sha256(raw_bytes),
            byte_size=len(raw_bytes),
        )
        if version.logical_source_id != logical_source_id:
            # The stored key no longer derives the stored id: a corrupted row or
            # a changed minting scheme, never an ordinary conflict.
            raise RegistryError("logical_source_conflict")

        with _connect(self._db_path) as conn, _write_txn(conn):
            self._require_live_scope(conn, scope_id)
            self._require_live_logical_source(conn, logical_source_id)
            registered_at = _utc_now()
            cursor = conn.execute(
                "INSERT INTO evidence_document_versions "
                "(document_version_id, scope_id, logical_source_id, content_sha256, "
                " byte_size, registered_at, retired_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL) "
                "ON CONFLICT(document_version_id) DO NOTHING",
                (
                    version.document_version_id,
                    version.scope_id,
                    version.logical_source_id,
                    version.content_sha256,
                    version.byte_size,
                    registered_at,
                ),
            )
            # rowcount is 1 only when this statement actually inserted; comparing
            # timestamps instead would call a concurrent duplicate "created" any
            # time two callers landed on the same microsecond.
            created = cursor.rowcount == 1
            stored = conn.execute(
                "SELECT * FROM evidence_document_versions WHERE document_version_id = ?",
                (version.document_version_id,),
            ).fetchone()
            if stored is None:
                raise RegistryError("storage_unavailable", "row vanished after insert")
            # Read back and re-verify: redundant on the happy path, and the only
            # thing that turns a corrupted or tampered row into a typed failure
            # instead of a silent wrong answer.
            if (
                stored["scope_id"] != version.scope_id
                or stored["logical_source_id"] != version.logical_source_id
                or stored["content_sha256"] != version.content_sha256
                or stored["byte_size"] != version.byte_size
            ):
                raise RegistryError("document_version_conflict")
            return _dv_row(stored), created

    def get_document_version(self, document_version_id: str) -> DocumentVersionRecord | None:
        with _connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM evidence_document_versions WHERE document_version_id = ?",
                (document_version_id,),
            ).fetchone()
        return _dv_row(row) if row is not None else None

    def list_document_versions(
        self, *, logical_source_id: str, include_retired: bool = False
    ) -> tuple[DocumentVersionRecord, ...]:
        """Versions of one logical source, newest registration first.

        Ordered by ``(registered_at DESC, document_version_id DESC)`` — durable
        stored fields only. **Never ``rowid``**: it is not a column of the row and
        is not preserved by ``VACUUM``, which is exactly how a client slot is
        saved (``VACUUM INTO``), so a rowid order would be silently unstable
        across a save/restore.

        ``document_version_id`` is only a deterministic tie-breaker when two
        versions share a ``registered_at``. It is a hash: it expresses **no**
        semantic version precedence, and no caller should read one into it.
        Which version is "current" is a version-selection semantic this phase
        deliberately does not define.
        """
        sql = "SELECT * FROM evidence_document_versions WHERE logical_source_id = ?"
        if not include_retired:
            sql += " AND retired_at IS NULL"
        sql += " ORDER BY registered_at DESC, document_version_id DESC"
        with _connect(self._db_path) as conn:
            rows = conn.execute(sql, (logical_source_id,)).fetchall()
        return tuple(_dv_row(r) for r in rows)

    def retire_document_version(self, document_version_id: str) -> DocumentVersionRecord:
        """Retire a version, one way. Nothing is ever hard-deleted."""
        with _connect(self._db_path) as conn, _write_txn(conn):
            row = conn.execute(
                "SELECT retired_at FROM evidence_document_versions "
                "WHERE document_version_id = ?",
                (document_version_id,),
            ).fetchone()
            if row is None:
                raise RegistryError("document_version_not_found")
            if row["retired_at"] is not None:
                raise RegistryError("document_version_retired")
            conn.execute(
                "UPDATE evidence_document_versions SET retired_at = ? "
                "WHERE document_version_id = ?",
                (_utc_now(), document_version_id),
            )
            stored = conn.execute(
                "SELECT * FROM evidence_document_versions WHERE document_version_id = ?",
                (document_version_id,),
            ).fetchone()
            return _dv_row(stored)

    # ── shared guards ───────────────────────────────────────────────────

    @staticmethod
    def _require_live_scope(conn: sqlite3.Connection, scope_id: str) -> None:
        row = conn.execute(
            "SELECT retired_at FROM evidence_scopes WHERE scope_id = ?", (scope_id,)
        ).fetchone()
        if row is None:
            raise RegistryError("scope_not_found")
        if row["retired_at"] is not None:
            raise RegistryError("scope_retired")

    @staticmethod
    def _require_live_logical_source(
        conn: sqlite3.Connection, logical_source_id: str
    ) -> None:
        row = conn.execute(
            "SELECT retired_at FROM evidence_logical_sources WHERE logical_source_id = ?",
            (logical_source_id,),
        ).fetchone()
        if row is None:
            raise RegistryError("logical_source_not_found")
        if row["retired_at"] is not None:
            raise RegistryError("logical_source_retired")
