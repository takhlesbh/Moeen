"""The registry's domain rules, schema shape and failure contract."""
from __future__ import annotations

import sqlite3

import pytest

from openexecutive.evidence import identity
from openexecutive.evidence.factory import (
    mint_document_version,
    rehydrate_document_version,
)
from openexecutive.evidence.registry import (
    MAX_BINDING_KEY,
    REGISTRY_TABLES,
    EvidenceRegistry,
    RegistryError,
    ScopeBindingKind,
    initialize_evidence_registry,
)
from openexecutive.fixtures.generator import derive_slug

EXPECTED_INDEXES = {
    "idx_evidence_scope_binding_live",
    "idx_evidence_ls_scope_key",
    "idx_evidence_ls_id_scope",
    "idx_evidence_ls_scope_live",
    "idx_evidence_dv_source_content",
    "idx_evidence_dv_scope_content",
    "idx_evidence_dv_source_registered",
}
EXPECTED_TRIGGERS = {
    "trg_evidence_scopes_immutable",
    "trg_evidence_logical_sources_immutable",
    "trg_evidence_document_versions_immutable",
}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "registry.db"
    initialize_evidence_registry(path)
    return path


@pytest.fixture
def reg(db):
    return EvidenceRegistry(db)


@pytest.fixture
def scope(reg):
    record, _ = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key="acme", display_name="Acme"
    )
    return record


def raw(db_path):
    """A connection with enforcement on, for tests that bypass the repository."""
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    return conn


# ── schema ──────────────────────────────────────────────────────────────


def test_initialize_creates_the_whole_schema(db):
    conn = raw(db)
    names = lambda kind: {  # noqa: E731
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = ?", (kind,)
        )
    }
    assert set(REGISTRY_TABLES) <= names("table")
    assert names("index") >= EXPECTED_INDEXES
    assert names("trigger") >= EXPECTED_TRIGGERS


def test_initialize_is_idempotent(db):
    initialize_evidence_registry(db)
    initialize_evidence_registry(db)
    conn = raw(db)
    assert {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_evidence%'"
    )} == EXPECTED_INDEXES


def test_initialize_over_an_existing_database_preserves_rows(tmp_path):
    path = tmp_path / "existing.db"
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("CREATE TABLE decisions (id INTEGER PRIMARY KEY, summary TEXT)")
    conn.execute("INSERT INTO decisions VALUES (1, 'pre-existing')")
    conn.close()

    initialize_evidence_registry(path)
    initialize_evidence_registry(path)

    conn = raw(path)
    assert conn.execute("SELECT summary FROM decisions").fetchone()[0] == "pre-existing"
    assert set(REGISTRY_TABLES) <= {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def test_wipe_table_order_is_children_before_parents():
    """The order `clients.slots` deletes in. Parents first would violate the FKs."""
    assert REGISTRY_TABLES == (
        "evidence_document_versions",
        "evidence_logical_sources",
        "evidence_scopes",
    )


# ── closed binding domain ───────────────────────────────────────────────


@pytest.mark.parametrize("kind,key", [
    (ScopeBindingKind.CLIENT_SLOT, "acme"),
    (ScopeBindingKind.FIXTURE, "demo_co"),
    (ScopeBindingKind.SINGLE_COMPANY, ""),
])
def test_every_member_of_the_closed_domain_is_accepted(reg, kind, key):
    record, created = reg.get_or_create_scope(
        binding_kind=kind, binding_key=key, display_name="X"
    )
    assert created and record.binding_kind is kind


@pytest.mark.parametrize("kind", [
    "analysis_workspace", "ANALYSIS_WORKSPACE", "client-slot", "", "tenant", "Client_Slot",
])
def test_unknown_binding_kind_is_refused_at_the_api(reg, kind):
    with pytest.raises(RegistryError) as exc:
        reg.get_or_create_scope(binding_kind=kind, binding_key="k", display_name="X")
    assert exc.value.check == "scope_binding_kind_unknown"


def test_unknown_binding_kind_is_refused_by_the_database_too(db):
    """A future binding cannot become valid just by writing a new string: the
    CHECK constraint forces a deliberate schema migration."""
    conn = raw(db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO evidence_scopes VALUES ('s1','analysis_workspace','w','W','t',NULL)"
        )


@pytest.mark.parametrize(
    "key", ["Acme", "with space", "a/b", "..", "x" * (MAX_BINDING_KEY + 1), "a\x00b"]
)
def test_malformed_slot_key_is_refused(reg, key):
    with pytest.raises(RegistryError) as exc:
        reg.get_or_create_scope(
            binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key=key, display_name="X"
        )
    assert exc.value.check == "scope_binding_key_invalid"


def test_single_company_key_must_be_empty(reg):
    with pytest.raises(RegistryError) as exc:
        reg.get_or_create_scope(
            binding_kind=ScopeBindingKind.SINGLE_COMPANY,
            binding_key="anything",
            display_name="X",
        )
    assert exc.value.check == "scope_binding_key_invalid"


def test_registry_key_shape_matches_the_product_slug_shape(reg):
    """Every slug the fixture loader would produce must be a legal binding key,
    short ones and long ones alike -- see the boundary-length tests below for
    why "short ones" alone is not sufficient evidence."""
    from openexecutive.cli.fixture_loader import _SAFE_NAME_RE

    samples = [
        "acme", "acme_2", "a-b-c", "x", "0", "a_2-b",
        # A real long display name, run through the real derivation -- not a
        # hand-typed string. This is the reviewer's counter-example.
        derive_slug(
            "Very Long International Holdings and Subsidiaries Group "
            "Corporation Limited"
        ),
        # The other worst case the product can actually construct: an
        # all-alphanumeric display name at exactly the API's 200-char cap,
        # where the substitution regex collapses nothing.
        derive_slug("a" * 200),
    ]
    for candidate in samples:
        assert _SAFE_NAME_RE.match(candidate), f"not a legal product slug: {candidate!r}"
        _, created = reg.get_or_create_scope(
            binding_kind=ScopeBindingKind.CLIENT_SLOT,
            binding_key=candidate,
            display_name="X",
        )
        assert created


def test_the_reviewers_75_character_slug_is_accepted(reg):
    """The exact counter-example from the independent review: a legal client
    display name whose derived slug the 64-character ceiling refused."""
    slug = derive_slug(
        "Very Long International Holdings and Subsidiaries Group "
        "Corporation Limited"
    )
    assert len(slug) == 75
    record, created = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key=slug, display_name="X"
    )
    assert created
    assert record.binding_key == slug


def test_a_slug_at_the_display_name_boundary_is_accepted(reg):
    """`CreateClientRequest.display_name` is capped at 200 characters -- the
    largest base `derive_client_slug` can ever hand the registry when nothing
    collapses. It must fit, because the product would otherwise successfully
    create a client the registry could never open a scope for."""
    slug = derive_slug("a" * 200)
    assert len(slug) == 200
    _, created = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key=slug, display_name="X"
    )
    assert created


def test_a_collision_suffixed_slug_is_accepted(reg):
    """`derive_slug`'s own collision loop, driven against a real `exists`
    predicate -- the same shape a second `POST /clients` with an identical
    display name would produce (`derive_client_slug` passes
    `_slot_dir(settings, s).exists()` as exactly this kind of callable)."""
    base_name = "a" * 200
    taken: set[str] = set()

    first_slug = derive_slug(base_name, taken.__contains__)
    taken.add(first_slug)
    reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key=first_slug, display_name="First"
    )

    suffixed = derive_slug(base_name, taken.__contains__)
    assert suffixed != first_slug
    assert suffixed.startswith(first_slug + "_")
    _, created = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key=suffixed, display_name="Second"
    )
    assert created


@pytest.mark.parametrize("length,expect_ok", [
    (MAX_BINDING_KEY, True),
    (MAX_BINDING_KEY + 1, False),
])
def test_the_binding_key_ceiling_is_pinned_exactly(reg, length, expect_ok):
    """255 is the OS filesystem's path-component ceiling (POSIX NAME_MAX),
    not a round number picked for its own sake -- see the docstring on
    MAX_BINDING_KEY. Pin both edges so a future change is a deliberate act."""
    key = "a" * length
    if expect_ok:
        _, created = reg.get_or_create_scope(
            binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key=key, display_name="X"
        )
        assert created
    else:
        with pytest.raises(RegistryError) as exc:
            reg.get_or_create_scope(
                binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key=key, display_name="X"
            )
        assert exc.value.check == "scope_binding_key_invalid"


def test_the_os_filesystem_ceiling_matches_the_chosen_bound(tmp_path):
    """The evidence for 255, reproduced in the test suite rather than only in
    a docstring: this is the actual point at which the product's own
    directory-per-slug mechanism (client slots, curated fixtures) stops being
    able to persist a name at all."""
    (tmp_path / ("a" * MAX_BINDING_KEY)).mkdir()
    with pytest.raises(OSError):
        (tmp_path / ("a" * (MAX_BINDING_KEY + 1))).mkdir()


@pytest.mark.parametrize("key", [
    "Acme", "with space", "a/b", "..", "../etc", "a\x00b", "a" * 300,
])
def test_invalid_characters_remain_rejected_regardless_of_length(reg, key):
    """Widening the ceiling must not widen the character class. Every one of
    these was already rejected before this repair and must still be."""
    with pytest.raises(RegistryError) as exc:
        reg.get_or_create_scope(
            binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key=key, display_name="X"
        )
    assert exc.value.check == "scope_binding_key_invalid"


def test_single_company_still_accepts_only_the_empty_key_at_any_ceiling(reg):
    """The ceiling change must not loosen single_company's exact-empty rule."""
    for key in ["", "a", "a" * MAX_BINDING_KEY]:
        if key == "":
            _, created = reg.get_or_create_scope(
                binding_kind=ScopeBindingKind.SINGLE_COMPANY, binding_key=key, display_name="X"
            )
            assert created
        else:
            with pytest.raises(RegistryError) as exc:
                reg.get_or_create_scope(
                    binding_kind=ScopeBindingKind.SINGLE_COMPANY,
                    binding_key=key,
                    display_name="X",
                )
            assert exc.value.check == "scope_binding_key_invalid"


def test_binding_key_ceiling_error_message_stays_value_free(reg):
    """A rejected long key must not echo the key itself -- only counts."""
    canary = "s3cret_" + "z" * 300
    with pytest.raises(RegistryError) as exc:
        reg.get_or_create_scope(
            binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key=canary, display_name="X"
        )
    assert canary not in str(exc.value)
    assert "s3cret" not in str(exc.value)
    assert exc.value.check == "scope_binding_key_invalid"


# ── scope identity ──────────────────────────────────────────────────────


def test_scope_id_is_opaque_and_unrelated_to_any_name(reg):
    a, _ = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key="one", display_name="Same"
    )
    b, _ = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key="two", display_name="Same"
    )
    assert a.scope_id != b.scope_id
    for record in (a, b):
        assert len(record.scope_id) == 32
        assert record.display_name.lower() not in record.scope_id
        assert record.binding_key not in record.scope_id


def test_get_or_create_is_stable_and_does_not_silently_rename(reg, scope):
    again, created = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT,
        binding_key="acme",
        display_name="A Different Name",
    )
    assert not created
    assert again.scope_id == scope.scope_id
    assert again.display_name == "Acme"


def test_renaming_a_scope_moves_no_identity(reg, scope):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="Q3")
    version, _ = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=source.logical_source_id, raw_bytes=b"x"
    )
    renamed = reg.rename_scope(scope_id=scope.scope_id, display_name="Acme Holdings")

    assert renamed.scope_id == scope.scope_id
    assert reg.get_logical_source(source.logical_source_id).logical_source_id == (
        source.logical_source_id
    )
    assert reg.get_document_version(version.document_version_id) == version


def test_retiring_a_scope_frees_the_binding_for_a_new_scope(reg, scope):
    retired = reg.retire_scope(scope.scope_id)
    assert retired.retired_at is not None

    fresh, created = reg.get_or_create_scope(
        binding_kind=ScopeBindingKind.CLIENT_SLOT, binding_key="acme", display_name="New Acme"
    )
    assert created and fresh.scope_id != scope.scope_id
    assert reg.get_scope(scope.scope_id).retired_at is not None


def test_scope_retirement_is_one_way(reg, scope):
    reg.retire_scope(scope.scope_id)
    with pytest.raises(RegistryError) as exc:
        reg.retire_scope(scope.scope_id)
    assert exc.value.check == "scope_retired"


# ── logical sources ─────────────────────────────────────────────────────


def test_same_display_label_creates_distinct_logical_sources(reg, scope):
    a = reg.create_logical_source(scope_id=scope.scope_id, display_label="Annual Report")
    b = reg.create_logical_source(scope_id=scope.scope_id, display_label="Annual Report")
    assert a.logical_source_id != b.logical_source_id


def test_logical_source_key_is_opaque_and_not_derived_from_the_label(reg, scope, db):
    label = "Quarterly Filing"
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label=label)
    key = raw(db).execute(
        "SELECT logical_source_key FROM evidence_logical_sources WHERE logical_source_id = ?",
        (source.logical_source_id,),
    ).fetchone()[0]

    assert len(key) == 32
    assert label not in key and label.lower().replace(" ", "") not in key
    # And the id really is minted from that key, not from anything human.
    assert source.logical_source_id == identity.mint_id(
        identity.TAG_LOGICAL_SOURCE, scope.scope_id, key
    )


def test_logical_source_key_is_not_exposed_on_the_record(reg, scope):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    assert not hasattr(source, "logical_source_key")


def test_relabel_changes_only_the_label(reg, scope):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="Old")
    updated = reg.relabel_logical_source(
        logical_source_id=source.logical_source_id, display_label="New"
    )
    assert updated.logical_source_id == source.logical_source_id
    assert updated.display_label == "New"
    assert updated.created_at == source.created_at


def test_logical_source_under_missing_or_retired_scope(reg, scope):
    with pytest.raises(RegistryError) as exc:
        reg.create_logical_source(scope_id="nonexistent", display_label="X")
    assert exc.value.check == "scope_not_found"

    reg.retire_scope(scope.scope_id)
    with pytest.raises(RegistryError) as exc:
        reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    assert exc.value.check == "scope_retired"


def test_logical_source_retirement_is_one_way_and_keeps_versions(reg, scope):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    version, _ = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=source.logical_source_id, raw_bytes=b"a"
    )
    reg.retire_logical_source(source.logical_source_id)

    assert reg.get_document_version(version.document_version_id) == version
    with pytest.raises(RegistryError) as exc:
        reg.retire_logical_source(source.logical_source_id)
    assert exc.value.check == "logical_source_retired"


# ── document versions ───────────────────────────────────────────────────


def test_registration_derives_identity_through_the_factory(reg, scope, db):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    record, created = reg.register_document_version(
        scope_id=scope.scope_id,
        logical_source_id=source.logical_source_id,
        raw_bytes=b"hello",
    )
    key = raw(db).execute(
        "SELECT logical_source_key FROM evidence_logical_sources WHERE logical_source_id = ?",
        (source.logical_source_id,),
    ).fetchone()[0]
    expected = mint_document_version(
        raw_bytes=b"hello", scope_id=scope.scope_id, logical_source_key=key
    )

    assert created
    assert record.document_version_id == expected.document_version_id
    assert record.content_sha256 == expected.content_sha256
    assert record.byte_size == 5


def test_identical_bytes_are_idempotent_and_rewrite_nothing(reg, scope):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    first, created_first = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=source.logical_source_id, raw_bytes=b"same"
    )
    second, created_second = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=source.logical_source_id, raw_bytes=b"same"
    )
    assert created_first and not created_second
    assert first == second
    assert len(reg.list_document_versions(logical_source_id=source.logical_source_id)) == 1


def test_new_bytes_add_a_version_without_touching_the_older_row(reg, scope):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    old, _ = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=source.logical_source_id, raw_bytes=b"v1"
    )
    before = reg.get_document_version(old.document_version_id)
    new, created = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=source.logical_source_id, raw_bytes=b"v2"
    )

    assert created and new.document_version_id != old.document_version_id
    # No supersession stamp is written anywhere: the older row is byte-identical.
    assert reg.get_document_version(old.document_version_id) == before
    assert len(reg.list_document_versions(logical_source_id=source.logical_source_id)) == 2


def test_same_bytes_under_two_logical_sources_stay_distinct(reg, scope):
    a = reg.create_logical_source(scope_id=scope.scope_id, display_label="A")
    b = reg.create_logical_source(scope_id=scope.scope_id, display_label="B")
    va, _ = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=a.logical_source_id, raw_bytes=b"dup"
    )
    vb, _ = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=b.logical_source_id, raw_bytes=b"dup"
    )
    assert va.document_version_id != vb.document_version_id
    assert va.content_sha256 == vb.content_sha256


def test_registration_rejects_a_missing_or_retired_source(reg, scope):
    with pytest.raises(RegistryError) as exc:
        reg.register_document_version(
            scope_id=scope.scope_id, logical_source_id="a" * 64, raw_bytes=b"x"
        )
    assert exc.value.check == "logical_source_not_found"

    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    reg.retire_logical_source(source.logical_source_id)
    with pytest.raises(RegistryError) as exc:
        reg.register_document_version(
            scope_id=scope.scope_id,
            logical_source_id=source.logical_source_id,
            raw_bytes=b"x",
        )
    assert exc.value.check == "logical_source_retired"


@pytest.mark.parametrize("payload", ["not bytes", 5, None])
def test_registration_rejects_non_bytes(reg, scope, payload):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    with pytest.raises(RegistryError) as exc:
        reg.register_document_version(
            scope_id=scope.scope_id,
            logical_source_id=source.logical_source_id,
            raw_bytes=payload,
        )
    assert exc.value.check == "bytes_invalid"


def test_version_retirement_is_one_way(reg, scope):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    version, _ = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=source.logical_source_id, raw_bytes=b"x"
    )
    retired = reg.retire_document_version(version.document_version_id)
    assert retired.retired_at is not None
    assert reg.get_document_version(version.document_version_id).retired_at is not None

    with pytest.raises(RegistryError) as exc:
        reg.retire_document_version(version.document_version_id)
    assert exc.value.check == "document_version_retired"


def test_a_corrupted_stored_row_is_reported_not_returned(reg, scope, db):
    """The read-back verification is what turns tampering into a typed failure.
    Without it the corrupted row is handed back as if it were authoritative."""
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    reg.register_document_version(
        scope_id=scope.scope_id,
        logical_source_id=source.logical_source_id,
        raw_bytes=b"payload",
    )
    conn = raw(db)
    conn.execute("DROP TRIGGER trg_evidence_document_versions_immutable")
    conn.execute("UPDATE evidence_document_versions SET byte_size = 99999")

    with pytest.raises(RegistryError) as exc:
        reg.register_document_version(
            scope_id=scope.scope_id,
            logical_source_id=source.logical_source_id,
            raw_bytes=b"payload",
        )
    assert exc.value.check == "document_version_conflict"


def test_a_logical_source_whose_key_no_longer_derives_its_id_is_refused(reg, scope, db):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    conn = raw(db)
    conn.execute("DROP TRIGGER trg_evidence_logical_sources_immutable")
    conn.execute(
        "UPDATE evidence_logical_sources SET logical_source_key = ? "
        "WHERE logical_source_id = ?",
        ("f" * 32, source.logical_source_id),
    )
    with pytest.raises(RegistryError) as exc:
        reg.register_document_version(
            scope_id=scope.scope_id,
            logical_source_id=source.logical_source_id,
            raw_bytes=b"x",
        )
    assert exc.value.check == "logical_source_conflict"


# ── rehydration ─────────────────────────────────────────────────────────


def test_rehydrate_reproduces_mint_exactly():
    minted = mint_document_version(
        raw_bytes=b"payload", scope_id="scope", logical_source_key="key"
    )
    rehydrated = rehydrate_document_version(
        scope_id="scope",
        logical_source_key="key",
        content_sha256=minted.content_sha256,
        byte_size=minted.byte_size,
    )
    assert rehydrated == minted


@pytest.mark.parametrize("bad_hash", ["short", "A" * 64, "g" * 64, "", "0" * 63])
def test_rehydrate_refuses_a_malformed_hash(bad_hash):
    from openexecutive.evidence.factory import EvidenceFactoryError

    with pytest.raises(EvidenceFactoryError):
        rehydrate_document_version(
            scope_id="s", logical_source_key="k", content_sha256=bad_hash, byte_size=1
        )


@pytest.mark.parametrize("bad_size", [-1, 64 * 1024 * 1024 + 1, True, "5"])
def test_rehydrate_refuses_a_malformed_byte_size(bad_size):
    from openexecutive.evidence.factory import EvidenceFactoryError

    with pytest.raises(EvidenceFactoryError):
        rehydrate_document_version(
            scope_id="s", logical_source_key="k", content_sha256="a" * 64, byte_size=bad_size
        )


# ── ordering ────────────────────────────────────────────────────────────


def test_listing_order_is_deterministic_and_independent_of_insertion_order(reg, scope, db):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    for payload in (b"a", b"b", b"c"):
        reg.register_document_version(
            scope_id=scope.scope_id,
            logical_source_id=source.logical_source_id,
            raw_bytes=payload,
        )
    # Force a shared timestamp so the tie-breaker, not the clock, decides.
    conn = raw(db)
    conn.execute("DROP TRIGGER trg_evidence_document_versions_immutable")
    conn.execute("UPDATE evidence_document_versions SET registered_at = '2026-01-01T00:00:00+00:00'")

    ordered = reg.list_document_versions(logical_source_id=source.logical_source_id)
    ids = [r.document_version_id for r in ordered]
    assert ids == sorted(ids, reverse=True)

    # Rewriting the rows in a different physical order must not change the answer.
    rows = conn.execute("SELECT * FROM evidence_document_versions").fetchall()
    conn.execute("DELETE FROM evidence_document_versions")
    for row in reversed(rows):
        conn.execute(
            "INSERT INTO evidence_document_versions VALUES (?,?,?,?,?,?,?)", tuple(row)
        )
    assert [
        r.document_version_id
        for r in reg.list_document_versions(logical_source_id=source.logical_source_id)
    ] == ids


def test_retired_versions_are_excluded_unless_asked_for(reg, scope):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    keep, _ = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=source.logical_source_id, raw_bytes=b"1"
    )
    drop, _ = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=source.logical_source_id, raw_bytes=b"2"
    )
    reg.retire_document_version(drop.document_version_id)

    live = reg.list_document_versions(logical_source_id=source.logical_source_id)
    assert [r.document_version_id for r in live] == [keep.document_version_id]
    assert len(
        reg.list_document_versions(
            logical_source_id=source.logical_source_id, include_retired=True
        )
    ) == 2


# ── immutability triggers ───────────────────────────────────────────────


@pytest.mark.parametrize("column,value", [
    ("scope_id", "other"), ("binding_kind", "fixture"),
    ("binding_key", "renamed"), ("created_at", "2020-01-01"),
])
def test_scope_immutable_columns_cannot_be_updated(reg, scope, db, column, value):
    conn = raw(db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"UPDATE evidence_scopes SET {column} = ? WHERE scope_id = ?",  # noqa: S608
            (value, scope.scope_id),
        )


@pytest.mark.parametrize("column,value", [
    ("logical_source_id", "x" * 64), ("scope_id", "other"),
    ("logical_source_key", "forged"), ("created_at", "2020-01-01"),
])
def test_logical_source_immutable_columns_cannot_be_updated(reg, scope, db, column, value):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    conn = raw(db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"UPDATE evidence_logical_sources SET {column} = ? "  # noqa: S608
            "WHERE logical_source_id = ?",
            (value, source.logical_source_id),
        )


@pytest.mark.parametrize("column,value", [
    ("document_version_id", "x" * 64), ("scope_id", "other"),
    ("logical_source_id", "x" * 64), ("content_sha256", "b" * 64),
    ("byte_size", 999), ("registered_at", "2020-01-01"),
])
def test_document_version_immutable_columns_cannot_be_updated(reg, scope, db, column, value):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    version, _ = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=source.logical_source_id, raw_bytes=b"x"
    )
    conn = raw(db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"UPDATE evidence_document_versions SET {column} = ? "  # noqa: S608
            "WHERE document_version_id = ?",
            (value, version.document_version_id),
        )


def test_display_columns_remain_mutable(reg, scope, db):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    conn = raw(db)
    conn.execute("UPDATE evidence_scopes SET display_name = 'ok' WHERE scope_id = ?",
                 (scope.scope_id,))
    conn.execute(
        "UPDATE evidence_logical_sources SET display_label = 'ok' WHERE logical_source_id = ?",
        (source.logical_source_id,),
    )


@pytest.mark.parametrize("table,key_column", [
    ("evidence_scopes", "scope_id"),
    ("evidence_logical_sources", "logical_source_id"),
    ("evidence_document_versions", "document_version_id"),
])
def test_retirement_cannot_be_undone_or_rewritten(reg, scope, db, table, key_column):
    source = reg.create_logical_source(scope_id=scope.scope_id, display_label="X")
    version, _ = reg.register_document_version(
        scope_id=scope.scope_id, logical_source_id=source.logical_source_id, raw_bytes=b"x"
    )
    key = {
        "evidence_scopes": scope.scope_id,
        "evidence_logical_sources": source.logical_source_id,
        "evidence_document_versions": version.document_version_id,
    }[table]
    conn = raw(db)
    conn.execute(
        f"UPDATE {table} SET retired_at = '2026-01-01' WHERE {key_column} = ?", (key,)  # noqa: S608
    )
    for attempt in (None, "2027-01-01"):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"UPDATE {table} SET retired_at = ? WHERE {key_column} = ?",  # noqa: S608
                (attempt, key),
            )


# ── failure-message hygiene ─────────────────────────────────────────────


def test_error_messages_never_echo_caller_values(reg, scope, tmp_path):
    canary = "CANARYVALUE"
    failures = []

    try:
        reg.get_or_create_scope(
            binding_kind="bogus_" + canary, binding_key=canary, display_name=canary
        )
    except RegistryError as exc:
        failures.append(exc)
    try:
        reg.get_or_create_scope(
            binding_kind=ScopeBindingKind.CLIENT_SLOT,
            binding_key=canary,
            display_name=canary,
        )
    except RegistryError as exc:
        failures.append(exc)
    try:
        reg.create_logical_source(scope_id=canary, display_label=canary)
    except RegistryError as exc:
        failures.append(exc)
    try:
        reg.create_logical_source(scope_id=scope.scope_id, display_label=canary + "\x00")
    except RegistryError as exc:
        failures.append(exc)

    assert len(failures) == 4
    for exc in failures:
        assert canary not in str(exc)
        assert str(tmp_path) not in str(exc)
        assert exc.check == exc.check.lower()


def test_check_codes_are_stable_lowercase_literals(reg):
    with pytest.raises(RegistryError) as exc:
        reg.get_or_create_scope(binding_kind="nope", binding_key="k", display_name="X")
    assert exc.value.check == "scope_binding_kind_unknown"
    assert str(exc.value).startswith("registry rejected: scope_binding_kind_unknown")
