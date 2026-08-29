"""Fail-closed preflight for the eval harness.

Two proven failure modes this module exists to make impossible:

1. The runner discovered zero scenarios and still exited 0, so `make eval`
   reported success having made no model call at all.
2. The runner resolved a relative ``VECTOR_STORE_PATH`` against the process
   CWD, and ``chromadb.PersistentClient`` **created** a brand-new empty store
   at that path. Every specialist then answered with zero retrieval while the
   run looked healthy.

Both share one root cause: the harness inferred state instead of asserting it.
Everything here therefore resolves explicitly, validates before any paid call,
and raises :class:`PreflightError` rather than degrading.

Scenario discovery is NOT implemented here — it delegates to
``openexecutive.evals.scenarios``, the authoritative loader also used by the
HTTP eval routes. A second discovery implementation is what broke.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Collections each scenario kind actually reads at run time.
#
# Deliberately NOT a universal "company_docs is required" rule: company_docs
# holds tenant-uploaded documents that a clean checkout legitimately lacks, and
# no built-in scenario asserts against them. Triage requires none at all —
# ``TriageAgent.triage`` performs no retrieval.
#
# A scenario may add to its kind's set with a ``requires_collections`` list.
_KIND_REQUIRED_COLLECTIONS: dict[str, frozenset[str]] = {
    "chat": frozenset({"builtin_knowledge"}),
    "mcp": frozenset({"builtin_knowledge"}),
    "workflow": frozenset({"builtin_knowledge"}),
    "triage": frozenset(),
}

# A real query run against the resolved store before any model call. Phrased to
# match the seeded MBA corpus (unit-economics / CAC material lives in the
# built-in knowledge base), so an empty or wrong store fails here rather than
# silently producing ungrounded specialist answers 40 scenarios later.
SMOKE_QUERY = "unit economics customer acquisition cost payback period"
# Enough hits to show the store is genuinely queryable and to record what came
# back as evidence; more would bloat every manifest for no diagnostic gain.
SMOKE_RESULT_LIMIT = 3

# Collections the chat retrieval path reads at run time, from
# knowledge/retriever.py. Only builtin_knowledge is REQUIRED (see
# _KIND_REQUIRED_COLLECTIONS); the rest are optional by design — company_docs
# is tenant data a clean checkout legitimately lacks. They are listed here so
# their pre-run state is recorded rather than discovered by accident:
# ChromaDBStore.query calls _get_or_create_collection, so any of these that is
# absent gets created empty during the run.
RETRIEVAL_COLLECTIONS = (
    "builtin_knowledge",
    "company_docs",
    "recent_research",
    "failure_cases",
)


class PreflightError(RuntimeError):
    """A precondition for a trustworthy eval run is not met.

    Always fatal: the alternative is a run whose numbers look like a result but
    measure a misconfigured system.
    """


@dataclass
class KnowledgeEvidence:
    """Everything observable about the store the run will actually read."""

    persist_path: str
    exists: bool
    required_collections: list[str]
    collections: dict[str, dict[str, Any]] = field(default_factory=dict)
    fingerprint: str = ""
    smoke_query: str = ""
    smoke_results: list[dict[str, Any]] = field(default_factory=list)
    embedding: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persist_path": self.persist_path,
            "exists": self.exists,
            "required_collections": self.required_collections,
            "collections": self.collections,
            "fingerprint": self.fingerprint,
            "smoke_query": self.smoke_query,
            "smoke_results": self.smoke_results,
            "embedding": self.embedding,
        }


# ---------------------------------------------------------------------------
# Knowledge store
# ---------------------------------------------------------------------------


def resolve_store_path(explicit: str | None, repo_root: Path) -> Path:
    """Resolve the evaluation knowledge store to an absolute path.

    Precedence, most explicit first:

    1. ``--vector-store`` on the command line,
    2. an operator-set ``VECTOR_STORE_PATH``,
    3. ``<repo_root>/chroma_db`` — the authoritative default that
       ``config.Settings`` documents.

    Every branch returns an ABSOLUTE path, and the caller exports it back into
    the environment before ``Settings`` is first read. That is the whole fix
    for defect 2: a relative value would be re-resolved against ``Path.cwd()``
    by ``Settings._resolve_paths``, which is how ``cd packages/core`` in the
    Makefile silently retargeted the store.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("VECTOR_STORE_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root / "chroma_db").resolve()


def assert_runtime_store_identity(validated: Path) -> Path:
    """Fail unless the store the RUNTIME will open is the one preflight validated.

    Necessary because ``config.get_settings()`` is NOT cached — it builds a
    fresh ``Settings`` on every call, and ``_resolve_paths`` re-resolves any
    relative path against ``Path.cwd()`` each time. Retrieval calls it
    independently (``retriever.py`` → ``ChromaDBStore(settings.vector_store_path)``),
    so validating one path and retrieving from another is a real possibility,
    not a theoretical one. Exporting an absolute value closes it; this asserts
    the closure instead of trusting it.
    """
    from openexecutive.config import get_settings

    runtime = Path(get_settings().vector_store_path).resolve()
    if runtime != validated.resolve():
        raise PreflightError(
            "Knowledge store identity drifted between preflight and runtime.\n"
            f"  validated: {validated.resolve()}\n"
            f"  runtime  : {runtime}\n"
            "Refusing to grade against a store that was never validated."
        )
    return runtime


def observe_collections(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    """Read-only census of ``names`` in an EXISTING store.

    Never creates anything: it lists what is there and reports the rest as
    absent, so a collection the run will auto-create later is visible in the
    manifest beforehand rather than appearing from nowhere.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )
    present = {c.name for c in client.list_collections()}
    out: dict[str, Any] = {}
    for name in names:
        if name in present:
            out[name] = {"present": True, "count": client.get_collection(name).count()}
        else:
            out[name] = {
                "present": False,
                "count": 0,
                "note": (
                    "absent at preflight; ChromaDBStore.query will create it "
                    "empty during the run (product behavior). Retrieval from it "
                    "yields nothing — recorded so that is not mistaken for a "
                    "validated source."
                ),
            }
    return out


def required_collections_for(scenarios: list[dict[str, Any]]) -> list[str]:
    """Collections this specific run must be able to read.

    Derived from the scenarios actually selected — their kind, plus any
    explicit ``requires_collections`` they declare — never from a blanket
    assumption about what a store ought to contain.
    """
    required: set[str] = set()
    for s in scenarios:
        kind = s.get("_kind") or "chat"
        required |= _KIND_REQUIRED_COLLECTIONS.get(kind, frozenset())
        declared = s.get("requires_collections") or []
        if isinstance(declared, list):
            required |= {str(c) for c in declared if str(c).strip()}
    return sorted(required)


def _assert_store_present(path: Path) -> None:
    """Fail unless a Chroma store already exists at ``path``.

    Checked on the filesystem BEFORE any Chroma client is constructed:
    ``PersistentClient`` creates its directory and SQLite file on
    instantiation, so touching it first would manufacture the very empty store
    this function exists to reject.

    Scope of the guarantee, stated precisely: preflight never CREATES a store,
    and never lets a run proceed against a missing or empty one. It does not
    make the run read-only — ``ChromaDBStore.query`` calls
    ``_get_or_create_collection``, so executing scenarios can still add empty
    ``company_docs`` / ``failure_cases`` collections to the target store. That
    is product behavior outside this harness's reach; evaluate against a copy
    if the target must remain untouched, and do not run against a store an API
    server holds open (Chroma does not support concurrent writers).
    """
    if not path.exists():
        raise PreflightError(
            f"Knowledge store does not exist: {path}\n"
            f"Refusing to create one — an empty store makes every specialist "
            f"answer with zero retrieval while the run still looks healthy.\n"
            f"Seed it (`make seed-knowledge`) or point --vector-store / "
            f"VECTOR_STORE_PATH at the intended store."
        )
    if not path.is_dir():
        raise PreflightError(f"Knowledge store path is not a directory: {path}")
    sqlite_file = path / "chroma.sqlite3"
    if not sqlite_file.exists():
        raise PreflightError(
            f"No chroma.sqlite3 under {path} — this is not a Chroma store.\n"
            f"Refusing to initialize one implicitly."
        )


def _fingerprint(collections: dict[str, dict[str, Any]]) -> str:
    """Stable digest of the knowledge state a run was scored against.

    Built from collection name, count, dimension and the sorted record ids —
    ids rather than contents so the value is cheap and stable, but still
    changes when the corpus is re-seeded or drifts. Two runs sharing a
    fingerprint were graded against the same knowledge.
    """
    payload = [
        {
            "name": name,
            "count": info.get("count"),
            "dimension": info.get("dimension"),
            "ids": info.get("_ids", []),
        }
        for name, info in sorted(collections.items())
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def _inspect_collection(col: Any) -> dict[str, Any]:
    """Observable facts about one collection: count, vector dimension, ids.

    Chroma returns embeddings as a numpy array, so ``or []`` and plain
    truthiness raise "truth value of an array is ambiguous" — every check here
    is an explicit None/len test for that reason.
    """
    count = col.count()
    info: dict[str, Any] = {"count": count}
    if count == 0:
        return info

    peek = col.get(limit=1, include=["embeddings"])
    vectors = peek.get("embeddings")
    has_vector = vectors is not None and len(vectors) > 0
    info["dimension"] = int(len(vectors[0])) if has_vector else None
    ids = col.get(include=[]).get("ids")
    info["_ids"] = sorted(ids) if ids is not None else []
    return info


def _embedding_identity(col: Any, info: dict[str, Any]) -> dict[str, Any]:
    """What can actually be observed about how this collection was embedded.

    The class name is the only identity Chroma exposes on a collection object;
    the concrete model is whatever that class resolves to at call time. We
    record what is readable here and do not assert a model name we cannot see.
    """
    ef = getattr(col, "_embedding_function", None)
    if ef is None:
        return {}
    return {"function": type(ef).__name__, "dimension": info.get("dimension")}


def validate_knowledge(
    path: Path, required: list[str], *, smoke: bool = True
) -> KnowledgeEvidence:
    """Assert the store is usable and capture what it contains.

    Order matters: filesystem existence is checked before a client is built,
    and every required collection is proven present AND non-empty before the
    caller is allowed to spend money on a model call.
    """
    _assert_store_present(path)

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )
    present = {c.name for c in client.list_collections()}

    evidence = KnowledgeEvidence(
        persist_path=str(path),
        exists=True,
        required_collections=list(required),
        smoke_query=SMOKE_QUERY if smoke and required else "",
    )

    missing = [c for c in required if c not in present]
    if missing:
        raise PreflightError(
            f"Required collection(s) absent from {path}: {', '.join(missing)}\n"
            f"Present: {', '.join(sorted(present)) or '(none)'}"
        )

    empty: list[str] = []
    for name in required:
        col = client.get_collection(name)
        info = _inspect_collection(col)
        evidence.collections[name] = info
        if info["count"] == 0:
            empty.append(name)
            continue
        if not evidence.embedding:
            evidence.embedding = _embedding_identity(col, info)

    if empty:
        raise PreflightError(
            f"Required collection(s) empty in {path}: {', '.join(empty)}\n"
            f"An empty collection yields zero retrieval — specialists would be "
            f"scored ungrounded. Seed the knowledge base before evaluating."
        )

    evidence.fingerprint = _fingerprint(evidence.collections)

    if smoke and required:
        evidence.smoke_results = _smoke_query(client, required[0])
        if not evidence.smoke_results:
            raise PreflightError(
                f"Retrieval smoke query returned nothing from "
                f"{required[0]!r} at {path}.\n"
                f"The store is populated but not queryable — refusing to run."
            )

    # Drop the raw id lists: they fed the fingerprint, and persisting a few
    # hundred ids per collection would bloat every manifest for no read value.
    for info in evidence.collections.values():
        info.pop("_ids", None)

    return evidence


def _smoke_query(client: Any, collection: str) -> list[dict[str, Any]]:
    """One real embedding + vector search against the resolved store.

    Uses Chroma's own local embedding function — no network call, no paid
    model — so this is safe to run unconditionally before a paid baseline.
    """
    col = client.get_collection(collection)
    res = col.query(
        query_texts=[SMOKE_QUERY],
        n_results=min(SMOKE_RESULT_LIMIT, col.count()),
        include=["metadatas", "distances"],
    )
    def _first(key: str) -> list[Any]:
        # Same numpy caveat as above: never rely on truthiness of a Chroma
        # result field.
        val = res.get(key)
        if val is None or len(val) == 0:
            return []
        return list(val[0])

    out: list[dict[str, Any]] = []
    ids = _first("ids")
    metas = _first("metadatas")
    dists = _first("distances")
    for i, chunk_id in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        out.append(
            {
                "id": chunk_id,
                "source": (meta or {}).get("filename"),
                "domain": (meta or {}).get("domain"),
                "distance": dists[i] if i < len(dists) else None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Scenario inventory
# ---------------------------------------------------------------------------


def load_inventory(
    kind: str, scenario_id: str | None
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    """Discovered inventory for this run, or raise.

    Delegates to ``openexecutive.evals.scenarios`` — the loader the HTTP eval
    routes already use. Validation the CLI needs on top of it lives here:
    a missing source directory, an unparseable user scenario, and an empty
    selection are all errors rather than an empty list that exits 0.
    """
    from openexecutive.evals.scenarios import (
        load_scenarios,
        scenarios_dir,
    )

    # Normalized: this path is written into the run manifest as evidence, and
    # a value full of ../.. segments is hard to compare between runs.
    source = scenarios_dir().resolve()
    if not source.exists():
        raise PreflightError(
            f"Scenario source directory does not exist: {source}\n"
            f"Set EVAL_SCENARIOS_PATH or restore the packaged _scenarios/ "
            f"directory. Refusing to report success on an empty inventory."
        )

    _assert_no_silently_skipped_user_scenarios()

    try:
        all_for_kind = load_scenarios(kind=kind)
    except yaml.YAMLError as exc:
        # The package loader lets built-in YAML errors propagate (correctly —
        # a corrupt built-in scenario must not be skipped). Wrap it so the CLI
        # reports it the same way as every other preflight failure instead of
        # dumping a bare traceback.
        raise PreflightError(
            f"A scenario file under {source} is not valid YAML:\n{exc}"
        ) from exc
    selected = (
        [s for s in all_for_kind if s.get("id") == scenario_id]
        if scenario_id
        else all_for_kind
    )

    if scenario_id and not selected:
        raise PreflightError(
            f"--scenario-id {scenario_id!r} matched nothing of kind {kind!r}. "
            f"Available: {', '.join(sorted(s['id'] for s in all_for_kind)) or '(none)'}"
        )
    if not selected:
        raise PreflightError(
            f"No scenarios of kind {kind!r} found in {source}.\n"
            f"A run with an empty inventory proves nothing — failing instead "
            f"of reporting success."
        )

    inventory = {
        "source_dir": str(source),
        "kind": kind,
        "scenario_id_filter": scenario_id,
        "discovered": sorted(s["id"] for s in selected),
        "discovered_count": len(selected),
        "kind_totals": _kind_totals(),
    }
    return selected, source, inventory


def _kind_totals() -> dict[str, int]:
    """Per-kind counts across the whole authoritative source.

    Derived from the loader, never hard-coded, so the reported inventory
    cannot drift from what is actually on disk.
    """
    from openexecutive.evals.scenarios import load_scenarios

    totals: dict[str, int] = {}
    for s in load_scenarios():
        totals[s.get("_kind", "chat")] = totals.get(s.get("_kind", "chat"), 0) + 1
    return dict(sorted(totals.items()))


def _assert_no_silently_skipped_user_scenarios() -> None:
    """Fail if a user scenario row exists that the loader could not parse.

    ``scenarios._load_user_scenarios`` swallows YAML errors with ``continue``
    so one bad row cannot break the eval UI. That is right for a web surface
    and wrong for a graded run: the scenario would vanish from the inventory
    with no signal. We re-read the rows and compare counts rather than change
    the shared loader's behavior.
    """
    import yaml

    try:
        from openexecutive.evals.persistence import list_user_scenarios
    except ImportError:
        # No DB layer in this install — there are no user rows to skip.
        return

    try:
        rows = list_user_scenarios()
    except sqlite3.OperationalError:
        # Table absent on a fresh checkout; built-in scenarios still load and
        # nothing could have been skipped. Narrow on purpose: a broad
        # ``except Exception`` here would disable the guard on any DB fault,
        # which is precisely when a scenario is most likely to go missing.
        return

    bad: list[str] = []
    for row in rows:
        try:
            parsed = yaml.safe_load(row["yaml"])
        except yaml.YAMLError as exc:
            bad.append(f"{row.get('id', '?')}: {exc}")
            continue
        if not isinstance(parsed, dict):
            bad.append(f"{row.get('id', '?')}: top-level YAML is not a mapping")

    if bad:
        raise PreflightError(
            "User scenario(s) failed to parse and would have been skipped "
            "silently:\n  " + "\n  ".join(bad)
        )
