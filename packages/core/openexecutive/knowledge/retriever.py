from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from openexecutive.audit import get_active_ids
from openexecutive.audit import log_event as _audit_log
from openexecutive.knowledge.review_store import (
    PRIORITY_ORDER,
    ContentType,
    Priority,
    ReviewStore,
)
from openexecutive.knowledge.store import ChromaDBStore

# Cosine distance threshold for the main retrieve() path. Hits with a
# distance > this are dropped before the top-K slice. Mirrors the value
# already used by retrieve_failures() — weak matches are noise that
# poisons grounding (e.g. a "Hi" greeting pulling a GitLab handbook
# chunk because it happens to be the closest seeded knowledge).
_DISTANCE_THRESHOLD = 0.55

# Minimum character length for RAG to fire. Below this we treat the
# message as a greeting / acknowledgement ("Hi", "ok") and skip the
# vector store entirely. Char count (not token count) because `\w+`
# matches a CJK sentence as a single token, which would incorrectly
# bypass RAG for meaningful Chinese/Japanese queries. Threshold sits
# at 3 so 3-letter business acronyms ("ROI", "CFO", "P&L") still fire.
_MIN_QUERY_CHARS = 3


def _passes_threshold(
    row: dict[str, Any], threshold: float = _DISTANCE_THRESHOLD
) -> bool:
    """True iff the Chroma row's cosine distance is within the relevance gate.

    Treats a missing/None distance as out-of-bounds (we don't surface chunks
    of unknown relevance). Uses an explicit None check rather than `... or
    1.0` because `0.0 or 1.0 == 1.0` would falsy-drop the strongest possible
    match — Chroma returns 0.0 for a verbatim hit. ``threshold`` defaults to
    the module constant but callers pass the settings-configured value.
    """
    distance = row.get("distance")
    if distance is None:
        return False
    return distance <= threshold


def _default_review_store() -> ReviewStore:
    from openexecutive.memory.episodic import DB_PATH

    return ReviewStore(db_path=DB_PATH)


def _dedupe_by_text(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate-text hits from a Chroma result list.

    Multi-domain OER sources fan each chunk out to one row per declared
    domain. A specialist query that filters by domain naturally gets one
    row per chunk, but an unfiltered call (e.g. the Executive's global
    retrieve) could see the same passage 2-5x. Preserve order so the most
    semantically relevant copy wins.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in results:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        out.append(r)
    return out


def _emit_retrieval_audit(
    *,
    query: str,
    domain_filter: list[str] | None,
    specialist_name: str | None,
    builtin_results: list[dict[str, Any]],
    company_results: list[dict[str, Any]],
    annotation_count: int,
    collection: str,
) -> None:
    """Fire-and-forget audit emit for a retrieval pass.

    Reads (session_id, turn_id) from the audit context vars set by the
    Executive at turn entry; emits None for both when called outside a
    turn (CLI, ad-hoc workflows) so the row is still captured but won't
    cluster into a session timeline. log_event already swallows.
    """
    session_id, turn_id = get_active_ids()

    def _chunks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "source": r.get("metadata", {}).get("filename"),
                "domain": r.get("metadata", {}).get("domain"),
                "distance": r.get("distance"),
                # First 400 chars is enough to recognise the passage in the
                # UI without bloating audit rows; full text lives in Chroma.
                "text_preview": (r.get("text") or "")[:400],
            }
            for r in rows
        ]

    total = len(builtin_results) + len(company_results)
    domain_str = ",".join(domain_filter) if domain_filter else "*"
    _audit_log(
        "knowledge_retrieval",
        f"retrieve({domain_str}) → {total} chunks: {query[:140]}",
        session_id=session_id,
        turn_id=turn_id,
        actor=specialist_name or "executive",
        details={
            "query": query[:300],
            "collection": collection,
            "domain_filter": domain_filter,
            "specialist": specialist_name,
            "builtin_count": len(builtin_results),
            "company_count": len(company_results),
            "annotation_count": annotation_count,
        },
        full={
            "query": query,
            "domain_filter": domain_filter,
            "specialist": specialist_name,
            "builtin_chunks": _chunks(builtin_results),
            "company_chunks": _chunks(company_results),
        },
    )


DOMAIN_ALIASES: dict[str, list[str]] = {
    "cso": ["strategy"],
    "cfo": ["finance"],
    "chro": ["hr"],
    "gc": ["legal"],
    "coo": ["operations"],
    "cmo": ["marketing"],
    "cpo": ["product", "strategy"],
    "board_comms": ["board", "finance"],
    # The talent specialist reuses the existing HR + strategy knowledge
    # domains until a dedicated `talent` knowledge corpus is seeded (Phase 2).
    "talent": ["hr", "strategy"],
}


class RetrievedEvidenceChunk(BaseModel):
    """One chunk actually supplied to a specialist in a single retrieval call.

    ``retrieval_id`` is the ONLY field that may authorise a provenance claim.
    It is minted here, per invocation, from :func:`secrets.token_urlsafe` — so
    the same underlying chunk retrieved twice carries two different tokens, and
    a token copied out of an earlier consultation matches nothing in the current
    set. That is what makes replay detectable rather than merely unlikely.

    Everything else is descriptive:

    * ``chunk_id`` — the store's persistent record id. Global and stable, hence
      useless as authority: a replayed one would still "exist".
    * ``document_label`` — the ``[filename]`` the model sees. Display text, and
      still not trustworthy as a *label*: it is now the document's real
      sanitized name (see ``document_identity`` in ``architecture-facts.yaml``)
      rather than the upload's temp-file name, but a name is guessable, is
      chosen by whoever uploaded or sent the file, and two documents from
      different ingest paths can share one. Identity lives in ``chunk_id`` /
      ``document_id``; authority lives only in ``retrieval_id``.
    * ``collection`` / ``chunk_index`` / ``distance`` — retrieval context.

    The token is NOT a secret and NOT a capability. It is an invocation-scoped
    nonce, and it deliberately encodes nothing: no filename, collection,
    ordinal, document identity, call sequence, or timestamp. Anything derivable
    from the token would be provenance the model could forge by construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    retrieval_id: str
    chunk_id: str | None = None
    document_label: str | None = None
    chunk_index: int | None = None
    text: str = ""
    collection: str = ""
    distance: float | None = None


class RetrievalSet(BaseModel):
    """The exact chunks handed to ONE specialist invocation.

    Membership in :meth:`allowed_ids` is the whole trust model: a model-written
    ``retrieval_id`` is provenance if and only if it is in this set, for this
    call. Never merge two sets, never cache one across calls, and never hold one
    on shared mutable state — a set that outlives its invocation turns replay
    from a detected attack into an accepted one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunks: tuple[RetrievedEvidenceChunk, ...] = ()

    def allowed_ids(self) -> frozenset[str]:
        """The tokens a specialist may legitimately cite on this call."""
        return frozenset(c.retrieval_id for c in self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)


# Bytes of entropy per token. 12 bytes → a 16-char urlsafe string: far past any
# realistic collision risk within one call (which is all that must hold), while
# staying short enough that a model copies it back without truncating.
_RETRIEVAL_ID_BYTES = 12

# Marker wrapping the token in the model-visible text. Distinct from the
# `[filename]` label so the mapping token → chunk needs no filename parsing,
# and a filename containing "ref:" cannot be mistaken for one.
_REF_PREFIX = "ref:"

_STRUCTURED_HEADER = (
    "### Evidence references\n\n"
    "Each retrieved passage below is tagged with an opaque reference token, "
    f"shown as [{_REF_PREFIX}<token>]. When a claim rests on a passage, copy "
    "that passage's token verbatim into the evidence entry's `retrieval_id`. "
    "Tokens are valid only for this request. Do not invent, alter, or reuse a "
    "token from an earlier request — an unrecognised token is discarded and "
    "the claim is left with no established source. A passage you did not use "
    "needs no token."
)


def _mint_retrieval_ids(count: int) -> list[str]:
    """``count`` distinct opaque tokens for one retrieval invocation.

    The uniqueness loop is not defensive theatre about entropy — at 96 bits a
    collision will not happen. It is here because *uniqueness within the set is
    a correctness precondition*: two chunks sharing a token would make the
    token→chunk mapping ambiguous, and a reader could not tell which passage a
    claim cited. Guaranteeing it structurally costs one set lookup.
    """
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < count:
        token = secrets.token_urlsafe(_RETRIEVAL_ID_BYTES)
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _coerce_chunk_index(value: Any) -> int | None:
    """``chunk_index`` from stored metadata, or None when it isn't an integer.

    Chroma metadata is untyped at read time. A bad value must become None
    ("not available") rather than a guess — the same rule the evidence contract
    applies to page/sheet/cell.
    """
    if isinstance(value, bool):  # bool is an int subclass; never a chunk index
        return None
    if isinstance(value, int):
        return value
    return None


def _coerce_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True)
class _Gathered:
    """Retrieval output after filtering, ranking, dedup and limiting.

    The single source of truth for both render paths. Everything that decides
    *which chunks survive* happens before this object exists, so a token is only
    ever minted for a chunk that is genuinely supplied to the model — a token on
    a dropped row would be authority for something the specialist never saw.

    A ``dataclass`` rather than a Pydantic model, deliberately, despite the
    repo-wide Pydantic default: the fields are raw Chroma rows built by the
    function directly above, never parsed from an external payload, so there is
    nothing here for validation to check. The public models this slice adds
    (:class:`RetrievedEvidenceChunk`, :class:`RetrievalSet`) DO cross a trust
    boundary and are Pydantic accordingly.

    ``frozen`` stops the *fields* being rebound. Read it as a hint, NOT as
    protection for the invariant that matters here, which is positional: the
    token map is keyed by ``(bucket, list index)``, so ``_build_chunks`` and
    ``_render`` must see the same lists in the same order. ``frozen`` does not
    prevent ``gathered.builtin.sort()`` or ``.pop()`` — the lists stay mutable.
    Anything that reorders, inserts, or drops a row between those two calls
    would shift every ``[ref:<token>]`` tag onto a neighbouring passage, and the
    result would be a *validated* token pointing at the wrong document: worse
    than an invalid one, because it reads as confirmed provenance and no test
    would fail (the token is still a genuine set member).

    So: do not add a pass between ``_build_chunks`` and ``_render`` in
    :func:`retrieve_structured`. If a later slice needs one, it must run before
    ``_build_chunks``, or the two walks must be fused into a single loop.
    """

    company: list[dict[str, Any]]
    research: list[dict[str, Any]]
    builtin: list[dict[str, Any]]
    annotations: list[Any]
    priority_map: dict[str, str]

    def is_empty(self) -> bool:
        return not (self.company or self.research or self.builtin or self.annotations)


def _gather(
    query: str,
    domain_filter: list[str] | None = None,
    specialist_name: str | None = None,
    n_builtin: int | None = None,
    n_company: int | None = None,
    store: ChromaDBStore | None = None,
    review_store: ReviewStore | None = None,
    distance_threshold: float | None = None,
) -> _Gathered:
    """Query, filter, rank and limit — everything except formatting.

    Extracted verbatim from the body of :func:`retrieve` so the legacy string
    path and the structured path share one pipeline; there is no second copy of
    the relevance gate that could drift.
    """
    effective_domains = domain_filter
    if effective_domains is None and specialist_name:
        effective_domains = DOMAIN_ALIASES.get(specialist_name)

    # Short-message bypass: greetings and acknowledgements never benefit
    # from semantic retrieval and reliably surface noise. Skip the ChromaDB
    # roundtrip entirely, but still emit audit so the flow chart records
    # "we considered RAG and gated it out". Longer-but-tangential queries
    # are caught by the distance threshold below, not here.
    if len(query.strip()) < _MIN_QUERY_CHARS:
        _emit_retrieval_audit(
            query=query,
            domain_filter=effective_domains,
            specialist_name=specialist_name,
            builtin_results=[],
            company_results=[],
            annotation_count=0,
            collection="builtin+company (bypassed: short query)",
        )
        return _Gathered([], [], [], [], {})

    # Resolve tunable retrieval params from settings when not explicitly
    # passed. Callers that pass values (e.g. the report workflows) keep
    # them; the chat path leaves them None and inherits the configured
    # defaults. get_settings() is uncached, so KNOWLEDGE_* env overrides
    # take effect on the next call — this is the lever the RAG ablation
    # harness toggles (KNOWLEDGE_BUILTIN_N_RESULTS=0 disables builtin RAG).
    from openexecutive.config import get_settings

    settings = get_settings()
    if n_builtin is None:
        n_builtin = settings.knowledge_builtin_n_results
    if n_company is None:
        n_company = settings.knowledge_company_n_results
    if distance_threshold is None:
        distance_threshold = settings.knowledge_distance_threshold

    if store is None:
        store = ChromaDBStore(persist_directory=settings.vector_store_path)

    rs = review_store or _default_review_store()
    rejected_builtin = rs.get_rejected_filenames(ContentType.BUILTIN)
    rejected_external = rs.get_rejected_source_ids()
    priority_map = rs.get_priority_map(ContentType.BUILTIN)

    # Over-fetch slightly so post-query text dedup (multi-domain chunks share
    # the same text across rows) still leaves us with the requested count.
    # n_builtin <= 0 disables builtin-knowledge retrieval entirely (the
    # lever the RAG ablation harness flips). Skip the query rather than
    # asking Chroma for 0 results.
    if n_builtin > 0:
        raw_builtin = _dedupe_by_text(
            store.query(
                query_text=query,
                collection=ChromaDBStore.BUILTIN_COLLECTION,
                domain_filter=effective_domains,
                n_results=n_builtin * 3,
            )
        )

        # Filter out rejected files and rejected OER sources, drop weak
        # matches, then sort by SME priority.
        filtered_builtin = [
            r
            for r in raw_builtin
            if r["metadata"].get("filename") not in rejected_builtin
            and r["metadata"].get("source_id") not in rejected_external
            and _passes_threshold(r, distance_threshold)
        ]
        filtered_builtin.sort(
            key=lambda r: PRIORITY_ORDER.get(
                priority_map.get(r["metadata"].get("filename", ""), Priority.NORMAL.value),
                1,
            )
        )
        builtin_results = filtered_builtin[:n_builtin]
    else:
        builtin_results = []

    if n_company > 0:
        raw_company = store.query(
            query_text=query,
            collection=ChromaDBStore.COMPANY_COLLECTION,
            domain_filter=effective_domains,
            n_results=n_company,
        )
        company_results = [
            r for r in raw_company if _passes_threshold(r, distance_threshold)
        ]
    else:
        company_results = []

    # Recent research artifacts — kept in a separate collection and ranked
    # BELOW curated company docs. These are unvetted, web-sourced summaries
    # from executive_research runs, so they are clearly labelled as such and
    # never blended into the company-documents section above.
    raw_research = store.query(
        query_text=query,
        collection=ChromaDBStore.RESEARCH_COLLECTION,
        domain_filter=None,  # research is cross-domain; never domain-scoped
        n_results=2,
    )
    research_results = [
        r for r in raw_research if _passes_threshold(r, distance_threshold)
    ]

    active_annotations = rs.list_annotations(domains=effective_domains, active_only=True)

    # Audit emit — always fire, even on empty results, so the flow chart
    # can show "we asked but found nothing" rather than silently omitting
    # the retrieval step. Fire-and-forget; never blocks/breaks the caller.
    _emit_retrieval_audit(
        query=query,
        domain_filter=effective_domains,
        specialist_name=specialist_name,
        builtin_results=builtin_results,
        company_results=company_results,
        annotation_count=len(active_annotations),
        collection="builtin+company",
    )

    return _Gathered(
        company=company_results,
        research=research_results,
        builtin=builtin_results,
        annotations=list(active_annotations),
        priority_map=priority_map,
    )


# Bucket order is load-bearing: it is the order the sections are rendered in,
# and (paired with the row index) the key that maps a minted token back to the
# exact chunk it identifies. `_render` and `_build_chunks` MUST walk it
# identically or a token would be printed beside the wrong passage.
_BUCKETS: tuple[tuple[str, str], ...] = (
    ("company", ChromaDBStore.COMPANY_COLLECTION),
    ("research", ChromaDBStore.RESEARCH_COLLECTION),
    ("builtin", ChromaDBStore.BUILTIN_COLLECTION),
)


def _render(gathered: _Gathered, tokens: dict[tuple[str, int], str] | None) -> str:
    """Format gathered chunks for a specialist.

    With ``tokens is None`` this emits the legacy string, byte for byte — that
    equivalence is pinned by a golden test, because ~57 call sites across ~40
    modules consume it and none of them are migrating in this slice.

    With a token map it additionally prefixes each chunk with
    ``[ref:<token>]``. Only chunk sections are tagged. SME annotations are not
    retrieved chunks — they have no store record and no ``retrieval_id`` — so
    they carry no token and can never establish retrieval provenance.
    """
    if gathered.is_empty():
        return ""

    def tag(bucket: str, index: int) -> str:
        if tokens is None:
            return ""
        token = tokens.get((bucket, index))
        return f"[{_REF_PREFIX}{token}] " if token else ""

    parts: list[str] = []
    if tokens:
        parts.append(_STRUCTURED_HEADER)

    if gathered.company:
        parts.append("### From your company documents:")
        for i, r in enumerate(gathered.company):
            filename = r["metadata"].get("filename", "unknown")
            parts.append(f"{tag('company', i)}[{filename}] {r['text']}")

    if gathered.research:
        parts.append(
            "### Recent research (unverified, web-sourced — weigh below "
            "company documents):"
        )
        for i, r in enumerate(gathered.research):
            created = r["metadata"].get("created_at", "")
            when = f" — {created}" if created else ""
            parts.append(f"{tag('research', i)}[recent research{when}] {r['text']}")

    if gathered.builtin:
        parts.append("### From executive knowledge base:")
        for i, r in enumerate(gathered.builtin):
            filename = r["metadata"].get("filename", "unknown")
            prio = gathered.priority_map.get(filename, Priority.NORMAL.value)
            prefix = "[verified - priority source] " if prio == Priority.HIGH.value else ""
            parts.append(f"{tag('builtin', i)}[{filename}] {prefix}{r['text']}")

    if gathered.annotations:
        parts.append("### SME corrections and context:")
        for ann in gathered.annotations:
            parts.append(f"[SME annotation] {ann.correction}")

    return "\n\n".join(parts)


def _build_chunks(
    gathered: _Gathered,
) -> tuple[list[RetrievedEvidenceChunk], dict[tuple[str, int], str]]:
    """Mint one token per retained chunk; return the chunks and the render map.

    Called only after ``_gather`` has finished filtering, so nothing here can
    mint a token for a row that was deduped, threshold-dropped, rejected by the
    review store, or sliced off past the top-K.
    """
    rows: list[tuple[str, int, str, dict[str, Any]]] = [
        (bucket, i, collection, row)
        for bucket, collection in _BUCKETS
        for i, row in enumerate(getattr(gathered, bucket))
    ]
    ids = _mint_retrieval_ids(len(rows))

    chunks: list[RetrievedEvidenceChunk] = []
    token_map: dict[tuple[str, int], str] = {}
    for (bucket, i, collection, row), token in zip(rows, ids, strict=True):
        token_map[(bucket, i)] = token
        meta = row.get("metadata") or {}
        distance = row.get("distance")
        # `isinstance(True, int)` holds, and a bool distance would silently
        # become 1.0 — a plausible-looking score for a value that is not one.
        if isinstance(distance, bool):
            distance = None
        chunks.append(
            RetrievedEvidenceChunk(
                retrieval_id=token,
                chunk_id=_coerce_str(row.get("id")),
                document_label=_coerce_str(meta.get("filename")),
                chunk_index=_coerce_chunk_index(meta.get("chunk_index")),
                text=row.get("text") or "",
                collection=collection,
                distance=float(distance) if isinstance(distance, int | float) else None,
            )
        )
    return chunks, token_map


def retrieve(
    query: str,
    domain_filter: list[str] | None = None,
    specialist_name: str | None = None,
    n_builtin: int | None = None,
    n_company: int | None = None,
    store: ChromaDBStore | None = None,
    review_store: ReviewStore | None = None,
    distance_threshold: float | None = None,
) -> str:
    """Legacy retrieval. Returns the formatted context string, unchanged.

    Signature, semantics and output bytes are exactly what they were before
    provenance tokens existed. Callers that want verifiable evidence identity
    use :func:`retrieve_structured` instead; nothing else needs to change.
    """
    return _render(
        _gather(
            query,
            domain_filter=domain_filter,
            specialist_name=specialist_name,
            n_builtin=n_builtin,
            n_company=n_company,
            store=store,
            review_store=review_store,
            distance_threshold=distance_threshold,
        ),
        None,
    )


def retrieve_structured(
    query: str,
    domain_filter: list[str] | None = None,
    specialist_name: str | None = None,
    n_builtin: int | None = None,
    n_company: int | None = None,
    store: ChromaDBStore | None = None,
    review_store: ReviewStore | None = None,
    distance_threshold: float | None = None,
) -> tuple[str, RetrievalSet]:
    """Retrieve, and mint a fresh provenance token for every retained chunk.

    Returns ``(context_string, retrieval_set)``. The string is the legacy
    rendering plus a ``[ref:<token>]`` tag on each chunk and a header telling
    the model what the tags are for; the set is what a model-written
    ``retrieval_id`` is later checked against.

    The set is valid for THIS call only. Pass it down the invocation that
    produced it and let it go — see :class:`RetrievalSet`.
    """
    gathered = _gather(
        query,
        domain_filter=domain_filter,
        specialist_name=specialist_name,
        n_builtin=n_builtin,
        n_company=n_company,
        store=store,
        review_store=review_store,
        distance_threshold=distance_threshold,
    )
    if gathered.is_empty():
        return "", RetrievalSet()
    chunks, token_map = _build_chunks(gathered)
    return _render(gathered, token_map), RetrievalSet(chunks=tuple(chunks))


def retrieve_failures(
    query: str,
    domain_filter: list[str] | None = None,
    specialist_name: str | None = None,
    n_results: int = 2,
    store: ChromaDBStore | None = None,
) -> str:
    """Query the failure_cases collection and return formatted context.

    Returns an empty string if no result clears the distance threshold —
    tangential failure stories are noise, so we prefer surfacing nothing
    over surfacing a poor match.
    """
    from openexecutive.config import get_settings

    settings = get_settings()
    if store is None:
        store = ChromaDBStore(persist_directory=settings.vector_store_path)

    effective_domains = domain_filter
    if effective_domains is None and specialist_name:
        effective_domains = DOMAIN_ALIASES.get(specialist_name)

    raw = store.query(
        query_text=query,
        collection=ChromaDBStore.FAILURES_COLLECTION,
        domain_filter=effective_domains,
        n_results=n_results * 2,
    )

    # Cosine distance threshold (configurable via KNOWLEDGE_DISTANCE_THRESHOLD):
    # a larger distance means the match is too weak to be useful.
    threshold = settings.knowledge_distance_threshold
    filtered = _dedupe_by_text([r for r in raw if r["distance"] <= threshold])
    results = filtered[:n_results]

    # Audit emit (failure cases collection). Fire even when empty so the
    # timeline shows we considered failure stories and rejected them.
    _emit_retrieval_audit(
        query=query,
        domain_filter=effective_domains,
        specialist_name=specialist_name,
        builtin_results=results,
        company_results=[],
        annotation_count=0,
        collection="failure_cases",
    )

    if not results:
        return ""

    parts = ["### Relevant failure cases:"]
    for r in results:
        filename = r["metadata"].get("filename", "unknown")
        parts.append(f"[{filename}] {r['text']}")
    return "\n\n".join(parts)
