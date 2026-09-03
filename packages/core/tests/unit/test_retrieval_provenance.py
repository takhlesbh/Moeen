"""Retrieval Provenance Phase A — the trust boundary and its compatibility.

One rule is under test throughout:

    a model-written ``retrieval_id`` is provenance IF AND ONLY IF it is in the
    retrieval set minted for THAT specialist invocation.

Everything else a model can write about a source — filename, label, page — is
display text with no authority, and stays that way whether or not a token is
present. The attack tests (A–F) each attempt to establish provenance by some
route other than set membership; all must fail.

The compatibility half pins the other half of the bargain: ~57 ``retrieve()``
call expressions across ~40 modules keep receiving the byte-identical string
they receive today.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from openexecutive.knowledge import retriever as R
from openexecutive.knowledge.retriever import (
    RetrievalSet,
    RetrievedEvidenceChunk,
    _mint_retrieval_ids,
    retrieve,
    retrieve_structured,
)
from openexecutive.knowledge.store import ChromaDBStore
from openexecutive.specialists.result_contract import (
    emit_specialist_result_tool,
    parse_specialist_result,
)

# ---------------------------------------------------------------------------
# Fixtures — a deterministic retrieval world with no chromadb/network/LLM.
# ---------------------------------------------------------------------------

BUILTIN_ROWS: list[dict[str, Any]] = [
    {
        "id": "md5-builtin-0",
        "text": "Builtin chunk one about unit economics.",
        "metadata": {"filename": "unit-economics.md", "domain": "finance", "chunk_index": 0},
        "distance": 0.10,
    },
    {
        "id": "md5-builtin-1",
        "text": "Builtin chunk two about burn multiple.",
        "metadata": {"filename": "hi-prio.md", "domain": "finance", "chunk_index": 3},
        "distance": 0.20,
    },
    # Same TEXT as row 0 — must be deduped away, and must never get a token.
    {
        "id": "md5-builtin-dupe",
        "text": "Builtin chunk one about unit economics.",
        "metadata": {"filename": "dupe.md", "domain": "strategy", "chunk_index": 9},
        "distance": 0.25,
    },
    # Beyond the distance threshold — dropped, and must never get a token.
    {
        "id": "md5-builtin-weak",
        "text": "Too far away to matter.",
        "metadata": {"filename": "weak.md", "domain": "finance"},
        "distance": 0.95,
    },
    # Rejected by the review store — dropped, and must never get a token.
    {
        "id": "md5-builtin-rejected",
        "text": "Rejected file chunk.",
        "metadata": {"filename": "rejected.md", "domain": "finance"},
        "distance": 0.11,
    },
]

COMPANY_ROWS: list[dict[str, Any]] = [
    {
        "id": "md5-company-0",
        "text": "Company doc chunk about Q3 runway.",
        "metadata": {"filename": "Q3-board-deck.pdf", "domain": "finance", "chunk_index": 2},
        "distance": 0.12,
    },
    {
        "id": "md5-company-weak",
        "text": "Company weak chunk.",
        "metadata": {"filename": "far.pdf", "domain": "finance"},
        "distance": 0.99,
    },
]

RESEARCH_ROWS: list[dict[str, Any]] = [
    {
        "id": "md5-research-0",
        "text": "Research artifact body.",
        "metadata": {
            "filename": "recent_research_2026-05-29",
            "created_at": "2026-05-29T00:00:00Z",
        },
        "distance": 0.30,
    },
]


class FixedStore:
    """Returns the fixture rows verbatim, so both paths see identical input."""

    def __init__(
        self,
        builtin: list[dict[str, Any]] | None = None,
        company: list[dict[str, Any]] | None = None,
        research: list[dict[str, Any]] | None = None,
    ) -> None:
        self.builtin = BUILTIN_ROWS if builtin is None else builtin
        self.company = COMPANY_ROWS if company is None else company
        self.research = RESEARCH_ROWS if research is None else research

    def query(
        self,
        *,
        query_text: str,
        collection: str,
        domain_filter: Any,
        n_results: int,
    ) -> list[dict[str, Any]]:
        if collection == ChromaDBStore.BUILTIN_COLLECTION:
            return list(self.builtin)[:n_results]
        if collection == ChromaDBStore.COMPANY_COLLECTION:
            return list(self.company)[:n_results]
        if collection == ChromaDBStore.RESEARCH_COLLECTION:
            return list(self.research)[:n_results]
        return []


class _Ann:
    def __init__(self, correction: str) -> None:
        self.correction = correction


def _review_store(annotations: list[_Ann] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        get_rejected_filenames=lambda _ct: {"rejected.md"},
        get_rejected_source_ids=lambda: set(),
        get_priority_map=lambda _ct: {"hi-prio.md": "high"},
        list_annotations=lambda domains=None, active_only=True: (
            [_Ann("SME says watch the churn cohort.")]
            if annotations is None
            else annotations
        ),
    )


@pytest.fixture(autouse=True)
def _no_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit is fire-and-forget I/O; irrelevant to every assertion here."""
    monkeypatch.setattr(R, "_emit_retrieval_audit", lambda **_kw: None)


def _call(structured: bool, store: FixedStore | None = None, **over: Any) -> Any:
    kwargs: dict[str, Any] = {
        "query": "What is our runway and burn multiple?",
        "specialist_name": "cfo",
        "n_builtin": 3,
        "n_company": 2,
        "store": store or FixedStore(),
        "review_store": _review_store(),
        "distance_threshold": 0.55,
    }
    kwargs.update(over)
    return retrieve_structured(**kwargs) if structured else retrieve(**kwargs)


# The exact legacy bytes, transcribed from a run against the baseline commit
# (e9033e5) BEFORE any change in this slice. Hard-coded rather than computed so
# the test cannot drift along with the implementation it is guarding.
LEGACY_GOLDEN = (
    "### From your company documents:\n"
    "\n"
    "[Q3-board-deck.pdf] Company doc chunk about Q3 runway.\n"
    "\n"
    "### Recent research (unverified, web-sourced — weigh below company documents):\n"
    "\n"
    "[recent research — 2026-05-29T00:00:00Z] Research artifact body.\n"
    "\n"
    "### From executive knowledge base:\n"
    "\n"
    "[hi-prio.md] [verified - priority source] Builtin chunk two about burn multiple.\n"
    "\n"
    "[unit-economics.md] Builtin chunk one about unit economics.\n"
    "\n"
    "### SME corrections and context:\n"
    "\n"
    "[SME annotation] SME says watch the churn cohort."
)


# ---------------------------------------------------------------------------
# Compatibility — the legacy string must not move.
# ---------------------------------------------------------------------------


def test_legacy_retrieve_is_byte_identical_to_baseline() -> None:
    """The whole no-migration bargain rests on this one assertion."""
    assert _call(structured=False) == LEGACY_GOLDEN


def test_legacy_retrieve_carries_no_provenance_tokens() -> None:
    out = _call(structured=False)
    assert "ref:" not in out
    assert "Evidence references" not in out


def test_store_query_row_consumers_tolerate_additive_id() -> None:
    """Row dicts gained an "id" key; existing consumers read by name."""
    from openexecutive.talent.graph import _result

    row = {
        "id": "md5-x",
        "text": "t",
        "metadata": {"candidate_id": "c1", "engagement_id": "e1", "stage": "screen"},
        "distance": 0.2,
    }
    assert _result(row)["candidate_id"] == "c1"


def test_store_query_pairs_ids_positionally_and_degrades_on_short_ids() -> None:
    """A short/absent ids array must yield None, never a mispaired id."""

    class FakeCollection:
        def count(self) -> int:
            return 2

        def query(self, **_kw: Any) -> dict[str, Any]:
            return {
                "ids": [["only-one"]],  # shorter than documents
                "documents": [["doc-a", "doc-b"]],
                "metadatas": [[{"filename": "a"}, {"filename": "b"}]],
                "distances": [[0.1, 0.2]],
            }

    store = ChromaDBStore.__new__(ChromaDBStore)
    store._get_or_create_collection = lambda _name: FakeCollection()  # type: ignore[method-assign]
    rows = store.query(query_text="q", collection="c")
    assert [r["id"] for r in rows] == ["only-one", None]
    assert [r["text"] for r in rows] == ["doc-a", "doc-b"]


# ---------------------------------------------------------------------------
# Token minting and set construction.
# ---------------------------------------------------------------------------


def test_structured_mints_one_token_per_retained_chunk_only() -> None:
    text, rset = _call(structured=True)
    # 1 company + 1 research + 2 builtin survive filtering; the deduped,
    # threshold-dropped and review-rejected rows must not be tokenised.
    assert len(rset) == 4
    labels = {c.document_label for c in rset.chunks}
    assert labels == {"Q3-board-deck.pdf", "recent_research_2026-05-29",
                      "hi-prio.md", "unit-economics.md"}
    for gone in ("dupe.md", "weak.md", "rejected.md", "far.pdf"):
        assert gone not in labels
    # The header itself contains a literal "[ref:<token>]" example, so count
    # the real markers rather than the substring.
    body = text.split("### From your company documents:", 1)[1]
    assert body.count("[ref:") == 4
    for chunk in rset.chunks:
        assert body.count(f"[ref:{chunk.retrieval_id}] ") == 1


def test_no_token_is_minted_for_discarded_rows() -> None:
    """Explicitly: a dropped row's chunk_id never appears in the set."""
    _text, rset = _call(structured=True)
    chunk_ids = {c.chunk_id for c in rset.chunks}
    for dropped in (
        "md5-builtin-dupe",
        "md5-builtin-weak",
        "md5-builtin-rejected",
        "md5-company-weak",
    ):
        assert dropped not in chunk_ids


def test_retrieval_ids_unique_within_one_invocation() -> None:
    _text, rset = _call(structured=True)
    ids = [c.retrieval_id for c in rset.chunks]
    assert len(set(ids)) == len(ids)
    assert rset.allowed_ids() == frozenset(ids)


def test_mint_returns_distinct_tokens() -> None:
    tokens = _mint_retrieval_ids(500)
    assert len(set(tokens)) == 500
    assert all(t for t in tokens)


def test_same_chunk_gets_a_different_token_each_invocation() -> None:
    """The property that turns replay from unlikely into detectable."""
    _t1, first = _call(structured=True)
    _t2, second = _call(structured=True)
    by_chunk_first = {c.chunk_id: c.retrieval_id for c in first.chunks}
    by_chunk_second = {c.chunk_id: c.retrieval_id for c in second.chunks}
    assert by_chunk_first.keys() == by_chunk_second.keys()  # same underlying chunks
    for chunk_id in by_chunk_first:
        assert by_chunk_first[chunk_id] != by_chunk_second[chunk_id]
    assert not (first.allowed_ids() & second.allowed_ids())


def test_token_encodes_nothing_about_the_chunk() -> None:
    """No filename, collection, ordinal, or index leaks into the token."""
    _text, rset = _call(structured=True)
    for chunk in rset.chunks:
        token = chunk.retrieval_id
        assert chunk.document_label not in token
        assert chunk.collection not in token
        assert str(chunk.chunk_id) not in token


def test_structured_text_maps_token_to_its_own_chunk() -> None:
    """Each token is printed immediately before the passage it identifies."""
    text, rset = _call(structured=True)
    for chunk in rset.chunks:
        marker = f"[ref:{chunk.retrieval_id}] "
        assert marker in text
        after = text.split(marker, 1)[1]
        # The tagged line runs to the next blank line; the chunk's own text
        # must be on it (not merely somewhere else in the block).
        line = after.split("\n\n", 1)[0]
        assert chunk.text in line


def test_sme_annotations_get_no_token() -> None:
    """Annotations are not retrieved chunks and cannot back a claim."""
    text, rset = _call(structured=True)
    sme_line = [ln for ln in text.split("\n\n") if ln.startswith("[SME annotation]")]
    assert sme_line and "ref:" not in sme_line[0]
    assert all("SME says" not in c.text for c in rset.chunks)


def test_empty_retrieval_returns_empty_string_and_empty_set() -> None:
    store = FixedStore(builtin=[], company=[], research=[])
    text, rset = retrieve_structured(
        query="What is our runway?",
        specialist_name="cfo",
        store=store,
        review_store=_review_store(annotations=[]),
        distance_threshold=0.55,
    )
    assert text == ""
    assert rset.allowed_ids() == frozenset()


def test_short_query_bypass_returns_empty_set() -> None:
    text, rset = _call(structured=True, query="Hi")
    assert text == ""
    assert len(rset) == 0


def test_retrieval_set_is_frozen() -> None:
    """Immutability is load-bearing: a set mutated after minting could be
    widened to admit a token that was never rendered to the model."""
    from pydantic import ValidationError

    _text, rset = _call(structured=True)
    with pytest.raises(ValidationError):
        rset.chunks = ()  # type: ignore[misc]
    with pytest.raises(ValidationError):
        rset.chunks[0].retrieval_id = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Parser plumbing for the attack tests.
# ---------------------------------------------------------------------------


class _Block:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def _tool_message(evidence: list[dict[str, Any]], narrative: str = "Analysis.") -> Any:
    return SimpleNamespace(
        content=[
            _Block(
                type="tool_use",
                name="emit_specialist_result",
                input={
                    "narrative": narrative,
                    "claims": [
                        {
                            "claim_id": "c1",
                            "text": "Runway is 14 months.",
                            "claim_type": "source_fact",
                            "attribution": "independent_evidence",
                            "evidence": evidence,
                        }
                    ],
                },
            )
        ]
    )


def _parse(evidence: list[dict[str, Any]], allowed: frozenset[str] | None) -> Any:
    return parse_specialist_result(
        _tool_message(evidence),
        specialist="cfo",
        model="test",
        allowed_retrieval_ids=allowed,
    )


def _only_ref(result: Any) -> Any:
    return result.claims[0].evidence[0]


# ---------------------------------------------------------------------------
# ATTACK A — fabricated filename, no valid token.
# ---------------------------------------------------------------------------


def test_attack_a_fabricated_filename_establishes_no_provenance() -> None:
    _text, rset = _call(structured=True)
    result = _parse(
        [{
            "kind": "document",
            "label": "[KPMG_Audit_2024.pdf]",
            "filename": "KPMG_Audit_2024.pdf",
        }],
        rset.allowed_ids(),
    )
    ref = _only_ref(result)
    assert ref.retrieval_id is None, "no token ⇒ no provenance"
    # The label survives as display text — but proves nothing.
    assert ref.filename == "KPMG_Audit_2024.pdf"
    assert not result.degraded


def test_attack_a_real_filename_without_token_still_proves_nothing() -> None:
    """Naming a genuinely retrieved document is not evidence of retrieval."""
    _text, rset = _call(structured=True)
    result = _parse(
        [{"kind": "document", "label": "[Q3-board-deck.pdf]",
          "filename": "Q3-board-deck.pdf"}],
        rset.allowed_ids(),
    )
    assert _only_ref(result).retrieval_id is None


# ---------------------------------------------------------------------------
# ATTACK B — real filename, token for a chunk not in this call.
# ---------------------------------------------------------------------------


def test_attack_b_wrong_chunk_token_is_stripped_and_reported() -> None:
    _text, rset = _call(structured=True)
    result = _parse(
        [{
            "kind": "document",
            "label": "[Q3-board-deck.pdf]",
            "filename": "Q3-board-deck.pdf",
            "retrieval_id": "totally-made-up-token",
        }],
        rset.allowed_ids(),
    )
    assert _only_ref(result).retrieval_id is None
    assert result.degraded
    assert "not in this call's retrieval set" in (result.degraded_reason or "")


def test_rejected_token_is_never_echoed_into_the_reason() -> None:
    """The reason is logged, persisted and UI-surfaced; it reports a count."""
    _text, rset = _call(structured=True)
    hostile = "TOKEN-" + ("A" * 5000)
    result = _parse(
        [{"kind": "document", "label": "x", "retrieval_id": hostile}],
        rset.allowed_ids(),
    )
    reason = result.degraded_reason or ""
    assert hostile not in reason
    assert "A" * 100 not in reason
    assert len(reason) < 500


# ---------------------------------------------------------------------------
# ATTACK C — two chunks sharing one display filename.
# ---------------------------------------------------------------------------


def test_attack_c_identical_filenames_remain_distinguishable_by_token() -> None:
    """Filename collision is real today (temp-name ingest); tokens survive it."""
    collide = [
        {"id": "md5-a", "text": "Version A: runway 14 months.",
         "metadata": {"filename": "report.pdf", "chunk_index": 0}, "distance": 0.10},
        {"id": "md5-b", "text": "Version B: runway 6 months.",
         "metadata": {"filename": "report.pdf", "chunk_index": 1}, "distance": 0.11},
    ]
    text, rset = _call(
        structured=True,
        store=FixedStore(builtin=[], company=collide, research=[]),
    )
    assert len(rset) == 2
    a, b = rset.chunks
    assert a.document_label == b.document_label == "report.pdf"
    assert a.retrieval_id != b.retrieval_id
    assert a.chunk_id != b.chunk_id

    # Citing the token for B must resolve to B, not to "some report.pdf".
    result = _parse(
        [{"kind": "document", "label": "[report.pdf]", "filename": "report.pdf",
          "retrieval_id": b.retrieval_id}],
        rset.allowed_ids(),
    )
    cited = _only_ref(result).retrieval_id
    assert cited == b.retrieval_id
    resolved = {c.retrieval_id: c for c in rset.chunks}[cited]
    assert resolved.text == "Version B: runway 6 months."
    assert f"[ref:{b.retrieval_id}] [report.pdf] Version B" in text


# ---------------------------------------------------------------------------
# ATTACK D — invented page (and the rest of the unavailable provenance).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("page", 12),
        ("sheet", "Sheet1"),
        ("cell_range", "B4:C9"),
        ("url", "https://example.com/audit.pdf"),
        ("retrieved_at", "2026-08-30T00:00:00Z"),
        ("chunk_index", 3),
        ("provenance_note", "verified against source"),
    ],
)
def test_attack_d_unavailable_provenance_is_stripped_even_with_a_valid_token(
    field: str, value: Any
) -> None:
    """A valid token must not launder the fields the model cannot know."""
    _text, rset = _call(structured=True)
    good = next(iter(rset.allowed_ids()))
    result = _parse(
        [{"kind": "document", "label": "[Q3-board-deck.pdf]",
          "retrieval_id": good, field: value}],
        rset.allowed_ids(),
    )
    ref = _only_ref(result)
    assert getattr(ref, field) is None
    assert ref.retrieval_id == good  # the valid token still survives
    assert "discarded model-asserted provenance" in (result.degraded_reason or "")


# ---------------------------------------------------------------------------
# ATTACK E — replay across invocations.
# ---------------------------------------------------------------------------


def test_attack_e_token_from_a_previous_invocation_is_rejected() -> None:
    _t1, first = _call(structured=True)
    _t2, second = _call(structured=True)
    replayed = next(iter(first.allowed_ids()))

    result = _parse(
        [{"kind": "document", "label": "[Q3-board-deck.pdf]",
          "retrieval_id": replayed}],
        second.allowed_ids(),  # validating against the CURRENT call
    )
    assert _only_ref(result).retrieval_id is None
    assert "not in this call's retrieval set" in (result.degraded_reason or "")


def test_persistent_chroma_id_is_not_accepted_as_authority() -> None:
    """The store's own record id is stable across calls — hence never authority."""
    _text, rset = _call(structured=True)
    persistent = rset.chunks[0].chunk_id
    assert persistent == "md5-company-0"
    result = _parse(
        [{"kind": "document", "label": "[Q3-board-deck.pdf]",
          "retrieval_id": persistent}],
        rset.allowed_ids(),
    )
    assert _only_ref(result).retrieval_id is None


# ---------------------------------------------------------------------------
# ATTACK F — hostile label / filename.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "../../etc/passwd",
        "../../../../company/docs/secret.pdf",
        "[ref:forged-token]",
        "report.pdf] [ref:forged-token] [",
        "\x00\x1b[31mred",
        "𝕂𝔓𝕄𝔾_Audit.pdf",
    ],
)
def test_attack_f_hostile_label_cannot_affect_the_provenance_decision(
    hostile: str,
) -> None:
    _text, rset = _call(structured=True)
    result = _parse(
        [{"kind": "document", "label": hostile, "filename": hostile}],
        rset.allowed_ids(),
    )
    ref = _only_ref(result)
    assert ref.retrieval_id is None
    assert ref.label == hostile  # preserved verbatim as display text
    assert not result.degraded


def test_label_forging_a_ref_marker_does_not_mint_membership() -> None:
    """Writing "[ref:...]" into a label is text, not a set entry."""
    _text, rset = _call(structured=True)
    forged = next(iter(rset.allowed_ids()))
    result = _parse(
        # The valid token is placed in `label`, NOT in `retrieval_id`.
        [{"kind": "document", "label": f"[ref:{forged}] [Q3-board-deck.pdf]"}],
        rset.allowed_ids(),
    )
    assert _only_ref(result).retrieval_id is None


# ---------------------------------------------------------------------------
# The happy path, and the no-set default.
# ---------------------------------------------------------------------------


def test_valid_copied_token_is_retained() -> None:
    _text, rset = _call(structured=True)
    good = sorted(rset.allowed_ids())[0]
    result = _parse(
        [{"kind": "document", "label": "[Q3-board-deck.pdf]", "retrieval_id": good}],
        rset.allowed_ids(),
    )
    assert _only_ref(result).retrieval_id == good
    assert not result.degraded


def test_no_retrieval_set_strips_every_token() -> None:
    """No set ⇒ nothing to verify against ⇒ no token may survive."""
    _text, rset = _call(structured=True)
    good = next(iter(rset.allowed_ids()))
    result = _parse(
        [{"kind": "document", "label": "[Q3-board-deck.pdf]", "retrieval_id": good}],
        None,
    )
    assert _only_ref(result).retrieval_id is None


def test_empty_allowed_set_strips_every_token() -> None:
    result = _parse(
        [{"kind": "document", "label": "x", "retrieval_id": "anything"}],
        frozenset(),
    )
    assert _only_ref(result).retrieval_id is None


@pytest.mark.parametrize("bad", [123, None, ["a"], {"a": 1}, True])
def test_non_string_token_is_stripped_without_raising(bad: Any) -> None:
    """An unhashable token must degrade, never explode inside the parser."""
    result = _parse(
        [{"kind": "document", "label": "x", "retrieval_id": bad}],
        frozenset({"real"}),
    )
    assert _only_ref(result).retrieval_id is None
    assert not result.claims[0].evidence[0].model_dump().get("retrieval_id")


def test_claim_with_no_evidence_field_is_unaffected() -> None:
    result = parse_specialist_result(
        _tool_message([]),
        specialist="cfo",
        model="t",
        allowed_retrieval_ids=frozenset({"x"}),
    )
    assert result.claims[0].evidence == ()
    assert not result.degraded


# ---------------------------------------------------------------------------
# Tool schema — cache stability.
# ---------------------------------------------------------------------------


def test_tool_schema_exposes_retrieval_id_in_sorted_position() -> None:
    tool = emit_specialist_result_tool()
    ev = tool["input_schema"]["properties"]["claims"]["items"]["properties"]["evidence"]
    props = ev["items"]["properties"]
    assert "retrieval_id" in props
    assert list(props) == sorted(props), "cache stability depends on sorted keys"
    assert ev["items"]["required"] == ["kind", "label"]


@pytest.mark.parametrize("forbidden", ["page", "sheet", "cell_range", "url",
                                       "retrieved_at", "provenance_note"])
def test_tool_schema_still_hides_unavailable_provenance(forbidden: str) -> None:
    tool = emit_specialist_result_tool()
    props = (tool["input_schema"]["properties"]["claims"]["items"]["properties"]
             ["evidence"]["items"]["properties"])
    assert forbidden not in props


def test_tool_schema_is_byte_stable_across_calls() -> None:
    import json

    a = json.dumps(emit_specialist_result_tool(), sort_keys=False)
    b = json.dumps(emit_specialist_result_tool(), sort_keys=False)
    assert a == b


# ---------------------------------------------------------------------------
# Router + CFO threading, including concurrency.
# ---------------------------------------------------------------------------


def test_router_only_takes_the_structured_path_for_opted_in_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.orchestrator import router as router_mod

    seen: list[str] = []

    def fake_retrieve(*, query: str, specialist_name: str) -> str:
        seen.append(f"legacy:{specialist_name}")
        return "LEGACY"

    def fake_structured(
        *, query: str, specialist_name: str
    ) -> tuple[str, RetrievalSet]:
        seen.append(f"structured:{specialist_name}")
        return "STRUCTURED", RetrievalSet(
            chunks=(RetrievedEvidenceChunk(retrieval_id="tok-1"),)
        )

    monkeypatch.setattr(R, "retrieve", fake_retrieve)
    monkeypatch.setattr(R, "retrieve_structured", fake_structured)

    cfo_text, cfo_set = asyncio.run(
        router_mod._retrieve_for_call({"specialist": "cfo", "query": "q"})
    )
    cso_text, cso_set = asyncio.run(
        router_mod._retrieve_for_call({"specialist": "cso", "query": "q"})
    )

    assert (cfo_text, cfo_set.allowed_ids()) == ("STRUCTURED", frozenset({"tok-1"}))
    assert (cso_text, cso_set) == ("LEGACY", None)
    assert seen == ["structured:cfo", "legacy:cso"]


def test_route_to_specialist_drops_the_set_for_non_opted_in_agents() -> None:
    """BaseAgent.analyze has no such parameter; passing it would TypeError."""
    from openexecutive.orchestrator import router as router_mod

    captured: dict[str, Any] = {}

    class Plain:
        async def analyze(self, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

    rset = RetrievalSet(chunks=(RetrievedEvidenceChunk(retrieval_id="t"),))
    router_mod.SPECIALIST_REGISTRY["_plain_test"] = Plain()  # type: ignore[assignment]
    try:
        out = asyncio.run(
            router_mod.route_to_specialist(
                specialist_name="_plain_test", query="q", retrieval_set=rset
            )
        )
    finally:
        del router_mod.SPECIALIST_REGISTRY["_plain_test"]
    assert out == "ok"
    assert "retrieval_set" not in captured


def test_route_to_specialist_forwards_the_set_to_cfo() -> None:
    from openexecutive.orchestrator import router as router_mod

    captured: dict[str, Any] = {}

    class OptedIn:
        accepts_retrieval_set = True

        async def analyze(self, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

    rset = RetrievalSet(chunks=(RetrievedEvidenceChunk(retrieval_id="t"),))
    router_mod.SPECIALIST_REGISTRY["_opted_test"] = OptedIn()  # type: ignore[assignment]
    try:
        asyncio.run(
            router_mod.route_to_specialist(
                specialist_name="_opted_test", query="q", retrieval_set=rset
            )
        )
    finally:
        del router_mod.SPECIALIST_REGISTRY["_opted_test"]
    assert captured["retrieval_set"] is rset


def test_prebuilt_knowledge_map_supplies_no_retrieval_set() -> None:
    """Caller-supplied text carries no tokens, so it must authorise nothing."""
    from openexecutive.orchestrator import router as router_mod
    from openexecutive.specialists.routed_output import RoutedSpecialistOutput

    captured: list[Any] = []

    # route_parallel dispatches through route_to_specialist_structured (Phase
    # 3B2); the structured dispatcher is what receives the retrieval set.
    async def fake_route(**kwargs: Any) -> RoutedSpecialistOutput:
        captured.append(kwargs.get("retrieval_set"))
        return RoutedSpecialistOutput(specialist="cfo", narrative="r")

    orig = router_mod.route_to_specialist_structured
    router_mod.route_to_specialist_structured = fake_route  # type: ignore[assignment]
    try:
        asyncio.run(
            router_mod.route_parallel(
                calls=[{"specialist": "cfo", "query": "q"}],
                retrieved_knowledge_map={"cfo": "plain text, no tokens"},
            )
        )
    finally:
        router_mod.route_to_specialist_structured = orig  # type: ignore[assignment]
    assert captured == [None]


def test_concurrent_cfo_calls_cannot_validate_against_each_others_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two interleaved turns on the ONE shared FinanceAgent must not cross-validate.

    ``SPECIALIST_REGISTRY`` holds a single ``FinanceAgent``, so this is a real
    production shape, not a contrived one: two users asking the CFO something at
    the same time hit the same object.

    Both calls cite ``token_a``. Call A owns it (it was minted for A's
    retrieval); call B does not — for B it is a replay. The correct outcome is
    that A keeps it and B loses it, *regardless of interleaving*. The two calls
    are forced to overlap: each awaits inside the provider call, so B is
    mid-flight while A is, and B finishes first.

    The assertions are on what the parser was actually handed per call, captured
    by wrapping ``parse_specialist_result``. Asserting on ``agent.last_result``
    would be meaningless (it races by construction — whichever call lands last
    wins), and asserting on the two ``allowed_ids()`` sets would be pure set
    arithmetic that holds even with the feature removed.
    """
    from openexecutive.agents import finance as finance_mod
    from openexecutive.agents.finance import FinanceAgent

    agent = FinanceAgent()  # ONE instance, as the registry holds
    _t1, set_a = _call(structured=True)
    _t2, set_b = _call(structured=True)
    token_a = sorted(set_a.allowed_ids())[0]

    seen: list[tuple[str, frozenset[str] | None, Any]] = []
    real_parse = finance_mod.parse_specialist_result

    def spy(message: Any, **kw: Any) -> Any:
        result = real_parse(message, **kw)
        # The narrative carries the call's name, which is how each parse is
        # attributed back to its originating invocation.
        seen.append((result.narrative, kw.get("allowed_retrieval_ids"), result))
        return result

    monkeypatch.setattr(finance_mod, "parse_specialist_result", spy)

    async def fake_tools(user_content: str, *_a: Any, **_kw: Any) -> Any:
        # Yield control mid-call so the sibling invocation is genuinely in
        # flight here; B sleeps less, so it completes first.
        await asyncio.sleep(0.02 if "CALL-A" in user_content else 0.01)
        return _tool_message(
            [{"kind": "document", "label": "[x]", "retrieval_id": token_a}],
            narrative="CALL-A" if "CALL-A" in user_content else "CALL-B",
        )

    agent.analyze_with_tools = fake_tools  # type: ignore[assignment]

    async def both() -> Any:
        return await asyncio.gather(
            agent.analyze(query="CALL-A", retrieval_set=set_a),
            agent.analyze(query="CALL-B", retrieval_set=set_b),
        )

    asyncio.run(both())

    by_call = {name: (allowed, res) for name, allowed, res in seen}
    assert set(by_call) == {"CALL-A", "CALL-B"}, "both calls must have parsed"

    # Each call was validated against ITS OWN set, not the other's.
    assert by_call["CALL-A"][0] == set_a.allowed_ids()
    assert by_call["CALL-B"][0] == set_b.allowed_ids()

    # The same token survives in A and is stripped in B — the outcome depends on
    # the argument, never on which call happened to finish last.
    assert by_call["CALL-A"][1].claims[0].evidence[0].retrieval_id == token_a
    assert by_call["CALL-B"][1].claims[0].evidence[0].retrieval_id is None
    assert "not in this call's retrieval set" in (
        by_call["CALL-B"][1].degraded_reason or ""
    )

    # And no retrieval state was parked on the shared instance.
    assert not any(
        isinstance(v, RetrievalSet) for v in vars(agent).values()
    ), "a RetrievalSet on the shared agent would leak across concurrent turns"


def test_cfo_validates_against_the_set_it_was_given() -> None:
    from openexecutive.agents.finance import FinanceAgent

    agent = FinanceAgent()
    _text, rset = _call(structured=True)
    good = next(iter(rset.allowed_ids()))

    async def fake_tools(*_a: Any, **_kw: Any) -> Any:
        return _tool_message([
            {"kind": "document", "label": "[ok]", "retrieval_id": good},
            {"kind": "document", "label": "[bad]", "retrieval_id": "forged"},
        ])

    agent.analyze_with_tools = fake_tools  # type: ignore[assignment]
    asyncio.run(agent.analyze(query="q", retrieval_set=rset))
    result = agent.last_result
    assert result is not None
    refs = result.claims[0].evidence
    assert refs[0].retrieval_id == good
    assert refs[1].retrieval_id is None


def test_cfo_without_a_set_behaves_as_before() -> None:
    """Every pre-existing caller (workflows, MCP, CLI) passes no set."""
    from openexecutive.agents.finance import FinanceAgent

    agent = FinanceAgent()

    async def fake_tools(*_a: Any, **_kw: Any) -> Any:
        return _tool_message([
            {"kind": "document", "label": "[x]", "retrieval_id": "whatever"}
        ])

    agent.analyze_with_tools = fake_tools  # type: ignore[assignment]
    out = asyncio.run(agent.analyze(query="q"))
    assert isinstance(out, str) and out
    result = agent.last_result
    assert result is not None
    assert result.claims[0].evidence[0].retrieval_id is None


def test_cfo_still_returns_str_and_claims_stay_internal() -> None:
    """Nothing structural may cross into the Executive in this slice."""
    from openexecutive.agents.finance import FinanceAgent

    agent = FinanceAgent()
    _text, rset = _call(structured=True)
    good = next(iter(rset.allowed_ids()))

    async def fake_tools(*_a: Any, **_kw: Any) -> Any:
        return _tool_message(
            [{"kind": "document", "label": "[x]", "retrieval_id": good}],
            narrative="Runway is 14 months.",
        )

    agent.analyze_with_tools = fake_tools  # type: ignore[assignment]
    out = asyncio.run(agent.analyze(query="q", retrieval_set=rset))
    assert isinstance(out, str)
    assert "Runway is 14 months." in out
    # The token is an internal handle; it must not leak into Executive prose.
    assert good not in out
    assert "ref:" not in out


def test_explicit_null_token_is_not_counted_as_an_attack() -> None:
    """`"retrieval_id": null` is the JSON way to say "I used no tagged passage".

    Counting it as a rejected reference would degrade every result from a
    compliant model and make `degraded_reason` unable to tell a well-behaved
    specialist from one citing a forged token.
    """
    _text, rset = _call(structured=True)
    result = _parse(
        [{"kind": "document", "label": "[x]", "retrieval_id": None}],
        rset.allowed_ids(),
    )
    assert _only_ref(result).retrieval_id is None
    assert not result.degraded
    assert "not in this call's retrieval set" not in (result.degraded_reason or "")


def test_a_forged_token_alongside_a_null_still_reports_exactly_one() -> None:
    """The count must reflect real rejections only."""
    _text, rset = _call(structured=True)
    result = _parse(
        [
            {"kind": "document", "label": "[a]", "retrieval_id": None},
            {"kind": "document", "label": "[b]", "retrieval_id": "forged"},
        ],
        rset.allowed_ids(),
    )
    assert "1 evidence reference(s) not in this call's retrieval set" in (
        result.degraded_reason or ""
    )
