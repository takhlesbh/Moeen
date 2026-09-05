"""The production chunk configuration, proved against the real algorithm.

Everything here is computed FROM the constants. No test asserts the literals
2000, 200, 1111 or 1250, so changing any constant into an unsafe combination
fails this file rather than production.
"""
from __future__ import annotations

import math

import pytest

from openexecutive.evidence.contracts import MAX_QUOTE_CODE_POINTS
from openexecutive.evidence.extraction_text import (
    BOUNDARY_LOOKBACK,
    MAX_CHUNK_CODE_POINTS,
    MAX_CHUNK_COUNT,
    MAX_TEXT_CODE_POINTS,
    PRODUCTION_MAX_CODE_POINTS,
    PRODUCTION_OVERLAP_CODE_POINTS,
    ChunkingError,
    chunk_with_offsets,
)


def min_progress(max_code_points: int, overlap: int) -> int:
    """Guaranteed advance per chunk.

    The boundary search may pull ``end`` back by up to ``BOUNDARY_LOOKBACK``
    before the next start is computed as ``end - overlap``. An invariant that
    omits the lookback term overstates progress and admits unsafe configurations.
    """
    return max_code_points - BOUNDARY_LOOKBACK - overlap


def worst_case_chunks(max_code_points: int, overlap: int, text_length: int) -> int:
    """Chunks produced at ``text_length`` under worst-case boundary placement.

    Non-final chunks are emitted while ``start + max < length``, each advancing
    at least ``min_progress``; the final chunk is emitted once the candidate end
    reaches the text end, where the boundary search is skipped entirely.
    """
    if text_length <= max_code_points:
        return 1
    progress = min_progress(max_code_points, overlap)
    return math.ceil((text_length - max_code_points) / progress) + 1


def adversarial_text(max_code_points: int, overlap: int, length: int) -> str:
    """Whitespace placed exactly at each boundary-search floor.

    That is the placement which minimises progress: every chunk gives back the
    full lookback window. Derived from the constants, so it stays worst-case if
    they change.
    """
    progress = min_progress(max_code_points, overlap)
    buf = bytearray(b"a" * length)
    index = 0
    while True:
        position = index * progress + max_code_points - BOUNDARY_LOOKBACK - 1
        if position >= length:
            break
        buf[position] = 0x20
        index += 1
    return buf.decode()


# ── the algebraic invariant ─────────────────────────────────────────────


def test_production_configuration_satisfies_every_clause_of_the_bound():
    max_cp = PRODUCTION_MAX_CODE_POINTS
    overlap = PRODUCTION_OVERLAP_CODE_POINTS

    assert 0 <= overlap < max_cp
    assert max_cp <= MAX_CHUNK_CODE_POINTS
    # A whole chunk must always be proposable as a single quote.
    assert max_cp <= MAX_QUOTE_CODE_POINTS
    assert min_progress(max_cp, overlap) >= 1
    assert worst_case_chunks(max_cp, overlap, MAX_TEXT_CODE_POINTS) <= MAX_CHUNK_COUNT


def test_the_bound_accounts_for_the_boundary_lookback():
    """A bound of `max - overlap` alone is wrong. Pin the difference so nobody
    reintroduces the weaker invariant."""
    max_cp = PRODUCTION_MAX_CODE_POINTS
    overlap = PRODUCTION_OVERLAP_CODE_POINTS
    assert min_progress(max_cp, overlap) == max_cp - overlap - BOUNDARY_LOOKBACK
    assert min_progress(max_cp, overlap) < max_cp - overlap


# ── the runtime proof, at the input ceiling ─────────────────────────────


def test_hostile_no_whitespace_input_at_the_ceiling_stays_within_the_cap():
    text = "a" * MAX_TEXT_CODE_POINTS
    chunks = chunk_with_offsets(
        text,
        max_code_points=PRODUCTION_MAX_CODE_POINTS,
        overlap_code_points=PRODUCTION_OVERLAP_CODE_POINTS,
    )
    assert len(chunks) <= MAX_CHUNK_COUNT
    assert len(chunks) <= worst_case_chunks(
        PRODUCTION_MAX_CODE_POINTS, PRODUCTION_OVERLAP_CODE_POINTS, MAX_TEXT_CODE_POINTS
    )


def test_adversarial_whitespace_at_the_ceiling_stays_within_the_cap():
    """The genuine worst case: every chunk gives back the whole lookback window."""
    text = adversarial_text(
        PRODUCTION_MAX_CODE_POINTS, PRODUCTION_OVERLAP_CODE_POINTS, MAX_TEXT_CODE_POINTS
    )
    chunks = chunk_with_offsets(
        text,
        max_code_points=PRODUCTION_MAX_CODE_POINTS,
        overlap_code_points=PRODUCTION_OVERLAP_CODE_POINTS,
    )
    predicted = worst_case_chunks(
        PRODUCTION_MAX_CODE_POINTS, PRODUCTION_OVERLAP_CODE_POINTS, MAX_TEXT_CODE_POINTS
    )
    assert len(chunks) == predicted, "the formula must be exact, not merely an upper bound"
    assert len(chunks) <= MAX_CHUNK_COUNT

    observed = min(
        chunks[i + 1].start_char - chunks[i].start_char for i in range(len(chunks) - 1)
    )
    assert observed == min_progress(
        PRODUCTION_MAX_CODE_POINTS, PRODUCTION_OVERLAP_CODE_POINTS
    )


@pytest.mark.parametrize("builder", ["plain", "adversarial"])
def test_structural_invariants_hold_at_the_ceiling(builder):
    text = (
        "a" * MAX_TEXT_CODE_POINTS
        if builder == "plain"
        else adversarial_text(
            PRODUCTION_MAX_CODE_POINTS, PRODUCTION_OVERLAP_CODE_POINTS, MAX_TEXT_CODE_POINTS
        )
    )
    chunks = chunk_with_offsets(
        text,
        max_code_points=PRODUCTION_MAX_CODE_POINTS,
        overlap_code_points=PRODUCTION_OVERLAP_CODE_POINTS,
    )
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text)
    for index, chunk in enumerate(chunks):
        # Every chunk is an exact slice -- the precondition for minting a span.
        assert text[chunk.start_char : chunk.end_char] == chunk.text
        assert 0 < len(chunk.text) <= MAX_QUOTE_CODE_POINTS
        if index:
            previous = chunks[index - 1]
            assert chunk.start_char > previous.start_char
            assert chunk.start_char <= previous.end_char  # no gap in coverage


# ── unsafe configurations must fail both ways ───────────────────────────


@pytest.mark.parametrize("max_cp,overlap", [(1200, 150), (1024, 128), (600, 100), (400, 40)])
def test_unsafe_configurations_are_rejected_by_formula_and_at_runtime(max_cp, overlap):
    """The algebraic predicate and the algorithm must agree. A configuration the
    formula calls unsafe really does blow the cap -- these would have passed an
    invariant that ignored BOUNDARY_LOOKBACK."""
    assert worst_case_chunks(max_cp, overlap, MAX_TEXT_CODE_POINTS) > MAX_CHUNK_COUNT

    with pytest.raises(ChunkingError) as exc:
        chunk_with_offsets(
            adversarial_text(max_cp, overlap, MAX_TEXT_CODE_POINTS),
            max_code_points=max_cp,
            overlap_code_points=overlap,
        )
    assert exc.value.check == "too_many_chunks"


def test_the_formula_is_exact_at_the_cap_boundary():
    """Find the configuration whose predicted worst case is exactly the cap, and
    confirm the algorithm produces exactly that -- neither one over nor under."""
    boundary = None
    for max_cp in range(BOUNDARY_LOOKBACK + 2, MAX_CHUNK_CODE_POINTS + 1):
        overlap = max_cp // 10
        if min_progress(max_cp, overlap) < 1:
            continue
        if worst_case_chunks(max_cp, overlap, MAX_TEXT_CODE_POINTS) == MAX_CHUNK_COUNT:
            boundary = (max_cp, overlap)
            break
    assert boundary is not None

    max_cp, overlap = boundary
    chunks = chunk_with_offsets(
        adversarial_text(max_cp, overlap, MAX_TEXT_CODE_POINTS),
        max_code_points=max_cp,
        overlap_code_points=overlap,
    )
    assert len(chunks) == MAX_CHUNK_COUNT


def test_the_chunker_is_still_unwired():
    """4A2b adds constants and a proof, not a call site. Ingestion still uses the
    legacy word-splitting chunker; replacing it belongs to a later phase."""
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[2] / "openexecutive"
    callers = []
    for path in package.rglob("*.py"):
        if path.name == "extraction_text.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "chunk_with_offsets"
            ):
                callers.append(str(path.relative_to(package)))
    assert callers == []
