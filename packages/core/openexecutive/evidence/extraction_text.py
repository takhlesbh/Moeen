"""Offset-preserving chunking: every chunk is an exact slice of its source.

Pure and in-memory. No filesystem, no network, no clock. **Nothing here mints a
canonical evidence record** — no ``DocumentVersion``, ``Extraction`` or
``SourceSpan``, no scope, no logical-source key.

**Why this exists.** The knowledge loader's ``chunk_text`` does
``text.split()`` and ``" ".join(...)``, which destroys character positions: the
chunk it hands back is not a substring of the document, so no offset can address
it and no quote can be verified against it. This module chunks by *offset*, so
``nfc_text[chunk.start_char:chunk.end_char] == chunk.text`` holds exactly, for
every chunk, byte for byte — which is the precondition
``factory.mint_source_span`` checks before it will mint a span at all.

So: **no split, no join, no strip, no whitespace collapse, no rewriting of any
kind.** Whitespace runs, CRLF, a BOM, U+2028 and bidi control marks all survive
in place, because a chunk that "cleaned up" its text would no longer be
addressable.

**Offsets are Python ``str`` indices — Unicode code points, never UTF-8 byte
positions** — the unit ``evidence.contracts`` defines for ``SourceSpan``.
``start_char`` is inclusive, ``end_char`` exclusive.

**No grapheme-cluster claim is made.** Chunk boundaries may fall inside a
combining sequence, a ZWJ emoji sequence or a variation-selector pair. That is
deliberate: ``unicodedata.combining()`` is not a grapheme segmenter (it reports
0 for ZWJ, U+FE0F, regional indicators and Hangul jamo), so a boundary rule
built on it would be an incomplete Unicode claim dressed up as a guarantee.
Provenance correctness needs exact code-point offsets and an exact quote, which
is what this provides; a rendering-safe boundary is a display concern and would
need a real segmentation dependency.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

MAX_CHUNK_CODE_POINTS = 4_096
"""Ceiling on ``max_code_points``. Matches ``contracts.MAX_QUOTE_CODE_POINTS``,
so a whole chunk can always be proposed as one quote."""

MAX_TEXT_CODE_POINTS = 2_000_000
"""Input ceiling. Matches ``factory.MAX_TEXT_CODE_POINTS``."""

MAX_CHUNK_COUNT = 2_000
"""Ceiling on the number of chunks one call may produce.

Input length alone does not bound the output. Progress per chunk is
``end - start - overlap_code_points``, which the ``overlap < max`` rule only
forces to be **at least one code point** — so a highly overlapping
configuration (say ``max=4096, overlap=4095``) advances one code point per
chunk while copying up to ``max`` code points into each. At the input ceiling
that is on the order of two million chunks and eight billion copied code
points from a two-megabyte input: a resource-exhaustion amplifier, not a
useful chunking.

Capping the count rather than narrowing the overlap range keeps every
reasonable configuration working and makes the pathological one **fail closed**
with :data:`ChunkingError` ``too_many_chunks`` instead of exhausting memory.
"""

BOUNDARY_LOOKBACK = 200
"""Fixed backward search window for a whitespace boundary. Bounded so the work
per chunk is O(1) and the whole pass stays O(n) — never a scan back to
``start``, and never a substring search."""

PRODUCTION_MAX_CODE_POINTS = 2_000
"""The chunk size production ingestion will use. Not wired to a call site yet.

Chosen against the *corrected* worst-case bound, which must include
:data:`BOUNDARY_LOOKBACK`: the boundary search can pull ``end`` back by up to
``BOUNDARY_LOOKBACK`` code points, so the guaranteed progress per chunk is

    P_min = max_code_points - BOUNDARY_LOOKBACK - overlap_code_points

and the chunk count at the input ceiling is

    C_worst = ceil((MAX_TEXT_CODE_POINTS - max) / P_min) + 1

At 2000/200 that is ``P_min = 1600`` and ``C_worst = 1250``, comfortably inside
:data:`MAX_CHUNK_COUNT` (2,000). An invariant that omitted the lookback term
would wrongly admit 1200/150, whose true worst case is 2,353 — over the cap.

2,000 also stays at or below :data:`MAX_CHUNK_CODE_POINTS`, so a whole chunk can
always be proposed as a single quote.
"""

PRODUCTION_OVERLAP_CODE_POINTS = 200
"""Production overlap: 10% of :data:`PRODUCTION_MAX_CODE_POINTS`.

Changing either constant without re-checking ``C_worst`` can push chunking over
:data:`MAX_CHUNK_COUNT` at the input ceiling, turning a maximum-size document
into a hard ``too_many_chunks`` failure. ``test_evidence_chunk_policy`` pins the
whole inequality, so an unsafe pair fails the suite rather than production.
"""


class ChunkingError(ValueError):
    """A chunking argument was rejected.

    ``check`` is a stable literal naming the failed check. ``detail`` carries
    only counts and offsets — **never source text**.
    """

    def __init__(self, check: str, detail: str = "") -> None:
        self.check = check
        super().__init__(
            f"chunking rejected: {check}" + (f" ({detail})" if detail else "")
        )


@dataclass(frozen=True, slots=True)
class OffsetChunk:
    """One chunk, and the exact range of NFC text it is a slice of.

    The invariant that gives this type its purpose:
    ``text == nfc_text[start_char:end_char]``, exactly.
    """

    start_char: int
    end_char: int
    text: str


def chunk_with_offsets(
    nfc_text: str,
    *,
    max_code_points: int,
    overlap_code_points: int,
) -> tuple[OffsetChunk, ...]:
    """Chunk already-NFC text into exact, overlapping, offset-addressed slices.

    Input must already be NFC. Non-NFC input is **rejected**, not normalized:
    normalizing here would shift every offset the caller is about to record
    against text it never saw, so the mismatch is surfaced instead of hidden.

    The algorithm, exactly — each step is load-bearing:

    1. The candidate end is ``min(start + max_code_points, len(text))``.
    2. If that is not the end of the text, search *backward* for a whitespace
       boundary — the largest ``e <= candidate`` where ``text[e - 1]`` is
       whitespace, so a chunk ends after a whitespace run rather than mid-token
       — over at most :data:`BOUNDARY_LOOKBACK` code points.
    3. No end is accepted that would leave ``end - start <= overlap``; the
       search floor enforces it. Without that floor the next start could land
       at or before the current one.
    4. With no boundary in the window, the candidate hard cut is used. A token
       longer than ``max_code_points`` is therefore cut mid-token, which is
       correct: an exact offset is worth more than an intact word.
    5. The next start is ``end - overlap_code_points``.
    6. Progress is asserted, not assumed.

    Coverage is total: consecutive chunks satisfy ``next.start <= prev.end``, so
    the union of the ranges is ``[0, len(text))`` with no gap. Starts strictly
    increase, ends never decrease, and every chunk is non-empty.

    Empty input yields an empty tuple — zero chunks, not one empty chunk, since
    an empty chunk could not be quoted or verified. All-whitespace input is
    ordinary text and is chunked exactly, never emptied.

    **Cost, stated honestly.** Runtime is O(input + produced output): the pass
    reads each code point a bounded number of times, and each chunk costs one
    slice of at most :data:`MAX_CHUNK_CODE_POINTS`. Produced chunks are capped
    at :data:`MAX_CHUNK_COUNT`, so total copied output is bounded by
    ``MAX_CHUNK_COUNT * MAX_CHUNK_CODE_POINTS`` — independent of how the
    overlap is chosen. Input length alone does *not* bound the count, because
    progress can be as little as one code point per chunk; a configuration that
    would exceed the cap **fails closed** with ``too_many_chunks`` rather than
    returning a truncated tuple, since a partial chunking silently covering
    only a prefix of the document is worse than a rejection.
    """
    if not isinstance(nfc_text, str):
        raise ChunkingError("text_not_str")
    for label, value in (
        ("max_code_points", max_code_points),
        ("overlap_code_points", overlap_code_points),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ChunkingError("argument_not_int", label)
    if not 1 <= max_code_points <= MAX_CHUNK_CODE_POINTS:
        raise ChunkingError("max_code_points_range", f"{max_code_points}")
    # Strictly less than max: equal overlap would make progress zero.
    if not 0 <= overlap_code_points < max_code_points:
        raise ChunkingError("overlap_code_points_range", f"{overlap_code_points}")
    if len(nfc_text) > MAX_TEXT_CODE_POINTS:
        raise ChunkingError("text_length", f"{len(nfc_text)} > {MAX_TEXT_CODE_POINTS}")
    if not unicodedata.is_normalized("NFC", nfc_text):
        raise ChunkingError("text_not_nfc")

    total = len(nfc_text)
    if total == 0:
        return ()

    chunks: list[OffsetChunk] = []
    start = 0
    while True:
        candidate_end = min(start + max_code_points, total)
        end = candidate_end
        if candidate_end < total:
            # `+ 1` keeps `end - start > overlap`, which is what guarantees
            # step 6 below can never fail. Dropping it admits `next_start ==
            # start` and an unterminated loop.
            floor = max(start + overlap_code_points + 1, candidate_end - BOUNDARY_LOOKBACK)
            probe = candidate_end
            while probe >= floor:
                if nfc_text[probe - 1].isspace():
                    end = probe
                    break
                probe -= 1
        # Checked *before* the slice below, so an over-limit substring is never
        # materialized and an over-limit object is never appended.
        if len(chunks) >= MAX_CHUNK_COUNT:
            raise ChunkingError("too_many_chunks", f"limit={MAX_CHUNK_COUNT}")
        # The one place a chunk's text is produced: an exact slice of the input.
        chunks.append(
            OffsetChunk(start_char=start, end_char=end, text=nfc_text[start:end])
        )
        if end >= total:
            break
        next_start = end - overlap_code_points
        if next_start <= start:
            raise ChunkingError("no_progress", f"start={start} end={end}")
        start = next_start
    return tuple(chunks)
