"""Offset-preserving chunking: exactness, bounds and Unicode transparency. Non-ASCII code points are
``\\uXXXX`` escapes pinned with ``ord()``: a literal glyph can be rewritten in
transit (U+2028 becomes a real line break), silently turning a Unicode test
into an ASCII one that still passes.
"""
from __future__ import annotations

import time
import unicodedata

import pytest

from openexecutive.evidence import extraction_text
from openexecutive.evidence.extraction_text import (
    BOUNDARY_LOOKBACK,
    MAX_CHUNK_CODE_POINTS,
    MAX_CHUNK_COUNT,
    MAX_TEXT_CODE_POINTS,
    ChunkingError,
    OffsetChunk,
    chunk_with_offsets,
)

ZWJ = "\u200d"
VS16 = "\ufe0f"
LINE_SEP = "\u2028"
BOM = "\ufeff"
RLM = "\u200f"
LRM = "\u200e"
NBSP = "\u00a0"
ALEF, LAM, MEEM = "\u0627", "\u0644", "\u0645"
WOMAN, LAPTOP = "\U0001f469", "\U0001f4bb"
ACUTE = "\u0301"
E_ACUTE = "\u00e9"
HEART = "\u2764"
STACKED = "\u00e1\u0302\u0303"  # NFC-stable: a-acute, circumflex, tilde


def test_pinned_code_points_survived_the_editing_transport() -> None:
    """If this fails, every Unicode assertion below is testing the wrong bytes."""
    assert [ord(c) for c in (ZWJ, VS16, LINE_SEP, BOM, RLM, LRM, NBSP)] == [
        0x200D, 0xFE0F, 0x2028, 0xFEFF, 0x200F, 0x200E, 0x00A0,
    ]
    assert [ord(c) for c in (ALEF, LAM, MEEM, ACUTE, E_ACUTE, HEART)] == [
        0x0627, 0x0644, 0x0645, 0x0301, 0x00E9, 0x2764,
    ]
    assert [ord(c) for c in (WOMAN, LAPTOP)] == [0x1F469, 0x1F4BB]
    assert [ord(c) for c in STACKED] == [0x00E1, 0x0302, 0x0303]
    # Whitespace classification the boundary search depends on.
    assert LINE_SEP.isspace() and NBSP.isspace()
    assert not (ZWJ.isspace() or VS16.isspace() or BOM.isspace() or RLM.isspace())


def assert_chunk_invariants(
    text: str, chunks: tuple[OffsetChunk, ...], max_code_points: int
) -> None:
    """Every structural invariant the module promises, checked at once."""
    if not text:
        assert chunks == ()
        return
    assert chunks, "non-empty input must produce at least one chunk"
    for item in chunks:
        assert item.text == text[item.start_char : item.end_char], "exact-slice"
        assert item.start_char < item.end_char, "every chunk is non-empty"
        assert item.end_char - item.start_char <= max_code_points, "length ceiling"
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert following.start_char > previous.start_char, "strict progress"
        assert following.end_char >= previous.end_char, "ends never decrease"
        assert following.start_char <= previous.end_char, "no coverage gap"
    assert chunks[0].start_char == 0 and chunks[-1].end_char == len(text)
    covered = bytearray(len(text))
    for item in chunks:
        covered[item.start_char : item.end_char] = b"\x01" * (
            item.end_char - item.start_char
        )
    assert all(covered), "union of chunk ranges must cover every code point"


def chunk(text: str, max_code_points: int = 16, overlap: int = 0) -> tuple[OffsetChunk, ...]:
    result = chunk_with_offsets(
        text, max_code_points=max_code_points, overlap_code_points=overlap
    )
    assert_chunk_invariants(text, result, max_code_points)
    return result


@pytest.mark.parametrize(
    "text,max_code_points,expected",
    [
        ("", 16, []),
        ("", 1, []),
        ("x", 16, [(0, 1)]),
        ("a" * 16, 16, [(0, 16)]),
        ("a" * 17, 16, [(0, 16), (16, 17)]),
        ("abcde", 1, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        ("a" * 100, MAX_CHUNK_CODE_POINTS, [(0, 100)]),
        ("z" * 5_000, 1_000, [(i * 1_000, i * 1_000 + 1_000) for i in range(5)]),
    ],
)
def test_chunk_ranges_at_size_boundaries(
    text: str, max_code_points: int, expected: list[tuple[int, int]]
) -> None:
    """Empty input yields zero chunks; a token longer than max is hard-cut."""
    result = chunk(text, max_code_points=max_code_points)
    assert [(c.start_char, c.end_char) for c in result] == expected


@pytest.mark.parametrize("overlap", [0, 3])
def test_chunks_are_contiguous_and_overlap_repeats_the_previous_tail(
    overlap: int,
) -> None:
    text = "".join(str(index % 10) for index in range(100))
    result = chunk(text, max_code_points=10, overlap=overlap)
    assert len(result) > 1
    for previous, following in zip(result, result[1:], strict=False):
        assert following.start_char == previous.end_char - overlap
        if overlap:
            assert following.text[:overlap] == previous.text[-overlap:]
    if not overlap:
        assert "".join(c.text for c in result) == text


@pytest.mark.parametrize("bad_max", [0, -1, MAX_CHUNK_CODE_POINTS + 1, 10_000])
def test_invalid_max_code_points_is_rejected(bad_max: int) -> None:
    with pytest.raises(ChunkingError) as excinfo:
        chunk_with_offsets("abc", max_code_points=bad_max, overlap_code_points=0)
    assert excinfo.value.check == "max_code_points_range"


@pytest.mark.parametrize("bad_overlap", [-1, 16, 17, 1_000])
def test_invalid_overlap_is_rejected(bad_overlap: int) -> None:
    """Overlap equal to max is rejected too: progress would be zero."""
    with pytest.raises(ChunkingError) as excinfo:
        chunk_with_offsets("abc", max_code_points=16, overlap_code_points=bad_overlap)
    assert excinfo.value.check == "overlap_code_points_range"


@pytest.mark.parametrize("bad_text", [None, b"bytes", 42, ["a"]])
def test_non_string_text_is_rejected(bad_text: object) -> None:
    with pytest.raises(ChunkingError) as excinfo:
        chunk_with_offsets(bad_text, max_code_points=16, overlap_code_points=0)  # type: ignore[arg-type]
    assert excinfo.value.check == "text_not_str"
    # Error messages carry check names and counts, never source text.
    secret = "CONFIDENTIAL-ACQUISITION-PRICE-4200000"
    with pytest.raises(ChunkingError) as excinfo:
        chunk_with_offsets(secret, max_code_points=0, overlap_code_points=0)
    assert secret not in str(excinfo.value) and "CONFIDENTIAL" not in str(excinfo.value)


@pytest.mark.parametrize("bad", [True, False, 1.5, "8", None])
def test_non_integer_bounds_are_rejected(bad: object) -> None:
    """``bool`` is rejected too: ``True`` is an ``int`` but not a real bound."""
    for kwargs in ({"max_code_points": bad}, {"overlap_code_points": bad}):
        call = {"max_code_points": 16, "overlap_code_points": 0, **kwargs}
        with pytest.raises(ChunkingError) as excinfo:
            chunk_with_offsets("abc", **call)  # type: ignore[arg-type]
        assert excinfo.value.check == "argument_not_int"


def test_text_length_ceiling_is_enforced() -> None:
    with pytest.raises(ChunkingError) as excinfo:
        chunk_with_offsets(
            "a" * (MAX_TEXT_CODE_POINTS + 1), max_code_points=4096, overlap_code_points=0
        )
    assert excinfo.value.check == "text_length"


def test_decomposed_input_is_rejected_until_normalized() -> None:
    decomposed = "cafe" + ACUTE
    assert not unicodedata.is_normalized("NFC", decomposed)
    with pytest.raises(ChunkingError) as excinfo:
        chunk_with_offsets(decomposed, max_code_points=16, overlap_code_points=0)
    assert excinfo.value.check == "text_not_nfc"

    composed = unicodedata.normalize("NFC", decomposed)
    assert composed == "caf" + E_ACUTE and (len(composed), len(decomposed)) == (4, 5)
    assert chunk(composed)[0].text == composed


def test_hostile_input_at_the_ceiling_is_bounded_time() -> None:
    """Exactly 2M code points with no whitespace: every backward search fails.

    Pins both that the ceiling is inclusive and that the search window is fixed
    rather than a scan back to ``start``, which would make this quadratic.
    """
    text = "x" * MAX_TEXT_CODE_POINTS
    started = time.monotonic()
    result = chunk_with_offsets(
        text, max_code_points=4096, overlap_code_points=BOUNDARY_LOOKBACK
    )
    assert time.monotonic() - started < 10.0, "chunking is not linear"
    assert_chunk_invariants(text, result, 4096)
    assert result[-1].end_char == MAX_TEXT_CODE_POINTS


def test_all_whitespace_input_is_kept_exactly_not_emptied() -> None:
    text = "   \t\n  \r\n " + NBSP + "  "
    assert len(text) == 13
    assert chunk(text, max_code_points=64) == (
        OffsetChunk(start_char=0, end_char=13, text=text),
    )
    for max_code_points in (1, 4, 8, 12):
        result = chunk(text, max_code_points=max_code_points)
        assert "".join(c.text for c in result) == text, "whitespace is never dropped"
    # Long runs are never collapsed, even when a chunk boundary falls inside one.
    runs = "a" + " " * 40 + "b" + "\t" * 40 + "c"
    split = chunk(runs, max_code_points=16, overlap=0)
    assert "".join(c.text for c in split) == runs
    assert sum(c.text.count(" ") for c in split) == 40
    assert sum(c.text.count("\t") for c in split) == 40


def test_crlf_and_bom_and_line_separator_are_preserved_exactly() -> None:
    text = BOM + "line one\r\nline two" + LINE_SEP + "three"
    result = chunk(text, max_code_points=128)
    assert result[0].text == text
    assert result[0].text[0] == BOM and ord(result[0].text[0]) == 0xFEFF
    assert result[0].text.count("\r\n") == 1
    assert result[0].text.count(LINE_SEP) == 1


def test_arabic_with_bidi_marks_is_preserved_exactly() -> None:
    word = ALEF + LAM + MEEM
    text = RLM + word + LRM + " " + word + RLM
    assert unicodedata.is_normalized("NFC", text)
    result = chunk(text, max_code_points=64)
    assert result[0].text == text
    assert result[0].text.count(RLM) == 2 and result[0].text.count(LRM) == 1


@pytest.mark.parametrize(
    "text,split_at,expected_chunks",
    [
        ((WOMAN + ZWJ + LAPTOP) * 10, 2, 15),   # boundary inside a ZWJ sequence
        ((HEART + VS16) * 12, 1, 24),           # base separated from VS16
        (STACKED * 8, 2, 12),                   # boundary inside combining marks
        ((ALEF + LAM + MEEM + RLM) * 6, 2, 12),  # Arabic with a bidi mark
    ],
)
def test_sequences_may_be_split_and_offsets_stay_exact(
    text: str, split_at: int, expected_chunks: int
) -> None:
    """No grapheme claim: a boundary inside a ZWJ, variation-selector or
    combining sequence is allowed, and exactness must hold on both sides."""
    assert unicodedata.is_normalized("NFC", text)
    assert chunk(text, max_code_points=64)[0].text == text
    split = chunk(text, max_code_points=split_at, overlap=0)
    assert len(split) == expected_chunks
    assert "".join(c.text for c in split) == text


def test_whitespace_boundary_is_preferred_near_the_target_end() -> None:
    result = chunk("alpha beta gamma delta epsilon zeta", max_code_points=12)
    assert result[0].text == "alpha beta " and result[0].end_char == 11


def test_boundary_search_never_yields_a_chunk_at_or_below_overlap() -> None:
    """Pins the ``start + overlap + 1`` search floor.

    ``text[3]`` is the only whitespace in reach of the first candidate end.
    Accepting it would make ``end - start == overlap``, so the next start would
    land back on the current one. The floor rejects it and the hard cut is used.
    """
    text = "abc efghij" + "klmnop"
    assert len(text) == 16
    assert text[3] == " " and not any(c.isspace() for c in text[4:])
    result = chunk(text, max_code_points=10, overlap=4)
    assert result[0].end_char == 10, "must hard-cut, not take the too-early boundary"
    assert result[0].text == "abc efghij"
    assert (result[1].start_char, result[1].end_char) == (6, 16)


def test_boundary_lookback_window_is_bounded() -> None:
    """Whitespace further back than the fixed window is not reached, and the
    final chunk takes the text end without any boundary search."""
    result = chunk("a " + "b" * 400, max_code_points=300, overlap=0)
    assert result[0].end_char == 300, "space at offset 1 is outside the window"
    assert result[-1].end_char == 402
    assert chunk("alpha beta gamma", max_code_points=64)[0].text == "alpha beta gamma"
    # An unbroken token longer than max is hard-cut under overlap too.
    assert chunk("q" * 1_003, max_code_points=100, overlap=10)[-1].end_char == 1_003


def test_repeated_identical_passages_get_distinct_offsets() -> None:
    """Placement is positional, so duplicate text must not collapse -- which is
    also why no implementation may use a substring search to place a chunk."""
    passage = "revenue grew twelve percent "
    text = passage * 12
    result = chunk(text, max_code_points=len(passage), overlap=0)
    assert [c.start_char for c in result] == [i * len(passage) for i in range(12)]
    assert len({c.text for c in result}) == 1, "text really is identical"
    # Same content under overlap still satisfies every invariant.
    chunk(text, max_code_points=40, overlap=12)


def test_output_is_deterministic_across_repeated_calls() -> None:
    text = "alpha beta " * 60 + ALEF + ZWJ + VS16
    first = chunk_with_offsets(text, max_code_points=64, overlap_code_points=8)
    for _ in range(4):
        assert chunk_with_offsets(text, max_code_points=64, overlap_code_points=8) == first
    # Chunks are frozen value objects, which is what makes equality meaningful.
    assert chunk("hello world", max_code_points=32) == (
        OffsetChunk(start_char=0, end_char=11, text="hello world"),
    )
    with pytest.raises(AttributeError):
        first[0].start_char = 5  # type: ignore[misc]


@pytest.mark.parametrize("max_code_points", [1, 2, 3, 7, 16, 64, 4096])
@pytest.mark.parametrize("overlap_fraction", [0.0, 0.25, 0.5, 0.9])
def test_invariants_hold_across_the_parameter_grid(
    max_code_points: int, overlap_fraction: float
) -> None:
    overlap = min(int(max_code_points * overlap_fraction), max_code_points - 1)
    text = (
        "Board minutes " + ALEF + LAM + " " + WOMAN + ZWJ + LAPTOP + " q3\r\n"
        + LINE_SEP + NBSP + "figures " + E_ACUTE + STACKED
    ) * 9
    assert unicodedata.is_normalized("NFC", text)
    result = chunk_with_offsets(
        text, max_code_points=max_code_points, overlap_code_points=overlap
    )
    assert_chunk_invariants(text, result, max_code_points)


def test_chunk_count_ceiling_is_pinned_at_two_thousand() -> None:
    """The production cap, not a test-local one. Bounds copied output at
    ``MAX_CHUNK_COUNT * MAX_CHUNK_CODE_POINTS`` regardless of overlap."""
    assert MAX_CHUNK_COUNT == 2_000
    assert extraction_text.MAX_CHUNK_COUNT == 2_000


def test_exactly_max_chunk_count_is_allowed_and_one_more_fails_closed() -> None:
    """The cap is inclusive: 2,000 chunks pass, the 2,001st is refused.

    ``max=1, overlap=0`` makes chunk count equal input length, so this pins the
    boundary exactly without a large input.
    """
    at_limit = "a" * MAX_CHUNK_COUNT
    result = chunk(at_limit, max_code_points=1, overlap=0)
    assert len(result) == MAX_CHUNK_COUNT
    assert result[-1].end_char == MAX_CHUNK_COUNT

    with pytest.raises(ChunkingError) as excinfo:
        chunk_with_offsets(
            "a" * (MAX_CHUNK_COUNT + 1), max_code_points=1, overlap_code_points=0
        )
    assert excinfo.value.check == "too_many_chunks"


def test_chunk_count_guard_reads_the_constant_not_a_hardcoded_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lowering the module constant must move the boundary with it."""
    monkeypatch.setattr(extraction_text, "MAX_CHUNK_COUNT", 3)
    assert len(chunk_with_offsets("abc", max_code_points=1, overlap_code_points=0)) == 3
    with pytest.raises(ChunkingError) as excinfo:
        chunk_with_offsets("abcd", max_code_points=1, overlap_code_points=0)
    assert excinfo.value.check == "too_many_chunks"


def test_one_code_point_progress_fails_closed_instead_of_growing_unbounded() -> None:
    """``max=2, overlap=1`` advances one code point per chunk.

    Unbounded, this input would yield ~4,999 chunks; the cap refuses at 2,000
    rather than returning an input-sized tuple.
    """
    text = "alpha beta " * 500
    assert unicodedata.is_normalized("NFC", text) and len(text) == 5_500
    with pytest.raises(ChunkingError) as excinfo:
        chunk_with_offsets(text, max_code_points=2, overlap_code_points=1)
    assert excinfo.value.check == "too_many_chunks"


def test_maximal_overlap_at_the_ceiling_is_bounded_by_count_not_by_input() -> None:
    """``max=4096, overlap=4095``: the review's amplification case, shrunk.

    Each chunk copies up to 4,096 code points while advancing one, so output
    grows ~4,096x input. Refused at the cap, far below the ~25,905 chunks the
    input would otherwise demand -- and in bounded time.
    """
    text = "z" * 30_000
    would_be_unbounded = len(text) - MAX_CHUNK_CODE_POINTS + 1
    assert would_be_unbounded > 10 * MAX_CHUNK_COUNT, "input must outrun the cap"
    started = time.monotonic()
    with pytest.raises(ChunkingError) as excinfo:
        chunk_with_offsets(
            text,
            max_code_points=MAX_CHUNK_CODE_POINTS,
            overlap_code_points=MAX_CHUNK_CODE_POINTS - 1,
        )
    assert excinfo.value.check == "too_many_chunks"
    assert time.monotonic() - started < 10.0, "failing closed must still be fast"


def test_too_many_chunks_error_carries_no_source_text() -> None:
    """Same contract as every other rejection: counts only, never content."""
    secret = "CONFIDENTIAL-ACQUISITION-PRICE-4200000 "
    text = secret * 200
    assert len(text) > MAX_CHUNK_COUNT
    with pytest.raises(ChunkingError) as excinfo:
        chunk_with_offsets(text, max_code_points=1, overlap_code_points=0)
    message = str(excinfo.value)
    assert excinfo.value.check == "too_many_chunks"
    assert secret not in message and "CONFIDENTIAL" not in message
    assert "4200000" not in message
    assert message == "chunking rejected: too_many_chunks (limit=2000)"


def test_existing_overlap_configurations_stay_under_the_cap() -> None:
    """``overlap == max - 1`` is still legal: small inputs are unaffected."""
    text = "alpha beta gamma delta"
    result = chunk_with_offsets(text, max_code_points=8, overlap_code_points=7)
    assert_chunk_invariants(text, result, 8)
    assert 0 < len(result) < MAX_CHUNK_COUNT
