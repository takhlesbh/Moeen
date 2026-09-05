"""Page mapping: separator ownership, empty pages, NFC assembly and lookup. Non-ASCII code points are
``\\uXXXX`` escapes pinned with ``ord()``: a literal glyph can be rewritten in
transit (U+2028 becomes a real line break), silently turning a Unicode test
into an ASCII one that still passes.
"""
from __future__ import annotations

import time
import unicodedata

import pytest

from openexecutive.evidence.page_map import (
    MAX_PAGE_COUNT,
    MAX_TEXT_CODE_POINTS,
    PAGE_SEPARATOR,
    PageMapError,
    PageMappedText,
    PageSpan,
    SeparatorSpan,
    build_page_mapped_text,
)

ZWJ = "\u200d"
VS16 = "\ufe0f"
LINE_SEP = "\u2028"
RLM = "\u200f"
LRM = "\u200e"
NBSP = "\u00a0"
ALEF, LAM, MEEM = "\u0627", "\u0644", "\u0645"
WOMAN, LAPTOP = "\U0001f469", "\U0001f4bb"
ACUTE = "\u0301"
E_ACUTE = "\u00e9"
HEART = "\u2764"
KA_NUKTA, KA, NUKTA = "\u0958", "\u0915", "\u093c"


def test_pinned_code_points_survived_the_editing_transport() -> None:
    """If this fails, every Unicode assertion below is testing the wrong bytes."""
    assert [ord(c) for c in (ZWJ, VS16, LINE_SEP, RLM, LRM, NBSP)] == [
        0x200D, 0xFE0F, 0x2028, 0x200F, 0x200E, 0x00A0,
    ]
    assert [ord(c) for c in (ALEF, LAM, MEEM, ACUTE, E_ACUTE, HEART)] == [
        0x0627, 0x0644, 0x0645, 0x0301, 0x00E9, 0x2764,
    ]
    assert [ord(c) for c in (WOMAN, LAPTOP, KA_NUKTA, KA, NUKTA)] == [
        0x1F469, 0x1F4BB, 0x0958, 0x0915, 0x093C,
    ]
    # U+0958 is a composition exclusion: NFC turns one code point into two.
    assert unicodedata.normalize("NFC", KA_NUKTA) == KA + NUKTA
    assert PAGE_SEPARATOR == "\n" and len(PAGE_SEPARATOR) == 1


def assert_map_invariants(mapped: PageMappedText, pages: list[str | None]) -> None:
    """Every structural invariant the assembler promises, checked at once."""
    assert mapped.page_count == len(pages) == len(mapped.pages)
    assert len(mapped.separators) == len(pages) - 1
    # Global NFC invariant: per-page normalization agrees with normalizing the
    # whole assembled document.
    assert mapped.nfc_text == unicodedata.normalize("NFC", mapped.raw_text)
    assert unicodedata.is_normalized("NFC", mapped.nfc_text)
    assert [s.page_number for s in mapped.pages] == list(range(1, len(pages) + 1))

    for span, source in zip(mapped.pages, pages, strict=True):
        expected = unicodedata.normalize("NFC", source or "")
        assert mapped.nfc_text[span.start_char : span.end_char] == expected
        assert span.is_empty == (expected == "")
    for separator in mapped.separators:
        assert separator.end_char - separator.start_char == 1
        assert mapped.nfc_text[separator.start_char] == PAGE_SEPARATOR
        for span in mapped.pages:
            assert not (
                span.start_char <= separator.start_char < span.end_char
            ), "no page may claim an inserted separator"
    # Non-empty pages and separators tile the text with no gap and no overlap.
    cursor = 0
    for start, end in sorted(
        [(s.start_char, s.end_char) for s in mapped.pages if not s.is_empty]
        + [(s.start_char, s.end_char) for s in mapped.separators]
    ):
        assert start == cursor, "gap or overlap in the offset map"
        cursor = end
    assert cursor == len(mapped.nfc_text)
    assert mapped.locatable == tuple(s for s in mapped.pages if not s.is_empty)


def build(*pages: str | None) -> PageMappedText:
    mapped = build_page_mapped_text(list(pages))
    assert_map_invariants(mapped, list(pages))
    return mapped


def test_single_and_multiple_pages_get_ordered_spans_and_separators() -> None:
    single = build("Quarterly revenue rose.")
    assert single.raw_text == single.nfc_text == "Quarterly revenue rose."
    assert single.pages == (PageSpan(page_number=1, start_char=0, end_char=23),)
    assert single.separators == () and single.locate(0, 23) == (1, 1)

    mapped = build("alpha", "beta", "gamma")
    assert mapped.nfc_text == "alpha\nbeta\ngamma"
    assert mapped.pages == (
        PageSpan(page_number=1, start_char=0, end_char=5),
        PageSpan(page_number=2, start_char=6, end_char=10),
        PageSpan(page_number=3, start_char=11, end_char=16),
    )
    assert mapped.separators == (
        SeparatorSpan(after_page=1, start_char=5, end_char=6),
        SeparatorSpan(after_page=2, start_char=10, end_char=11),
    )
    # Identical page text must not collapse pages into one another.
    repeated = build("same", "same", "same")
    assert [repeated.locate(*r) for r in ((0, 4), (5, 9), (10, 14))] == [
        (1, 1), (2, 2), (3, 3)
    ]


def test_page_text_is_never_stripped_and_none_is_an_empty_page() -> None:
    mapped = build("  leading and trailing  ", "\t\ttabs\t\t")
    assert mapped.nfc_text == "  leading and trailing  \n\t\ttabs\t\t"
    assert mapped.nfc_text[mapped.pages[1].start_char] == "\t"

    none_page = build(None, "content")
    assert none_page.pages[0] == PageSpan(page_number=1, start_char=0, end_char=0)
    assert none_page.nfc_text == "\ncontent" and none_page.locate(1, 8) == (2, 2)
    assert build(None, "x").pages == build("", "x").pages


def test_page_ending_with_lf_and_next_beginning_with_lf() -> None:
    """The inserted separator must stay distinguishable from page-owned LFs.

    Three consecutive LFs: page 1's own trailing LF, the inserted separator,
    then page 2's own leading LF. Only the middle one is synthetic, and only it
    locates to no page.
    """
    mapped = build("a\n", "\nb")
    assert mapped.nfc_text == "a\n\n\nb" and len(mapped.nfc_text) == 5
    assert mapped.pages[0] == PageSpan(page_number=1, start_char=0, end_char=2)
    assert mapped.pages[1] == PageSpan(page_number=2, start_char=3, end_char=5)
    assert mapped.separators == (SeparatorSpan(after_page=1, start_char=2, end_char=3),)
    assert mapped.locate(1, 2) == (1, 1)
    assert mapped.locate(2, 3) == (None, None)
    assert mapped.locate(3, 4) == (2, 2)


@pytest.mark.parametrize(
    "start,end,expected",
    [
        (0, 5, (1, 1)),          # page 1 exactly
        (6, 10, (2, 2)),         # page 2 exactly
        (11, 16, (3, 3)),        # page 3 exactly
        (15, 16, (3, 3)),        # the exact text end
        (5, 6, (None, None)),    # separator 1 only
        (10, 11, (None, None)),  # separator 2 only
        (5, 10, (2, 2)),         # separator plus the page after it
        (0, 6, (1, 1)),          # page plus the separator after it
        (4, 7, (1, 2)),          # crosses one boundary
        (4, 12, (1, 3)),         # crosses two boundaries
        (0, 16, (1, 3)),         # whole document
    ],
)
def test_locate_semantics(start: int, end: int, expected: tuple[int | None, ...]) -> None:
    assert build("alpha", "beta", "gamma").locate(start, end) == expected


@pytest.mark.parametrize(
    "pages,text,separator_only,whole,expected",
    [
        (["", "content"], "\ncontent", (0, 1), (0, 8), (2, 2)),   # empty first
        (["a", ""], "a\n", (1, 2), (0, 2), (1, 1)),               # empty last
        (["a", "", "b"], "a\n\nb", (1, 3), (0, 4), (1, 3)),       # empty middle
    ],
)
def test_empty_pages_never_win_a_lookup(
    pages: list[str],
    text: str,
    separator_only: tuple[int, int],
    whole: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    mapped = build(*pages)
    assert mapped.nfc_text == text
    assert mapped.locate(*separator_only) == (None, None), "separators only"
    assert mapped.locate(*whole) == expected, "empty page must not win"


def test_consecutive_empty_pages_stay_distinct_records() -> None:
    mapped = build("a", "", "", "b")
    assert mapped.nfc_text == "a\n\n\nb" and mapped.page_count == 4
    empty = [span for span in mapped.pages if span.is_empty]
    assert [span.page_number for span in empty] == [2, 3]
    assert len(set(empty)) == 2, "distinct records, not deduplicated"
    assert len(mapped.separators) == 3
    assert mapped.locate(1, 4) == (None, None), "separators and empty pages only"
    assert mapped.locate(0, 5) == (1, 4)


def test_empty_page_span_may_coincide_with_a_separator_boundary() -> None:
    """An empty page's zero-length span sits exactly where a separator starts,
    so coincident offsets must not make the two records interchangeable."""
    mapped = build("a", "", "b")
    assert mapped.pages[1].start_char == mapped.pages[1].end_char == 2
    assert mapped.separators[1].start_char == 2
    assert mapped.pages[1] not in mapped.locatable


def test_one_all_empty_page_yields_empty_text_and_no_locatable_range() -> None:
    mapped = build("")
    assert mapped.raw_text == mapped.nfc_text == ""
    assert mapped.pages == (PageSpan(page_number=1, start_char=0, end_char=0),)
    assert mapped.separators == () and mapped.locatable == ()
    # No range is valid at all, so there is nothing to answer wrongly.
    with pytest.raises(PageMapError) as excinfo:
        mapped.locate(0, 1)
    assert excinfo.value.check == "locate_offset_range"


def test_multiple_all_empty_pages_yield_separator_only_text() -> None:
    mapped = build("", "", "")
    assert mapped.nfc_text == "\n\n" and mapped.page_count == 3
    assert mapped.locatable == () and len(mapped.separators) == 2
    for span in ((0, 2), (0, 1), (1, 2)):
        assert mapped.locate(*span) == (None, None)
    assert build(None, None).locate(0, 1) == (None, None)


def test_bidi_zwj_variation_selector_and_line_separator_are_preserved() -> None:
    """U+2028 is whitespace but is NOT the page separator: it must not create a
    page boundary or be mistaken for one."""
    page_one = RLM + ALEF + LAM + MEEM + LRM
    page_two = WOMAN + ZWJ + LAPTOP + HEART + VS16 + NBSP + "note" + LINE_SEP + "end"
    mapped = build(page_one, page_two)
    assert mapped.nfc_text == page_one + "\n" + page_two
    assert all(c in mapped.nfc_text for c in (RLM, LRM, ZWJ, VS16, LINE_SEP, NBSP))
    assert LINE_SEP.isspace() and mapped.page_count == 2
    assert len(mapped.separators) == 1
    index = mapped.nfc_text.index(LINE_SEP)
    assert mapped.locate(index, index + 1) == (2, 2)
    assert mapped.locate(0, 1) == (1, 1)


@pytest.mark.parametrize(
    "page,raw_len,nfc_page",
    [
        ("caf" + "e" + ACUTE, 5, "caf" + E_ACUTE),  # NFC shrinks 5 -> 4
        (KA_NUKTA, 1, KA + NUKTA),                  # NFC grows 1 -> 2
    ],
)
def test_nfc_normalization_changes_length_and_offsets_follow(
    page: str, raw_len: int, nfc_page: str
) -> None:
    mapped = build(page, "x")
    size = len(nfc_page)
    assert mapped.raw_text == page + "\nx" and len(page) == raw_len
    assert mapped.nfc_text == nfc_page + "\nx"
    assert mapped.pages[0] == PageSpan(page_number=1, start_char=0, end_char=size)
    assert mapped.separators[0].start_char == size
    assert mapped.locate(0, size) == (1, 1)
    assert mapped.locate(size, size + 1) == (None, None)


@pytest.mark.parametrize(
    "pages",
    [
        ["a", "b"],
        ["", ""],
        [None, "x", None],
        ["e" + ACUTE, ACUTE + "e"],
        [ACUTE, "a"],
        ["a", ACUTE],
        [KA_NUKTA, KA_NUKTA],
        [RLM + ALEF, LINE_SEP, NBSP],
        ["a\n", "\nb", "\n"],
        [WOMAN + ZWJ + LAPTOP, HEART + VS16],
    ],
)
def test_global_nfc_invariant_holds_across_page_shapes(pages: list[str | None]) -> None:
    assert_map_invariants(build_page_mapped_text(pages), pages)


@pytest.mark.parametrize(
    "pages,expected_check",
    [
        ([], "empty_page_sequence"),
        ("not a list", "pages_not_a_sequence"),
        (b"bytes", "pages_not_a_sequence"),
        (42, "pages_not_a_sequence"),
        (None, "pages_not_a_sequence"),
        (["ok", 42], "page_not_str"),
        (["ok", b"bytes"], "page_not_str"),
        (["ok", 1.5], "page_not_str"),
        (["ok", ["nested"]], "page_not_str"),
        (["ok", True], "page_not_str"),
        (["x"] * (MAX_PAGE_COUNT + 1), "page_count"),
        (["x" * (MAX_TEXT_CODE_POINTS + 1)], "page_text_length"),
        (["x" * (MAX_TEXT_CODE_POINTS // 2)] * 3, "raw_text_length"),
    ],
)
def test_rejected_inputs(pages: object, expected_check: str) -> None:
    with pytest.raises(PageMapError) as excinfo:
        build_page_mapped_text(pages)  # type: ignore[arg-type]
    assert excinfo.value.check == expected_check


def test_nfc_expansion_over_the_limit_is_rejected() -> None:
    """Raw text is inside the limit; NFC doubles it and pushes it over."""
    page = KA_NUKTA * 1_200_000
    assert len(page) <= MAX_TEXT_CODE_POINTS
    with pytest.raises(PageMapError) as excinfo:
        build_page_mapped_text([page])
    assert excinfo.value.check == "nfc_text_length"


def test_records_are_immutable() -> None:
    mapped = build("alpha", "beta")
    for target, attribute in (
        (mapped.pages[0], "page_number"),
        (mapped.separators[0], "after_page"),
        (mapped, "nfc_text"),
    ):
        with pytest.raises(AttributeError):
            setattr(target, attribute, "tampered")


@pytest.mark.parametrize(
    "start,end", [(-1, 3), (0, 0), (3, 3), (10, 10), (3, 2), (0, 100), (10, 11), (16, 17)]
)
def test_invalid_and_zero_length_locate_ranges_are_rejected(start: int, end: int) -> None:
    with pytest.raises(PageMapError) as excinfo:
        build("alpha", "beta").locate(start, end)
    assert excinfo.value.check == "locate_offset_range"


@pytest.mark.parametrize("bad_offset", [1.5, "0", None, True, False])
def test_non_integer_locate_offsets_are_rejected(bad_offset: object) -> None:
    with pytest.raises(PageMapError) as excinfo:
        build("alpha", "beta").locate(bad_offset, 3)  # type: ignore[arg-type]
    assert excinfo.value.check == "locate_offset_type"


def test_error_messages_never_echo_page_text() -> None:
    secret = "CONFIDENTIAL-TERMINATION-CLAUSE-7"
    with pytest.raises(PageMapError) as excinfo:
        build_page_mapped_text([secret] * (MAX_PAGE_COUNT + 1))
    assert secret not in str(excinfo.value)
    with pytest.raises(PageMapError) as excinfo:
        build_page_mapped_text([secret, 42])  # type: ignore[list-item]
    message = str(excinfo.value)
    assert secret not in message and "CONFIDENTIAL" not in message
    assert "page 2" in message, "position is reportable, content is not"


def test_build_and_locate_stay_bounded_at_the_page_limit() -> None:
    pages = [f"page {index} body text" for index in range(MAX_PAGE_COUNT)]
    started = time.monotonic()
    mapped = build_page_mapped_text(pages)
    assert time.monotonic() - started < 5.0, "build is not linear"
    assert mapped.page_count == MAX_PAGE_COUNT, "the page limit is inclusive"
    started = time.monotonic()
    for span in mapped.pages:
        assert mapped.locate(*(span.start_char, span.end_char)) == (
            span.page_number, span.page_number)
    assert time.monotonic() - started < 5.0, "lookup is not logarithmic"


def test_locate_agrees_with_a_brute_force_scan() -> None:
    """Differential check of the binary search against the plain definition."""
    mapped = build("alpha", "", "beta", None, "gamma", "", "")
    total = len(mapped.nfc_text)
    for start in range(total):
        for end in range(start + 1, total + 1):
            hits = [
                span.page_number
                for span in mapped.pages
                if not span.is_empty and span.start_char < end and start < span.end_char
            ]
            assert mapped.locate(start, end) == (
                (hits[0], hits[-1]) if hits else (None, None)
            ), (start, end)
