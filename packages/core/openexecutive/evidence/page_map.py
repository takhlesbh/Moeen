"""Page-mapped NFC document text: the coordinate system spans address.

Pure and in-memory. No filesystem, no network, no clock, no PDF library, no
ambient configuration — a caller hands over already-extracted per-page text and
gets back the assembled text plus the offset map. Extraction itself (pypdf and
friends) stays outside this module, so the mapping is testable without a
document.

**Nothing here mints a canonical evidence record.** No ``DocumentVersion``, no
``Extraction``, no ``SourceSpan``, no scope and no logical-source key. This
module produces *coordinates*; :mod:`openexecutive.evidence.factory` remains the
only path that mints identity, and a later slice wires the two together.

**The separator is owned by no page.** Pages are joined with a fixed
``PAGE_SEPARATOR`` that the assembler *inserts*; it is not part of any page's
extracted text, so attributing it to the preceding page would let a range that
touched only inserted whitespace report a page number that no source page
supports. Each inserted separator therefore gets its own explicit
:class:`SeparatorSpan`, and :meth:`PageMappedText.locate` returns
``(None, None)`` for a range that intersects nothing else. **A page locator is
never guessed.**

**Empty pages are real structure, not absence.** A page whose extractor returned
``None`` or ``""`` keeps its page number and gets a zero-length span, so page
numbering stays faithful to the document. A zero-length span can never
*intersect* a non-empty half-open range, so empty pages are excluded from
lookup rather than special-cased inside it.

**Offsets are Python ``str`` indices — Unicode code points, never UTF-8 byte
positions** — matching the offset unit ``evidence.contracts`` defines for
``SourceSpan``. ``start_char`` is inclusive, ``end_char`` exclusive.
"""
from __future__ import annotations

import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field

PAGE_SEPARATOR = "\n"
"""Fixed. Not caller-supplied: the separator participates in every offset in the
map, so letting a caller vary it would make two maps of the same document
silently incomparable. One code point, so every separator span is non-empty."""

MAX_TEXT_CODE_POINTS = 2_000_000
"""Assembled raw and NFC text ceiling. Matches ``factory.MAX_TEXT_CODE_POINTS``
so text that maps here cannot later be refused by the minting path."""

MAX_PAGE_COUNT = 10_000
"""Conservative fixed page ceiling. Well above any real business document
(a 10,000-page PDF is not a report), low enough that the per-page records and
the two span lists stay small and bounded."""


class PageMapError(ValueError):
    """A page sequence or a lookup range was rejected.

    ``check`` is a stable literal naming the failed check, for tests and logs to
    assert on. ``detail`` carries only counts and offsets — **never page text**:
    these failures get logged, and the input is a document body.
    """

    def __init__(self, check: str, detail: str = "") -> None:
        self.check = check
        super().__init__(
            f"page map rejected: {check}" + (f" ({detail})" if detail else "")
        )


@dataclass(frozen=True, slots=True)
class PageSpan:
    """One source page's half-open range in the assembled NFC text.

    ``page_number`` is 1-based, as a document's pages are cited. An empty page
    has ``start_char == end_char``; two empty pages stay distinct records even
    where their offsets coincide, because the page number is part of the record.
    """

    page_number: int
    start_char: int
    end_char: int

    @property
    def is_empty(self) -> bool:
        return self.start_char == self.end_char


@dataclass(frozen=True, slots=True)
class SeparatorSpan:
    """One *inserted* separator's range. Synthetic: no page owns it.

    ``after_page`` is the 1-based number of the page it follows, which locates
    the separator without granting either neighbour a claim on it.
    """

    after_page: int
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class PageMappedText:
    """Assembled document text and its complete offset map.

    Carries its own coordinate text: ``nfc_text`` is the string every span
    addresses, so a map can never be read against text it did not describe.
    """

    raw_text: str
    nfc_text: str
    pages: tuple[PageSpan, ...]
    separators: tuple[SeparatorSpan, ...]
    locatable: tuple[PageSpan, ...] = field(repr=False)
    """Non-empty pages only, in order — the lookup domain. Zero-length pages are
    absent by construction, which is what stops an empty page winning a lookup."""

    @property
    def page_count(self) -> int:
        """Number of source pages, empty ones included."""
        return len(self.pages)

    def locate(self, start_char: int, end_char: int) -> tuple[int | None, int | None]:
        """First and last non-empty page numbers a half-open range intersects.

        Returns ``(None, None)`` when the range lies entirely within inserted
        separators — never a guessed neighbour. A range touching exactly one
        page returns ``(n, n)``; one crossing pages returns ``(first, last)``.

        Offsets are code points into :attr:`nfc_text`, ``end_char`` exclusive.
        An empty range is rejected rather than answered: ``start == end``
        intersects nothing, so no page number could be honest.
        """
        for label, value in (("start_char", start_char), ("end_char", end_char)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise PageMapError("locate_offset_type", label)
        if not 0 <= start_char < end_char <= len(self.nfc_text):
            raise PageMapError(
                "locate_offset_range",
                f"[{start_char},{end_char}) len={len(self.nfc_text)}",
            )

        spans = self.locatable
        if not spans:
            return (None, None)
        # Spans are disjoint and ordered, so both bounds are a binary search.
        # `first`: earliest span whose end is past `start_char`.
        # `last`:  latest span whose start is before `end_char`.
        # Half-open intersection is exactly `span.start < end and start < span.end`.
        first = bisect_right(spans, start_char, key=lambda span: span.end_char)
        last = bisect_left(spans, end_char, key=lambda span: span.start_char) - 1
        if first > last:
            return (None, None)
        return (spans[first].page_number, spans[last].page_number)


def build_page_mapped_text(pages: Sequence[str | None]) -> PageMappedText:
    """Assemble per-page extracted text into page-mapped NFC text.

    ``None`` means the extractor returned nothing for that page and is treated
    as empty — the page still counts. Page text is **never** stripped, collapsed
    or rewritten: only NFC normalization is applied, exactly once per page, so
    offsets address a representation the caller can reproduce.

    An empty page sequence is rejected. A PDF has pages; an empty list is a
    caller or extractor defect, and inventing a zero-page document would let
    every later lookup answer against text that describes nothing.
    """
    if isinstance(pages, str | bytes) or not isinstance(pages, Sequence):
        raise PageMapError("pages_not_a_sequence")
    count = len(pages)
    if count == 0:
        raise PageMapError("empty_page_sequence")
    if count > MAX_PAGE_COUNT:
        raise PageMapError("page_count", f"{count} > {MAX_PAGE_COUNT}")

    # Cheap structural checks and a running length total BEFORE any join or
    # normalization, so a hostile page list fails without first materializing
    # a multi-megabyte string.
    raw_pages: list[str] = []
    raw_total = count - 1  # the separators this assembly will insert
    for index, page in enumerate(pages):
        if page is None:
            raw_pages.append("")
            continue
        if not isinstance(page, str):
            raise PageMapError("page_not_str", f"page {index + 1}")
        if len(page) > MAX_TEXT_CODE_POINTS:
            raise PageMapError("page_text_length", f"page {index + 1}")
        raw_total += len(page)
        if raw_total > MAX_TEXT_CODE_POINTS:
            raise PageMapError("raw_text_length", f"> {MAX_TEXT_CODE_POINTS}")
        raw_pages.append(page)

    # Normalize each page exactly once. NFC can *grow* a page (U+0958 composes
    # out to two code points), so the assembled NFC length is re-checked rather
    # than assumed to be bounded by the raw total.
    nfc_pages = [unicodedata.normalize("NFC", page) for page in raw_pages]
    nfc_total = sum(len(page) for page in nfc_pages) + count - 1
    if nfc_total > MAX_TEXT_CODE_POINTS:
        raise PageMapError("nfc_text_length", f"> {MAX_TEXT_CODE_POINTS}")

    raw_text = PAGE_SEPARATOR.join(raw_pages)
    nfc_text = PAGE_SEPARATOR.join(nfc_pages)
    # Per-page normalization must agree with normalizing the whole document.
    # It does, because U+000A has combining class 0 and composes with nothing,
    # so it terminates every combining sequence and no reordering crosses it.
    # That is a property of this separator, not a general law about NFC and
    # concatenation, so it is verified rather than trusted — one linear pass.
    if nfc_text != unicodedata.normalize("NFC", raw_text):
        raise PageMapError("nfc_join_invariant")

    page_spans: list[PageSpan] = []
    separator_spans: list[SeparatorSpan] = []
    cursor = 0
    for index, text in enumerate(nfc_pages):
        if index > 0:
            # Claimed before the page that follows it, and by neither neighbour.
            separator_spans.append(
                SeparatorSpan(
                    after_page=index,
                    start_char=cursor,
                    end_char=cursor + len(PAGE_SEPARATOR),
                )
            )
            cursor += len(PAGE_SEPARATOR)
        page_spans.append(
            PageSpan(page_number=index + 1, start_char=cursor, end_char=cursor + len(text))
        )
        cursor += len(text)
    if cursor != len(nfc_text):
        raise PageMapError("span_coverage", f"{cursor} != {len(nfc_text)}")

    return PageMappedText(
        raw_text=raw_text,
        nfc_text=nfc_text,
        pages=tuple(page_spans),
        separators=tuple(separator_spans),
        locatable=tuple(span for span in page_spans if not span.is_empty),
    )
