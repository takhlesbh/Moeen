"""Identity encoding: injective framing, domain separation, determinism."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from openexecutive.evidence import identity
from openexecutive.evidence.identity import (
    DISCRIMINATOR_BYTES,
    DISCRIMINATOR_INT,
    DISCRIMINATOR_STR,
    ID_PREFIX,
    MAX_COMPONENT_BYTES,
    MAX_ID_COMPONENTS,
    TAG_DOCUMENT_VERSION,
    TAG_EXTRACTION,
    TAG_LOGICAL_SOURCE,
    TAG_SOURCE_SPAN,
    IdentityError,
    content_sha256,
    mint_id,
    text_sha256,
)

ALL_TAGS = [TAG_LOGICAL_SOURCE, TAG_DOCUMENT_VERSION, TAG_EXTRACTION, TAG_SOURCE_SPAN]


def test_content_sha256_is_plain_sha256_of_the_bytes() -> None:
    assert content_sha256(b"hello") == hashlib.sha256(b"hello").hexdigest()


def test_text_sha256_is_sha256_of_utf8_with_no_normalization() -> None:
    decomposed = "e\u0301"
    composed = "\u00e9"
    assert text_sha256(decomposed) == hashlib.sha256(decomposed.encode("utf-8")).hexdigest()
    assert text_sha256(decomposed) != text_sha256(composed)


def test_text_sha256_rejects_lone_surrogate() -> None:
    with pytest.raises(IdentityError):
        text_sha256("a\ud800b")


@pytest.mark.parametrize("bad", [b"bytes", 1, None, 1.5])
def test_text_sha256_requires_str(bad: object) -> None:
    with pytest.raises(IdentityError):
        text_sha256(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["str", 1, None])
def test_content_sha256_requires_bytes(bad: object) -> None:
    with pytest.raises(IdentityError):
        content_sha256(bad)  # type: ignore[arg-type]


def test_length_framing_makes_components_unambiguous() -> None:
    """Without the length prefix these two would digest identically."""
    assert mint_id(TAG_EXTRACTION, b"ab", b"c") != mint_id(TAG_EXTRACTION, b"a", b"bc")


def test_component_count_changes_the_id() -> None:
    assert mint_id(TAG_EXTRACTION, b"a") != mint_id(TAG_EXTRACTION, b"a", b"")


@pytest.mark.parametrize("left", ALL_TAGS)
@pytest.mark.parametrize("right", ALL_TAGS)
def test_distinct_type_tags_never_collide(left: bytes, right: bytes) -> None:
    same = mint_id(left, "scope", "key") == mint_id(right, "scope", "key")
    assert same == (left == right)


def _frame(kind: bytes, payload: bytes) -> bytes:
    """The documented typed frame: discriminator + 4-byte BE length + payload."""
    return kind + len(payload).to_bytes(4, "big") + payload


def test_digest_matches_the_documented_typed_framing() -> None:
    manual = hashlib.sha256()
    manual.update(ID_PREFIX)
    manual.update(_frame(DISCRIMINATOR_BYTES, TAG_EXTRACTION))
    manual.update(_frame(DISCRIMINATOR_BYTES, b"a"))
    manual.update(_frame(DISCRIMINATOR_STR, "\u00e9".encode()))
    manual.update(_frame(DISCRIMINATOR_INT, (7).to_bytes(8, "big")))
    assert mint_id(TAG_EXTRACTION, b"a", "\u00e9", 7) == manual.hexdigest()
    assert (DISCRIMINATOR_BYTES, DISCRIMINATOR_STR, DISCRIMINATOR_INT) == (b"b", b"s", b"i")


def test_str_and_bytes_never_collide() -> None:
    assert mint_id(TAG_EXTRACTION, "a") != mint_id(TAG_EXTRACTION, b"a")
    assert mint_id(TAG_EXTRACTION, "\u00e9") != mint_id(TAG_EXTRACTION, "\u00e9".encode())


def test_int_and_bytes_never_collide() -> None:
    assert mint_id(TAG_SOURCE_SPAN, 0) != mint_id(TAG_SOURCE_SPAN, b"\x00" * 8)
    assert mint_id(TAG_SOURCE_SPAN, 1) != mint_id(TAG_SOURCE_SPAN, (1).to_bytes(8, "big"))


def test_int_and_str_never_collide() -> None:
    assert mint_id(TAG_SOURCE_SPAN, "\x00" * 8) != mint_id(TAG_SOURCE_SPAN, 0)
    assert mint_id(TAG_SOURCE_SPAN, "\x00" * 7 + "\x01") != mint_id(TAG_SOURCE_SPAN, 1)


@pytest.mark.parametrize("tag", ALL_TAGS)
@pytest.mark.parametrize("value", [0, 1, 2**64 - 1])
def test_every_cross_type_pair_differs_at_the_same_position(tag: bytes, value: int) -> None:
    as_int = mint_id(tag, "ctx", value)
    as_bytes = mint_id(tag, "ctx", value.to_bytes(8, "big"))
    as_str = mint_id(tag, "ctx", value.to_bytes(8, "big").decode("latin-1"))
    assert len({as_int, as_bytes, as_str}) == 3


def test_int_components_are_unsigned_eight_byte_big_endian() -> None:
    assert mint_id(TAG_SOURCE_SPAN, 0) != mint_id(TAG_SOURCE_SPAN, 1)
    assert mint_id(TAG_SOURCE_SPAN, 2**64 - 1)


@pytest.mark.parametrize("bad", [-1, 2**64, True, False, None, 1.5, ["a"]])
def test_unsupported_or_out_of_range_components_are_rejected(bad: object) -> None:
    with pytest.raises(IdentityError):
        mint_id(TAG_EXTRACTION, bad)  # type: ignore[arg-type]


def test_component_size_bound() -> None:
    assert mint_id(TAG_EXTRACTION, b"x" * MAX_COMPONENT_BYTES)
    with pytest.raises(IdentityError):
        mint_id(TAG_EXTRACTION, b"x" * (MAX_COMPONENT_BYTES + 1))


def test_component_count_bound_includes_the_tag() -> None:
    assert mint_id(TAG_EXTRACTION, *([b"a"] * (MAX_ID_COMPONENTS - 1)))
    with pytest.raises(IdentityError):
        mint_id(TAG_EXTRACTION, *([b"a"] * MAX_ID_COMPONENTS))


def test_type_tag_must_be_bytes() -> None:
    with pytest.raises(IdentityError):
        mint_id("extraction", b"a")  # type: ignore[arg-type]


def test_lone_surrogate_in_a_str_component_is_rejected() -> None:
    with pytest.raises(IdentityError):
        mint_id(TAG_EXTRACTION, "a\ud800b")


def test_ids_are_stable_across_separate_python_processes() -> None:
    script = (
        "from openexecutive.evidence.identity import mint_id, TAG_EXTRACTION;"
        "print(mint_id(TAG_EXTRACTION, 'scope', b'bytes', 7))"
    )
    core = Path(identity.__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=core,
    )
    assert completed.stdout.strip() == mint_id(TAG_EXTRACTION, "scope", b"bytes", 7)


def test_mint_id_returns_lowercase_hex_sha256() -> None:
    minted = mint_id(TAG_EXTRACTION, b"a")
    assert len(minted) == 64
    assert minted == minted.lower()
    assert all(c in "0123456789abcdef" for c in minted)
