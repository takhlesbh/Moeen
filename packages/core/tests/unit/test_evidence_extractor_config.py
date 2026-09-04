"""Extractor configuration: a closed, bounded value space with a canonical hash."""
from __future__ import annotations

from typing import Any

import pytest

from openexecutive.evidence import extractor_config as ec
from openexecutive.evidence.extractor_config import (
    MAX_DEPTH,
    MAX_INT_MAGNITUDE,
    MAX_ITEMS,
    MAX_SERIALIZED_BYTES,
    MAX_STRING_CODE_POINTS,
    ExtractorConfigError,
    canonical_config_bytes,
    extractor_config_sha256,
)


def test_key_order_does_not_change_the_hash() -> None:
    assert extractor_config_sha256({"a": 1, "b": 2}) == extractor_config_sha256({"b": 2, "a": 1})


def test_nested_key_order_does_not_change_the_hash() -> None:
    left = {"outer": {"z": [1, 2], "a": "x"}}
    right = {"outer": {"a": "x", "z": [1, 2]}}
    assert extractor_config_sha256(left) == extractor_config_sha256(right)


def test_changed_value_changes_the_hash() -> None:
    assert extractor_config_sha256({"a": 1}) != extractor_config_sha256({"a": 2})


def test_list_order_does_change_the_hash() -> None:
    assert extractor_config_sha256([1, 2]) != extractor_config_sha256([2, 1])


def test_canonical_form_is_compact_sorted_and_not_ascii_escaped() -> None:
    assert canonical_config_bytes({"b": 1, "a": "\u00e9"}) == b'{"a":"\xc3\xa9","b":1}'


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        {"k": float("nan")},
        [float("inf")],
    ],
)
def test_non_finite_floats_are_rejected(value: Any) -> None:
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256(value)


@pytest.mark.parametrize("key", [1, None, (1, 2), True])
def test_non_string_dict_keys_are_rejected(key: Any) -> None:
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256({key: "v"})


@pytest.mark.parametrize("value", [object(), {1, 2}, b"bytes", (1, 2), 1j])
def test_arbitrary_objects_are_rejected(value: Any) -> None:
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256(value)


def test_depth_limit() -> None:
    ok: Any = "leaf"
    for _ in range(MAX_DEPTH):
        ok = [ok]
    assert extractor_config_sha256(ok)
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256([ok])


def test_item_count_limit() -> None:
    assert extractor_config_sha256([0] * MAX_ITEMS)
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256([0] * (MAX_ITEMS + 1))


def test_item_count_is_summed_across_nested_collections() -> None:
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256([[0] * 200, [0] * 200])


def test_string_length_limit() -> None:
    assert extractor_config_sha256("a" * MAX_STRING_CODE_POINTS)
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256("a" * (MAX_STRING_CODE_POINTS + 1))


def test_dict_key_length_limit() -> None:
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256({"a" * (MAX_STRING_CODE_POINTS + 1): 1})


def test_integer_magnitude_limit() -> None:
    assert extractor_config_sha256(MAX_INT_MAGNITUDE)
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256(MAX_INT_MAGNITUDE + 1)
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256(-MAX_INT_MAGNITUDE - 1)


def test_serialized_size_limit() -> None:
    too_big = {"k": "a" * MAX_STRING_CODE_POINTS}
    too_big.update({f"k{i}": "a" * MAX_STRING_CODE_POINTS for i in range(20)})
    with pytest.raises(ExtractorConfigError) as excinfo:
        extractor_config_sha256(too_big)
    assert str(MAX_SERIALIZED_BYTES) in str(excinfo.value)


def test_lone_surrogate_in_a_string_is_rejected() -> None:
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256({"k": "a\ud800b"})


def test_lone_surrogate_in_a_key_is_rejected() -> None:
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256({"a\ud800b": 1})


@pytest.mark.parametrize(
    "value",
    [float("nan"), {1: "v"}, object(), [[[[[[[[["deep"]]]]]]]]], [0] * (MAX_ITEMS + 1)],
)
def test_structural_failures_occur_before_canonical_serialization(
    value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_: Any) -> bytes:
        raise AssertionError("serialization must not be reached")

    monkeypatch.setattr(ec, "_canonical_json", boom)
    with pytest.raises(ExtractorConfigError):
        extractor_config_sha256(value)


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "", [], {}, {"a": [1, {"b": None}]}])
def test_the_closed_value_space_is_accepted(value: Any) -> None:
    assert len(extractor_config_sha256(value)) == 64
