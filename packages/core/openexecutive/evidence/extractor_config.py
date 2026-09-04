"""Canonical, bounded hashing of an extractor's configuration.

The configuration hash participates in ``extraction_id``, so its determinism is
an identity property, not a convenience.

*Closed.* Only ``None``, ``bool``, ``int``, ``float``, ``str``, ``list`` and
``dict`` with string keys. An arbitrary object has no canonical serialization,
so it is rejected rather than coerced by ``repr``.

*Bounded.* Depth, item count, string length, integer magnitude and serialized
size are capped, and the value is walked **before** any serialization, so a
hostile configuration fails a cheap structural check rather than inside the JSON
encoder. Integers cap at ±2^53, the range JSON numbers round-trip exactly.

Canonical form is ``json.dumps`` with sorted keys, no whitespace,
``ensure_ascii=False``, ``allow_nan=False``, UTF-8. Key order in the caller's
dict is irrelevant to the hash; the values are not.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, TypeAlias

ConfigValue: TypeAlias = (
    "None | bool | int | float | str | list[ConfigValue] | dict[str, ConfigValue]"
)

MAX_STRING_CODE_POINTS = 1_024
MAX_DEPTH = 8
MAX_ITEMS = 256
MAX_SERIALIZED_BYTES = 16 * 1024
MAX_INT_MAGNITUDE = 2**53


class ExtractorConfigError(ValueError):
    """An extractor configuration was outside the closed, bounded value space."""


def _check_string(value: str, what: str) -> None:
    if len(value) > MAX_STRING_CODE_POINTS:
        raise ExtractorConfigError(f"{what} exceeds {MAX_STRING_CODE_POINTS} code points")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ExtractorConfigError(f"{what} contains a lone surrogate") from exc


def _walk(value: Any, depth: int, counter: list[int]) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > MAX_INT_MAGNITUDE:
            raise ExtractorConfigError("int exceeds +/- 2^53")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExtractorConfigError("float must be finite (NaN and Infinity are rejected)")
        return
    if isinstance(value, str):
        _check_string(value, "string value")
        return
    if isinstance(value, list | dict):
        if depth > MAX_DEPTH:
            raise ExtractorConfigError(f"nesting deeper than {MAX_DEPTH}")
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in items:
            counter[0] += 1
            if counter[0] > MAX_ITEMS:
                raise ExtractorConfigError(f"more than {MAX_ITEMS} collection items")
            if isinstance(value, dict):
                if not isinstance(key, str):
                    raise ExtractorConfigError("dict keys must be str")
                _check_string(key, "dict key")
            _walk(item, depth + 1, counter)
        return
    raise ExtractorConfigError(f"unsupported config type: {type(value).__name__}")


def validate_config(value: ConfigValue) -> None:
    """Walk the whole value and raise on the first bound or type violation."""
    _walk(value, 1, [0])


def _canonical_json(value: ConfigValue) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_config_bytes(value: ConfigValue) -> bytes:
    """Validate, then serialize to the canonical byte form."""
    validate_config(value)
    encoded = _canonical_json(value)
    if len(encoded) > MAX_SERIALIZED_BYTES:
        raise ExtractorConfigError(f"canonical form exceeds {MAX_SERIALIZED_BYTES} bytes")
    return encoded


def extractor_config_sha256(value: ConfigValue) -> str:
    """Hex SHA-256 of the canonical byte form. Feeds ``extraction_id``."""
    return hashlib.sha256(canonical_config_bytes(value)).hexdigest()
