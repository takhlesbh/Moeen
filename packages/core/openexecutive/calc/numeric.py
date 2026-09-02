"""The numeric boundary: how a figure is allowed to enter a calculation.

Parsing lives here, in *validation*. No arithmetic does. A value is checked,
normalised, and given a canonical form; it is never added, scaled, or converted.

The rules below are the ones the controlled CFO evaluation showed are needed.
A model that reported an energy cost of "TND 75M" where the source said
"TND 650,000-750,000/year" did not make an arithmetic mistake — it made a
*parsing* mistake, one order of magnitude at a time. Every rule here removes one
way for a magnitude to change silently between a document and a calculation:

* **No float, ever.** ``0.1 + 0.2`` is not ``0.3`` in binary floating point, and
  a contract whose entire purpose is exactness cannot accept a type that has
  already lost information before validation sees it. Floats are rejected with
  an explicit error rather than coerced.
* **No implicit separators.** ``"1,234"`` is one thousand two hundred and
  thirty-four to a British reader and one point two three four to a French one.
  Version 1 refuses to guess: the plain format forbids commas entirely, and
  comma-thousands must be asked for by name.
* **No decimal comma.** Deliberately unsupported rather than supported-and-
  ambiguous. ``"1.234,56"`` is rejected with a message saying why.
* **No automatic scaling.** Nothing here turns thousands into millions. A
  magnitude change requires an explicit unit conversion the engine can refuse.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Literal

# --- contract-level limits -------------------------------------------------

MAX_NUMERIC_STRING_LEN = 64
"""Longest accepted numeric literal. A 64-digit financial figure is a bug or an
attack, not a number anyone typed."""

MAX_ADJUSTED_EXPONENT = 30
"""Bound on ``Decimal.adjusted()`` magnitude, applied to EVERY value.

Rejects ``1e400``, ``1e-400`` and ``0e-400`` alike. The last is the one that
matters: a zero carrying a huge negative exponent compares equal to zero, so any
guard written as "check unless it is zero" lets it through — and positional
formatting then expands it into a string the size of the exponent. 10^30 is far
beyond any real capital structure."""

MAX_PRECISION_REQUEST = 50
"""Ceiling on a caller-requested working precision, for the Phase 2 engine."""

MAX_SCALE = 28
"""Ceiling on requested output decimal places."""

NumberFormat = Literal["plain", "comma_thousands"]
"""Closed enum. Not a locale name.

Accepting arbitrary locale identifiers would reintroduce exactly the ambiguity
this module exists to remove, and would make parsing depend on process locale
state. ``plain`` means digits, one optional ``.``, an optional exponent, and
nothing else. ``comma_thousands`` additionally permits ``,`` in the
integer part in strict groups of three.
"""

_PLAIN_RE = re.compile(r"\A[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?\Z")
_COMMA_THOUSANDS_RE = re.compile(r"\A[+-]?[0-9]{1,3}(,[0-9]{3})+(\.[0-9]+)?\Z")
_DECIMAL_COMMA_HINT_RE = re.compile(r"[0-9],[0-9]{1,2}\Z")
# ``[0-9]`` not ``\d``: ``\d`` is Unicode-aware, so ``\d`` would accept
# "\u0663.\u0665" and "\uff11\uff12\uff13" as numbers. Those map to faithful values and
# canonicalise to ASCII, so no magnitude changes — but the stored canonical
# string would differ textually from the source literal, and this boundary is
# specified as digit literals. ``\A``/``\Z`` rather than ``^``/``$`` because
# ``$`` also matches before a terminal newline.


class NumericPolicyError(ValueError):
    """Raised when a value may not enter a calculation. Always says why."""


def parse_numeric(raw: object, number_format: NumberFormat = "plain") -> Decimal:
    """Validate and normalise one operand value into an exact ``Decimal``.

    Accepts ``str`` and ``int``. Rejects ``float``, ``bool``, ``Decimal``
    instances built elsewhere, and everything else.

    ``int`` is accepted under an explicit lossless policy: a Python ``int`` has
    unbounded exact precision, so ``Decimal(str(value))`` round-trips it without
    loss. ``bool`` is rejected even though it subclasses ``int`` — ``True`` as a
    financial operand is a bug every time.

    ``Decimal`` is rejected too, which is deliberate rather than pedantic: a
    ``Decimal`` arriving here has already been constructed by someone else,
    possibly from a float (``Decimal(0.1)`` is ``0.1000000000000000055511...``),
    and this function's contract is that it is the *only* place a value becomes
    a Decimal.
    """
    if isinstance(raw, bool):
        raise NumericPolicyError(
            "bool is not a numeric operand; pass an explicit number as a string"
        )
    if isinstance(raw, float):
        raise NumericPolicyError(
            "float is rejected: binary floating point cannot represent decimal "
            "figures exactly. Pass the value as a canonical string, e.g. "
            '"30140000.00" rather than 30140000.0'
        )
    if isinstance(raw, Decimal):
        raise NumericPolicyError(
            "Decimal is rejected at the boundary: it may have been built from a "
            "float upstream. Pass the originating string instead."
        )
    if isinstance(raw, int):
        text = str(raw)
    elif isinstance(raw, str):
        text = raw
    else:
        raise NumericPolicyError(
            f"unsupported numeric type {type(raw).__name__}; expected str or int"
        )

    if text == "" or text.strip() != text:
        raise NumericPolicyError(
            "numeric value must be a non-empty string with no surrounding "
            "whitespace"
        )
    if len(text) > MAX_NUMERIC_STRING_LEN:
        raise NumericPolicyError(
            f"numeric value exceeds {MAX_NUMERIC_STRING_LEN} characters "
            f"({len(text)})"
        )

    if _DECIMAL_COMMA_HINT_RE.search(text):
        raise NumericPolicyError(
            f"{text!r} looks like a decimal comma. Version 1 does not interpret "
            "decimal commas because the same string means different magnitudes "
            "in different conventions. Supply the value with a '.' decimal point."
        )

    if number_format == "plain":
        if "," in text:
            raise NumericPolicyError(
                f"{text!r} contains a separator but number_format is 'plain'. "
                "Separators are never inferred; declare "
                "number_format='comma_thousands' if that is what is meant."
            )
        if not _PLAIN_RE.match(text):
            raise NumericPolicyError(f"{text!r} is not a canonical decimal literal")
        cleaned = text
    else:
        if "," in text:
            if not _COMMA_THOUSANDS_RE.match(text):
                raise NumericPolicyError(
                    f"{text!r} is not valid comma-thousands notation: groups "
                    "must be exactly three digits"
                )
            cleaned = text.replace(",", "")
        else:
            if not _PLAIN_RE.match(text):
                raise NumericPolicyError(f"{text!r} is not a canonical decimal literal")
            cleaned = text

    try:
        value = Decimal(cleaned)
    except InvalidOperation as exc:
        # Reachable, despite the regex: ``Decimal()`` raises on an exponent past
        # its own parseable range, e.g. "0e-9999999999999999999", which
        # ``_PLAIN_RE`` matches happily. A draft carried a "regex already
        # guards" pragma here; it did not.
        raise NumericPolicyError(f"{text!r} is not a decimal number") from exc

    # The regexes already exclude the NaN/Infinity spellings, so this is defence
    # in depth rather than the primary guard — and it is the check that would
    # still hold if a regex were ever loosened.
    if not value.is_finite():
        raise NumericPolicyError(
            "NaN and Infinity are not calculable figures and are rejected"
        )
    # NOT ``if value != 0 and ...``. A draft exempted zero, and zero is exactly
    # where the exemption bites: ``Decimal("0e-2000000000") == 0``, so its
    # adjusted exponent of -2000000000 went unchecked, and
    # ``canonical_numeric_string`` — ``format(value, "f")`` — then expanded it
    # positionally into a two-billion-character string. Measured: 4 GB resident
    # from a 13-character literal, well inside MAX_NUMERIC_STRING_LEN, on the
    # exact wire shape a model emits. The length backstop runs after the
    # expansion, so it cannot help. ``Decimal.adjusted()`` is well defined for
    # zero (``Decimal("0").adjusted() == 0``), so the exemption bought nothing.
    if abs(value.adjusted()) > MAX_ADJUSTED_EXPONENT:
        raise NumericPolicyError(
            f"exponent out of range: adjusted exponent {value.adjusted()} "
            f"exceeds the permitted magnitude of {MAX_ADJUSTED_EXPONENT}"
        )
    if value == 0 and value.is_signed():
        # Negative zero canonicalises to "-0", which would fingerprint
        # differently from "0" while being the same quantity. Scale is evidence
        # here and is preserved deliberately; a sign on zero is not — no
        # document states a figure as negative zero. ``copy_abs`` only clears
        # the sign bit: it is not an arithmetic operation and sets no Inexact or
        # Rounded flag, so it does not breach this package's no-arithmetic rule.
        return value.copy_abs()
    return value


def canonical_numeric_string(value: Decimal) -> str:
    """The stable serialized form of a value, for storage and fingerprinting.

    Plain positional notation, never exponent notation, never locale-dependent,
    and **scale-preserving**: ``1.50`` and ``1.5`` render differently and are
    therefore fingerprinted differently. That is correct rather than a wart — a
    figure a document states to two decimal places is not the same evidence as
    one it states to one, and collapsing them would erase a real difference
    between two sources.
    """
    return format(value, "f")
