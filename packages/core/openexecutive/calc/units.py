"""Closed unit registry and dimension model for the calculation contracts.

This module is **data and type predicates only**. It performs no conversion and
no dimensional arithmetic: it declares which units exist, what dimension each
belongs to, and the exact factor a future engine will use. Nothing here
multiplies, divides, or converts a value. That is deliberate — Phase 1 ships the
vocabulary the Phase 2 engine will be held to, so the engine can be reviewed
against a contract that already exists rather than one it invents for itself.

Why a closed registry rather than free-text units. The controlled CFO
evaluation produced three unit failures that a string field cannot catch: a
production figure understated by exactly 10^4 (hectares treated as square
metres), an energy cost misquoted by 10^2 (thousands read as millions), and a
gross-margin claim compared against an operating-cost percentage as though a
percentage and a percentage-point difference were the same quantity. Each is a
*dimension* error. Making dimension part of the type turns all three from
judgement calls into validation failures.

Three rules the registry exists to make structural:

1. **No implicit guessing.** An unknown code is a validation error, never a
   default. There is no "assume dimensionless" path.
2. **``pct`` and ``pct_point`` are different dimensions**, not different labels
   for one. "Margin is 60%" and "the gap is 25 percentage points" are not
   additively compatible, and the type system says so.
3. **Conversion factors are exact.** Every factor is a :class:`~decimal.Decimal`
   built from an integer string. No float ever touches this module, so no
   conversion the engine later performs can carry binary rounding error that
   originated here.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal
from types import MappingProxyType
from typing import Literal

from pydantic import Field, field_validator, model_validator

from openexecutive.calc._model import ContractModel

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

Dimension = Literal[
    "dimensionless",
    "currency",
    "mass",
    "area",
    "time",
    "percentage",
    "percentage_point",
    "mass_per_area",
]
"""The closed set of dimension families in Version 1.

``percentage`` and ``percentage_point`` are separate members on purpose; see the
module docstring. ``mass_per_area`` exists so a yield figure (``kg_per_m2``) has
a real dimension rather than being smuggled through as dimensionless.
"""

ConversionPolicy = Literal["exact", "explicit_required", "none"]
"""How a future engine is permitted to convert within a dimension.

* ``exact`` — the ratio is a defined integer constant (hectare/square metre,
  tonne/kilogram). Conversion is lossless and needs no caller decision.
* ``explicit_required`` — a ratio exists but is a *convention*, not a fact. A
  month is not 1/12 of a financial year under every day-count basis, and a
  monthly figure annualised without saying how is the "monthly vs annual"
  failure this contract is meant to catch. The engine must be told the policy;
  it may not assume one.
* ``none`` — conversion within this dimension is undefined. Currency lives here:
  TND and EUR share the ``currency`` dimension but no rate authority exists, and
  Version 1 introduces none.
"""


class UnitSpec(ContractModel):
    """One registered unit: its identity, dimension, and exact base factor.

    ``factor_to_base`` is the multiplier onto the dimension's base unit, held as
    an exact ``Decimal``. It is *registry data*: this module never applies it.
    """

    code: str = Field(min_length=1, max_length=32)
    dimension: Dimension
    base_code: str = Field(min_length=1, max_length=32)
    factor_to_base: Decimal
    conversion_policy: ConversionPolicy
    display: str = Field(max_length=32)

    @field_validator("factor_to_base", mode="before")
    @classmethod
    def _factor_must_not_be_a_float(cls, value: object) -> object:
        """Pydantic would coerce ``0.1`` to ``Decimal("0.1000000000000000055…")``.

        The module claims no float ever touches it; without this the claim held
        for ``parse_numeric`` and not for the registry's own type.
        """
        if isinstance(value, float):
            raise ValueError(
                "factor_to_base must not be a float; conversion factors are "
                "exact and are built from string or int literals"
            )
        return value


def _d(literal: str) -> Decimal:
    """Exact Decimal from a string literal. The only way factors are built here."""
    return Decimal(literal)


_MUTABLE_REGISTRY: dict[str, UnitSpec] = {
    spec.code: spec
    for spec in (
        UnitSpec(
            code="one", dimension="dimensionless", base_code="one",
            factor_to_base=_d("1"), conversion_policy="exact", display="",
        ),
        UnitSpec(
            code="pct", dimension="percentage", base_code="pct",
            factor_to_base=_d("1"), conversion_policy="exact", display="%",
        ),
        UnitSpec(
            code="pct_point", dimension="percentage_point", base_code="pct_point",
            factor_to_base=_d("1"), conversion_policy="exact", display="pp",
        ),
        UnitSpec(
            code="kg", dimension="mass", base_code="kg",
            factor_to_base=_d("1"), conversion_policy="exact", display="kg",
        ),
        UnitSpec(
            code="t", dimension="mass", base_code="kg",
            factor_to_base=_d("1000"), conversion_policy="exact", display="t",
        ),
        UnitSpec(
            code="m2", dimension="area", base_code="m2",
            factor_to_base=_d("1"), conversion_policy="exact", display="m²",
        ),
        UnitSpec(
            code="ha", dimension="area", base_code="m2",
            factor_to_base=_d("10000"), conversion_policy="exact", display="ha",
        ),
        UnitSpec(
            code="month", dimension="time", base_code="month",
            factor_to_base=_d("1"), conversion_policy="explicit_required", display="month",
        ),
        UnitSpec(
            code="year", dimension="time", base_code="month",
            factor_to_base=_d("12"), conversion_policy="explicit_required", display="year",
        ),
        UnitSpec(
            code="kg_per_m2", dimension="mass_per_area", base_code="kg_per_m2",
            factor_to_base=_d("1"), conversion_policy="exact", display="kg/m²",
        ),
    )
}

_REGISTRY: Mapping[str, UnitSpec] = MappingProxyType(_MUTABLE_REGISTRY)
del _MUTABLE_REGISTRY
"""Read-only view. ``UnitSpec`` is frozen, but a plain dict is not: without
this, a same-process caller could insert ``_REGISTRY["TND"]`` and make the bare
ISO code this module deliberately rejects suddenly resolve. Same "conventional,
not a boundary" class as everything else in Python, but the proxy costs nothing
and removes the accidental case."""

CURRENCY_PREFIX = "currency:"
"""Currency units are written ``currency:XXX`` — never a bare ``TND``.

A bare three-letter code is indistinguishable from a future registry unit and
invites exactly the implicit guessing rule 1 forbids. The prefix makes a
currency operand self-describing and makes ``currency:TND`` vs ``currency:EUR``
a visible difference rather than a string that happens not to match.
"""

_ISO_CURRENCY_RE = re.compile(r"\A[A-Z]{3}\Z")
"""ISO-4217 *shape*, not membership.

Deliberately a shape check. Pinning a list of live currency codes would make
this module wrong every time a currency is added or withdrawn, and the contract
has no business being the authority on that. What it must reject is empty,
lowercase, numeric, over- or under-length, and whitespace codes.

Anchored ``\\A``/``\\Z``, not ``^``/``$``. In Python ``$`` also matches
immediately before a terminal newline, so ``^[A-Z]{3}$`` accepted
``"TND\\n"`` — which would have produced two distinct ``Unit.code`` values for
one currency, fingerprinted the same reconciliation differently, refused a
legitimate TND-vs-TND comparison as cross-currency, and put a newline into
``display`` and therefore into logs.
"""


class Unit(ContractModel):
    """A unit reference: one validated code, resolved against the registry.

    Two forms are accepted and nothing else:

    * a registry code (``"kg"``, ``"ha"``, ``"pct_point"``, …);
    * ``currency:XXX`` where ``XXX`` is an ISO-shaped currency code.

    Anything else fails validation. There is no free-text unit and no default.
    """

    code: str = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _code_must_resolve(self) -> Unit:
        if self.code.startswith(CURRENCY_PREFIX):
            iso = self.code[len(CURRENCY_PREFIX):]
            if not _ISO_CURRENCY_RE.match(iso):
                raise ValueError(
                    f"malformed currency code {iso!r}: expected three uppercase "
                    f"letters, e.g. {CURRENCY_PREFIX}TND"
                )
            return self
        if self.code not in _REGISTRY:
            raise ValueError(
                f"unknown unit {self.code!r}. Known units: "
                f"{', '.join(sorted(_REGISTRY))}, or {CURRENCY_PREFIX}<ISO>. "
                "Units are never inferred."
            )
        return self

    # -- resolved properties ------------------------------------------------

    @property
    def is_currency(self) -> bool:
        return self.code.startswith(CURRENCY_PREFIX)

    @property
    def currency_code(self) -> str | None:
        """The ISO code for a currency unit, else ``None``."""
        return self.code[len(CURRENCY_PREFIX):] if self.is_currency else None

    @property
    def dimension(self) -> Dimension:
        if self.is_currency:
            return "currency"
        return _REGISTRY[self.code].dimension

    @property
    def conversion_policy(self) -> ConversionPolicy:
        if self.is_currency:
            # No rate authority exists in Version 1, so no currency conversion
            # is definable — not even between a currency and itself's "base",
            # because currency has no shared base.
            return "none"
        return _REGISTRY[self.code].conversion_policy

    @property
    def factor_to_base(self) -> Decimal | None:
        """Exact factor onto the dimension's base unit; ``None`` for currency."""
        if self.is_currency:
            return None
        return _REGISTRY[self.code].factor_to_base

    @property
    def display(self) -> str:
        """Human label. Never used for identity — :attr:`code` is identity."""
        if self.is_currency:
            return self.currency_code or ""
        return _REGISTRY[self.code].display


# ---------------------------------------------------------------------------
# Type predicates (no arithmetic, no conversion)
# ---------------------------------------------------------------------------


def known_unit_codes() -> tuple[str, ...]:
    """Every non-currency registry code, sorted. Currency codes are open-ended."""
    return tuple(sorted(_REGISTRY))


def unit_spec(code: str) -> UnitSpec | None:
    """Registry lookup for a non-currency code. ``None`` when unknown."""
    return _REGISTRY.get(code)


def same_dimension(a: Unit, b: Unit) -> bool:
    """Whether two units belong to the same dimension family."""
    return a.dimension == b.dimension


def additively_compatible(a: Unit, b: Unit) -> tuple[bool, str | None]:
    """Whether ``a`` and ``b`` may be added, subtracted, or compared.

    Returns ``(compatible, reason_if_not)``. This is a *type predicate*: it
    inspects dimensions and currency codes and computes nothing.

    Two units are compatible when they share a dimension **and**, for currency,
    name the same ISO code. Cross-currency is rejected rather than converted —
    Version 1 ships no exchange-rate authority, and silently treating TND and
    EUR as interchangeable is precisely the class of error this contract exists
    to prevent.
    """
    if a.dimension != b.dimension:
        return False, f"dimension mismatch: {a.dimension} vs {b.dimension}"
    if a.dimension == "currency" and a.currency_code != b.currency_code:
        return False, (
            f"cross-currency comparison {a.currency_code} vs {b.currency_code} "
            "requires an exchange-rate authority, which does not exist"
        )
    if a.code != b.code and (
        a.conversion_policy == "explicit_required"
        or b.conversion_policy == "explicit_required"
    ):
        # Compatible, but not silently. Comparing a monthly figure with an
        # annual one is the failure this module names in its own docstring, and
        # returning a bare ``(True, None)`` here would hand a Phase 2 engine an
        # unqualified yes for exactly that. The sibling predicate warns; so does
        # this one.
        return True, (
            f"{a.code} and {b.code} share a dimension but converting between "
            "them requires an explicit period policy; the engine must be told "
            "the basis rather than assuming one"
        )
    return True, None


def convertible(a: Unit, b: Unit) -> tuple[bool, str | None]:
    """Whether a future engine may convert ``a`` to ``b``, and on what terms.

    Returns ``(convertible, note)``. ``note`` is populated for the
    ``explicit_required`` case so a caller cannot treat "convertible" as
    "convertible without saying how".
    """
    if a.dimension != b.dimension:
        return False, f"dimension mismatch: {a.dimension} vs {b.dimension}"
    if a.dimension == "currency":
        return False, "no exchange-rate authority exists in Version 1"
    if a.conversion_policy == "explicit_required" or b.conversion_policy == "explicit_required":
        return True, (
            "conversion requires an explicit period policy; the engine must be "
            "told the basis rather than assuming one"
        )
    return True, None


# The one composition Version 1 needs: yield x area -> mass. Declared as data so
# the Phase 2 engine reads a rule rather than hard-coding a special case, and so
# a test can assert the rule exists before any engine does.
_MULTIPLICATIVE_COMPOSITIONS: dict[tuple[Dimension, Dimension], Dimension] = {
    ("mass_per_area", "area"): "mass",
    ("area", "mass_per_area"): "mass",
}

MULTIPLICATIVE_COMPOSITIONS: Mapping[tuple[Dimension, Dimension], Dimension] = (
    MappingProxyType(_MULTIPLICATIVE_COMPOSITIONS)
)
"""Read-only, for the same reason as the unit registry.

``composed_dimension`` reads this table live, so a plain dict would let a
same-process caller declare ``("mass", "time") -> "currency"`` and change what
the engine believes a multiplication produces. Hardening one lookup table and
leaving its sibling open is worse than hardening neither: it reads as a
guarantee that does not hold."""
del _MULTIPLICATIVE_COMPOSITIONS


def composed_dimension(a: Unit, b: Unit) -> Dimension | None:
    """Dimension produced by multiplying ``a`` and ``b``, or ``None``.

    ``None`` means Version 1 declares no result dimension for that pairing; the
    engine must report a unit mismatch rather than inventing one.
    """
    if a.dimension == "dimensionless":
        return b.dimension
    if b.dimension == "dimensionless":
        return a.dimension
    return MULTIPLICATIVE_COMPOSITIONS.get((a.dimension, b.dimension))
