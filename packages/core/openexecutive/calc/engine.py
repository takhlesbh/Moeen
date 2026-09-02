"""The deterministic calculation engine.

This is the authority Phase 1 was written against: the thing entitled to say a
figure was computed. It executes a closed set of twelve operations over
``Decimal`` values with declared arity and dimensional signatures, converts
units by exact integer factors, derives its own fingerprint, and returns a typed
result — success or failure — for every request it is given.

Why an engine at all. A controlled evaluation measured a specialist model
identifying the correct operands and the correct formula for a production
calculation — ``52 kg/m2``, ``11 ha = 110,000 m2``, "yield per m2 x area" — and
then reporting an answer off by a factor of ten; a second calculation came back
carrying the literal placeholder ``XX``. A prompt-only remedy was implemented,
measured, and rejected. Arithmetic execution therefore moves here, where it is
deterministic and testable with no model in the loop.

What this module does **not** do, deliberately:

* it is not connected to any specialist, provider, prompt, or model path, and
  nothing in production imports it;
* it evaluates no expressions — there is no parser, and the operation is
  selected from a closed mapping, never looked up from model text;
* it touches no filesystem, no network, no clock for identity, and no global
  mutable state.

**Authority, stated precisely.** Against a *model*, the boundary is structural:
a model speaks only through :class:`~openexecutive.calc.contract.CalculationRequest`,
which has no field for a status, an authority stamp, a fingerprint, or a
timestamp, and ``extra="forbid"`` rejects the attempt rather than dropping it.
Against a *same-process caller*, it is a convention: Python has no private
constructor, and ``model_construct`` bypasses pydantic for any model. This
module does not claim otherwise. What it does guarantee is that every result it
issues was built here, from arithmetic performed here, with a fingerprint
derived here — and that it forwards no caller-supplied authority field of any
kind.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from decimal import (
    ROUND_HALF_EVEN,
    Clamped,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    FloatOperation,
    Inexact,
    InvalidOperation,
    Overflow,
    Rounded,
    Subnormal,
    Underflow,
    localcontext,
)
from types import MappingProxyType
from typing import Literal, NamedTuple

from pydantic import ValidationError

from openexecutive.calc.authority import current_authority
from openexecutive.calc.contract import (
    MAX_EXPRESSION_LEN,
    MAX_OPERANDS_PER_REQUEST,
    MAX_REQUESTS_PER_BATCH,
    ArithmeticStatus,
    CalculationBatch,
    CalculationError,
    CalculationRequest,
    CalculationResult,
    ConflictClass,
    InputEvidenceSummary,
    NormalizedOperand,
    Operand,
    OperationId,
    RoundingMode,
)
from openexecutive.calc.fingerprint import fingerprint_for
from openexecutive.calc.numeric import (
    MAX_ADJUSTED_EXPONENT,
    MAX_NUMERIC_STRING_LEN,
    MAX_PRECISION_REQUEST,
    MAX_SCALE,
    NumericPolicyError,
    canonical_numeric_string,
    parse_numeric,
)
from openexecutive.calc.units import (
    Dimension,
    Unit,
    additively_compatible,
    base_unit_for,
    composed_dimension,
    convertible,
    is_base_unit,
)

ENGINE_PRECISION = MAX_PRECISION_REQUEST
"""Working precision for every execution, and the enforcement of Phase 1's
declared ceiling — which until now had no consumer at all.

Intermediates run at this precision and the caller-visible rounding happens once,
at the final ``quantize``.

Stated accurately rather than as a slogan: this is not the *only* rounding.
Each operand is normalised to its target unit before the operation, so a
division-based conversion rounds to 50 digits there and again at the boundary.
Measured divergence from a single correctly-rounded computation appears at digit
50 — far below any reported scale, which is capped at 28 — so it is invisible in
practice, but "rounding happens once" was an overstatement and is not repeated.
"""

def engine_context(precision: int = ENGINE_PRECISION) -> Context:
    """A fully specified ``Decimal`` context, built from scratch every time.

    **Never** derived from :func:`decimal.getcontext`. ``localcontext()`` with no
    argument *copies* the caller's thread-local context, so an earlier version
    inherited whatever the process had set and only overrode ``prec`` and three
    traps. Two consequences, both reproduced:

    * ``getcontext().rounding = ROUND_UP`` — a legal, documented global any
      dependency may set — changed the 50th digit of a normalized operand on
      ``divide``, ``percentage_of`` and ``month->year``. That digit *is* a
      fingerprint payload field, so byte-identical input produced two different
      fingerprints and a dedup index would have failed to recognise the same
      calculation.
    * ``getcontext().traps[Inexact] = True`` turned every division-based
      operation into ``CALCULATION_UNAVAILABLE`` process-wide. This repository's
      own test suite sets exactly that trap, so the hazard was already in-tree.

    Every field is therefore pinned explicitly — precision, rounding, exponent
    range, clamp, and the full trap set — rather than left to inheritance.
    ``Inexact``, ``Rounded`` and ``Subnormal`` are deliberately **not** trapped:
    they are the normal condition of division, and trapping them would make
    ``1/3`` a failure. ``Emin``/``Emax`` are set well outside
    ``MAX_ADJUSTED_EXPONENT`` so the engine's own bound is what rejects an
    out-of-range figure, with a typed status, rather than a Decimal trap firing
    first and reporting something less specific.
    """
    # ``rounding=ROUND_HALF_EVEN`` is stated even though it is currently the
    # ``decimal`` module default, so deleting it is a semantic no-op that no
    # test can detect. It stays because this context's whole purpose is that no
    # field is left to inheritance — including inheritance from a library
    # default that a future Python could change.
    return Context(
        prec=precision,
        rounding=ROUND_HALF_EVEN,
        Emin=-999999,
        Emax=999999,
        capitals=1,
        clamp=0,
        flags=[],
        # All nine signals, explicitly. The constructor requires the complete
        # set — an eight-entry dict raises ``KeyError: 'invalid signal dict'`` —
        # and that strictness is welcome here: it makes an omission impossible
        # to write, which is exactly how the ambient inheritance arose.
        traps={
            Clamped: False,
            DivisionByZero: True,
            FloatOperation: True,
            Inexact: False,
            InvalidOperation: True,
            Overflow: True,
            Rounded: False,
            Subnormal: False,
            Underflow: False,
        },
    )


PER_REQUEST_BUDGET_S = 0.25
PER_BATCH_BUDGET_S = 2.0
"""Wall-clock budgets, measured with a monotonic clock.

Stated honestly: these are **measured and reported, not preemptive**. A
synchronous Python call cannot be interrupted from inside itself without threads
or signals, and the Council's directive rules both out — correctly, since a
signal-based timeout in a library is a footgun and a thread pool is
infrastructure this package has no business owning.

The real protection is that every authorised operation is *bounded by
construction*: at most 64 operands, numeric literals of at most 64 characters
with ``|adjusted exponent| <= 30``, no nesting, no loops over caller-controlled
counts, and a fixed 50-digit precision. There is no input in the authorised set
whose execution time is superlinear in its length. The budget is therefore a
tripwire that catches a wrong assumption, not the thing standing between the
process and a hostile input — the pre-validation is.
"""


class EngineLimits(NamedTuple):
    """The limits this engine enforces, exposed so tests can assert them."""

    max_operands: int = MAX_OPERANDS_PER_REQUEST
    max_requests: int = MAX_REQUESTS_PER_BATCH
    max_precision: int = MAX_PRECISION_REQUEST
    max_adjusted_exponent: int = MAX_ADJUSTED_EXPONENT
    max_expression_len: int = MAX_EXPRESSION_LEN
    nested_operations: int = 0
    per_request_budget_s: float = PER_REQUEST_BUDGET_S
    per_batch_budget_s: float = PER_BATCH_BUDGET_S


LIMITS = EngineLimits()

_MUTABLE_ROUNDING: dict[RoundingMode, str] = {
    "ROUND_HALF_EVEN": "ROUND_HALF_EVEN",
    "ROUND_HALF_UP": "ROUND_HALF_UP",
    "ROUND_DOWN": "ROUND_DOWN",
}

_ROUNDING: Mapping[RoundingMode, str] = MappingProxyType(_MUTABLE_ROUNDING)
del _MUTABLE_ROUNDING

SUPPORTED_TIME_CONVERSION_POLICIES: frozenset[str] = frozenset(
    {"calendar_12_months_per_year"}
)
"""The closed set of accepted month<->year bases.

``Literal`` is erased at runtime, so the annotation alone stopped nothing: an
unvalidated ``"actual/360"`` was accepted, the registry's ``x1/12`` factor was
applied regardless, and the record then asserted a basis the engine had never
used — the "annualised without saying how" failure inverted, saying *how* and
saying it falsely. Every distinct string also produced a distinct fingerprint
for byte-identical arithmetic, fragmenting identity without bound.

The sibling ``evidence`` parameter at the same boundary already gets an explicit
runtime check for exactly this reason; these two were in the identical position
and had none.
"""

SUPPORTED_WEIGHT_POLICIES: frozenset[str] = frozenset({"normalized_by_engine"})
"""The closed set of accepted weighting policies. Validated, not just annotated."""

TimeConversionPolicy = Literal["calendar_12_months_per_year"]
"""The only month<->year basis this version accepts, and it must be asked for.

A month is not one twelfth of a financial year under every day-count
convention, so annualising without saying how is exactly the silent error the
unit registry marks ``explicit_required``. Naming the policy makes the
assumption part of the record and part of the fingerprint.
"""

WeightPolicy = Literal["normalized_by_engine"]
"""Weights are divided by their own sum; they need not total 1 or 100.

Declared as a policy rather than assumed, and included in the fingerprint, so a
weighted average computed under a different future policy cannot collide with
one computed under this one.
"""


FALLBACK_COMPUTED_AT = "1970-01-01T00:00:00Z"
"""The timestamp a failure record carries when the caller's is unusable.

A failure result still has to be *constructible*: an earlier version passed the
caller's rejected ``computed_at`` straight back into ``_failure``, so the
recovery handler re-raised the very ``ValidationError`` it existed to convert
and the exception escaped ``execute`` untyped. The Unix epoch is chosen because
it is unmistakably not a real computation time — a reader seeing 1970 knows the
clock value was refused, and the accompanying error says so.
"""

# ``[0-9]``, never ``\d``. ``\d`` matches every Unicode decimal digit, so an
# Arabic-Indic, Devanagari or fullwidth timestamp was ACCEPTED here and then
# refused by the contract's ASCII-only validator — and the recovery path
# re-canonicalised the same value successfully, so the epoch fallback never
# fired, ``issue_calculation_result`` raised again, and the ValidationError
# escaped ``execute`` untyped and aborted whole batches. This module states the
# rule twice elsewhere (``numeric.py``: "``[0-9]`` not ``\d``: ``\d`` is
# Unicode-aware"); these two patterns were the one place that did not follow it.
# They must stay character-for-character equivalent to
# ``contract._ISO_UTC_RE``; a test asserts that agreement rather than trusting
# it.
_ISO_UTC_CANONICAL = re.compile(
    r"\A([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"(\.[0-9]{1,6})?Z\Z"
)
_ISO_UTC_OFFSET = re.compile(
    r"\A([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"(\.[0-9]{1,6})?(?:Z|([+-])([0-9]{2}):?([0-9]{2}))\Z"
)
_MAX_TIMESTAMP_LEN = 64


def canonical_computed_at(value: object) -> tuple[str | None, str | None]:
    """Normalise a caller timestamp to the contract's ``…Z`` form.

    Returns ``(canonical, None)`` on success or ``(None, reason)`` on failure —
    it never raises, because the caller's clock format must not be able to
    abort a calculation.

    ``datetime.now(timezone.utc).isoformat()`` produces ``+00:00``, which is the
    single most likely thing an integrator will pass and which the contract's
    ``…Z`` pattern rejects. Accepting it (and any other zero offset) is not
    leniency: it is the same instant, written the way the standard library
    writes it. A **non-zero** offset is refused rather than shifted, because
    converting it would mean this module doing calendar arithmetic to produce a
    timestamp the caller did not write.
    """
    if not isinstance(value, str):
        return None, f"computed_at must be a string, not {type(value).__name__}"
    if not value:
        return None, "computed_at is empty"
    if len(value) > _MAX_TIMESTAMP_LEN:
        return None, (
            f"computed_at is {len(value)} characters, over the "
            f"{_MAX_TIMESTAMP_LEN}-character limit"
        )
    if _ISO_UTC_CANONICAL.fullmatch(value):
        return value, None
    match = _ISO_UTC_OFFSET.fullmatch(value)
    if match is None:
        return None, (
            "computed_at is not an ISO-8601 UTC instant "
            "(YYYY-MM-DDTHH:MM:SS[.ffffff]Z or a +00:00 offset)"
        )
    date, clock, fraction, sign, hours, minutes = match.groups()
    if sign is not None and (hours, minutes) != ("00", "00"):
        return None, (
            f"computed_at carries a non-zero UTC offset ({sign}{hours}:{minutes}); "
            "supply the instant in UTC rather than having the engine shift it"
        )
    return f"{date}T{clock}{fraction or ''}Z", None


class EngineError(Exception):
    """Internal control flow, carrying the status a failure should report.

    Never escapes a public entry point: :func:`execute` converts it into a typed
    :class:`CalculationResult`. It exists so the many validation sites can say
    *which* typed status they mean without threading a result object through
    every one of them.
    """

    def __init__(self, status: ArithmeticStatus, code: str, detail: str,
                 operand_id: str | None = None) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.operand_id = operand_id


# ---------------------------------------------------------------------------
# Operation signatures
# ---------------------------------------------------------------------------


class OperationSignature(NamedTuple):
    """The closed declaration of what one operation accepts and produces.

    Phase 1 deliberately shipped no per-operation dimensional table: encoding an
    operation's signature means encoding its *semantics*, and a draft that did so
    inside the contract got ``divide`` wrong and rejected this package's own
    motivating calculation. Semantics belong with the code that executes them,
    which is here — and here each one arrives with its arity, its accepted
    dimensions, and its own tests.
    """

    operation: OperationId
    min_arity: int
    max_arity: int
    requires_target_unit: bool
    requires_stated_value: bool
    # Deliberately no ``order_matters``. A draft declared and exported one,
    # set False for add and sum_components, and never read it — so a Phase 3
    # consumer reading ``signature_for("add").order_matters is False`` would
    # reasonably conclude the engine canonicalises commutative operand order.
    # It does not, by design: labels and provenance ride with position, so
    # reordering would re-pair a value with another operand's label. Dead
    # metadata that reads as a guarantee is worse than no metadata.
    # None means "derived from the operands"; a Dimension pins it.
    fixed_output_dimension: Dimension | None = None
    # Inputs must all share this dimension when set.
    fixed_input_dimension: Dimension | None = None
    # Inputs must be pairwise additively compatible (same dimension, same
    # currency code) when True.
    requires_compatible_inputs: bool = False


_MUTABLE_SIGNATURES: dict[str, OperationSignature] = {
    "add": OperationSignature(
        "add", 2, 2, requires_target_unit=True,
        requires_stated_value=False, requires_compatible_inputs=True,
    ),
    "subtract": OperationSignature(
        "subtract", 2, 2, requires_target_unit=True,
        requires_stated_value=False, requires_compatible_inputs=True,
    ),
    "multiply": OperationSignature(
        "multiply", 2, 2, requires_target_unit=True,
        requires_stated_value=False,
    ),
    "divide": OperationSignature(
        "divide", 2, 2, requires_target_unit=True,
        requires_stated_value=False,
    ),
    "sum_components": OperationSignature(
        "sum_components", 1, MAX_OPERANDS_PER_REQUEST,
        requires_target_unit=True, requires_stated_value=False,
        requires_compatible_inputs=True,
    ),
    "percentage_of": OperationSignature(
        "percentage_of", 2, 2, requires_target_unit=True,
        requires_stated_value=False, fixed_output_dimension="percentage",
        requires_compatible_inputs=True,
    ),
    "percentage_point_difference": OperationSignature(
        "percentage_point_difference", 2, 2,
        requires_target_unit=True, requires_stated_value=False,
        fixed_output_dimension="percentage_point",
        fixed_input_dimension="percentage",
    ),
    "ratio": OperationSignature(
        "ratio", 2, 2, requires_target_unit=True,
        requires_stated_value=False, fixed_output_dimension="dimensionless",
        requires_compatible_inputs=True,
    ),
    "weighted_average": OperationSignature(
        "weighted_average", 2, MAX_OPERANDS_PER_REQUEST,
        requires_target_unit=True, requires_stated_value=False,
    ),
    "variance": OperationSignature(
        "variance", 1, 1, requires_target_unit=True,
        requires_stated_value=True,
    ),
    "convert_unit": OperationSignature(
        "convert_unit", 1, 1, requires_target_unit=True,
        requires_stated_value=False,
    ),
    "interval_implied_total": OperationSignature(
        "interval_implied_total", 4, 4,
        requires_target_unit=True, requires_stated_value=False,
    ),
}


_SIGNATURES: Mapping[str, OperationSignature] = MappingProxyType(_MUTABLE_SIGNATURES)
del _MUTABLE_SIGNATURES
"""Read-only, matching the unit registry's policy.

``units.py`` argues in its own docstring that "hardening one lookup table and
leaving its sibling open is worse than hardening neither: it reads as a
guarantee that does not hold". This module calls ``signature_for`` a "closed
lookup" and says the operation "is selected from a closed mapping", so leaving
``_DISPATCH["add"] = evil`` a one-liner was the same inconsistency."""


def signature_for(operation: str) -> OperationSignature | None:
    """Closed lookup. An unknown name resolves to ``None``, never to a callable."""
    return _SIGNATURES.get(operation)


# ---------------------------------------------------------------------------
# Numeric and unit helpers
# ---------------------------------------------------------------------------


def _to_decimal(value: str, field: str, operand_id: str | None = None) -> Decimal:
    """Parse through the Phase 1 boundary, converting its error to a typed one."""
    try:
        return parse_numeric(value)
    except NumericPolicyError as exc:
        raise EngineError("INVALID_INPUT", "bad_numeric", f"{field}: {exc}",
                          operand_id) from exc


def _bounded(value: Decimal, field: str, scale: int) -> str | None:
    """Canonical text for a *derived* figure, or ``None`` when it cannot fit.

    Returns ``None`` rather than raising. A derived comparison — a percentage
    difference, a ratio — is commentary on the primary result, and losing it
    must never discard the calculation itself. An earlier version let its
    last-resort ``quantize`` escape, so a ``variance`` of ``1e30`` against
    ``1e-30`` produced *no record at all* for the largest possible
    order-of-magnitude discrepancy, reported as ``INVALID_INPUT`` — blaming
    inputs that were both perfectly valid.

    Rendering, honestly described: the value is rendered at **full precision**
    whenever that fits the 64-character field, so what a reader sees is the
    computed figure, not a figure quantized to ``scale``. Only when it does not
    fit are significant digits trimmed, and only as a last resort is it
    quantized to ``scale``. Trimming precision rather than quantizing is
    deliberate: a ``-3.3e-13`` percentage difference quantized at scale 2 reads
    ``"0.00"``, which erases the very finding the field exists to report.
    """
    try:
        rendered = _render(value, field, scale)
    except EngineError:
        rendered = ""
    if rendered and len(rendered) <= MAX_NUMERIC_STRING_LEN:
        return rendered
    for digits in (28, 20, 12, 6):
        with localcontext(engine_context(digits)):
            # Same explicit context as execution: the trimming step must not
            # round differently from the arithmetic that produced the value.
            #
            # Mutation-tested honestly: reverting this to a bare
            # ``localcontext()`` currently changes no observable output, because
            # every value reaching here was produced at 50 digits inside the
            # engine context, so ``+value`` at 28 or fewer digits is exact
            # regardless of ambient rounding. It stays explicit because that is
            # a property of today's call sites, not of this function — a future
            # caller passing an inexact value would silently inherit ambient
            # rounding into ``exact_result``, ``ratio`` and the interval text,
            # none of which enter the fingerprint. That failure mode is two
            # records with different reported answers under one identity, which
            # is worse for a dedup index than a fingerprint split.
            trimmed = +value
        try:
            candidate = _render(trimmed, field, scale)
        except EngineError:
            continue
        if len(candidate) <= MAX_NUMERIC_STRING_LEN:
            return candidate
    try:
        quantum = Decimal(1).scaleb(-scale)
        return _render(value.quantize(quantum, rounding=ROUND_HALF_EVEN), field, scale)
    except (EngineError, DecimalException):
        # Genuinely unrepresentable in this field. The caller reports it as an
        # unavailable derived figure and keeps the primary result.
        return None


def _render(value: Decimal, field: str, scale_hint: int = 0) -> str:
    """Canonical string, with the positional-expansion hazard closed first.

    ``canonical_numeric_string`` is ``format(v, "f")``, which expands
    positionally: the output length is the exponent. Phase 1 bounds every value
    that arrives through ``parse_numeric``, but an engine *result* is a fresh
    Decimal that never passed through it — dividing by a very small number can
    produce an exponent no input carried. Checking here rather than trusting the
    inputs is the difference between a bound and an assumption.
    """
    # Zero needs its own rule, and a *narrow* one. The exponent bound exists to
    # stop positional expansion, and ``Decimal("0E-50")`` compares equal to zero
    # so a "check unless it is zero" guard skips it — that was the original F3
    # bug, where ``v - v`` for a 31-decimal ``v`` returned no result at all.
    #
    # The first repair over-corrected: it rewrote every zero to ``"0"``, which
    # silently changed a ``"0.00"`` operand's stored value. ``NormalizedOperand``
    # then refused it ("a conversion that does not convert units cannot alter
    # the number") and eleven of the twelve operations failed on any scaled
    # zero — a cost stack with one ``0.00`` line among them.
    #
    # So: bound the rendered LENGTH, and otherwise return the value untouched.
    # The length is computed from the exponent and never by rendering first,
    # because ``format(Decimal("0E-2000000000"), "f")`` genuinely produces two
    # billion characters — measured, while diagnosing this.
    # ``value == 0`` alone is sufficient: ``Decimal("NaN") == 0`` and
    # ``Decimal("Infinity") == 0`` are both False, so a non-finite value cannot
    # enter this branch and the extra ``is_finite()`` was redundant. The
    # exponent of any Decimal satisfying ``== 0`` is therefore an int.
    if value == 0:
        exponent = value.as_tuple().exponent
        assert isinstance(exponent, int), "a zero always has an integer exponent"
        width = 2 + (-exponent) if exponent < 0 else 1
        # No ``+1`` for a sign. A signed zero only reaches this width check when
        # ``|exponent| <= MAX_ADJUSTED_EXPONENT`` (32 characters at the very
        # most); anything longer takes the clamp branch below, where the sign is
        # gone. Adding one here was an unreachable adjustment that read as a
        # live correction — mutation-tested and confirmed dead.
        if width > MAX_NUMERIC_STRING_LEN:
            raise EngineError(
                "RESOURCE_LIMIT_EXCEEDED", "result_too_long",
                f"{field}: a zero at exponent {exponent} would render as "
                f"{width} characters, over the {MAX_NUMERIC_STRING_LEN}-character "
                "field limit",
            )
        if abs(exponent) <= MAX_ADJUSTED_EXPONENT:
            # Within the contract's own numeric bound, so hand it back exactly
            # as it arrived. This is the operand case: ``parse_numeric`` already
            # bounded every operand, so a "0.00" keeps its two decimals and the
            # NormalizedOperand validator sees an unchanged value.
            return canonical_numeric_string(value)
        # Beyond the bound, so this is a *computed* zero — ``v - v`` for a
        # 31-decimal ``v`` is ``0E-31``. No operand can reach here, because the
        # numeric boundary refuses one (deliberately: dropping the zero
        # exemption there is what closed a 4 GB positional-expansion DoS). A
        # zero has no magnitude to lose, so it is clamped to the scale the
        # caller asked for rather than refused.
        # ``scale_hint`` needs no clamping: ``CalculationRequest.scale`` is
        # already ``le=MAX_SCALE``, so a ``min()`` here was an unreachable
        # branch that read as a live guard. The assertion documents the
        # dependency instead of hiding it behind dead code.
        assert scale_hint <= MAX_SCALE, "scale is bounded by the request contract"
        return canonical_numeric_string(Decimal(0).scaleb(-scale_hint))
    if value.is_finite() and abs(value.adjusted()) > MAX_ADJUSTED_EXPONENT:
        raise EngineError(
            "RESOURCE_LIMIT_EXCEEDED", "result_exponent",
            f"{field}: adjusted exponent {value.adjusted()} exceeds "
            f"{MAX_ADJUSTED_EXPONENT}",
        )
    if not value.is_finite():
        raise EngineError("INVALID_INPUT", "non_finite",
                          f"{field}: result is not finite")
    rendered = canonical_numeric_string(value)
    if len(rendered) > MAX_NUMERIC_STRING_LEN:
        # Never let an over-long string reach a 64-character field: the caller
        # gets a typed status naming the field, not a framework traceback
        # stored in an audit record.
        raise EngineError(
            "RESOURCE_LIMIT_EXCEEDED", "result_too_long",
            f"{field}: canonical form is {len(rendered)} characters, over the "
            f"{MAX_NUMERIC_STRING_LEN}-character field limit",
        )
    return rendered


def _same_unit(a: Unit, b: Unit) -> bool:
    return a.code == b.code


def _require_compatible(a: Unit, b: Unit, what: str) -> None:
    ok, reason = additively_compatible(a, b)
    if not ok:
        raise EngineError("UNIT_MISMATCH", "incompatible_units",
                          f"{what}: {a.code} and {b.code} — {reason}")


def _convert(
    value: Decimal,
    source: Unit,
    target: Unit,
    *,
    time_policy: TimeConversionPolicy | None,
) -> tuple[Decimal, str | None]:
    """Exact conversion between two units of one dimension.

    Returns the converted value and a human description of what was applied
    (``None`` for an identity conversion). Every factor is an exact integer from
    the registry, so ``ha -> m2`` and ``t -> kg`` are lossless; nothing here goes
    through float.
    """
    if _same_unit(source, target):
        return value, None
    ok, note = convertible(source, target)
    if not ok:
        raise EngineError("UNIT_MISMATCH", "not_convertible",
                          f"cannot convert {source.code} to {target.code}: {note}")
    if source.dimension == "currency" or target.dimension == "currency":
        # Unreachable via ``convertible`` (which refuses currency outright); kept
        # so a future registry change cannot silently open a rate-free path.
        raise EngineError("UNIT_MISMATCH", "currency_conversion",
                          "currency conversion requires an exchange-rate authority")
    if note is not None and time_policy is None:
        raise EngineError(
            "INVALID_INPUT", "missing_time_policy",
            f"converting {source.code} to {target.code} requires an explicit "
            "time_conversion_policy; a month is not one twelfth of a financial "
            "year under every day-count convention",
        )
    src_factor = source.factor_to_base
    tgt_factor = target.factor_to_base
    if src_factor is None or tgt_factor is None or tgt_factor == 0:
        raise EngineError("UNIT_MISMATCH", "no_factor",
                          f"no exact factor between {source.code} and {target.code}")
    converted = value * src_factor / tgt_factor
    return converted, f"{source.code}->{target.code} (x{src_factor}/{tgt_factor})"


def _normalize(
    operand: Operand,
    target: Unit,
    *,
    time_policy: TimeConversionPolicy | None,
) -> tuple[NormalizedOperand, Decimal]:
    """Convert one operand into ``target`` and record how."""
    raw = _to_decimal(operand.value, "operand value", operand.operand_id)
    converted, applied = _convert(raw, operand.unit, target, time_policy=time_policy)
    normalized = NormalizedOperand(
        operand_id=operand.operand_id,
        label=operand.label,
        original_value=canonical_numeric_string(raw),
        original_unit=operand.unit,
        normalized_value=_render(converted, f"operand {operand.operand_id}"),
        normalized_unit=target,
        conversion_applied=applied,
        basis=operand.basis,
        role=operand.role,
    )
    return normalized, converted


# ---------------------------------------------------------------------------
# Operation implementations
# ---------------------------------------------------------------------------


class Computed(NamedTuple):
    """One operation's outcome, before rounding and result construction."""

    exact: Decimal
    unit: Unit
    expression: str
    normalized: tuple[NormalizedOperand, ...]
    stated_value: str | None = None
    absolute_difference: str | None = None
    percentage_difference: str | None = None
    ratio: str | None = None
    conflict: ConflictClass = "NONE"
    warnings: tuple[str, ...] = ()
    used_time_policy: bool = False
    """Whether a month<->year conversion actually consumed the caller's policy.

    Only then does the policy belong in the fingerprint: it changed the
    arithmetic. Passing it to an ``add`` of two masses changes nothing, and
    letting it alter the identity would fragment the very calculations a
    fingerprint exists to recognise as the same."""


def _inputs(request: CalculationRequest) -> tuple[Operand, ...]:
    return tuple(o for o in request.operands if o.role == "input")


def _target(request: CalculationRequest) -> Unit:
    if request.target_unit is None:
        raise EngineError("INVALID_INPUT", "missing_target_unit",
                          f"{request.operation} requires an explicit target_unit")
    return request.target_unit


def _expr(parts: list[str], op: str) -> str:
    text = f" {op} ".join(parts)
    return text if len(text) <= MAX_EXPRESSION_LEN else text[:MAX_EXPRESSION_LEN]


def _additive(
    request: CalculationRequest, policy: TimeConversionPolicy | None, symbol: str
) -> Computed:
    target = _target(request)
    operands = _inputs(request)
    for operand in operands:
        _require_compatible(operand.unit, target, request.operation)
    normalized: list[NormalizedOperand] = []
    values: list[Decimal] = []
    for operand in operands:
        norm, value = _normalize(operand, target, time_policy=policy)
        normalized.append(norm)
        values.append(value)
    total = sum(values[1:], values[0]) if symbol == "+" else values[0] - values[1]
    parts = [f"{n.normalized_value} {target.display or target.code}" for n in normalized]
    used_policy = any(_uses_time_policy(o.unit, target) for o in operands)
    return Computed(total, target, _expr(parts, symbol), tuple(normalized),
                    used_time_policy=used_policy)


def _op_add(request: CalculationRequest, policy: TimeConversionPolicy | None) -> Computed:
    return _additive(request, policy, "+")


def _op_subtract(request: CalculationRequest, policy: TimeConversionPolicy | None) -> Computed:
    return _additive(request, policy, "-")


def _op_sum_components(
    request: CalculationRequest, policy: TimeConversionPolicy | None
) -> Computed:
    computed = _additive(request, policy, "+")
    # Component identity is the point of this operation: a stack that sums to
    # 65% is only auditable if the reader can see the six lines that made it.
    parts = [
        f"{n.label}={n.normalized_value}" for n in computed.normalized
    ]
    return computed._replace(expression=_expr(parts, "+"))


def _uses_time_policy(source: Unit, target: Unit) -> bool:
    """Whether converting ``source`` to ``target`` consumes a time policy."""
    if source.code == target.code:
        return False
    ok, note = convertible(source, target)
    return ok and note is not None


def _normalize_factor(
    operand: Operand,
    *,
    time_policy: TimeConversionPolicy | None,
) -> tuple[NormalizedOperand, Decimal, Unit]:
    """Convert one multiplicative factor onto its own dimension's base unit.

    Multiplication and division cannot convert a factor into the *target* unit —
    a product's unit is not any factor's unit, so there is nothing to convert
    to. An earlier version therefore skipped normalisation entirely and checked
    the target by dimension alone, which accepted ``52 kg/m2 x 11 ha -> kg`` and
    returned 572: the exact 10^4 hectare error this package was built to
    prevent, carrying a verified status, an authority stamp and a fingerprint.

    Normalising each factor onto its dimension's base (``ha -> m2``,
    ``t -> kg``) makes the coefficients commensurable before they are combined,
    and the caller then requires the target to be that base too.
    """
    raw = _to_decimal(operand.value, "operand value", operand.operand_id)
    base = base_unit_for(operand.unit)
    if base is None:
        # Currency: no factor relates two codes, so the operand is its own base.
        base = operand.unit
    # The caller's policy must reach here. A draft hard-coded ``None``, which
    # made ``divide 12 month / 1 year`` refuse the exact quotient ``ratio``
    # computed from the same operands, left ``multiply`` with a ``year`` operand
    # unreachable in every target shape, and told the integrator to supply a
    # parameter they had already supplied and the code discarded.
    converted, applied = _convert(raw, operand.unit, base, time_policy=time_policy)
    normalized = NormalizedOperand(
        operand_id=operand.operand_id, label=operand.label,
        original_value=canonical_numeric_string(raw), original_unit=operand.unit,
        normalized_value=_render(converted, f"operand {operand.operand_id}"),
        normalized_unit=base, conversion_applied=applied,
        basis=operand.basis, role=operand.role,
    )
    return normalized, converted, base


def _require_base_target(target: Unit, operation: str) -> None:
    """A multiplicative result must be expressed in its dimension's base unit.

    Checking the target by *dimension* alone was the second half of the 10^4
    bug: ``kg`` and ``t`` share a dimension, so ``-> t`` was accepted for a
    kilogram result. Requiring the base removes the ambiguity without inventing
    a scale conversion the engine has no mandate to perform — a caller wanting
    tonnes issues an explicit ``convert_unit``, which is auditable.
    """
    if not is_base_unit(target):
        base = base_unit_for(target)
        raise EngineError(
            "UNIT_MISMATCH", "non_base_target",
            f"{operation} must produce its dimension's base unit "
            f"({base.code if base else target.code}), not {target.code}; "
            "convert the result explicitly if another scale is wanted",
        )


def _require_same_currency(a: Unit, b: Unit, operation: str) -> None:
    """Two currency units in one calculation must name the same ISO code.

    Dimension equality is not enough. ``composed_dimension`` and the division
    table both collapse a dimensionless operand to "the other operand's
    dimension", so a target check written as ``target.dimension != produced``
    compares only ``"currency" != "currency"`` — and
    ``1,000,000 TND x 1 -> EUR`` was verified at par. The engine refused the
    honest conversion and permitted the dishonest one.
    """
    if a.dimension != "currency" or b.dimension != "currency":
        return
    if a.currency_code != b.currency_code:
        raise EngineError(
            "UNIT_MISMATCH", "cross_currency",
            f"{operation}: {a.currency_code} and {b.currency_code} cannot be "
            "related without an exchange-rate authority, which does not exist",
        )


def _op_multiply(request: CalculationRequest, policy: TimeConversionPolicy | None) -> Computed:
    target = _target(request)
    left, right = _inputs(request)
    composed = composed_dimension(left.unit, right.unit)
    if composed is None:
        raise EngineError(
            "UNIT_MISMATCH", "undeclared_composition",
            f"multiplying {left.unit.code} by {right.unit.code} has no declared "
            "result dimension",
        )
    if target.dimension != composed:
        raise EngineError(
            "UNIT_MISMATCH", "wrong_output_dimension",
            f"multiplying {left.unit.code} by {right.unit.code} produces "
            f"{composed}; target_unit {target.code} is {target.dimension}",
        )
    # A currency factor scaled by a dimensionless one keeps its own code.
    for factor in (left.unit, right.unit):
        _require_same_currency(factor, target, "multiply")
    _require_same_currency(left.unit, right.unit, "multiply")
    _require_base_target(target, "multiply")
    normalized: list[NormalizedOperand] = []
    values: list[Decimal] = []
    used_policy = any(
        _uses_time_policy(o.unit, base) for o in (left, right)
        if (base := base_unit_for(o.unit)) is not None
    )
    for operand in (left, right):
        norm, value, _base = _normalize_factor(operand, time_policy=policy)
        normalized.append(norm)
        values.append(value)
    product = values[0] * values[1]
    parts = [
        f"{n.normalized_value} {n.normalized_unit.display or n.normalized_unit.code}"
        for n in normalized
    ]
    return Computed(product, target, _expr(parts, "x"), tuple(normalized),
                    used_time_policy=used_policy)


def _divide_dimension(numerator: Unit, denominator: Unit) -> Dimension | None:
    """The declared V1 divisions, and only those."""
    if denominator.dimension == "dimensionless":
        return numerator.dimension
    if numerator.dimension == denominator.dimension:
        if numerator.dimension == "currency" and (
            numerator.currency_code != denominator.currency_code
        ):
            return None
        return "dimensionless"
    if numerator.dimension == "mass" and denominator.dimension == "area":
        return "mass_per_area"
    return None


def _op_divide(request: CalculationRequest, policy: TimeConversionPolicy | None) -> Computed:
    target = _target(request)
    numerator, denominator = _inputs(request)
    produced = _divide_dimension(numerator.unit, denominator.unit)
    if produced is None:
        raise EngineError(
            "UNIT_MISMATCH", "undeclared_division",
            f"dividing {numerator.unit.code} by {denominator.unit.code} has no "
            "declared result dimension",
        )
    if target.dimension != produced:
        raise EngineError(
            "UNIT_MISMATCH", "wrong_output_dimension",
            f"dividing {numerator.unit.code} by {denominator.unit.code} produces "
            f"{produced}; target_unit {target.code} is {target.dimension}",
        )
    _require_same_currency(numerator.unit, denominator.unit, "divide")
    _require_same_currency(numerator.unit, target, "divide")
    _require_base_target(target, "divide")
    normalized: list[NormalizedOperand] = []
    values: list[Decimal] = []
    used_policy = any(
        _uses_time_policy(o.unit, base) for o in (numerator, denominator)
        if (base := base_unit_for(o.unit)) is not None
    )
    for operand in (numerator, denominator):
        norm, value, _base = _normalize_factor(operand, time_policy=policy)
        normalized.append(norm)
        values.append(value)
    if values[1] == 0:
        raise EngineError("DIVISION_BY_ZERO", "zero_denominator",
                          f"{normalized[1].label} is zero")
    quotient = values[0] / values[1]
    parts = [
        f"{n.normalized_value} {n.normalized_unit.display or n.normalized_unit.code}"
        for n in normalized
    ]
    return Computed(quotient, target, _expr(parts, "/"), tuple(normalized),
                    used_time_policy=used_policy)


def _op_percentage_of(
    request: CalculationRequest, policy: TimeConversionPolicy | None
) -> Computed:
    target = _target(request)
    if target.dimension != "percentage":
        raise EngineError("UNIT_MISMATCH", "wrong_output_dimension",
                          f"percentage_of must produce pct; got {target.code}")
    part, whole = _inputs(request)
    _require_compatible(part.unit, whole.unit, "percentage_of")
    normalized: list[NormalizedOperand] = []
    values: list[Decimal] = []
    for operand in (part, whole):
        norm, value = _normalize(operand, part.unit, time_policy=policy)
        normalized.append(norm)
        values.append(value)
    if values[1] == 0:
        raise EngineError("DIVISION_BY_ZERO", "zero_denominator",
                          f"{normalized[1].label} is zero")
    result = values[0] / values[1] * Decimal(100)
    parts = [f"{normalized[0].normalized_value} / {normalized[1].normalized_value}"]
    return Computed(result, target, _expr(parts, "") + " x 100", tuple(normalized),
                    used_time_policy=_uses_time_policy(whole.unit, part.unit))


def _op_percentage_point_difference(
    request: CalculationRequest, policy: TimeConversionPolicy | None
) -> Computed:
    target = _target(request)
    if target.dimension != "percentage_point":
        raise EngineError("UNIT_MISMATCH", "wrong_output_dimension",
                          f"percentage_point_difference must produce pct_point; "
                          f"got {target.code}")
    left, right = _inputs(request)
    for operand in (left, right):
        if operand.unit.dimension != "percentage":
            raise EngineError(
                "UNIT_MISMATCH", "wrong_input_dimension",
                f"percentage_point_difference requires percentage inputs; "
                f"{operand.operand_id} is {operand.unit.dimension}",
            )
    normalized: list[NormalizedOperand] = []
    values: list[Decimal] = []
    for operand in (left, right):
        raw = _to_decimal(operand.value, "operand value", operand.operand_id)
        normalized.append(NormalizedOperand(
            operand_id=operand.operand_id, label=operand.label,
            original_value=canonical_numeric_string(raw), original_unit=operand.unit,
            normalized_value=_render(raw, f"operand {operand.operand_id}"),
            normalized_unit=operand.unit, basis=operand.basis, role=operand.role,
        ))
        values.append(raw)
    # Direct subtraction of the percentage *values*, never a relative change.
    difference = values[0] - values[1]
    parts = [f"{n.normalized_value}%" for n in normalized]
    return Computed(difference, target, _expr(parts, "-"), tuple(normalized))


def _op_ratio(request: CalculationRequest, policy: TimeConversionPolicy | None) -> Computed:
    target = _target(request)
    if target.dimension != "dimensionless":
        raise EngineError("UNIT_MISMATCH", "wrong_output_dimension",
                          f"ratio must produce a dimensionless result; got {target.code}")
    numerator, denominator = _inputs(request)
    _require_compatible(numerator.unit, denominator.unit, "ratio")
    normalized: list[NormalizedOperand] = []
    values: list[Decimal] = []
    for operand in (numerator, denominator):
        norm, value = _normalize(operand, numerator.unit, time_policy=policy)
        normalized.append(norm)
        values.append(value)
    if values[1] == 0:
        raise EngineError("DIVISION_BY_ZERO", "zero_denominator",
                          f"{normalized[1].label} is zero")
    quotient = values[0] / values[1]
    parts = [n.normalized_value for n in normalized]
    return Computed(quotient, target, _expr(parts, "/"), tuple(normalized),
                    used_time_policy=_uses_time_policy(
                        denominator.unit, numerator.unit))


def _op_weighted_average(
    request: CalculationRequest, policy: TimeConversionPolicy | None
) -> Computed:
    """Values and weights are distinguished by ``role``, never by position.

    Interleaving them positionally would make a mis-ordered request compute a
    plausible wrong answer instead of failing, which is precisely the class of
    silent error this package exists to remove. Values carry ``role="input"``;
    weights carry ``role="stated_comparison"``, the only other role the contract
    defines, reused here as "not an addend" rather than inventing a role the
    model-facing schema would have to grow a field for.
    """
    target = _target(request)
    values_in = _inputs(request)
    weights_in = tuple(o for o in request.operands if o.role == "stated_comparison")
    if not values_in or not weights_in:
        raise EngineError(
            "INVALID_INPUT", "weighted_average_shape",
            "weighted_average needs value operands (role='input') and weight "
            "operands (role='stated_comparison')",
        )
    if len(values_in) != len(weights_in):
        raise EngineError(
            "INVALID_INPUT", "weighted_average_shape",
            f"{len(values_in)} values but {len(weights_in)} weights",
        )
    weight_dimensions = {w.unit.dimension for w in weights_in}
    if len(weight_dimensions) != 1:
        raise EngineError(
            "UNIT_MISMATCH", "mixed_weight_units",
            f"weights mix representations: {sorted(weight_dimensions)}",
        )
    weight_dimension = weight_dimensions.pop()
    if weight_dimension not in ("dimensionless", "percentage"):
        raise EngineError("UNIT_MISMATCH", "bad_weight_unit",
                          f"weights must be dimensionless or pct; got {weight_dimension}")
    normalized: list[NormalizedOperand] = []
    values: list[Decimal] = []
    for operand in values_in:
        _require_compatible(operand.unit, target, "weighted_average")
        norm, value = _normalize(operand, target, time_policy=policy)
        normalized.append(norm)
        values.append(value)
    weights: list[Decimal] = []
    for operand in weights_in:
        raw = _to_decimal(operand.value, "weight value", operand.operand_id)
        normalized.append(NormalizedOperand(
            operand_id=operand.operand_id, label=operand.label,
            original_value=canonical_numeric_string(raw), original_unit=operand.unit,
            normalized_value=_render(raw, f"weight {operand.operand_id}"),
            normalized_unit=operand.unit, basis=operand.basis, role=operand.role,
        ))
        weights.append(raw)
    for operand, weight in zip(weights_in, weights, strict=True):
        if weight < 0:
            # A negative weight makes the "average" leave the convex hull of its
            # inputs: 10 and 20 weighted -2 and 1 gives 0. One sign error in a
            # model-emitted weight would produce a verified, fingerprinted
            # figure that is arbitrarily wrong and undetectable downstream.
            raise EngineError(
                "INVALID_INPUT", "negative_weight",
                f"weight {operand.operand_id!r} is negative ({weight}); a "
                "weighted average must lie between its smallest and largest "
                "value, which negative weights do not guarantee",
                operand.operand_id,
            )
    total_weight = sum(weights[1:], weights[0])
    if total_weight <= 0:
        raise EngineError("DIVISION_BY_ZERO", "zero_weight_total",
                          "weights sum to zero")
    weighted = sum(
        (v * w for v, w in list(zip(values, weights, strict=True))[1:]),
        values[0] * weights[0],
    )
    average = weighted / total_weight
    parts = [f"{v}x{w}" for v, w in zip(
        [n.normalized_value for n in normalized[:len(values)]],
        [n.normalized_value for n in normalized[len(values):]],
        strict=True,
    )]
    expression = _expr(parts, "+") + f" / {total_weight}"
    return Computed(average, target, expression[:MAX_EXPRESSION_LEN], tuple(normalized),
                    used_time_policy=any(
                        _uses_time_policy(o.unit, target) for o in values_in))


def _classify(stated: Decimal, calculated: Decimal) -> tuple[ConflictClass, Decimal | None]:
    """How a calculated figure relates to the applicant's stated one.

    ``ORDER_OF_MAGNITUDE`` is separated from a generic conflict because it is the
    specific error that survived a full evaluation unnoticed: a stated 572
    tonnes against a computed 5,720. A reviewer scanning statuses should be able
    to find that class without reading every difference.

    A ratio alone never proves which figure is right — the classification says
    *how far apart they are*, and the record keeps both values so a human decides.
    """
    if stated == calculated:
        return "EXACT_MATCH", Decimal(1) if stated != 0 else None
    if stated == 0:
        return "CONFLICT_DETECTED", None
    ratio = calculated / stated
    if (stated < 0) != (calculated < 0):
        return "SIGN_MISMATCH", ratio
    magnitude = abs(ratio)
    if magnitude >= 10 or magnitude <= Decimal("0.1"):
        return "ORDER_OF_MAGNITUDE", ratio
    relative = abs(calculated - stated) / abs(stated)
    if relative <= Decimal("0.0001"):
        return "WITHIN_TOLERANCE", ratio
    return "CONFLICT_DETECTED", ratio


def _op_variance(request: CalculationRequest, policy: TimeConversionPolicy | None) -> Computed:
    """Compare an applicant-stated value with an independently calculated one.

    The single ``input`` operand is the calculated figure; the request's
    ``stated_comparison`` operand is what the applicant claimed. Both must be
    unit-compatible — comparing 5,720 kg against 572 EUR is not a variance.
    """
    target = _target(request)
    calculated_in = _inputs(request)
    stated_in = tuple(o for o in request.operands if o.role == "stated_comparison")
    if len(calculated_in) != 1 or len(stated_in) != 1:
        raise EngineError(
            "INVALID_INPUT", "variance_shape",
            "variance needs exactly one calculated operand (role='input') and "
            "one stated operand (role='stated_comparison')",
        )
    calculated_op, stated_op = calculated_in[0], stated_in[0]
    _require_compatible(calculated_op.unit, target, "variance")
    _require_compatible(stated_op.unit, target, "variance")
    calc_norm, calculated = _normalize(calculated_op, target, time_policy=policy)
    stated_norm, stated = _normalize(stated_op, target, time_policy=policy)
    conflict, ratio = _classify(stated, calculated)
    difference = calculated - stated
    derived_warnings: list[str] = []
    percentage = None
    if stated != 0:
        percentage = _bounded(difference / stated * Decimal(100),
                              "percentage_difference", request.scale)
        if percentage is None:
            derived_warnings.append(
                "percentage_difference unavailable: the value cannot be "
                "represented within the field's limits"
            )
    absolute = _bounded(abs(difference), "absolute_difference", request.scale)
    if absolute is None:
        derived_warnings.append(
            "absolute_difference unavailable: the value cannot be represented "
            "within the field's limits"
        )
    ratio_text = _bounded(ratio, "ratio", request.scale) if ratio is not None else None
    if ratio is not None and ratio_text is None:
        derived_warnings.append(
            "ratio unavailable: the value cannot be represented within the "
            "field's limits"
        )
    return Computed(
        exact=difference,
        unit=target,
        expression=_expr(
            [f"calculated {calc_norm.normalized_value}",
             f"stated {stated_norm.normalized_value}"], "-",
        ),
        normalized=(calc_norm, stated_norm),
        stated_value=stated_norm.normalized_value,
        absolute_difference=absolute,
        percentage_difference=percentage,
        ratio=ratio_text,
        conflict=conflict,
        warnings=tuple(derived_warnings),
        used_time_policy=any(
            _uses_time_policy(o.unit, target) for o in (calculated_op, stated_op)
        ),
    )


def _op_convert_unit(
    request: CalculationRequest, policy: TimeConversionPolicy | None
) -> Computed:
    target = _target(request)
    (operand,) = _inputs(request)
    if operand.unit.dimension == "percentage" and target.dimension == "percentage_point":
        raise EngineError(
            "UNIT_MISMATCH", "pct_is_not_pct_point",
            "pct and pct_point are different concepts, not scale variants",
        )
    norm, converted = _normalize(operand, target, time_policy=policy)
    return Computed(
        converted, target,
        f"{norm.original_value} {operand.unit.code} -> {target.code}",
        (norm,),
        used_time_policy=_uses_time_policy(operand.unit, target),
    )


def _op_interval_implied_total(
    request: CalculationRequest, policy: TimeConversionPolicy | None
) -> Computed:
    """Volume interval / coverage interval -> implied total-production interval.

    Four operands in order: volume low, volume high, coverage low, coverage high.

    The endpoint pairing is the whole point. Division is decreasing in its
    denominator, so the extremes of ``[a,b] / [c,d]`` with ``c,d > 0`` are
    ``a/d`` and ``b/c`` — **cross-paired**. The naive same-index pairing
    (``a/c``, ``b/d``) yields a spuriously narrow band that can exclude the true
    value, which is how a cross-check silently passes when it should fail.

    Both bounds are computed and the result carries the lower one, with the
    upper recorded in the expression; a caller wanting the pair reads the
    expression or issues two divisions.
    """
    target = _target(request)
    operands = _inputs(request)
    vol_low_op, vol_high_op, cov_low_op, cov_high_op = operands
    for operand in (vol_low_op, vol_high_op):
        _require_compatible(operand.unit, target, "interval_implied_total")
    for operand in (cov_low_op, cov_high_op):
        if operand.unit.dimension not in ("percentage", "dimensionless"):
            raise EngineError(
                "UNIT_MISMATCH", "bad_coverage_unit",
                f"coverage bounds must be pct or dimensionless; "
                f"{operand.operand_id} is {operand.unit.dimension}",
            )
    normalized: list[NormalizedOperand] = []
    vol_low_n, vol_low = _normalize(vol_low_op, target, time_policy=policy)
    vol_high_n, vol_high = _normalize(vol_high_op, target, time_policy=policy)
    normalized += [vol_low_n, vol_high_n]
    fractions: list[Decimal] = []
    for operand in (cov_low_op, cov_high_op):
        raw = _to_decimal(operand.value, "coverage bound", operand.operand_id)
        # Percentages become fractions explicitly. Nothing here infers a scale.
        fraction = raw / Decimal(100) if operand.unit.dimension == "percentage" else raw
        normalized.append(NormalizedOperand(
            operand_id=operand.operand_id, label=operand.label,
            original_value=canonical_numeric_string(raw), original_unit=operand.unit,
            normalized_value=_render(raw, f"coverage {operand.operand_id}"),
            normalized_unit=operand.unit, basis=operand.basis, role=operand.role,
        ))
        fractions.append(fraction)
    cov_low, cov_high = fractions
    if vol_low > vol_high:
        raise EngineError("INVALID_INPUT", "interval_order",
                          "volume lower bound exceeds its upper bound")
    if cov_low > cov_high:
        raise EngineError("INVALID_INPUT", "interval_order",
                          "coverage lower bound exceeds its upper bound")
    if cov_low <= 0 or cov_high <= 0:
        raise EngineError(
            "INVALID_INPUT", "interval_denominator",
            "coverage interval must be strictly positive; an interval touching "
            "or crossing zero makes the quotient unbounded",
        )
    implied_low = vol_low / cov_high
    implied_high = vol_high / cov_low
    # Canonical positional form, like every numeric field in this contract. A
    # draft used ``str(Decimal)`` and rendered the upper bound as "5.0000E+5" —
    # asking a reviewer to re-read the exact magnitude this package exists to
    # protect, in the one figure that lives only in free text.
    upper_text = _bounded(implied_high, "interval upper bound", request.scale) or "unavailable"
    expression = (
        f"[{canonical_numeric_string(vol_low)}, {canonical_numeric_string(vol_high)}]"
        f" / [{canonical_numeric_string(cov_low)}, {canonical_numeric_string(cov_high)}]"
        f" = [{_bounded(implied_low, 'interval lower bound', request.scale) or 'unavailable'}, "
        f"{upper_text}]"
    )
    return Computed(
        implied_low, target, expression[:MAX_EXPRESSION_LEN], tuple(normalized),
        warnings=(f"upper bound {upper_text}",),
        used_time_policy=any(
            _uses_time_policy(o.unit, target) for o in (vol_low_op, vol_high_op)
        ),
    )


_MUTABLE_DISPATCH: dict[
    str, Callable[[CalculationRequest, TimeConversionPolicy | None], Computed]
] = {
    "add": _op_add,
    "subtract": _op_subtract,
    "multiply": _op_multiply,
    "divide": _op_divide,
    "sum_components": _op_sum_components,
    "percentage_of": _op_percentage_of,
    "percentage_point_difference": _op_percentage_point_difference,
    "ratio": _op_ratio,
    "weighted_average": _op_weighted_average,
    "variance": _op_variance,
    "convert_unit": _op_convert_unit,
    "interval_implied_total": _op_interval_implied_total,
}

_DISPATCH: Mapping[
    str, Callable[[CalculationRequest, TimeConversionPolicy | None], Computed]
] = MappingProxyType(_MUTABLE_DISPATCH)
del _MUTABLE_DISPATCH


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _validate_shape(request: CalculationRequest) -> OperationSignature:
    """Everything checkable before any Decimal work happens.

    Ordering matters: size and exponent limits are enforced *first*, so a
    hostile input is rejected before it can reach an allocation. That, not the
    wall-clock budget, is what makes every authorised operation bounded.
    """
    signature = signature_for(request.operation)
    if signature is None:
        raise EngineError("UNSUPPORTED_OPERATION", "unknown_operation",
                          f"{request.operation!r} is not an authorised operation")
    inputs = _inputs(request)
    if not (signature.min_arity <= len(inputs) <= signature.max_arity):
        raise EngineError(
            "INVALID_INPUT", "arity",
            f"{request.operation} accepts {signature.min_arity}..{signature.max_arity} "
            f"input operands; got {len(inputs)}",
        )
    if len(request.operands) > MAX_OPERANDS_PER_REQUEST:
        raise EngineError("RESOURCE_LIMIT_EXCEEDED", "too_many_operands",
                          f"{len(request.operands)} operands exceeds "
                          f"{MAX_OPERANDS_PER_REQUEST}")
    if request.scale > LIMITS.max_precision:
        raise EngineError("RESOURCE_LIMIT_EXCEEDED", "scale",
                          f"scale {request.scale} exceeds the working precision")
    if signature.requires_target_unit and request.target_unit is None:
        raise EngineError("INVALID_INPUT", "missing_target_unit",
                          f"{request.operation} requires a target_unit")
    stated = [o for o in request.operands if o.role == "stated_comparison"]
    if signature.requires_stated_value and not stated:
        raise EngineError("INVALID_INPUT", "missing_stated_value",
                          f"{request.operation} requires a stated_comparison operand")
    if not signature.requires_stated_value and stated and request.operation not in (
        "weighted_average",
    ):
        raise EngineError(
            "INVALID_INPUT", "unexpected_stated_value",
            f"{request.operation} does not take a stated_comparison operand",
        )
    if signature.fixed_input_dimension is not None:
        for operand in inputs:
            if operand.unit.dimension != signature.fixed_input_dimension:
                raise EngineError(
                    "UNIT_MISMATCH", "wrong_input_dimension",
                    f"{request.operation} requires "
                    f"{signature.fixed_input_dimension} inputs; "
                    f"{operand.operand_id} is {operand.unit.dimension}",
                )
    if (
        signature.fixed_output_dimension is not None
        and request.target_unit is not None
        and request.target_unit.dimension != signature.fixed_output_dimension
    ):
        raise EngineError(
            "UNIT_MISMATCH", "wrong_output_dimension",
            f"{request.operation} must produce "
            f"{signature.fixed_output_dimension}; target_unit "
            f"{request.target_unit.code} is {request.target_unit.dimension}",
        )
    if signature.requires_compatible_inputs and len(inputs) > 1:
        first = inputs[0].unit
        for operand in inputs[1:]:
            _require_compatible(first, operand.unit, request.operation)
    return signature


def _failure(
    request: CalculationRequest,
    status: ArithmeticStatus,
    code: str,
    detail: str,
    computed_at: str,
    operand_id: str | None = None,
) -> CalculationResult:
    """A typed failure result. No fingerprint — see ``FAILURE_FINGERPRINT_RULE``.

    ``computed_at`` is not assumed to be canonical. Some callers reach here
    after ``execute`` has already canonicalised it, but others do not:
    ``execute_batch``'s over-budget path calls this with the caller's raw
    value. So the timestamp is canonicalised here defensively, falling back to
    ``FALLBACK_COMPUTED_AT`` when the value cannot be canonicalised at all.
    That is what makes this function total: it is the recovery path, so it must
    be constructible for *any* input, including the case where the caller's
    clock value is itself what failed. Handing a rejected value back to the
    constructor that had just refused it is what made the handler re-raise the
    exception it existed to convert.
    """
    from openexecutive.calc.authority import issue_calculation_result

    # Reachable, and an earlier comment claiming otherwise was wrong:
    # ``execute_batch``'s over-budget path calls this with the caller's RAW
    # value, bypassing ``execute``'s canonicalisation entirely. The pragma that
    # marked it uncovered is removed with it — an unreachability claim is
    # load-bearing documentation, and a false one is worse than none.
    #
    # It is also what makes this function total: it is the recovery path, so it
    # must be constructible for any input, including the case where the caller's
    # clock value is itself what failed. Handing a rejected value back to the
    # constructor that had just refused it is precisely the bug this prevents.
    safe_computed_at, _ = canonical_computed_at(computed_at)
    if safe_computed_at is None:
        safe_computed_at = FALLBACK_COMPUTED_AT

    return issue_calculation_result(
        request_id=request.request_id,
        operation=request.operation,
        correlation=request.correlation,
        arithmetic_status=status,
        evidence=InputEvidenceSummary(status="EVIDENCE_UNAVAILABLE"),
        computed_at=safe_computed_at,
        errors=(CalculationError(code=code, detail=detail[:400],
                                 operand_id=operand_id),),
    )


def execute(
    request: CalculationRequest,
    *,
    computed_at: str,
    evidence: InputEvidenceSummary | None = None,
    time_conversion_policy: TimeConversionPolicy | None = None,
    weight_policy: WeightPolicy = "normalized_by_engine",
) -> CalculationResult:
    """Execute one request and return a typed result — success or failure.

    ``computed_at`` is a required argument rather than a clock read: this package
    takes no ambient dependencies, and a caller that must pass a timestamp is a
    caller whose tests can pin one. It is excluded from the fingerprint.

    ``evidence`` is the **application-validated** binding, supplied through this
    non-model path. It defaults to ``EVIDENCE_UNAVAILABLE``: the engine verifies
    arithmetic, never source truth, and it derives nothing from an operand's
    model-proposed ``SourceHint``. A retrieval id that exists only as text in a
    hint can never become ``ALL_SUPPORTED`` here.

    Note what is absent from this signature: no fingerprint, no arithmetic
    status, no authority stamp, no engine version, no pre-executed expression,
    no verified result. Those are constructed internally, every time.
    """
    started = time.monotonic()
    canonical_at, clock_error = canonical_computed_at(computed_at)
    if canonical_at is None:
        # A typed record, not an exception: the caller's clock format is not a
        # reason to lose the calculation's outcome.
        return _failure(request, "INVALID_INPUT", "invalid_computed_at",
                        clock_error or "computed_at is not usable",
                        FALLBACK_COMPUTED_AT)
    computed_at = canonical_at
    if (
        time_conversion_policy is not None
        and time_conversion_policy not in SUPPORTED_TIME_CONVERSION_POLICIES
    ):
        raise ValueError(
            f"unsupported time_conversion_policy {time_conversion_policy!r}; "
            f"expected one of {sorted(SUPPORTED_TIME_CONVERSION_POLICIES)}. The "
            "value names the basis on the durable record and enters the "
            "fingerprint, so an unrecognised one would assert a conversion the "
            "engine did not perform."
        )
    if weight_policy not in SUPPORTED_WEIGHT_POLICIES:
        raise ValueError(
            f"unsupported weight_policy {weight_policy!r}; expected one of "
            f"{sorted(SUPPORTED_WEIGHT_POLICIES)}"
        )
    if evidence is not None and not isinstance(evidence, InputEvidenceSummary):
        # A bare status string must not slip through. ``evidence or default``
        # would forward whatever was passed, and a caller writing
        # ``evidence="ALL_SUPPORTED"`` would get a result whose ``.status`` is a
        # string that no validator ever saw — the one field on the record that
        # decides whether a figure counts as supported evidence.
        raise TypeError(
            "evidence must be an InputEvidenceSummary constructed by application "
            f"code, not {type(evidence).__name__}; a status string is not a "
            "validated binding"
        )
    try:
        with localcontext(engine_context()):

            _validate_shape(request)
            handler = _DISPATCH.get(request.operation)
            if handler is None:  # pragma: no cover - _validate_shape covers it
                raise EngineError("UNSUPPORTED_OPERATION", "unknown_operation",
                                  f"{request.operation!r} has no handler")
            computed = handler(request, time_conversion_policy)

            exact = computed.exact
            quantum = Decimal(1).scaleb(-request.scale)
            try:
                rounded = exact.quantize(quantum, rounding=_ROUNDING[request.rounding])
            except InvalidOperation as exc:
                # ``quantize`` raises when the result needs more than ``prec``
                # digits. That is the engine's limit, not a bad input, and
                # reporting it as INVALID_INPUT told a reviewer the wrong thing.
                raise EngineError(
                    "RESOURCE_LIMIT_EXCEEDED", "result_out_of_range",
                    f"result cannot be represented at scale {request.scale} "
                    f"within {ENGINE_PRECISION} digits of precision",
                ) from exc
            # ``exact_result`` is not a derived commentary field: if it cannot
            # be represented, the calculation has no reportable answer. The
            # ``_render`` call immediately below raises for the same input, so
            # this is stated as an invariant rather than duplicated as a second
            # guard that mutation testing showed nothing depends on.
            exact_text = _bounded(exact, "exact_result", request.scale)
            rounded_text = _render(rounded, "result_value", request.scale)
            if exact_text is None:  # pragma: no cover - _render above raises first
                raise EngineError(
                    "RESOURCE_LIMIT_EXCEEDED", "result_out_of_range",
                    "the exact result cannot be represented within the "
                    "field's limits",
                )

        fingerprint = fingerprint_for(
            operation=request.operation,
            normalized_operands=computed.normalized,
            target_unit=computed.unit,
            scale=request.scale,
            rounding=request.rounding,
            authority=current_authority(),
            stated_value=computed.stated_value,
            # Both policies are gated on actually having been consumed. A
            # caller kwarg that touched nothing must not split the identity of
            # an identical calculation — that is precisely the dedup-index
            # failure the payload allowlist exists to prevent.
            time_conversion_policy=(
                time_conversion_policy if computed.used_time_policy else None
            ),
            weight_policy=weight_policy if request.operation == "weighted_average" else None,
        )
        elapsed = time.monotonic() - started
        warnings = computed.warnings
        if elapsed > PER_REQUEST_BUDGET_S:
            warnings = (*warnings, f"execution took {elapsed:.3f}s, over the "
                                   f"{PER_REQUEST_BUDGET_S}s budget")
        from openexecutive.calc.authority import issue_calculation_result

        return issue_calculation_result(
            request_id=request.request_id,
            operation=request.operation,
            correlation=request.correlation,
            arithmetic_status="ARITHMETIC_VERIFIED",
            evidence=evidence or InputEvidenceSummary(status="EVIDENCE_UNAVAILABLE"),
            computed_at=computed_at,
            normalized_operands=computed.normalized,
            expression_executed=computed.expression,
            exact_result=exact_text,
            result_value=rounded_text,
            result_unit=computed.unit,
            scale_applied=request.scale,
            rounding_applied=request.rounding,
            stated_value=computed.stated_value,
            absolute_difference=computed.absolute_difference,
            percentage_difference=computed.percentage_difference,
            ratio=computed.ratio,
            conflict=computed.conflict,
            warnings=warnings,
            fingerprint=fingerprint,
        )
    except EngineError as exc:
        return _failure(request, exc.status, exc.code, exc.detail, computed_at,
                        exc.operand_id)
    except DivisionByZero as exc:
        return _failure(request, "DIVISION_BY_ZERO", "decimal_division_by_zero",
                        str(exc) or "division by zero", computed_at)
    except Overflow as exc:
        return _failure(request, "RESOURCE_LIMIT_EXCEEDED", "decimal_overflow",
                        str(exc) or "overflow", computed_at)
    except InvalidOperation as exc:
        return _failure(request, "INVALID_INPUT", "decimal_invalid_operation",
                        str(exc) or "invalid operation", computed_at)
    except DecimalException:
        # Unreachable as the code now stands, and stated as such rather than
        # marked with a pragma that was previously false. Every trapped signal
        # (InvalidOperation, DivisionByZero, Overflow, FloatOperation) has its
        # own handler above; the remaining five are untrapped by
        # ``engine_context`` and cannot raise. The route that once reached here
        # was an inherited ambient trap, which the engine-owned context closed.
        #
        # It stays as defence in depth against a future signal being trapped
        # without a handler, and it no longer echoes ``str(exc)`` — that wrote
        # "Inexact: [<class 'decimal.Inexact'>]" into a durable audit record,
        # which its three sibling handlers deliberately avoid. Because it is
        # unreachable, no test can pin it; that is a property of the guard, not
        # a coverage gap.
        return _failure(request, "CALCULATION_UNAVAILABLE", "decimal_error",
                        "the calculation raised an unexpected decimal condition",
                        computed_at)
    except ValidationError:
        # A contract validator refused the result the engine built. The record
        # must say that in the engine's own words: pydantic's message carries a
        # traceback-shaped string and a truncated echo of a computed value, and
        # a durable audit record is the wrong home for either.
        return _failure(request, "CALCULATION_UNAVAILABLE", "result_rejected",
                        "the computed result did not satisfy the result contract",
                        computed_at)
    except (MemoryError, OverflowError, RecursionError, ValueError) as exc:
        # Request data must never produce an untyped escape. MemoryError and
        # OverflowError are here because a Decimal expansion or an oversized int
        # can raise them from CPython itself rather than from the decimal
        # module, and ValueError because helpers below this boundary may raise
        # CPython's own (e.g. the 4,300-digit int-to-str conversion limit).
        # Deliberately does NOT echo ``str(exc)``: CPython's messages for these
        # embed the offending value, and this text lands in a durable record.
        return _failure(request, "RESOURCE_LIMIT_EXCEEDED", "resource_limit",
                        f"the request exceeded an engine resource limit "
                        f"({type(exc).__name__})", computed_at)


def execute_batch(
    batch: CalculationBatch,
    *,
    computed_at: str,
    evidence_by_request: Mapping[str, InputEvidenceSummary] | None = None,
    time_conversion_policy: TimeConversionPolicy | None = None,
    weight_policy: WeightPolicy = "normalized_by_engine",
) -> tuple[CalculationResult, ...]:
    """Execute a bounded batch, one result per request, in request order.

    ``evidence_by_request`` is keyed by ``request_id`` — there is deliberately no
    batch-wide ``evidence`` parameter. An earlier version took one and forwarded
    it to every request, which was a real hole rather than a stylistic choice:
    ``operand_id`` is unique only *within* a request, and models reuse generic
    ids (``a``, ``total``, ``revenue``). Evidence validated for request 1 landed
    on any sibling whose ids happened to collide, and the result's own
    cross-check — that the named ids equal the recorded ids — is an identity
    test a colliding id set satisfies trivially. Operands no evidence layer had
    ever seen reached ``is_verified_evidence()``.

    Request scoping fixes the addressing: a binding names the one request it was
    validated for, and a request with no entry gets ``EVIDENCE_UNAVAILABLE``.

    Failure isolation is the other property that matters: a request that fails
    produces a typed failure in its own slot and does not remove, reorder, or
    invalidate its siblings. Phase 1 measured what the alternative costs — a
    single malformed claim discarded all eleven in a specialist result.

    No state is stored on a module global or a shared instance; everything lives
    in this call frame, so two concurrent batches cannot see each other's work.
    """
    if evidence_by_request is not None:
        if not isinstance(evidence_by_request, Mapping):
            raise TypeError(
                "evidence_by_request must be a mapping of request_id -> "
                f"InputEvidenceSummary, not {type(evidence_by_request).__name__}"
            )
        known = {request.request_id for request in batch.requests}
        for request_id, summary in evidence_by_request.items():
            if not isinstance(summary, InputEvidenceSummary):
                raise TypeError(
                    f"evidence for {request_id!r} must be an InputEvidenceSummary "
                    f"constructed by application code, not {type(summary).__name__}"
                )
            if request_id not in known:
                # Silently ignoring it would let a caller believe a binding was
                # applied when the request it names is not in this batch.
                raise ValueError(
                    f"evidence names request {request_id!r}, which is not in this "
                    "batch"
                )
    started = time.monotonic()
    results: list[CalculationResult] = []
    over_budget = False
    for request in batch.requests:
        if over_budget:
            results.append(_failure(
                request, "RESOURCE_LIMIT_EXCEEDED", "batch_budget",
                f"batch exceeded its {PER_BATCH_BUDGET_S}s budget before this "
                "request was executed", computed_at,
            ))
            continue
        results.append(execute(
            request, computed_at=computed_at,
            evidence=(evidence_by_request or {}).get(request.request_id),
            time_conversion_policy=time_conversion_policy,
            weight_policy=weight_policy,
        ))
        if time.monotonic() - started > PER_BATCH_BUDGET_S:
            # Checked *between* requests: a synchronous call cannot be preempted
            # from inside itself without threads or signals, both of which this
            # package deliberately does not own. Remaining requests fail typed
            # rather than silently vanishing.
            over_budget = True
    return tuple(results)
