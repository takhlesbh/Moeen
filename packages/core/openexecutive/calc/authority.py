"""Who may issue a calculation result, and how strong that boundary actually is.

Two adversaries, two very different answers. Conflating them is how a design
document ends up claiming a guarantee the runtime does not provide, so they are
separated here explicitly.

**Against the model — structural, and it holds.**
A model communicates with this package through exactly one type,
:class:`~openexecutive.calc.contract.CalculationRequest`. That type has no
field for an arithmetic status, an evidence status, an authority stamp, a
fingerprint, a computed timestamp, or a trusted source location. Because the
contract models are ``extra="forbid"``, a payload that tries to carry one is
*rejected* rather than silently stripped, so the attempt is visible instead of
being quietly discarded. There is no wire on which a model can assert that its
own arithmetic was checked. This mirrors the guarantee the specialist result
contract already relies on for ``verified_result``: the strongest available
form of "the model cannot claim this" is that the wire format gives it no field
in which to claim it.

**Against a same-process caller — conventional, and it does not hold absolutely.**
Python has no private constructor. ``CalculationResult.model_construct(...)``
and ``object.__setattr__`` bypass pydantic for any model, exactly as they do for
every other model in this codebase. This module therefore does **not** claim
unforgeability. What :func:`issue_calculation_result` provides is a single
named, greppable, reviewable channel: application code that goes through it
gets its authority stamp filled in from a module constant rather than from a
caller argument, and a code review can ask why any other construction site
exists. That is a convention with teeth, not a security boundary, and calling it
anything stronger would be the same overclaim this whole programme exists to
stop.

Phase 1 additionally leaves one thing genuinely unbuildable: no engine exists,
so no caller in this package can produce a real ``ARITHMETIC_VERIFIED`` result
from actual arithmetic. The status is constructible — it must be, so the
contract can express "arithmetic correct, inputs unsupported" and be tested
against it — but nothing in production computes one, and nothing imports this
package.
"""
from __future__ import annotations

from openexecutive.calc.contract import (
    KNOWN_AUTHORITY_IDS,
    KNOWN_AUTHORITY_VERSIONS,
    ApplicationAuthority,
    ArithmeticStatus,
    CalculationError,
    CalculationResult,
    ConflictClass,
    Correlation,
    InputEvidenceSummary,
    NormalizedOperand,
    OperationId,
    RoundingMode,
)
from openexecutive.calc.units import Unit

AUTHORITY_ID = "openexecutive.calc"
"""Stable identity of the application-side calculation authority.

Must be a member of ``KNOWN_AUTHORITY_IDS``, which the contract validates, so a
stamp naming an authority that never existed cannot survive deserialization."""
assert AUTHORITY_ID in KNOWN_AUTHORITY_IDS

AUTHORITY_VERSION = "0.2.0-engine"
"""Bumped from ``0.1.0-contract`` in the commit that added the engine.

The version is part of the fingerprint payload, so a result computed by this
engine can never collide with one hand-built under the contract-only phase —
which is the whole point of the suffix, and was briefly untrue when the engine
shipped while the stamp still read ``-contract``.

Must be a member of ``KNOWN_AUTHORITY_VERSIONS``, which the contract validates,
so a replayed result cannot be re-stamped with a version that never existed.
"""
assert AUTHORITY_VERSION in KNOWN_AUTHORITY_VERSIONS


def current_authority() -> ApplicationAuthority:
    """The authority stamp for this build. Not a caller-supplied value."""
    return ApplicationAuthority(
        authority_id=AUTHORITY_ID, authority_version=AUTHORITY_VERSION
    )


def issue_calculation_result(
    *,
    request_id: str,
    operation: OperationId,
    correlation: Correlation,
    arithmetic_status: ArithmeticStatus,
    evidence: InputEvidenceSummary,
    computed_at: str,
    normalized_operands: tuple[NormalizedOperand, ...] = (),
    expression_executed: str | None = None,
    exact_result: str | None = None,
    result_value: str | None = None,
    result_unit: Unit | None = None,
    scale_applied: int | None = None,
    rounding_applied: RoundingMode | None = None,
    stated_value: str | None = None,
    absolute_difference: str | None = None,
    percentage_difference: str | None = None,
    ratio: str | None = None,
    conflict: ConflictClass = "NONE",
    warnings: tuple[str, ...] = (),
    errors: tuple[CalculationError, ...] = (),
    fingerprint: str | None = None,
) -> CalculationResult:
    """The sanctioned channel for constructing a :class:`CalculationResult`.

    The authority stamp is **not** a parameter. It is filled from
    :func:`current_authority`, so a caller cannot label a result as having come
    from an engine version it did not come from — which is the one part of the
    stamp that would otherwise be trivially wrong rather than merely forgeable.

    ``computed_at`` is a required argument rather than a call to the clock: this
    package takes no ambient dependencies, and a caller that must pass a
    timestamp is a caller whose tests can pin one.
    """
    return CalculationResult(
        request_id=request_id,
        operation=operation,
        correlation=correlation,
        normalized_operands=normalized_operands,
        expression_executed=expression_executed,
        exact_result=exact_result,
        result_value=result_value,
        result_unit=result_unit,
        scale_applied=scale_applied,
        rounding_applied=rounding_applied,
        arithmetic_status=arithmetic_status,
        evidence=evidence,
        stated_value=stated_value,
        absolute_difference=absolute_difference,
        percentage_difference=percentage_difference,
        ratio=ratio,
        conflict=conflict,
        warnings=warnings,
        errors=errors,
        authority=current_authority(),
        fingerprint=fingerprint,
        computed_at=computed_at,
    )
