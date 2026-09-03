"""Engine-owned calculation identity.

Phase 1 defined *what* a fingerprint means — which fields constitute the
identity of a calculation and which are incidental — and pinned that with tests.
It deliberately shipped no digest, because no canonical-hash utility existed in
this repository to reuse. This module supplies the digest and nothing else.

The rule the whole module exists to enforce: **a fingerprint is computed, never
accepted**. :class:`~openexecutive.calc.contract.CalculationResult` has a
``fingerprint`` field and Phase 1's ``issue_calculation_result`` takes it as a
parameter, which means a caller can put any 64-hex string there. That was fine
while nothing consumed it. It stops being fine the moment a fingerprint is used
to recognise a repeated calculation, so the engine derives its own and never
forwards a caller's.

Identity is "what was computed", not "who asked or when". One consequence is
worth stating outright rather than leaving for whoever builds a dedup index to
discover: the payload carries only the **normalized** operand value, unit,
basis and role — never ``original_unit``, ``original_value`` or
``conversion_applied``. So ``convert_unit 100 kg -> t`` and ``0.1 t -> t``
produce the same fingerprint. That is deliberate. The fingerprint identifies the
*calculation*, and the result is a pure function of the payload, so two
calculations with differing results cannot collide; what collides is two routes
to the same computed figure. A consumer asking "was this figure already
computed?" is answered correctly; a consumer asking "was this request already
made?" must read the record, not the fingerprint.

The included and excluded fields:

* **included** — schema version, engine authority id and version, operation,
  ordered normalized operands (value, unit, basis, role), target unit, scale,
  rounding mode, the applicant's stated comparison value, and any
  operation-specific parameters;
* **excluded** — request id, case id, run id, specialist, claim id, timestamps,
  labels, purpose prose, and anything else that varies without the arithmetic
  varying.

Two runs of the same reconciliation in different cases therefore fingerprint
identically, which is the property that makes the value useful at all.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from openexecutive.calc.contract import (
    ApplicationAuthority,
    NormalizedOperand,
    OperationId,
    RoundingMode,
    canonical_payload_json,
    fingerprint_payload,
)
from openexecutive.calc.units import Unit

FINGERPRINT_ALGORITHM = "sha256"
"""Named so a later change of algorithm is a visible, versioned decision.

SHA-256 is used for *identity*, not for secrecy or authentication. Nothing here
resists a determined forger — a same-process caller can write any field on any
Python object — and the module says so rather than implying a guarantee it
cannot provide. What it does provide is that two different calculations do not
collide, and that the same calculation reproduces the same value in any process.
"""

FAILURE_FINGERPRINT_RULE = "absent"
"""Failed calculations carry **no** fingerprint. Documented, not incidental.

The Council asked for one of two options: deterministic-and-namespaced, or
absent with a stated rule. Absent is chosen because a fingerprint identifies
*a computed answer*, and a calculation that did not run has none. Minting an
identity for a failure would create a value that looks joinable with real
results — and a dedup index keyed on it would happily return "we already
computed this" for something never computed. The contract already enforces the
half of this that matters: a non-verified status may not carry a result.
"""


def _operation_parameters(
    *,
    time_conversion_policy: str | None = None,
    weight_policy: str | None = None,
) -> dict[str, Any]:
    """Operation-specific identity parameters, sorted and stable.

    Kept as an explicit allowlist rather than a free-form mapping: anything that
    changes the *meaning* of a calculation must change its identity, and
    anything else must not be able to leak in and fragment it.

    Both entries are policy *declarations*. Stated precisely, because a draft of
    this docstring overstated it: ``time_conversion_policy`` does not select a
    basis — the registry's exact factor is the only one available — it is a
    permission token recording that the caller accepted a month/year conversion
    at all, which is the assumption a reader needs to see. ``weight_policy``
    likewise records how weights were treated. Both belong in the identity
    because a calculation made under a different declaration is a different
    claim about the world; neither is free text, and the engine validates both
    against a closed set before they reach here.
    """
    params: dict[str, Any] = {}
    if time_conversion_policy is not None:
        params["time_conversion_policy"] = time_conversion_policy
    if weight_policy is not None:
        params["weight_policy"] = weight_policy
    return params


def build_payload(
    *,
    operation: OperationId,
    normalized_operands: tuple[NormalizedOperand, ...],
    target_unit: Unit | None,
    scale: int,
    rounding: RoundingMode,
    authority: ApplicationAuthority,
    stated_value: str | None = None,
    time_conversion_policy: str | None = None,
    weight_policy: str | None = None,
) -> dict[str, Any]:
    """The canonical payload for one calculation.

    Delegates the shared fields to Phase 1's
    :func:`~openexecutive.calc.contract.fingerprint_payload` rather than
    restating them, so the declared ``FINGERPRINT_INCLUDED_FIELDS`` contract and
    the digest cannot drift apart. Only the operation-specific parameters are
    added here, and only when present — an ``add`` and a ``subtract`` that
    differ in nothing else must not be separated by an empty ``parameters`` key
    appearing in one and not the other.
    """
    payload = fingerprint_payload(
        operation=operation,
        normalized_operands=normalized_operands,
        target_unit=target_unit,
        scale=scale,
        rounding=rounding,
        authority=authority,
        stated_value=stated_value,
    )
    parameters = _operation_parameters(
        time_conversion_policy=time_conversion_policy,
        weight_policy=weight_policy,
    )
    if parameters:
        payload["parameters"] = parameters
    return payload


def canonical_json(payload: dict[str, Any]) -> str:
    """Deterministic bytes for a payload.

    Reuses Phase 1's serializer (``sort_keys``, fixed separators,
    ``ensure_ascii``) so mapping-key order, whitespace and encoding are
    identical across processes, platforms and Python versions. Operand order is
    a *list* and is therefore preserved — only mapping keys are sorted.
    """
    return canonical_payload_json(payload)


def compute_fingerprint(payload: dict[str, Any]) -> str:
    """SHA-256 of the canonical payload, lowercase hex.

    ``json.dumps`` is called through :func:`canonical_json`; the only reason
    ``json`` is imported here at all is the round-trip assertion below, which
    keeps a payload containing something unserialisable from failing later at a
    less obvious place.
    """
    text = canonical_json(payload)
    # Cheap, and it fails at the point of construction rather than at the point
    # of storage: a payload that cannot round-trip is not a stable identity.
    json.loads(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_for(
    *,
    operation: OperationId,
    normalized_operands: tuple[NormalizedOperand, ...],
    target_unit: Unit | None,
    scale: int,
    rounding: RoundingMode,
    authority: ApplicationAuthority,
    stated_value: str | None = None,
    time_conversion_policy: str | None = None,
    weight_policy: str | None = None,
) -> str:
    """Build the payload and hash it. The engine's only fingerprint entry point.

    There is deliberately no parameter through which a caller can supply a
    precomputed value.
    """
    return compute_fingerprint(
        build_payload(
            operation=operation,
            normalized_operands=normalized_operands,
            target_unit=target_unit,
            scale=scale,
            rounding=rounding,
            authority=authority,
            stated_value=stated_value,
            time_conversion_policy=time_conversion_policy,
            weight_policy=weight_policy,
        )
    )
