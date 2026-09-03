"""What ``route_parallel`` hands back per specialist. Application-owned.

The transport is the asyncio **task result**: ``route_parallel`` gathers one
child task per specialist and returns this object for each, in call order.
Nothing here rides on a ContextVar, an instance attribute or a registry — a
value set inside a gathered child is invisible to the awaiting parent (each
task runs in a copy of the parent's context), and a channel that fails *stale*
rather than *empty* is exactly how a prior turn's figures would be misread as
this turn's. Returning the object is the one mechanism that reads correctly and
fails empty.

Ownership is preserved by NESTING, and the wording is precise on purpose:

* ``specialist_result`` is model-owned intent, parsed from the model's payload;
* ``calculations`` is gateway-owned — request ids, correlations, and the
  engine's records;
* ``frame`` and ``diagnostics`` are application-owned.

Nesting keeps those separate. It does **not** authenticate provenance. Like
``SpecialistCalculations``, this type establishes no provenance on its own: a
frozen dataclass runs no validators, pickle and pydantic's dataclass handling
reconstitute one from any bytes, and a deserialized instance proves nothing
about where its records came from. In-process provenance comes from one source
only — the explicit call path that built this object in this task, from a
gateway call it made itself. A consumer receiving one from an API body, a
database row, a cache, a queue replay or another service must re-establish
provenance by its own means.

``diagnostics`` is a closed vocabulary of literals, never model text and never
exception detail, matching the gateway's ``dropped`` discipline.
"""
from __future__ import annotations

from dataclasses import dataclass

from openexecutive.specialists.calculation_gateway import SpecialistCalculations
from openexecutive.specialists.result_contract import SpecialistResult

CALC_DISABLED = "calc_disabled"
"""The feature flag is off: no schema change, no parse, no gateway."""

CALC_NO_PROPOSALS = "calc_no_proposals"
"""The model proposed nothing; nothing to execute."""

CALC_SKIPPED_STRUCTURE_LOST = "calc_skipped_structure_lost"
"""The parser reached its lost-structure exit: no claim set to authorize
against and no proposals survive that exit, so the gateway is not called."""

CALC_TOO_MANY_PROPOSALS = "calc_too_many_proposals"
"""More proposals than the engine's batch limit. A declared design condition
with its own code — not an engine failure — so an operator can tell the two
apart. Fails closed: nothing in the batch executes."""

CALC_CONTEXT_UNAVAILABLE = "calc_context_unavailable"
"""No usable session/turn identity in the audit context. Fail closed: no
request id is ever minted under a constant fallback identity."""

CALC_CLAIM_SET_UNSAFE = "calc_claim_set_unsafe"
"""A final claim_id is not a usable identifier. The whole authorization set is
refused rather than filtered; partial authorization is not offered."""

CALC_GATEWAY_RAISED = "calc_gateway_raised"
"""The gateway raised. Contained here so it can never fail the turn."""

DIAGNOSTIC_CODES = frozenset(
    {
        CALC_DISABLED,
        CALC_NO_PROPOSALS,
        CALC_SKIPPED_STRUCTURE_LOST,
        CALC_TOO_MANY_PROPOSALS,
        CALC_CONTEXT_UNAVAILABLE,
        CALC_CLAIM_SET_UNSAFE,
        CALC_GATEWAY_RAISED,
    }
)


@dataclass(frozen=True)
class CorrelationFrame:
    """The application-supplied frame a gateway call was minted under.

    ``case_id`` and ``run_id`` are the audit context's session and turn ids,
    each screened as a usable identifier before this object exists;
    ``computed_at`` is a real UTC clock reading passed to the engine as given.
    None of the three is ever derived from model text.
    """

    case_id: str
    run_id: str
    computed_at: str


@dataclass(frozen=True)
class RoutedSpecialistOutput:
    """One specialist's output for one turn, returned as the task result.

    ``narrative`` is exactly the string the legacy boundary returns for this
    call — for the CFO, the same selection rule its ``analyze`` applies; for
    every other specialist, the string ``route_to_specialist`` returned. The
    Executive and the Committee read this field and nothing else.

    ``specialist_result`` and ``calculations`` are ``None`` for specialists on
    the legacy path. ``frame`` is ``None`` whenever the gateway was not called,
    and ``diagnostics`` then says why. See the module docstring for what this
    type does and does not establish.
    """

    specialist: str
    narrative: str
    specialist_result: SpecialistResult | None = None
    calculations: SpecialistCalculations | None = None
    frame: CorrelationFrame | None = None
    diagnostics: tuple[str, ...] = ()
