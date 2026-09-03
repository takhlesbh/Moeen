"""The only production module that executes a deterministic calculation.

It mints every authoritative identifier, calls the engine once, and returns an
application-owned record. It holds no state and touches no narrative. Its one
production caller is ``FinanceAgent.analyze_structured`` (Phase 3B2), behind
a default-off setting; a test pins that count.

Rationale and review history: ``architecture/architecture-facts.yaml``.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from openexecutive.calc.authority import issue_calculation_result
from openexecutive.calc.contract import (
    MAX_ID_LEN,
    MAX_REQUESTS_PER_BATCH,
    CalculationBatch,
    CalculationError,
    CalculationRequest,
    CalculationResult,
    Correlation,
    InputEvidenceSummary,
)
from openexecutive.calc.engine import (
    FALLBACK_COMPUTED_AT,
    canonical_computed_at,
    execute_batch,
)
from openexecutive.specialists.calculation_proposal import (
    CalculationProposal,
    is_safe_identifier,
)

logger = logging.getLogger(__name__)

REQUEST_DOMAIN = "openexecutive.calc.request/v1"
"""Domain tag in the minted-id pre-image, so a request id cannot collide with an
identifier minted for any other purpose."""

_SEP = "\x1f"
_GATEWAY_ERROR_CODE = "GATEWAY_UNAVAILABLE"
_INVALID_COMPUTED_AT_CODE = "invalid_computed_at"
_ENGINE_FAILED_DETAIL = "calculation engine unavailable"


@dataclass(frozen=True)
class SpecialistCalculations:
    """One specialist's calculations for one turn. A frozen transport container.

    That is the entire claim. It establishes **no provenance whatsoever**, and
    the list of ways it does not is worth writing out because two earlier
    versions of this docstring asserted a guarantee and a review falsified each:

    * direct construction puts anything in it — a frozen dataclass runs no
      validators;
    * ``pickle`` round-trips it, so a queue replay reconstitutes whatever was
      serialised;
    * **pydantic builds a validator for stdlib dataclasses**, so declaring this
      type as a field on a ``BaseModel`` deserializes an untrusted body straight
      into it. ``model_validate`` not being an attribute of the class proves
      nothing; pydantic never needed it to be;
    * persisting and reloading one attests to nothing about where its records
      came from.

    Trust in the ``CalculationResult`` records inside comes from one place only:
    this module's own code path obtained them from the engine in this process.
    A consumer that receives a ``SpecialistCalculations`` from anywhere else —
    an API body, a database row, a cache, another service — has learned nothing
    about provenance and **must not treat this type alone as evidence**. Phase
    3B2 in particular must re-establish provenance by its own means rather than
    relying on the type.

    ``dropped`` holds bounded reason codes built from a literal and a position,
    never from model text or exception detail.
    """

    specialist: str
    requests: tuple[CalculationRequest, ...] = ()
    results: tuple[CalculationResult, ...] = ()
    dropped: tuple[str, ...] = ()


def _digest(*parts: str) -> str:
    """SHA-256 over a length-prefixed tuple.

    Length-prefixed rather than merely separator-joined: a plain join is
    ambiguous whenever a component can contain the separator, so distinct
    tuples could share a pre-image. With a prefix the framing is
    self-describing and the parse back to components is unique.
    """
    framed = _SEP.join(f"{len(part)}:{part}" for part in parts)
    return hashlib.sha256(framed.encode("utf-8")).hexdigest()


def _canonical_content(proposal: CalculationProposal) -> str:
    """The proposal's content, byte-stable across processes.

    ``sort_keys`` and ``ensure_ascii`` so the same proposal yields the same
    string under any hash seed or locale. The proposal carries no id and no
    correlation, so the whole object is content — nothing has to be excluded.
    """
    return json.dumps(
        proposal.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def mint_request_id(
    *,
    case_id: str,
    run_id: str,
    specialist: str,
    position: int,
    claim_ref: str | None,
    content: str,
) -> str:
    """Derive one request's identifier. The only source of request ids.

    Deterministic in its arguments and nothing else — no clock, no counter, no
    salt — so a replayed turn produces byte-identical ids. A model contributes
    ``content`` and ``claim_ref``; it cannot choose the frame, and ``specialist``
    is inside the pre-image, so two specialists proposing identical arithmetic
    get different identifiers.
    """
    return _digest(
        REQUEST_DOMAIN,
        case_id,
        run_id,
        specialist,
        str(position),
        claim_ref or "",
        content,
    )


def _authorized_claim_ids(known_claim_ids: object) -> frozenset[str]:
    """Normalise the authorization set, refusing anything that is not one.

    This is the only gate deciding whether a computed figure may be attached to
    a specialist's claim, so it is validated rather than trusted to its type
    annotation. Each refusal below is a way the check silently *passes* when it
    should fail:

    * ``str`` and ``bytes`` are ``Collection``s whose ``in`` is **substring**
      containment, so ``known_claim_ids="c1,c9x"`` authorises ``"c9"`` — a claim
      ref nobody issued. Refused rather than coerced, because coercing guesses
      at a delimiter the caller never specified.
    * A ``Mapping`` iterates its keys, so passing ``{"c1": <claim>}`` would work
      by accident and then change meaning the day someone passes
      ``{<claim>: "c1"}``.
    * A generator or one-shot iterator would be consumed by the first membership
      test and read empty afterwards, silently denying every later proposal.
      ``Collection`` requires ``__len__``, which excludes them.
    * An element that is not a valid identifier cannot have been issued by
      anything that respects the same rules, so its presence means the set did
      not come from where the caller thinks it did.
    """
    if isinstance(known_claim_ids, str | bytes | bytearray):
        raise TypeError(
            "known_claim_ids must be a collection of claim ids, not a string: "
            "membership on a string is substring containment, which would "
            "authorise ids nobody issued"
        )
    if isinstance(known_claim_ids, Mapping):
        raise TypeError("known_claim_ids must be a collection of ids, not a mapping")
    if not isinstance(known_claim_ids, Collection):
        raise TypeError(
            "known_claim_ids must be a finite, re-readable collection; a "
            f"{type(known_claim_ids).__name__} would be consumed by the first "
            "membership test"
        )
    # Materialised ONCE, then validated, then returned — the same object
    # throughout. An earlier version validated one read and built the frozenset
    # from a second, so a collection yielding different values per call passed
    # validation with "c1" and authorised "c9": the set that was checked was not
    # the set that authorised.
    snapshot = frozenset(known_claim_ids)
    for element in snapshot:
        if not is_safe_identifier(element, max_length=MAX_ID_LEN):
            raise ValueError(
                "known_claim_ids contains an entry that is not a usable claim id"
            )
    return snapshot


def _fallback_computed_at(value: object) -> tuple[str, str | None]:
    """A timestamp safe to build a record with, and a code if one was substituted.

    Used **only** when constructing ``CALCULATION_UNAVAILABLE`` records after an
    unexpected engine exception — never before normal execution.

    That distinction is the fix. An earlier version canonicalised up front and
    handed the substitute to ``execute_batch``, so the engine's own
    ``INVALID_INPUT`` / ``invalid_computed_at`` record could never fire: a
    refused clock came back ``ARITHMETIC_VERIFIED`` stamped 1970-01-01 with
    empty ``errors``, ``warnings`` and ``dropped`` — a verified audit record at
    a fabricated instant. The original value now reaches the engine, which
    rejects it properly. This helper exists because the fallback path calls
    ``issue_calculation_result`` directly, where an unusable stamp would raise
    from inside the very ``except`` that keeps a failed batch typed.

    Returns ``(stamp, substitution_code)``; the code is non-``None`` exactly
    when a substitution happened, so a fabricated instant is never silent.
    """
    canonical, _ = canonical_computed_at(value)
    if canonical is not None:
        return canonical, None
    return FALLBACK_COMPUTED_AT, _INVALID_COMPUTED_AT_CODE


def _unavailable(
    request: CalculationRequest, computed_at: str, detail: str
) -> CalculationResult:
    """A typed 'did not run' record: real stamp, no fingerprint, no figures."""
    return issue_calculation_result(
        request_id=request.request_id,
        operation=request.operation,
        correlation=request.correlation,
        arithmetic_status="CALCULATION_UNAVAILABLE",
        evidence=InputEvidenceSummary(status="EVIDENCE_UNAVAILABLE"),
        computed_at=computed_at,
        errors=(CalculationError(code=_GATEWAY_ERROR_CODE, detail=detail),),
    )


MAX_PROPOSALS_PER_CALL = MAX_REQUESTS_PER_BATCH
"""The most proposals one specialist call may hand to :func:`execute_proposals`
(the engine's batch limit). Exposed so the caller can gate on it as a declared
condition rather than learn of it by exception; the caller imports this module
only and never the calc package."""


def is_usable_correlation_id(value: object) -> bool:
    """Whether ``value`` may serve as a ``case_id`` / ``run_id`` for a request.

    The same screen ``execute_proposals`` applies to those arguments, exposed so
    the caller can fail closed *before* minting anything rather than learn by
    exception. Kept here so the caller imports this module only — it never
    needs the calc package for the rule.
    """
    return is_safe_identifier(value, max_length=MAX_ID_LEN)


def execute_proposals(
    *,
    specialist: str,
    proposals: Sequence[CalculationProposal],
    case_id: str,
    run_id: str,
    computed_at: str,
    known_claim_ids: Collection[str] = frozenset(),
) -> SpecialistCalculations:
    """Mint identity for each proposal, execute the batch, return the record.

    ``specialist``, ``case_id``, ``run_id`` and ``known_claim_ids`` are
    application-supplied and screened here. The first three reach a persisted
    ``Correlation`` and a log line, so an unusable one is refused rather than
    hashed into something untraceable. ``known_claim_ids`` is the
    **authorization boundary** — it alone decides whether a computed figure may
    be attached to a claim — so it is normalised to a real ``frozenset`` of
    validated strings before any membership test runs.

    ``computed_at`` is passed to the engine **as given**, so an unusable value
    comes back as the engine's own typed ``INVALID_INPUT`` record rather than
    being silently replaced. A substitute is used only when an unexpected engine
    exception forces this module to build records itself, and then the
    substitution is reported.

    Every proposal is **re-validated** here, not merely type-checked:
    ``model_construct`` bypasses pydantic entirely, so an ``isinstance`` test
    alone would let a forged object's unsafe text reach a ``CalculationRequest``.

    No evidence is passed to the engine. No resolver exists in this slice, so
    every result honestly reports ``EVIDENCE_UNAVAILABLE`` and
    ``is_verified_evidence()`` is ``False``.

    One bad proposal costs its own slot and nothing else. A ``claim_ref`` that
    names no known claim drops that proposal — computing it anyway would attach
    a number to nothing — and its siblings still execute.
    """
    for name, value in (
        ("specialist", specialist),
        ("case_id", case_id),
        ("run_id", run_id),
    ):
        if not is_safe_identifier(value, max_length=MAX_ID_LEN):
            raise ValueError(f"{name} is not a usable identifier")
    authorized = _authorized_claim_ids(known_claim_ids)
    if len(proposals) > MAX_REQUESTS_PER_BATCH:
        raise ValueError(
            f"{len(proposals)} proposals exceeds the batch limit of "
            f"{MAX_REQUESTS_PER_BATCH}"
        )

    requests: list[CalculationRequest] = []
    dropped: list[str] = []
    for position, proposal in enumerate(proposals):
        if not isinstance(proposal, CalculationProposal):
            dropped.append(f"proposal_{position}_not_a_proposal")
            continue
        # EVERYTHING derived from the proposal happens inside this guard —
        # revalidation, the claim check, the correlation, the minted id. An
        # earlier version built the ``Correlation`` outside it, so a tampered
        # ``claim_ref`` that satisfied frozenset membership but was not a
        # ``str`` raised uncaught and destroyed every sibling.
        try:
            # ``model_construct`` bypasses every validator, so an object can be
            # an instance of the type and still carry unsafe text. Re-validating
            # here is what makes "unsafe text cannot reach a CalculationRequest"
            # true of the boundary rather than of the happy path.
            checked = CalculationProposal.model_validate(proposal)
            if checked.claim_ref is not None and checked.claim_ref not in authorized:
                dropped.append(f"proposal_{position}_unknown_claim_ref")
                continue
            requests.append(
                CalculationRequest(
                    request_id=mint_request_id(
                        case_id=case_id,
                        run_id=run_id,
                        specialist=specialist,
                        position=position,
                        claim_ref=checked.claim_ref,
                        content=_canonical_content(checked),
                    ),
                    operation=checked.operation,
                    operands=checked.operands,
                    target_unit=checked.target_unit,
                    scale=checked.scale,
                    rounding=checked.rounding,
                    purpose=checked.purpose,
                    correlation=Correlation(
                        specialist=specialist,
                        case_id=case_id,
                        run_id=run_id,
                        claim_id=checked.claim_ref,
                    ),
                )
            )
        except Exception:  # noqa: BLE001 - one bad proposal, never the batch
            # A LITERAL and a position. Nothing derived from the exception, and
            # nothing derived from the proposal. ``logger.exception`` emits a
            # traceback and a pydantic ValidationError renders its ``input``
            # values, so model text would reach the log unbounded; a class name
            # is attacker-influenceable too, since it can carry a newline and
            # forge an audit line. The ``dropped`` code carries the diagnosis.
            logger.warning(
                "calculation gateway: proposal %d could not become a request",
                position,
            )
            dropped.append(f"proposal_{position}_invalid_request")

    if not requests:
        return SpecialistCalculations(specialist=specialist, dropped=tuple(dropped))

    frozen = tuple(requests)
    try:
        # The caller's value, unaltered. If it is unusable the engine returns
        # its own typed INVALID_INPUT / invalid_computed_at record per request,
        # which is strictly more informative than a substituted stamp.
        results = execute_batch(
            CalculationBatch(requests=frozen), computed_at=computed_at
        )
    except Exception:  # noqa: BLE001 - a failed batch stays typed, never empty
        # A literal; see the note on the per-proposal handler above.
        logger.warning("calculation gateway: batch execution failed")
        stamp, substitution = _fallback_computed_at(computed_at)
        results = tuple(
            _unavailable(request, stamp, _ENGINE_FAILED_DETAIL) for request in frozen
        )
        dropped.append("batch_execution_failed")
        if substitution is not None:
            # The fabricated instant is never silent: a reader can tell a record
            # stamped with the epoch because the clock was refused from one
            # computed at the epoch.
            dropped.append(substitution)

    return SpecialistCalculations(
        specialist=specialist,
        requests=frozen,
        results=results,
        dropped=tuple(dropped),
    )
