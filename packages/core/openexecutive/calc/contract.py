"""Typed calculation request and result contracts.

**No arithmetic is performed anywhere in this module or this package.** There is
no engine in Phase 1. What ships here is the vocabulary a Phase 2 engine will be
held to, plus the authority boundary that decides who may say a figure was
checked.

The shape of this contract is a direct response to measured failure. In a
controlled evaluation the CFO specialist identified the right operands and the
right formula for a production calculation — ``52 kg/m2``, ``11 ha = 110,000
m2``, "yield per m2 x area" — and then reported a result off by an order of
magnitude, while a second calculation was returned with the literal placeholder
``XX`` in place of a number. Neither failure is fixable by asking the model to
try harder. Both are fixed by moving execution somewhere deterministic and
making the *record* distinguish four things the old string boundary could not:

* what the model proposed;
* what an engine computed;
* whether the inputs were backed by evidence;
* whether the computed figure conflicts with what the applicant claimed.

Those are four independent axes and this module keeps them independent.

Authority, stated precisely, because it differs by adversary:

* **Against the model: structural.** A model speaks only through
  :class:`CalculationRequest`. That type has no field for an arithmetic status,
  an evidence status, an authority stamp, a fingerprint, a timestamp, or a
  trusted source location, and ``extra="forbid"`` means a payload carrying one
  is *rejected* rather than silently stripped. There is no wire on which a model
  can assert that its own arithmetic was checked.
* **Against a same-process caller: conventional.** Python has no private
  constructor. ``model_construct`` and ``object.__setattr__`` bypass pydantic
  for any model. See :mod:`openexecutive.calc.authority` for exactly what the
  factory boundary does and does not buy.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, model_validator

from openexecutive.calc._model import ContractModel
from openexecutive.calc.numeric import (
    MAX_NUMERIC_STRING_LEN,
    MAX_SCALE,
    NumberFormat,
    canonical_numeric_string,
    parse_numeric,
)
from openexecutive.calc.units import Unit, convertible

_ISO_UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]{1,6})?Z")
"""``computed_at`` must look like an ISO-8601 UTC instant.

Shape only — this contract cannot know whether a clock was right. But an
unvalidated ``str`` accepted "whenever it suits me" as the timestamp on a
durable audit record, and a field documented as a timestamp should at least be
one."""

SCHEMA_VERSION: Literal["calc/1"] = "calc/1"
"""Bumped when the contract changes shape. Part of the fingerprint payload, so a
figure computed under an older contract can never collide with a newer one."""

# --- contract-level resource limits ---------------------------------------

MAX_OPERANDS_PER_REQUEST = 64
"""The largest real case seen is a six-line operating-cost stack; 64 leaves an
order of magnitude of headroom while keeping a malformed batch cheap to reject."""

MAX_REQUESTS_PER_BATCH = 32
"""One specialist turn reconciling a full investment case needs well under ten
calculations. 32 bounds a runaway emitter without constraining real work."""

MAX_EXPRESSION_LEN = 2048
"""Engine-authored, not caller-supplied, and derived from the operand cap: 64
operands rendered with units cannot approach this. A backstop, not a budget."""

MAX_LABEL_LEN = 200
"""A semantic label ("Total Sources") is a phrase. Generous for one, and far too
small to smuggle a document through."""

MAX_PURPOSE_LEN = 500
"""One or two sentences of intent. Bounded for the same reason as the label:
these are free-text fields a model writes, so they are the natural place to try
to push payload."""

MAX_ID_LEN = 64
"""Matches a sha256 hex digest — the longest identifier this contract will ever
legitimately carry."""

MAX_WARNINGS = 32
MAX_ERRORS = 32
"""Per result. A calculation producing more than 32 distinct problems is a
malformed request, and the record should say so rather than grow unboundedly."""
NESTED_OPERATION_DEPTH = 0
"""Version 1 operations do not nest: an operand is always a literal value, never
another calculation. That removes recursion depth as a category of limit rather
than bounding it, and it means chaining is the caller's job and is visible in
the record as separate fingerprinted requests."""


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

OperationId = Literal[
    "add",
    "subtract",
    "multiply",
    "divide",
    "sum_components",
    "percentage_of",
    "percentage_point_difference",
    "ratio",
    "weighted_average",
    "variance",
    "convert_unit",
    "interval_implied_total",
]
"""The closed Version 1 operation set. **Declared, not implemented.**

Closed rather than an expression grammar: a grammar buys arbitrary formulas,
which Version 1 does not want, in exchange for the only meaningful
code-execution surface in the system. Every member here is a named operation
with a fixed arity and a declared dimensional signature, so an unknown
``operation`` is a validation error and never something to parse.

Absent on purpose: IRR, XIRR, NPV, CAGR, debt schedules, and sensitivity
tables. Each needs a cash-flow *series* contract — timing, periodicity, day
count — that this scalar operand model does not carry, and none of them is
needed by the failures this phase exists to address.
"""

NON_COMMUTATIVE_OPERATIONS: frozenset[str] = frozenset(
    {
        "subtract",
        "divide",
        "percentage_of",
        "percentage_point_difference",
        "ratio",
        "variance",
        "convert_unit",
        "interval_implied_total",
    }
)
"""Operations where operand order changes the meaning.

Recorded as data so the Phase 2 engine and the fingerprint reader agree. Note
what this list is *not* used for here: operands are never reordered, for any
operation. Even for a commutative operation, each operand carries a semantic
label and its own provenance, so reordering would silently re-pair labels with
values. Request order is preserved, full stop.
"""


# ---------------------------------------------------------------------------
# Request side — model-proposed
# ---------------------------------------------------------------------------


class SourceHint(ContractModel):
    """A model's *claim* about where an operand came from. Never trusted.

    This is deliberately **not** the ``EvidenceRef`` type in
    ``openexecutive.specialists``,
    and the reason is a real hazard rather than a preference. ``EvidenceRef``
    carries ``page``, ``sheet``, ``cell_range``, ``url``, ``retrieved_at``,
    ``chunk_index``, and ``provenance_note``. Those fields are safe there only
    because one specific function — ``parse_specialist_result`` — strips them
    from model output via ``_MODEL_FORBIDDEN_EVIDENCE_FIELDS``. The *type*
    permits them. Embedding that type in a model-authored request would mean
    any other deserialization path — a test fixture, a queue replay, a future
    second caller — carries a model-asserted spreadsheet cell straight through
    as though the application had confirmed it.

    So this type simply has no field in which to assert those things. A model
    can say which document it thinks a number came from; it cannot say which
    cell, and it cannot mint a timestamp.

    ``retrieval_id_hint`` is named for what it is. It is not a validated
    ``retrieval_id`` and must never be copied into one without being checked
    against the retrieval set for that specific invocation. Application-
    validated binding lives on the result side, in
    :class:`InputEvidenceSummary`, and is written by application code only.
    """

    document_label: str | None = Field(default=None, max_length=MAX_LABEL_LEN)
    filename: str | None = Field(default=None, max_length=MAX_LABEL_LEN)
    retrieval_id_hint: str | None = Field(default=None, max_length=MAX_ID_LEN)
    quoted_text: str | None = Field(default=None, max_length=MAX_LABEL_LEN)


OperandBasis = Literal["applicant_stated", "independently_derived"]
"""Where the figure came from, as the requester understands it.

``applicant_stated`` is the default reading for anything a subject says about
itself. ``independently_derived`` means a third-party or corroborating source.
Neither value asserts that evidence was *validated* — that is a separate axis
the model cannot write.
"""

OperandRole = Literal["input", "stated_comparison"]
"""``stated_comparison`` marks the applicant's own claimed answer, supplied so a
``variance`` operation can compare against it. It is an input to the comparison,
never an input to the arithmetic."""


class Operand(ContractModel):
    """One typed value entering a calculation.

    ``value`` is a canonical string, validated through
    :func:`~openexecutive.calc.numeric.parse_numeric` at construction and stored
    back in canonical form. ``unit`` is mandatory and has no default: a figure
    without a unit is not calculable, and guessing one is the failure mode this
    contract exists to remove.
    """

    operand_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    label: str = Field(min_length=1, max_length=MAX_LABEL_LEN)
    value: str = Field(min_length=1, max_length=MAX_NUMERIC_STRING_LEN)
    unit: Unit
    basis: OperandBasis
    role: OperandRole = "input"
    number_format: NumberFormat = "plain"
    source_hint: SourceHint | None = None

    @model_validator(mode="after")
    def _value_must_parse_and_canonicalise(self) -> Operand:
        parsed = parse_numeric(self.value, self.number_format)
        canonical = canonical_numeric_string(parsed)
        if len(canonical) > MAX_NUMERIC_STRING_LEN:
            # ``max_length`` bounds the INPUT; this validator then rewrites the
            # field, and positional expansion of a small-exponent literal is
            # unbounded relative to what arrived ("1.<58 digits>e-30" is 64 chars
            # in and 90 out). Without this check the object would store a value
            # violating its own declared constraint and would then fail to
            # round-trip through model_validate, model_copy, or a request — data
            # the constructor had accepted.
            raise ValueError(
                f"canonical form of {self.value!r} is {len(canonical)} characters, "
                f"exceeding the {MAX_NUMERIC_STRING_LEN}-character bound; supply "
                "a value whose positional form fits"
            )
        if canonical != self.value:
            # Rebuild rather than mutate: the model is frozen, and routing back
            # through __init__ would re-enter this validator. object.__setattr__
            # is the documented pydantic escape for a normalising validator.
            object.__setattr__(self, "value", canonical)
        return self

    @property
    def decimal_value(self) -> Decimal:
        """Re-parse to an exact ``Decimal``. Reading, not arithmetic."""
        return parse_numeric(self.value, self.number_format)


class Correlation(ContractModel):
    """Where a calculation sits in the wider run. Never part of its identity.

    Every field here is excluded from the fingerprint payload: the same
    calculation over the same inputs must fingerprint identically whether it was
    run for one case or another. See :func:`fingerprint_payload`.
    """

    specialist: str = Field(min_length=1, max_length=MAX_ID_LEN)
    case_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    run_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    claim_id: str | None = Field(default=None, max_length=MAX_ID_LEN)


RoundingMode = Literal["ROUND_HALF_EVEN", "ROUND_HALF_UP", "ROUND_DOWN"]
"""Closed set, mapping onto ``decimal`` rounding constants in Phase 2.
``ROUND_HALF_EVEN`` is the default because it does not accumulate upward bias
across a long schedule."""


class CalculationRequest(ContractModel):
    """What a specialist asks to have calculated. Model-proposed, never trusted.

    The authority rule for this type is simply that it has nowhere to put an
    answer, a status, or a stamp. Combined with ``extra="forbid"``, a request
    that tries to carry one fails validation loudly.
    """

    schema_version: Literal["calc/1"] = SCHEMA_VERSION
    request_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    operation: OperationId
    operands: tuple[Operand, ...]
    target_unit: Unit | None = None
    scale: int = Field(default=2, ge=0, le=MAX_SCALE)
    rounding: RoundingMode = "ROUND_HALF_EVEN"
    purpose: str = Field(min_length=1, max_length=MAX_PURPOSE_LEN)
    correlation: Correlation
    # No request-level ``number_format``: a draft had one and nothing read it,
    # so a request declaring ``comma_thousands`` still rejected "1,234" while a
    # request declaring ``plain`` still got comma parsing from its operands. In
    # a module whose rule is "separators are never inferred", a separator
    # control that looks set and does nothing is the exact silent-magnitude
    # hazard it exists to remove. Each operand declares its own format, which is
    # the only place the value is actually used.

    @model_validator(mode="after")
    def _bounded_and_unique(self) -> CalculationRequest:
        if not self.operands:
            raise ValueError("a calculation request needs at least one operand")
        if len(self.operands) > MAX_OPERANDS_PER_REQUEST:
            raise ValueError(
                f"{len(self.operands)} operands exceeds the limit of "
                f"{MAX_OPERANDS_PER_REQUEST}"
            )
        ids = [o.operand_id for o in self.operands]
        if len(ids) != len(set(ids)):
            raise ValueError("operand_id must be unique within a request")
        return self


class CalculationBatch(ContractModel):
    """A bounded set of requests from one specialist turn."""

    schema_version: Literal["calc/1"] = SCHEMA_VERSION
    requests: tuple[CalculationRequest, ...]

    @model_validator(mode="after")
    def _bounded(self) -> CalculationBatch:
        if len(self.requests) > MAX_REQUESTS_PER_BATCH:
            raise ValueError(
                f"{len(self.requests)} requests exceeds the batch limit of "
                f"{MAX_REQUESTS_PER_BATCH}"
            )
        ids = [r.request_id for r in self.requests]
        if len(ids) != len(set(ids)):
            raise ValueError("request_id must be unique within a batch")
        return self


# ---------------------------------------------------------------------------
# Result side — application-authored
# ---------------------------------------------------------------------------

ArithmeticStatus = Literal[
    "ARITHMETIC_VERIFIED",
    "CALCULATION_UNAVAILABLE",
    "DIVISION_BY_ZERO",
    "UNIT_MISMATCH",
    "RESOURCE_LIMIT_EXCEEDED",
    "INVALID_INPUT",
    "UNSUPPORTED_OPERATION",
]
"""Whether the arithmetic ran, and if not, why.

**``CONFLICT_DETECTED`` is deliberately not a member.** The Council asked for a
decision between a primary status and an orthogonal field; this is the decision
and the reason is that a result is routinely *both*. Recomputing an applicant's
sources and uses gives a correct figure (``ARITHMETIC_VERIFIED``) that also
disagrees with the applicant's stated total (a conflict). Collapsing those into
one enum forces a choice that discards one of the two facts a reviewer needs.
The vocabulary is preserved on the orthogonal axis: see :data:`ConflictClass`,
whose ``CONFLICT_DETECTED`` member carries exactly that meaning.

Note what ``ARITHMETIC_VERIFIED`` does **not** mean. It says an engine executed
the expression. It says nothing about whether the operands were true. That is
:data:`InputEvidenceStatus`, and the two are never merged.
"""

InputEvidenceStatus = Literal[
    "ALL_SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONFLICTING_SOURCES",
    "EVIDENCE_UNAVAILABLE",
]
"""Whether the *inputs* were backed, assigned by application code after checking
each operand's binding. ``EVIDENCE_UNAVAILABLE`` is distinct from
``UNSUPPORTED``: the former means no evidence layer ran, the latter means one
ran and found nothing."""

ConflictClass = Literal[
    "NONE",
    "EXACT_MATCH",
    "WITHIN_TOLERANCE",
    "CONFLICT_DETECTED",
    "ORDER_OF_MAGNITUDE",
    "SIGN_MISMATCH",
]
"""How a computed figure relates to the applicant's stated one.

``ORDER_OF_MAGNITUDE`` is called out separately from a generic conflict because
it is the specific error that survived a full evaluation unnoticed: a stated
572 tonnes against a computed 5,720. A reviewer scanning statuses should be able
to find that class without reading every difference.

``EXACT_MATCH`` and ``NONE`` are not the same thing and the difference is
load-bearing: ``NONE`` means no comparison was made, ``EXACT_MATCH`` means one
was made and the figures agreed to the digit. Collapsing them would make "we
checked and it reconciles" indistinguishable from "we never checked", which is
the distinction a reviewer most needs.
"""


def _validated_numeric_field(value: str | None, field_name: str) -> str | None:
    """Run a result-side numeric string through the same boundary as an operand.

    Without this, every numeric field on the result side is a bare ``str`` with
    only a length bound, and ``result_value="XX"`` is a legal
    ``ARITHMETIC_VERIFIED`` record. ``XX`` is one of the two measured failures
    this package exists to prevent — a specialist returned it in place of a
    number — so allowing it to round-trip through the record, through
    :meth:`CalculationResult.is_verified_evidence`, and into the fingerprint
    payload would defeat the point of shipping the contract first.
    """
    if value is None:
        return None
    parsed = parse_numeric(value)
    canonical = canonical_numeric_string(parsed)
    if len(canonical) > MAX_NUMERIC_STRING_LEN:
        raise ValueError(
            f"{field_name}: canonical form of {value!r} exceeds "
            f"{MAX_NUMERIC_STRING_LEN} characters"
        )
    return canonical


class NormalizedOperand(ContractModel):
    """An operand as the engine actually used it, after any unit conversion."""

    operand_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    label: str = Field(min_length=1, max_length=MAX_LABEL_LEN)
    original_value: str = Field(min_length=1, max_length=MAX_NUMERIC_STRING_LEN)
    original_unit: Unit
    normalized_value: str = Field(min_length=1, max_length=MAX_NUMERIC_STRING_LEN)
    normalized_unit: Unit
    conversion_applied: str | None = Field(default=None, max_length=MAX_LABEL_LEN)
    basis: OperandBasis
    role: OperandRole = "input"

    @model_validator(mode="after")
    def _values_are_real_numbers(self) -> NormalizedOperand:
        # Written out rather than looped through ``getattr``: the package's own
        # test bans invoking ``getattr``. (Writing via ``object.__setattr__``
        # with a variable name, as ``CalculationResult`` does for its numeric
        # fields, is not banned and not equivalent — the names there come from a
        # hardcoded tuple and nothing is *read* dynamically.)
        original = _validated_numeric_field(self.original_value, "original_value")
        if original is not None and original != self.original_value:
            object.__setattr__(self, "original_value", original)
        normalized = _validated_numeric_field(self.normalized_value, "normalized_value")
        if normalized is not None and normalized != self.normalized_value:
            object.__setattr__(self, "normalized_value", normalized)

        # The units must describe a conversion that could actually happen.
        # Without this, "11 hectares normalised to 11 Tunisian dinars, no
        # conversion applied" is a legal record: the dimension predicates exist
        # but nothing calls them, so making dimension part of the type buys
        # nothing for the record. ``convertible`` is a pure predicate — it reads
        # dimensions and computes nothing — so enforcing it here introduces no
        # arithmetic.
        if self.original_unit.code != self.normalized_unit.code:
            ok, note = convertible(self.original_unit, self.normalized_unit)
            if not ok:
                raise ValueError(
                    f"cannot normalise {self.original_unit.code} to "
                    f"{self.normalized_unit.code}: {note}"
                )
            if note is not None and not (self.conversion_applied or "").strip():
                # ``convertible`` returns a note only for explicit_required
                # dimensions (month/year). Reading just ``ok`` and discarding
                # the note is how "annualised without saying how" becomes a
                # recordable fact — the failure the time policy exists to name.
                raise ValueError(
                    f"converting {self.original_unit.code} to "
                    f"{self.normalized_unit.code} requires conversion_applied to "
                    f"state the basis: {note}"
                )
        elif self.normalized_value != self.original_value:
            raise ValueError(
                f"{self.operand_id}: unit is unchanged ({self.original_unit.code}) "
                "but the value changed; a conversion that does not convert units "
                "cannot alter the number"
            )
        return self


# ---------------------------------------------------------------------------
# Why there is no per-operation dimensional rule table here
# ---------------------------------------------------------------------------
#
# A draft of this module carried one: which operations require additively
# compatible inputs, which produce a fixed result dimension, what `multiply`
# composes to. It was removed, and the reason is worth keeping.
#
# It was out of the authorized scope for this phase ("define the types and
# registry required for the later engine", not "implement dimensional
# arithmetic"), and it was wrong in the way that scope existed to prevent.
# Encoding an operation's dimensional signature means encoding its *semantics*,
# and getting `divide` wrong made this package reject the very calculation it
# was built for: 5,720,000 kg / 110,000 m2 = 52 kg/m2 — the yield figure whose
# 10^4 misreading is the measured failure cited throughout this package — was
# refused as "dimension mismatch: mass vs area". Six review findings in one
# round traced to that table, three of them holes it opened rather than closed.
#
# The predicates in `units` are the vocabulary; the engine that knows what each
# operation MEANS is the thing entitled to apply them, and it arrives with its
# own arity declarations and its own tests. What this contract enforces is the
# part that needs no operation semantics: a single operand's own
# `original_unit -> normalized_unit` step must be a conversion that could
# actually happen (see `NormalizedOperand`).


class InputEvidenceSummary(ContractModel):
    """Application-validated evidence binding. Written by application code only.

    This is the counterpart to :class:`SourceHint`: the hint is what the model
    said, this is what the application confirmed. ``bound_operand_ids`` lists
    operands whose evidence was checked against the retrieval set for the
    invocation that produced them; anything absent from that tuple was not
    confirmed, whatever its hint claimed.

    The validator below is load-bearing rather than tidy. Without it a partial
    evidence layer could write ``status="ALL_SUPPORTED"`` while correctly
    listing every operand as unbound, and :meth:`CalculationResult.is_verified_evidence`
    — the single accessor for the strongest claim in this contract — would
    report that record as verified evidence. That is the axis collapse the whole
    module exists to prevent, so the status and the tuples must agree.
    """

    status: InputEvidenceStatus
    bound_operand_ids: tuple[str, ...] = ()
    unbound_operand_ids: tuple[str, ...] = ()
    note: str | None = Field(default=None, max_length=MAX_PURPOSE_LEN)

    @model_validator(mode="after")
    def _status_matches_the_bindings(self) -> InputEvidenceSummary:
        for field_name, ids in (
            ("bound_operand_ids", self.bound_operand_ids),
            ("unbound_operand_ids", self.unbound_operand_ids),
        ):
            if len(ids) > MAX_OPERANDS_PER_REQUEST:
                raise ValueError(
                    f"{field_name} exceeds {MAX_OPERANDS_PER_REQUEST} entries: a "
                    "result cannot bind more operands than a request may carry"
                )
            for oid in ids:
                if not oid or len(oid) > MAX_ID_LEN:
                    raise ValueError(
                        f"{field_name} contains an empty or over-long operand id"
                    )
        overlap = set(self.bound_operand_ids) & set(self.unbound_operand_ids)
        if overlap:
            raise ValueError(
                f"operand(s) {sorted(overlap)} are listed as both bound and "
                "unbound; an operand's evidence was either confirmed or it was not"
            )
        if self.status == "ALL_SUPPORTED":
            if self.unbound_operand_ids:
                raise ValueError(
                    "ALL_SUPPORTED contradicts a non-empty unbound_operand_ids: "
                    "this is the state that would let unconfirmed figures be "
                    "reported as verified evidence"
                )
            if not self.bound_operand_ids:
                raise ValueError(
                    "ALL_SUPPORTED requires a non-empty bound_operand_ids. The "
                    "empty tuple is the zero-argument default — exactly what a "
                    "partial evidence layer that sets the status but not the "
                    "ids produces — and it would report the strongest claim in "
                    "this contract with no operand actually confirmed."
                )
            if len(set(self.bound_operand_ids)) != len(self.bound_operand_ids):
                raise ValueError("bound_operand_ids contains duplicates")
        if self.status == "UNSUPPORTED" and self.bound_operand_ids:
            raise ValueError(
                "UNSUPPORTED contradicts a non-empty bound_operand_ids"
            )
        if self.status == "PARTIALLY_SUPPORTED" and not (
            self.bound_operand_ids and self.unbound_operand_ids
        ):
            raise ValueError(
                "PARTIALLY_SUPPORTED requires both a bound and an unbound "
                "operand; otherwise it is ALL_SUPPORTED or UNSUPPORTED"
            )
        if self.status == "CONFLICTING_SOURCES" and not (
            self.bound_operand_ids or self.unbound_operand_ids
        ):
            raise ValueError(
                "CONFLICTING_SOURCES must name the operand(s) whose sources "
                "disagree; it is the only status that otherwise says nothing "
                "about what conflicted"
            )
        if self.status == "EVIDENCE_UNAVAILABLE" and (
            self.bound_operand_ids or self.unbound_operand_ids
        ):
            raise ValueError(
                "EVIDENCE_UNAVAILABLE means no evidence layer ran, so it cannot "
                "list bindings; use UNSUPPORTED when one ran and found nothing"
            )
        return self


class CalculationError(ContractModel):
    """One typed failure. Free-text is confined to ``detail`` and is bounded."""

    code: str = Field(min_length=1, max_length=MAX_ID_LEN)
    detail: str = Field(min_length=1, max_length=MAX_PURPOSE_LEN)
    operand_id: str | None = Field(default=None, max_length=MAX_ID_LEN)


KNOWN_AUTHORITY_VERSIONS: frozenset[str] = frozenset({"0.1.0-contract", "0.2.0-engine"})
"""The closed set of authority versions that may appear on a result.

``authority_id`` alone was not enough. ``authority_version`` is a *fingerprint
identity field*, and it is the part the factory's own docstring calls "the one
part of the stamp that would otherwise be trivially wrong": leaving it free text
let a replayed contract-phase result be re-stamped ``2.0.0-engine`` through the
ordinary ``model_validate`` path, defeating the whole point of the ``-contract``
suffix. ``0.2.0-engine`` arrived with the engine; ``0.1.0-contract`` is retained
so a result stored during the contract-only phase still validates on reload
rather than becoming unreadable."""

KNOWN_AUTHORITY_IDS: frozenset[str] = frozenset({"openexecutive.calc"})
"""The closed set of identities that may appear on a result.

Declared here rather than in ``authority.py`` so the contract can enforce it
without a circular import. It does not make the stamp unforgeable — nothing in
Python can — but it removes the free-text case, where ``model_validate`` on a
stored or replayed payload would happily accept an authority that never
existed. ``model_validate`` is the ordinary deserialization path, not one of the
two documented escapes, so it is worth closing."""


class ApplicationAuthority(ContractModel):
    """Who computed a result. Application-owned; no model may supply it."""

    authority_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    authority_version: str = Field(min_length=1, max_length=MAX_ID_LEN)

    @model_validator(mode="after")
    def _id_must_be_known(self) -> ApplicationAuthority:
        if self.authority_id not in KNOWN_AUTHORITY_IDS:
            raise ValueError(
                f"unknown authority_id {self.authority_id!r}; expected one of "
                f"{sorted(KNOWN_AUTHORITY_IDS)}"
            )
        if self.authority_version not in KNOWN_AUTHORITY_VERSIONS:
            raise ValueError(
                f"unknown authority_version {self.authority_version!r}; expected "
                f"one of {sorted(KNOWN_AUTHORITY_VERSIONS)}"
            )
        return self


class CalculationResult(ContractModel):
    """The record of one calculation. Constructed by application code only.

    Read the two status axes together and never separately:

    * ``arithmetic_status == "ARITHMETIC_VERIFIED"`` means an engine executed
      the expression.
    * ``evidence.status`` means whether the operands were backed.

    A result may legitimately be ``ARITHMETIC_VERIFIED`` with
    ``evidence.status == "UNSUPPORTED"``. That combination is not a
    contradiction and must not be summarised as "verified": correct arithmetic
    over unsupported inputs is still unsupported investment evidence. There is
    no field on this type that collapses the two, and :meth:`is_verified_evidence`
    is the only place the conjunction is expressed.
    """

    schema_version: Literal["calc/1"] = SCHEMA_VERSION
    request_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    operation: OperationId
    correlation: Correlation

    normalized_operands: tuple[NormalizedOperand, ...] = ()
    expression_executed: str | None = Field(default=None, max_length=MAX_EXPRESSION_LEN)
    exact_result: str | None = Field(default=None, max_length=MAX_NUMERIC_STRING_LEN)
    result_value: str | None = Field(default=None, max_length=MAX_NUMERIC_STRING_LEN)
    result_unit: Unit | None = None
    scale_applied: int | None = Field(default=None, ge=0, le=MAX_SCALE)
    rounding_applied: RoundingMode | None = None

    arithmetic_status: ArithmeticStatus
    evidence: InputEvidenceSummary

    stated_value: str | None = Field(default=None, max_length=MAX_NUMERIC_STRING_LEN)
    absolute_difference: str | None = Field(default=None, max_length=MAX_NUMERIC_STRING_LEN)
    percentage_difference: str | None = Field(default=None, max_length=MAX_NUMERIC_STRING_LEN)
    ratio: str | None = Field(default=None, max_length=MAX_NUMERIC_STRING_LEN)
    conflict: ConflictClass = "NONE"

    warnings: tuple[str, ...] = ()
    errors: tuple[CalculationError, ...] = ()

    authority: ApplicationAuthority
    fingerprint: str | None = Field(default=None, min_length=64, max_length=64)
    computed_at: str = Field(min_length=1, max_length=MAX_ID_LEN)

    @model_validator(mode="after")
    def _status_coherence(self) -> CalculationResult:
        for field_name, raw in (
            ("exact_result", self.exact_result),
            ("result_value", self.result_value),
            ("stated_value", self.stated_value),
            ("absolute_difference", self.absolute_difference),
            ("percentage_difference", self.percentage_difference),
            ("ratio", self.ratio),
        ):
            canonical = _validated_numeric_field(raw, field_name)
            if canonical is not None and canonical != raw:
                object.__setattr__(self, field_name, canonical)
        operand_ids = [o.operand_id for o in self.normalized_operands]
        if len(operand_ids) != len(set(operand_ids)):
            raise ValueError(
                "normalized_operands contains duplicate operand_id(s). A request "
                "enforces uniqueness, so a result carrying duplicates did not "
                "come from one — and it would let a single evidence binding "
                "satisfy the 'every recorded operand is bound' check for two "
                "different values."
            )
        if len(self.normalized_operands) > MAX_OPERANDS_PER_REQUEST:
            raise ValueError(
                f"{len(self.normalized_operands)} normalized operands exceeds "
                f"the request limit of {MAX_OPERANDS_PER_REQUEST}: a result "
                "cannot have consumed more operands than a request may carry"
            )
        if len(self.warnings) > MAX_WARNINGS:
            raise ValueError(f"more than {MAX_WARNINGS} warnings")
        for warning in self.warnings:
            # Capping the count alone bounds nothing that matters: one
            # multi-megabyte warning string reaches a log, a database row and a
            # response body just as effectively as a thousand short ones.
            if len(warning) > MAX_PURPOSE_LEN:
                raise ValueError(
                    f"a warning exceeds {MAX_PURPOSE_LEN} characters; the record "
                    "is meant to be durable and reviewable, not a payload"
                )
        if len(self.errors) > MAX_ERRORS:
            raise ValueError(f"more than {MAX_ERRORS} errors")
        if self.fingerprint is not None and not all(
            c in "0123456789abcdef" for c in self.fingerprint
        ):
            raise ValueError("fingerprint must be lowercase hex (sha256 shape)")

        if self.arithmetic_status == "ARITHMETIC_VERIFIED":
            # A verified result without an executed expression, a value, or the
            # operands it consumed would be a status with nothing behind it —
            # the exact thing the old string boundary allowed. Couple them
            # structurally rather than trusting the engine to be consistent.
            if not self.expression_executed:
                raise ValueError(
                    "ARITHMETIC_VERIFIED requires expression_executed: a status "
                    "with no executed expression asserts a check that did not "
                    "happen"
                )
            if self.result_value is None or self.result_unit is None:
                raise ValueError(
                    "ARITHMETIC_VERIFIED requires result_value and result_unit"
                )
            if not self.normalized_operands:
                raise ValueError(
                    "ARITHMETIC_VERIFIED requires normalized_operands: every "
                    "request carries at least one operand, so a verified result "
                    "recording none did not come from one. Without them the "
                    "fingerprint has no inputs to identify and a reader cannot "
                    "see what was actually computed."
                )
            if self.errors:
                raise ValueError(
                    "ARITHMETIC_VERIFIED cannot carry errors; use a failure status"
                )
        else:
            # A failure status must not also assert that work succeeded. Each
            # field below is a claim the engine got somewhere: a DIVISION_BY_ZERO
            # result carrying "5 / 0", an exact_result and a ratio lets any
            # downstream renderer show a number for a calculation that never ran,
            # and exact_result is the *unrounded answer*, so a UI preferring it
            # would display precisely the figure the status denies exists.
            claimed = [
                name
                for name, present in (
                    ("result_value", self.result_value is not None),
                    ("exact_result", self.exact_result is not None),
                    ("expression_executed", self.expression_executed is not None),
                    ("result_unit", self.result_unit is not None),
                    ("scale_applied", self.scale_applied is not None),
                    ("rounding_applied", self.rounding_applied is not None),
                    ("absolute_difference", self.absolute_difference is not None),
                    ("percentage_difference", self.percentage_difference is not None),
                    ("ratio", self.ratio is not None),
                    ("conflict", self.conflict != "NONE"),
                )
                if present
            ]
            if claimed:
                raise ValueError(
                    f"{self.arithmetic_status} must not carry {', '.join(claimed)}: "
                    "a calculation that did not run has nothing to report"
                )
        named_ids = set(self.evidence.bound_operand_ids) | set(
            self.evidence.unbound_operand_ids
        )
        if not self.normalized_operands and named_ids:
            raise ValueError(
                f"evidence names operand(s) {sorted(named_ids)} but the result "
                "records none; a binding must refer to an operand that exists"
            )
        if self.normalized_operands:
            known = {o.operand_id for o in self.normalized_operands}
            unknown = named_ids - known
            if unknown:
                raise ValueError(
                    f"evidence names operand(s) {sorted(unknown)} that this "
                    "result does not record; a binding must refer to an operand "
                    "that was actually used"
                )
            if self.evidence.status == "ALL_SUPPORTED" and set(
                self.evidence.bound_operand_ids
            ) != known:
                raise ValueError(
                    "ALL_SUPPORTED requires every recorded operand to be bound; "
                    f"unbound: {sorted(known - set(self.evidence.bound_operand_ids))}"
                )
        # ``re.fullmatch`` rather than a pydantic ``pattern``: pydantic compiles
        # patterns with the Rust regex engine, which rejects \A/\Z, and bare
        # ^/$ semantics differ between the two engines. Matching in Python keeps
        # the anchoring unambiguous.
        if not _ISO_UTC_RE.fullmatch(self.computed_at):
            raise ValueError(
                f"computed_at {self.computed_at!r} is not an ISO-8601 UTC "
                "instant (YYYY-MM-DDTHH:MM:SS[.ffffff]Z). Shape only — this "
                "contract cannot know whether a clock was right — but a field "
                "documented as a timestamp should at least be one."
            )
        if self.conflict != "NONE" and self.stated_value is None:
            raise ValueError(
                "a conflict classification requires the stated_value it conflicts with"
            )
        # The reverse is deliberately NOT enforced. A `variance` operation that
        # compares against an applicant figure and finds them in agreement
        # legitimately records the stated_value with conflict left at "NONE" —
        # "we checked and they match" is a result worth keeping, and requiring a
        # conflict class to justify a stated value would erase it.
        return self

    def is_verified_evidence(self) -> bool:
        """True only when the arithmetic ran and the evidence status is ALL_SUPPORTED.

        The single place the two axes are conjoined, so a caller wanting the
        strong claim has to ask for it by name rather than reading
        ``arithmetic_status`` and assuming.

        The strength of the second half rests on
        :class:`InputEvidenceSummary`'s validator, which refuses
        ``ALL_SUPPORTED`` alongside any unbound operand. This method does not
        re-derive binding from ``normalized_operands`` — it trusts the summary
        the evidence layer wrote, which is a real trust boundary and is named
        here rather than papered over.
        """
        return (
            self.arithmetic_status == "ARITHMETIC_VERIFIED"
            and self.evidence.status == "ALL_SUPPORTED"
        )


# ---------------------------------------------------------------------------
# Fingerprint payload (payload only — no hashing in Phase 1)
# ---------------------------------------------------------------------------

FINGERPRINT_INCLUDED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "operation",
    "operands",
    "target_unit",
    "scale",
    "rounding",
    "stated_value",
    "authority_id",
    "authority_version",
)
"""The key set :func:`fingerprint_payload` returns.

Phase 2's engine adds one more, ``parameters``, carrying the operation policies
that were actually consumed — see :data:`FINGERPRINT_OPTIONAL_FIELDS`. Keeping
that out of this tuple and out of the pinning test let the declared contract
drift from the real payload: a Phase 3 consumer reading this list would not know
the policies participate in identity.

Kept honest by a test comparing this tuple against the live payload keys rather
than restating them. A declared field list that can drift from the function it
documents is worse than no list: it reads as a guarantee while silently becoming
a lie the first time a field is added to one and not the other."""

FINGERPRINT_OPTIONAL_FIELDS: tuple[str, ...] = ("parameters",)
"""Keys the engine adds only when they carry meaning.

``parameters`` appears only when an operation actually consumed a policy. It is
omitted otherwise on purpose: an empty ``parameters`` key present on one
calculation and absent on another would separate two identical calculations,
which is the fragmentation the allowlist exists to prevent."""

FINGERPRINT_EXCLUDED_FIELDS: tuple[str, ...] = (
    "request_id",
    "case_id",
    "run_id",
    "specialist",
    "claim_id",
    "computed_at",
    "purpose",
    "label",
    "display",
    "warnings",
)
"""Excluded because they vary without the calculation varying.

Identity is "what was computed", not "who asked or when". Two runs of the same
reconciliation in different cases must fingerprint identically, or the
fingerprint cannot be used to recognise a repeated calculation. ``purpose`` and
``label`` are prose; ``computed_at`` is a clock.
"""


def fingerprint_payload(
    *,
    operation: OperationId,
    normalized_operands: tuple[NormalizedOperand, ...],
    target_unit: Unit | None,
    scale: int,
    rounding: RoundingMode,
    authority: ApplicationAuthority,
    stated_value: str | None = None,
) -> dict[str, Any]:
    """Build the canonical payload a Phase 2 fingerprint will hash.

    Field selection and ordering only — **nothing is hashed here**, because no
    canonical-hash utility exists in this repository to reuse and inventing one
    is Phase 2's job. What Phase 1 owns is the harder half: deciding what
    identity *means*, and pinning it with tests.

    Operand order is preserved exactly as requested, for every operation
    including commutative ones. Reordering would let two requests that pair
    different labels with different provenance collapse onto one fingerprint.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "operands": [
            {
                "value": o.normalized_value,
                "unit": o.normalized_unit.code,
                "basis": o.basis,
                "role": o.role,
            }
            for o in normalized_operands
        ],
        "target_unit": target_unit.code if target_unit is not None else None,
        "scale": scale,
        "rounding": rounding,
        "stated_value": stated_value,
        "authority_id": authority.authority_id,
        "authority_version": authority.authority_version,
    }


def canonical_payload_json(payload: dict[str, Any]) -> str:
    """Deterministic serialization of a fingerprint payload.

    ``sort_keys`` and fixed separators so the same payload produces the same
    bytes in any process; ``ensure_ascii`` so a unit display character can never
    change the encoding. Note that *operand order* is a list and is therefore
    preserved — only mapping keys are sorted.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
