"""Typed contract for what a specialist hands back to the Executive.

Today the boundary is a plain string (``BaseAgent.analyze -> str``,
``agents/base.py``). Prose is all the Executive gets, so a figure quoted from a
board deck, a number the model computed in its head, and an opinion are
indistinguishable by the time they reach synthesis. This module defines the
structure that fixes that.

**Nothing here is called from production yet.** No existing file imports it. It
ships as schema + parser + renderer so it can be reviewed and tested on its own,
and wired in behind a compatibility renderer in a later slice.

Three rules shape the whole design:

1. **Missing provenance stays missing.** Optional fields default to ``None``,
   never to a placeholder. The retrieval layer cannot supply page / sheet /
   cell / URL / retrieval-date today (PDF text is flattened across pages before
   chunking — ``knowledge/loader.py``), so those fields exist but are inert. A
   ``None`` here means "not available".

2. **LLM output is not calculation authority** (ADR 0001). A model may *state* a
   number; that is recorded as ``model_stated_result`` and is never promoted to
   an authoritative result. ``verified_result`` is reserved for a deterministic
   calculation authority that does not exist yet, the tool schema does not
   expose it, the parser overwrites it and reports the attempt, and — because
   no authority exists yet — the model layer refuses to construct a verified
   figure at all, by any route that runs validation.

3. **Every model is immutable, and every boundary re-validates.**
   ``frozen=True`` with tuple collections; a ``model_copy`` override on
   ``_ContractModel`` (pydantic's ``mode="after"`` validators run at
   construction only, and ``frozen=True`` alone stops ``__setattr__`` but not
   ``model_copy(update=...)``, which writes fields unvalidated); and
   ``revalidate_instances="always"``, so a nested model is re-checked when it is
   placed inside a parent rather than trusted.

   Scope, stated honestly: a *top-level* ``model_construct`` or
   ``object.__setattr__`` still bypasses pydantic, as it does for any model —
   this is not a same-process security boundary. Rule 2 is what holds
   regardless, because it leaves a forged verified figure with no legitimate
   value to forge. Freezing matches how the codebase already treats
   trust-boundary objects (``FeatureSpec`` in ``providers/feature_gate.py`` is a
   frozen dataclass for the same reason).
"""
from __future__ import annotations

import json
import logging
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

# What kind of statement a claim is. The Executive needs this to weigh claims
# differently: a source_fact can be checked, an assessment cannot, an
# unsupported claim should be treated as a lead rather than an input, and a
# conflict needs resolving before anything downstream depends on it.
ClaimType = Literal[
    "source_fact",
    "derived_calculation",
    "assessment",
    "unsupported",
    "conflict",
]

# WHO stands behind a claim, kept deliberately separate from ClaimType. "The
# promoter says revenue is $4M" and "the audited accounts show revenue is $4M"
# are both source_fact; only attribution distinguishes them, and that
# distinction is the entire point of the contract for Moeen.
Attribution = Literal[
    "applicant_asserted",
    "independent_evidence",
    "specialist_judgement",
    "unknown",
]

ConfidenceLevel = Literal["low", "medium", "high"]

# Where a piece of evidence came from. ``none`` is explicit rather than implied
# by an empty list, so "the specialist had nothing" is stated, not inferred.
EvidenceKind = Literal[
    "document",
    "web",
    "company_profile",
    "user_supplied",
    "none",
]

# Lifecycle of a derived figure. Only a deterministic authority may move a
# calculation out of ``unverified``; no such authority exists in this slice, so
# in practice everything parsed from a model today is ``unverified``.
VerificationStatus = Literal[
    "unverified",
    "verified",
    "refuted",
    "not_applicable",
]

# Shared config for every model here. ``frozen`` is the load-bearing half — see
# rule 3 in the module docstring. ``extra="forbid"`` makes an off-schema field a
# loud failure instead of a silent drop.
# ``revalidate_instances="always"`` matters as much as the other two: without
# it pydantic trusts an already-constructed nested model, so a
# ``model_construct``-forged CalculationProvenance could be wrapped in a Claim
# and pass a fully-validating SpecialistResult(...). With it, every nested model
# is re-checked at each boundary, which is what makes rule 2 true as written.
_CONTRACT_CONFIG = ConfigDict(
    extra="forbid", frozen=True, revalidate_instances="always"
)


class _ContractModel(BaseModel):
    """Base for every model here: frozen, and re-validating on ``model_copy``.

    ``frozen=True`` only blocks ``__setattr__``. Pydantic's
    ``model_copy(update=...)`` writes fields straight onto the new object
    **without running validators**, which would let a caller fabricate
    provenance or a dangling ``conflicts_with`` on a copy. Routing the update
    through ``model_validate`` applies the same rules a constructor would.

    Scope, stated honestly: this makes the *documented* invariants hold for the
    object's lifetime. Combined with ``revalidate_instances="always"`` it also
    catches a forged nested model on its way into a parent. It is still not a
    security boundary against a caller in the same process — a top-level
    ``model_construct`` or ``object.__setattr__`` bypasses pydantic entirely, as
    for any model. What protects the calculation-authority rule from those is
    that ``verified_result`` has no legitimate value in this slice at all (see
    :class:`CalculationProvenance`), not that the field is hard to write.
    """

    model_config = _CONTRACT_CONFIG

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if not update:
            return super().model_copy(deep=deep)
        # Round-trips through validation, so an update cannot write a field
        # the constructor would have rejected. Two knock-on differences from
        # BaseModel worth knowing: the copy is always deep (nested models are
        # rebuilt), and ``model_fields_set`` becomes "all fields" rather than
        # the caller's subset — so do not rely on ``exclude_unset`` after a
        # copy-with-update.
        return type(self).model_validate({**self.model_dump(), **update})


class EvidenceRef(_ContractModel):
    """A pointer to whatever backed a claim.

    Every field except ``kind`` and ``label`` is optional and defaults to
    ``None``. That is a load-bearing decision: the retrieval layer currently
    renders chunks as ``[filename] text`` (``knowledge/retriever.py``) and
    carries no page, sheet, cell, or URL. Defaulting any of those to a
    plausible-looking value would manufacture provenance the system does not
    have — the exact failure this contract exists to prevent.

    ``label`` and ``filename`` are, on the parse path, **model-asserted**: they
    are what the specialist says it saw, not something this module verified. The
    structured fields below are stripped from model output entirely; these two
    cannot be, because they are the only handle a human has for tracing a claim.
    Cross-checking them against the chunk set actually retrieved needs the
    retrieval result in scope, which arrives with the wiring slice.
    """

    kind: EvidenceKind
    # What the specialist reports having seen, e.g. "[Q3-board-deck.pdf]".
    label: str

    # --- supplied by the retrieval layer today ---------------------------
    filename: str | None = None
    # Present in chunk metadata (knowledge/loader.py) but not currently rendered
    # into the text the specialist sees, so expect None until that is surfaced.
    chunk_index: int | None = None

    # --- declared, NOT available today -----------------------------------
    # Never populated from model output; see _MODEL_FORBIDDEN_EVIDENCE_FIELDS.
    page: int | None = None
    sheet: str | None = None
    cell_range: str | None = None
    url: str | None = None
    retrieved_at: str | None = None

    # System-authored explanation of why a field above is None — e.g. "page
    # unavailable: PDF pages are flattened before chunking". NOT a model field:
    # it is stripped from model output, because a model writing free text here
    # would route around the stripping of the structured fields above.
    provenance_note: str | None = None


class CalculationProvenance(_ContractModel):
    """How a derived figure was arrived at, and whether anyone checked it.

    The split between ``model_stated_result`` and ``verified_result`` is the
    mechanism that keeps ADR 0001 ("LLM output is not calculation authority")
    true as a property of the data rather than a hope about the prompt:

    * ``model_stated_result`` — what the model said the answer is. Recorded so
      it can be checked and so a reader can see what was claimed. Never
      authoritative.
    * ``verified_result`` — what a deterministic calculation authority computed.
      No such authority exists in this slice, so this stays ``None``.

    A reader that wants a trustworthy number reads ``verified_result``. If it is
    ``None``, there is no trustworthy number — which is the honest answer.
    """

    # Human-readable references to what fed the calculation. Free text on
    # purpose: the inputs are often prose from a document, not typed values.
    inputs: tuple[str, ...] = ()
    # The formula or method as stated, e.g. "net burn / net new ARR".
    method: str

    model_stated_result: str | None = None
    verified_result: str | None = None
    verification_status: VerificationStatus = "unverified"

    @model_validator(mode="after")
    def _nothing_may_claim_verification_yet(self) -> CalculationProvenance:
        """Refuse any verified figure: there is no authority that could produce one.

        A lockstep rule — "``verified_result`` and ``verification_status`` must
        agree" — is not enough, and it took three review rounds to see why. It
        constrains the *pair* without constraining *who set it*, so any caller
        that writes both fields together produces an object that passes
        validation and is then indistinguishable from a checked figure. Given
        this slice ships no deterministic calculation engine, every such object
        would be a lie by construction.

        So the rule here is absolute rather than relational: while no authority
        exists, ``verified_result`` is always ``None`` and
        ``verification_status`` is never ``'verified'``. That is unforgeable
        because it depends on no provenance marker a caller could supply.

        The slice that introduces a real calculation authority relaxes this to
        a single sanctioned constructor (e.g. ``CalculationProvenance.verified(
        ..., computed_by=<authority>)``) — and until then the fields exist only
        to declare the eventual shape, so downstream readers can already be
        written against it.
        """
        if self.verified_result is not None:
            raise ValueError(
                "verified_result cannot be set: no deterministic calculation "
                "authority exists in this slice, so any value here would be a "
                "model's own arithmetic presented as a checked figure "
                "(ADR 0001). Record it as model_stated_result instead."
            )
        if self.verification_status == "verified":
            raise ValueError(
                "verification_status cannot be 'verified': nothing in this "
                "slice is able to verify a calculation."
            )
        return self


class Claim(_ContractModel):
    """One statement a specialist is making, with whatever backs it."""

    # Unique within ONE SpecialistResult and nowhere else. Deliberately a plain
    # string chosen by the emitter (e.g. "c1"): conflict references need to
    # survive reordering and filtering, which positional indices do not, but a
    # global identity scheme would drag in registries and storage this slice has
    # no use for.
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    claim_type: ClaimType
    evidence: tuple[EvidenceRef, ...] = ()
    attribution: Attribution = "unknown"
    confidence: ConfidenceLevel | None = None
    calculation: CalculationProvenance | None = None
    # Other claim_ids in the same result that this claim contradicts.
    # Cross-result references are out of scope; SpecialistResult validates that
    # every id here resolves locally.
    conflicts_with: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _calculation_matches_claim_type(self) -> Claim:
        """A derived_calculation must show its work; nothing else may pretend to.

        Both directions matter. A derived_calculation without provenance is an
        unaudited number wearing a trustworthy label, and a non-calculation
        carrying CalculationProvenance would let a plain assertion inherit the
        credibility of a computed one.
        """
        if self.claim_type == "derived_calculation" and self.calculation is None:
            raise ValueError(
                "claim_type 'derived_calculation' requires a calculation "
                "provenance (inputs + method); a derived figure with no shown "
                "work cannot be audited."
            )
        if self.claim_type != "derived_calculation" and self.calculation is not None:
            raise ValueError(
                f"claim_type {self.claim_type!r} must not carry calculation "
                "provenance; only 'derived_calculation' may."
            )
        return self

    @model_validator(mode="after")
    def _no_self_conflict(self) -> Claim:
        if self.claim_id in self.conflicts_with:
            raise ValueError(f"claim {self.claim_id!r} cannot conflict with itself")
        return self


class SpecialistResult(_ContractModel):
    """What one specialist returns for one consultation.

    ``narrative`` is required and is the compatibility surface: every existing
    consumer of the current plain-string boundary keeps working by reading it
    (see :func:`render_for_executive`). ``claims`` is additive on top.
    """

    specialist: str = Field(min_length=1)
    # Prose for the Executive. Required even when claims are present — the
    # synthesis path consumes text today, and making this optional would force
    # every call site to change at once.
    narrative: str
    claims: tuple[Claim, ...] = ()
    model: str = ""

    # True when structured output was unavailable, unusable, or only partly
    # usable. Surfaced rather than silently absorbed, matching the
    # capability-honest handling the provider layer already uses for dropped
    # features: a caller must be able to tell "the specialist made no claims"
    # apart from "claims were made and this module could not read them".
    degraded: bool = False
    degraded_reason: str | None = None

    @model_validator(mode="after")
    def _claim_ids_unique(self) -> SpecialistResult:
        seen: set[str] = set()
        for claim in self.claims:
            if claim.claim_id in seen:
                raise ValueError(f"duplicate claim_id {claim.claim_id!r}")
            seen.add(claim.claim_id)
        return self

    @model_validator(mode="after")
    def _conflicts_resolve_locally(self) -> SpecialistResult:
        """Every ``conflicts_with`` target must exist in this result.

        A dangling reference is worse than no reference: it looks like a
        recorded contradiction while pointing at nothing, so a reader cannot
        tell whether the conflicting claim was dropped or never existed.
        """
        known = {claim.claim_id for claim in self.claims}
        for claim in self.claims:
            unknown = [ref for ref in claim.conflicts_with if ref not in known]
            if unknown:
                raise ValueError(
                    f"claim {claim.claim_id!r} references unknown claim_id(s) "
                    f"{sorted(unknown)}; conflicts_with must resolve within the "
                    "same SpecialistResult"
                )
        return self

    @model_validator(mode="after")
    def _degraded_reason_accompanies_degraded(self) -> SpecialistResult:
        if self.degraded and not self.degraded_reason:
            raise ValueError("degraded=True requires a degraded_reason")
        return self


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

# Anthropic-shape tool definition the specialist would be given to emit a
# structured result. NOT registered or called anywhere in this slice.
#
# Note what is absent: ``verified_result`` and ``verification_status`` are not
# in this schema at all. That is the strongest available guarantee for ADR 0001
# — the model cannot assert a verified figure because the wire format gives it
# no field to assert it in. The parser enforces the same rule again as defence
# in depth.
#
# Properties are declared in sorted order so the serialized tool block is byte-
# stable across processes, which is what prompt caching keys on.
#
# This is a plain module-level dict and is therefore mutable by any importer.
# Harmless while unwired; before the wiring slice puts it inside a
# ``cache_control`` block, hand callers a deep copy (or freeze it), because an
# in-place reorder by one consumer would invalidate the cached prefix for every
# subsequent request.
EMIT_SPECIALIST_RESULT_TOOL: dict[str, Any] = {
    "name": "emit_specialist_result",
    "description": (
        "Return your analysis as structured claims plus a prose narrative. "
        "Separate what a document states from what you calculated and from "
        "what you judge. Never present a figure you computed as a sourced "
        "fact. If you have no evidence for a statement, mark it 'unsupported' "
        "rather than omitting it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "description": "Structured claims. May be empty.",
                "items": {
                    "type": "object",
                    "properties": {
                        "attribution": {
                            "type": "string",
                            "enum": [
                                "applicant_asserted",
                                "independent_evidence",
                                "specialist_judgement",
                                "unknown",
                            ],
                            "description": (
                                "Who stands behind this claim. Use "
                                "'applicant_asserted' for anything the subject "
                                "of the analysis says about itself."
                            ),
                        },
                        "calculation": {
                            "type": "object",
                            "description": (
                                "Required for claim_type 'derived_calculation', "
                                "forbidden otherwise. Show the inputs and the "
                                "method. Any figure you state here is recorded "
                                "as unverified."
                            ),
                            "properties": {
                                "inputs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "What fed the calculation.",
                                },
                                "method": {
                                    "type": "string",
                                    "description": "Formula or method used.",
                                },
                                "model_stated_result": {
                                    "type": "string",
                                    "description": (
                                        "The result you computed. This is "
                                        "recorded as UNVERIFIED and is not "
                                        "treated as authoritative."
                                    ),
                                },
                            },
                            "required": ["method"],
                        },
                        "claim_id": {
                            "type": "string",
                            "description": (
                                "Short id unique within this result, e.g. 'c1'. "
                                "Used by conflicts_with."
                            ),
                        },
                        "claim_type": {
                            "type": "string",
                            "enum": [
                                "source_fact",
                                "derived_calculation",
                                "assessment",
                                "unsupported",
                                "conflict",
                            ],
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                        "conflicts_with": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "claim_ids in THIS result that this claim "
                                "contradicts."
                            ),
                        },
                        "evidence": {
                            "type": "array",
                            "description": (
                                "What backs this claim. Give the source label "
                                "exactly as you saw it. Omit any field you do "
                                "not actually know — never guess a page number, "
                                "sheet, cell, URL, or date."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "filename": {"type": "string"},
                                    "kind": {
                                        "type": "string",
                                        "enum": [
                                            "document",
                                            "web",
                                            "company_profile",
                                            "user_supplied",
                                            "none",
                                        ],
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": (
                                            "Source label verbatim as it "
                                            "appeared to you."
                                        ),
                                    },
                                },
                                "required": ["kind", "label"],
                            },
                        },
                        "text": {"type": "string"},
                    },
                    "required": ["claim_id", "text", "claim_type"],
                },
            },
            "narrative": {
                "type": "string",
                "description": (
                    "Your analysis in prose, as you would normally answer. "
                    "Always required."
                ),
            },
        },
        "required": ["narrative"],
    },
}


def emit_specialist_result_tool() -> dict[str, Any]:
    """A fresh, independent copy of the tool schema, safe to hand to a provider.

    Use this rather than the module constant at any call site. The constant is a
    plain nested dict, so the usual idiom for tagging a tool —
    ``{**TOOL, "cache_control": ...}`` (see ``orchestrator/executive.py``) — is a
    *shallow* copy that still shares ``input_schema`` by reference. One caller
    reordering or editing that nested dict would change the schema every other
    caller sees, and once this tool lands inside a ``cache_control`` block a
    reordering also invalidates the cached prefix for every subsequent request.

    A deep copy per call is the smallest fix that needs no discipline from
    callers. The cost is negligible next to the provider round trip it precedes,
    and the constant stays exported so existing tests can assert against the
    canonical template.
    """
    return deepcopy(EMIT_SPECIALIST_RESULT_TOOL)


# Fields the model is never allowed to populate, even if it invents them. The
# tool schema above omits them, so reaching this list means the model went
# off-schema; the parser drops them rather than trusting them.
_MODEL_FORBIDDEN_CALCULATION_FIELDS = ("verified_result", "verification_status")

# Evidence fields the model may not populate. The first six are provenance the
# retrieval layer cannot supply — a model cannot know a page it was never shown,
# so anything it offers is invention. ``provenance_note`` is here for a
# different reason: it is a system-authored explanation of why a field is
# missing, and leaving it writable would give the model one free-text field in
# which to restate every datum the other six entries strip.
_MODEL_FORBIDDEN_EVIDENCE_FIELDS = (
    "page",
    "sheet",
    "cell_range",
    "url",
    "retrieved_at",
    "chunk_index",
    "provenance_note",
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _blocks(message: Any) -> list[Any]:
    """Content blocks of a provider message, as a concrete list.

    Materialized once and reused, for two reasons. A non-list ``content`` (a
    scalar from a provider shim) would otherwise raise ``TypeError`` out of a
    ``for`` loop, and a lazy/generator ``content`` would be exhausted by the
    first pass — making the tool_use block invisible to the second and
    producing a "no tool block" degradation for a response that had one.
    """
    raw = getattr(message, "content", None)
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, (str, bytes, dict)):
        # Iterable, but not a block sequence. A plain-string ``content`` is the
        # OpenAI message shape; iterating it would walk it character by
        # character and match nothing. ``_message_text`` reads it instead.
        return []
    try:
        return list(raw)
    except TypeError:
        return []


def _block_field(block: Any, name: str, default: Any = "") -> Any:
    """Read one field from a content block, object- or dict-shaped.

    ``getattr`` alone returns the default for a ``dict``, which made a
    dict-shaped message (raw provider JSON, a ``model_dump()``, a replayed
    cache entry) completely invisible: every block failed the ``type`` check,
    so the narrative AND every claim were dropped while the reason said the
    model never called the tool. Today's providers hand us object blocks
    (``providers/translator.py``), so this is tolerance for the shapes this
    function already claims to accept rather than a live fix.
    """
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _message_text(message: Any, blocks: list[Any]) -> tuple[str, list[str]]:
    """The prose of a provider message, whatever shape it arrives in.

    Handles both the Anthropic block list and a plain-string ``content`` (the
    OpenAI shape). Without the string case the narrative would be silently
    discarded for any caller that has not been through the Anthropic-shape
    translator — losing the specialist's entire answer and reporting it as "no
    tool block", which is both lossy and misleading.

    Joins every text block rather than taking only the first (which is what the
    current boundary in ``agents/base.py`` does), so a degraded result keeps all
    the prose the specialist actually produced.
    """
    raw = getattr(message, "content", None)
    if isinstance(raw, str):
        return raw, []

    chunks: list[str] = []
    unreadable = 0
    for block in blocks:
        # Guarded per block: attribute access on a provider object can itself
        # raise, and one hostile block must not cost us the prose in all the
        # others — losing the whole narrative is the worst available outcome.
        # Counted, not merely skipped: prose we could not read is lost
        # structure like any other, and must be reported.
        try:
            if _block_field(block, "type") != "text":
                continue
            text = _block_field(block, "text")
        except Exception:  # noqa: BLE001 - hostile block object
            unreadable += 1
            continue
        if isinstance(text, str):
            if text:
                chunks.append(text)
        else:
            # A text block whose ``.text`` is not a string (a list or int from a
            # provider shim) is prose we could not read — the same loss as a
            # block that raised, and it must be counted the same way.
            unreadable += 1
    problems = (
        [f"{unreadable} content block(s) could not be read"] if unreadable else []
    )
    return "\n\n".join(chunks), problems


def _tool_payloads(blocks: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """All ``emit_specialist_result`` payloads, plus reasons any were unusable.

    Returns every matching block rather than only the first so a model that
    emits two cannot have the second silently dropped. The reasons list
    distinguishes "no tool block at all" from "a tool block arrived and could
    not be read" — collapsing those two into one message tells an operator
    debugging a truncating provider that the model never called the tool.
    """
    payloads: list[dict[str, Any]] = []
    problems: list[str] = []
    for block in blocks:
        try:
            if _block_field(block, "type") != "tool_use":
                continue
            if _block_field(block, "name") != EMIT_SPECIALIST_RESULT_TOOL["name"]:
                continue
            payload = _block_field(block, "input", None)
        except Exception as exc:  # noqa: BLE001 - hostile block object
            problems.append(f"unreadable content block ({type(exc).__name__})")
            continue

        if isinstance(payload, dict):
            payloads.append(payload)
        elif isinstance(payload, str):
            # Some backends hand back the arguments still JSON-encoded.
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                problems.append("tool payload was a string but not valid JSON")
                continue
            if isinstance(decoded, dict):
                payloads.append(decoded)
            else:
                problems.append(
                    f"tool payload JSON was {type(decoded).__name__}, expected object"
                )
        else:
            problems.append(
                f"tool payload was {type(payload).__name__}, expected object"
            )
    return payloads, problems


def _scrub_evidence(raw: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Build evidence dicts, dropping provenance the model cannot know.

    Also reports what was lost. ``_scrub_evidence`` returning an empty list for
    a malformed input would otherwise be indistinguishable from "this claim had
    no evidence" — which would let a claim marked ``independent_evidence`` reach
    the Executive with zero refs and no sign that any were dropped.
    """
    problems: list[str] = []
    if raw is None:
        return [], problems
    if not isinstance(raw, list):
        return [], [f"evidence was {type(raw).__name__}, expected array"]

    out: list[dict[str, Any]] = []
    dropped = 0
    invented: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            dropped += 1
            continue
        invented.update(k for k in item if k in _MODEL_FORBIDDEN_EVIDENCE_FIELDS)
        out.append({
            k: v for k, v in item.items()
            if k not in _MODEL_FORBIDDEN_EVIDENCE_FIELDS
        })
    if dropped:
        problems.append(f"{dropped} evidence entr(y/ies) were not objects")
    if invented:
        # A model asserting provenance it was never shown is worth surfacing,
        # not just silently correcting: stripping it keeps the data honest, but
        # only a signal makes the behaviour visible in production.
        problems.append(
            "discarded model-asserted provenance: " + ", ".join(sorted(invented))
        )
    return out, problems


def _scrub_calculation(raw: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Build a calculation dict that cannot claim verification.

    Anything the model says about ``verified_result`` / ``verification_status``
    is discarded here — not merely ignored downstream — so a model that invents
    those fields cannot launder a guess into a checked figure.

    Reports problems for the same reason ``_scrub_evidence`` does: returning a
    bare ``None`` for a malformed payload makes "no calculation" and "a
    calculation I could not read" identical to the caller. Note the asymmetry
    that motivates this — a *well-formed* calculation on a non-derived claim
    fails loudly in ``Claim._calculation_matches_claim_type``, so without this a
    malformed one would be treated more leniently than a valid one.
    """
    if raw is None:
        return None, []
    if not isinstance(raw, dict):
        return None, [f"calculation was {type(raw).__name__}, expected object"]
    problems: list[str] = []
    invented = [k for k in raw if k in _MODEL_FORBIDDEN_CALCULATION_FIELDS]
    if invented:
        problems.append(
            "discarded model-asserted verification fields: " + ", ".join(sorted(invented))
        )
    cleaned = {
        k: v for k, v in raw.items()
        if k not in _MODEL_FORBIDDEN_CALCULATION_FIELDS
    }
    cleaned["verification_status"] = "unverified"
    cleaned["verified_result"] = None
    return cleaned, problems


def _validation_summary(exc: ValidationError) -> str:
    """Field paths and a count — never the offending values.

    ``degraded_reason`` is a plain field that will be logged, persisted, and
    possibly surfaced in a UI. Pydantic's ``str(exc)`` embeds ``input_value``,
    which here is model output derived from company documents, so the full error
    goes to the log and only the shape of the failure goes in the field.
    """
    paths: set[str] = set()
    for err in exc.errors():
        loc = list(err["loc"])
        # For extra_forbidden the LAST path component IS the model-supplied key
        # — arbitrary model-authored text. Substituting a constant keeps the
        # useful part (which claim, which sub-object) without echoing content.
        if err.get("type") == "extra_forbidden" and loc:
            loc[-1] = "<extra>"
        # A model-level (``mode="after"``) validator reports an empty ``loc``,
        # which would render as a dangling "at: " with nothing after it. Those
        # are the cross-claim rules — duplicate ids, dangling conflicts — so
        # name the object rather than pointing at nothing.
        paths.add(".".join(_safe_path_part(p) for p in loc) if loc else "<result>")

    ordered = sorted(paths)
    shown = ", ".join(ordered[:5])
    if len(ordered) > 5:
        shown += f", … (+{len(ordered) - 5} more)"
    return f"{exc.error_count()} validation error(s) at: {shown}"


# Path components are bounded and control-stripped: this string is logged and
# persisted, and a model-influenced component containing a newline could
# otherwise forge log lines (e.g. a fake "verification passed" record).
_MAX_PATH_PART_CHARS = 40


# Every individual contributor to a reason is already bounded; this bounds the
# aggregate. Without it a payload with hundreds of distinct per-claim failures
# produces a reason of unbounded length — and this field is logged, persisted,
# and surfaced per consultation.
_MAX_REASON_SEGMENTS = 8
_MAX_REASON_CHARS = 400


def _join(problems: list[str]) -> str:
    """One bounded reason string, first-occurrence order, no repeats.

    The same category can fire once per claim, and a sentence repeated five
    times carries no more information than one.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for problem in problems:
        if problem and problem not in seen:
            seen.add(problem)
            unique.append(problem)
    shown = unique[:_MAX_REASON_SEGMENTS]
    if len(unique) > _MAX_REASON_SEGMENTS:
        shown.append(f"… (+{len(unique) - _MAX_REASON_SEGMENTS} more)")
    joined = "; ".join(shown)
    if len(joined) > _MAX_REASON_CHARS:
        joined = joined[:_MAX_REASON_CHARS] + "…"
    return joined


def _safe_path_part(part: Any) -> str:
    try:
        text = str(part)
    except Exception:  # noqa: BLE001 - __str__ can raise on a hostile object
        return "<unprintable>"
    # Category-based, not ``ch >= " "``: that comparison is on codepoints, so it
    # keeps U+0085 NEL, U+2028/U+2029 (which ``str.splitlines`` treats as line
    # breaks — enough to forge a log line), U+009B (C1 CSI, an ANSI escape
    # introducer for anyone tailing logs), and U+202E RLO (display reordering in
    # any UI that renders this field). ``Cs`` (lone surrogates, reachable via
    # the JSON-string payload path) is stripped too: those raise on
    # ``.encode("utf-8")``, so one would crash whoever persists the result.
    text = "".join(
        " " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp", "Cs") else ch
        for ch in text
    )
    if len(text) > _MAX_PATH_PART_CHARS:
        text = text[:_MAX_PATH_PART_CHARS] + "…"
    return text


# Keys the tool payload is allowed to carry at its top level. Anything else is
# structure this module did not read — see the `unknown key` problem below.
_KNOWN_PAYLOAD_KEYS = frozenset({"narrative", "claims"})

# Stand-in when a caller passes an unusable specialist name. The parser's whole
# job is to not raise, so it must be able to build a degraded result even when
# its own arguments are wrong.
_UNKNOWN_SPECIALIST = "unknown_specialist"


def _degraded(
    specialist: str, narrative: str, reason: str, model: str
) -> SpecialistResult:
    """A result that carries prose and says plainly that structure was lost.

    Claims are always empty here. Inventing even one claim from unstructured
    text would put a fabricated provenance record into the very contract built
    to prevent that.

    Coerces its own arguments rather than validating them. This is the parser's
    last-resort path, so it must not be able to fail: an unusable ``specialist``
    would otherwise raise here, be caught by the caller's outer handler, and
    raise again from the retry — a double fault that escapes the "never raises"
    contract precisely when something has already gone wrong.
    """
    return SpecialistResult(
        specialist=specialist if isinstance(specialist, str) and specialist.strip()
        else _UNKNOWN_SPECIALIST,
        narrative=narrative if isinstance(narrative, str) else "",
        claims=(),
        model=model if isinstance(model, str) else "",
        degraded=True,
        degraded_reason=reason or "structured output unavailable",
    )


def parse_specialist_result(
    message: Any,
    *,
    specialist: str,
    model: str = "",
) -> SpecialistResult:
    """Turn a provider message into a :class:`SpecialistResult`.

    Never raises for bad model output. A specialist that returns something
    unexpected must not take down the Executive's tool loop, so every failure
    path degrades to prose plus an explicit reason.

    Degrades when the tool block is missing (the expected case on backends that
    silently drop ``tool_choice``), when its payload is unusable, when
    validation fails, **and** when structure was partially lost — a claim or an
    evidence entry that could not be read is reported, never quietly dropped.
    """
    # Everything, including reading the message's own shape, runs under the
    # guard: a provider object can raise on attribute access, and that is bad
    # model output rather than a caller bug, so it must degrade like any other.
    narrative_text = ""
    try:
        blocks = _blocks(message)
        narrative_text, text_problems = _message_text(message, blocks)
        payloads, problems = _tool_payloads(blocks)
        problems = text_problems + problems

        if not payloads:
            # Through ``_join`` like every other site: this branch fires when
            # the response is MOST broken (e.g. 500 unusable tool blocks), so
            # leaving it as a raw join made the one unbounded reason the one
            # most likely to be produced.
            reason = _join(
                problems or ["no emit_specialist_result tool_use block in response"]
            )
            return _degraded(specialist, narrative_text, reason, model)

        if len(payloads) > 1:
            problems.append(
                f"{len(payloads)} emit_specialist_result blocks; used the first"
            )
        payload = payloads[0]

        # Claim- and evidence-level extra keys are caught by extra="forbid",
        # but the top level of the payload is read key-by-key, so an
        # unrecognized one would be dropped with no trace. That is the worst
        # available failure: a model that emits its findings under "Claims" (a
        # capitalisation slip, or an instruction planted in an indexed
        # document) would hand the Executive claims=() with degraded=False,
        # which reads as "the specialist genuinely made no claims".
        unknown_keys = set(payload) - _KNOWN_PAYLOAD_KEYS
        if unknown_keys:
            # The COUNT, never the keys. A key is model-authored text, and the
            # claim-level path already refuses to echo one (``<extra>`` in
            # ``_validation_summary``) — echoing it here would reopen that hole
            # on the one field the module says is logged, persisted and
            # UI-surfaced. Sanitising is not enough either: `;` is the segment
            # separator (a key can forge a second reason), and a lone surrogate
            # from the JSON-string payload path would make the whole field
            # un-encodable downstream.
            problems.append(
                f"{len(unknown_keys)} unrecognized top-level payload key(s)"
            )

        narrative = payload.get("narrative")
        if not isinstance(narrative, str) or not narrative.strip():
            # Fall back to the message's own text so the specialist's work is
            # not thrown away just because the narrative field was malformed.
            # Reported in every case, absence included: ``narrative`` is
            # `required` in the tool schema, so a missing or null one means the
            # specialist broke the contract, and a consumer needs to know the
            # prose it received came from a fallback.
            problems.append(
                "narrative field was missing, null, empty, or not a string"
            )
            narrative = narrative_text
        if not narrative:
            # Through ``_join`` with the accumulated problems, like the branch
            # above: returning the bare constant would throw away every loss
            # already detected — including "used the first of 3 tool blocks",
            # which is exactly what an operator needs when a second, well-formed
            # block carried the real answer.
            return _degraded(
                specialist,
                narrative_text,
                _join(problems + ["tool payload carried no usable narrative"]),
                model,
            )

        raw_claims = payload.get("claims")
        claims: list[dict[str, Any]] = []
        if raw_claims is None:
            pass
        elif not isinstance(raw_claims, list):
            problems.append(f"claims was {type(raw_claims).__name__}, expected array")
        else:
            dropped = 0
            for item in raw_claims:
                if not isinstance(item, dict):
                    dropped += 1
                    continue
                claim: dict[str, Any] = {
                    k: v for k, v in item.items()
                    if k not in ("evidence", "calculation")
                }
                evidence, evidence_problems = _scrub_evidence(item.get("evidence"))
                problems.extend(evidence_problems)
                claim["evidence"] = evidence
                calculation, calculation_problems = _scrub_calculation(
                    item.get("calculation")
                )
                problems.extend(calculation_problems)
                if calculation is not None:
                    claim["calculation"] = calculation
                claims.append(claim)
            if dropped:
                problems.append(f"{dropped} claim entr(y/ies) were not objects")

        try:
            result = SpecialistResult(
                specialist=specialist,
                narrative=narrative,
                claims=claims,  # type: ignore[arg-type]
                model=model,
            )
        except ValidationError as exc:
            # Deliberately NOT ``exc``: ``str(ValidationError)`` embeds the raw
            # ``loc`` components and ``input_value``, i.e. exactly the
            # model-authored text the summary exists to keep out of the record.
            # Logging it here would leak company-document content into the log
            # stream and let an injected key forge log lines — undoing the
            # sanitisation applied one line below.
            logger.warning(
                "specialist %s: structured claims failed validation: %s",
                specialist,
                _validation_summary(exc),
            )
            # Everything gathered so far is carried into the reason alongside
            # the validation failure. Reporting only the failure would discard
            # real, already-detected losses — an operator would see one claim's
            # dangling reference and never learn another claim's evidence had
            # been dropped on the way here.
            problems.append(
                f"structured claims failed validation: {_validation_summary(exc)}"
            )
            return _degraded(specialist, narrative, _join(problems), model)

        if problems:
            # Structure was readable but incomplete. Rebuilt through the normal
            # constructor so every validator runs — deliberately not
            # ``model_copy(update=...)``, which in stock pydantic writes fields
            # without validating.
            return SpecialistResult(
                specialist=specialist,
                narrative=narrative,
                claims=result.claims,
                model=model,
                degraded=True,
                degraded_reason=_join(problems),
            )
        return result

    except Exception as exc:  # noqa: BLE001 - nothing may escape into the loop
        logger.exception("specialist %s: result parsing failed", specialist)
        return _degraded(
            specialist,
            narrative_text,
            f"parser failed: {type(exc).__name__}",
            model,
        )


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def render_for_executive(result: SpecialistResult) -> str:
    """Render a result as the string the current boundary returns.

    Verbatim ``narrative``, deliberately. This slice changes no behaviour: the
    Executive, the 27 workflow modules, the committee reviewers, and the MCP
    tool all keep receiving exactly what they receive today. Claims ride along
    in the object for consumers that opt in later.

    Do not start appending rendered claims here — that would change the
    Executive's prompt and, with it, its synthesis behaviour, which is out of
    scope for this slice.
    """
    return result.narrative
