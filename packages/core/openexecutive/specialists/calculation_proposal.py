"""What a specialist asks to have computed. Intent only — no authority.

Rationale and the review history behind each rule live in
``architecture/architecture-facts.yaml`` under ``calc``, not here.
"""
from __future__ import annotations

from typing import Any, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openexecutive.calc.contract import (
    MAX_ID_LEN,
    MAX_LABEL_LEN,
    MAX_OPERANDS_PER_REQUEST,
    MAX_PURPOSE_LEN,
    Operand,
    OperandBasis,
    OperandRole,
    OperationId,
    RoundingMode,
)
from openexecutive.calc.numeric import MAX_SCALE, NumberFormat
from openexecutive.calc.units import Unit


def _is_bounded_printable(value: object, *, max_length: int) -> bool:
    """The rule both policies share: a bounded, printable, non-empty string.

    ``str.isprintable()`` carries the character screen. It is false for every
    Unicode *Other* or *Separator* character except the ASCII space, so ``Cc``
    (LF, CR, ESC, NUL), ``Cf`` (NEL, the bidi overrides, ZWJ, BOM), ``Zl``
    (U+2028) and ``Zp`` (U+2029) are all rejected — as are surrogates, private
    use, unassigned codepoints and the non-ASCII spaces. Being category-defined,
    it covers codepoints added to Unicode later with no code change.
    """
    return isinstance(value, str) and 1 <= len(value) <= max_length and value.isprintable()


def is_safe_identifier(value: object, *, max_length: int) -> bool:
    """Screen for a string that is MATCHED, hashed, correlated or stored.

    Adds one rule to the shared base: no surrounding whitespace, so ``"c1"`` and
    ``" c1"`` cannot be two spellings of one identifier. A trimming validator
    would turn an equality check into a near-equality check and identity would
    stop being decidable.

    Rejection, never rewriting — a rewritten identifier is a *different*
    identifier, so silently repairing one would break the equality the caller
    relies on.
    """
    # ``isinstance`` re-checked rather than relying on the helper above: it
    # narrows the type for the strip comparison, and a reader should not have to
    # trace another function to see why ``.strip()`` is safe here.
    return (
        _is_bounded_printable(value, max_length=max_length)
        and isinstance(value, str)
        and value == value.strip()
    )


def is_safe_descriptive_text(value: object, *, max_length: int) -> bool:
    """Screen for a string that is READ but never matched.

    Bounded and printable, and nothing more. In particular ``"; "`` is
    **allowed**: an earlier version refused it here because
    ``result_contract._join`` composes degradation reasons with that delimiter,
    which coupled this module's input validation to another module's private
    output format. A specialist writing "runway; then covenant headroom" was
    refused for a reason that had nothing to do with its own field, and the
    coupling would silently invert the day ``_join`` changed its separator.

    Whatever renders these fields is responsible for escaping its own
    delimiters. This module's job is to refuse text that cannot appear in *any*
    audit record — line and control characters — not to guess at every
    consumer's syntax.
    """
    return _is_bounded_printable(value, max_length=max_length)


class CalculationProposal(BaseModel):
    """A model-shaped request for arithmetic, carrying no authority of any kind.

    Note what is absent, because the absence *is* the guarantee: no
    ``request_id``, no ``correlation``, no result, status, evidence,
    fingerprint, authority stamp or timestamp. With ``extra="forbid"`` a payload
    attempting any of them is rejected loudly rather than stripped silently, so
    the attempt is visible.

    Identity and correlation are minted by the gateway from this object's
    canonical content plus a frame the model does not control.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always"
    )

    operation: OperationId
    operands: tuple[Operand, ...]
    target_unit: Unit | None = None
    scale: int = Field(default=2, ge=0, le=MAX_SCALE)
    rounding: RoundingMode = "ROUND_HALF_EVEN"
    purpose: str = Field(min_length=1, max_length=MAX_PURPOSE_LEN)
    # A reference to a claim the specialist made. Checked against that
    # specialist's own claim ids by the gateway before it reaches a Correlation.
    claim_ref: str | None = Field(default=None, max_length=MAX_ID_LEN)

    @model_validator(mode="after")
    def _bounded_and_audit_safe(self) -> CalculationProposal:
        if not 1 <= len(self.operands) <= MAX_OPERANDS_PER_REQUEST:
            raise ValueError(
                f"a proposal carries 1..{MAX_OPERANDS_PER_REQUEST} operands, "
                f"not {len(self.operands)}"
            )
        ids = [operand.operand_id for operand in self.operands]
        if len(ids) != len(set(ids)):
            raise ValueError("operand_id must be unique within a proposal")

        self._require_text("purpose", self.purpose, MAX_PURPOSE_LEN)
        if self.claim_ref is not None:
            self._require_id("claim_ref", self.claim_ref, MAX_ID_LEN)
        for operand in self.operands:
            self._require_id("operand_id", operand.operand_id, MAX_ID_LEN)
            self._require_text("operand label", operand.label, MAX_LABEL_LEN)
            hint = operand.source_hint
            if hint is None:
                continue
            self._require_text("document_label", hint.document_label, MAX_LABEL_LEN)
            self._require_text("filename", hint.filename, MAX_LABEL_LEN)
            self._require_id("retrieval_id_hint", hint.retrieval_id_hint, MAX_ID_LEN)
            # Descriptive, but single-line: a quotation spanning lines is a real
            # thing a specialist may want to cite, and this slice deliberately
            # refuses it rather than flattening it. Preserving line structure
            # needs a quotation contract that says how a multi-line excerpt is
            # bounded, escaped and rendered; inventing one here would mean
            # guessing at a format no consumer has agreed to. Until that exists,
            # a rejected quote is honest and a flattened one is not.
            self._require_text("quoted_text", hint.quoted_text, MAX_LABEL_LEN)
        return self

    @staticmethod
    def _require_id(field_name: str, value: object, max_length: int) -> None:
        """Reject an unsafe identifier. ``None`` is absence, not unsafe text."""
        if value is None or is_safe_identifier(value, max_length=max_length):
            return
        raise ValueError(
            f"{field_name} is not a usable identifier: it must be printable, "
            f"unpadded, non-empty and at most {max_length} characters. It is "
            "rejected rather than rewritten."
        )

    @staticmethod
    def _require_text(field_name: str, value: object, max_length: int) -> None:
        """Reject unsafe descriptive text. ``None`` is absence, not unsafe text."""
        if value is None or is_safe_descriptive_text(value, max_length=max_length):
            return
        raise ValueError(
            f"{field_name} is not audit-safe: it must be printable, non-empty "
            f"and at most {max_length} characters, with no line or control "
            "characters. It is rejected rather than rewritten."
        )


def _enum(literal: Any) -> list[str]:
    """The tool-schema enum for a contract ``Literal``, derived, never retyped."""
    return list(get_args(literal))


CALCULATION_REQUESTS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": (
        "Arithmetic you want computed deterministically. Each entry is a "
        "PROPOSAL: name the operation, the typed operands with units, and the "
        "claim it supports. You do not supply the answer, an id, a status or a "
        "timestamp — the application computes and records those. Any figure "
        "you also state in prose stays unverified."
    ),
    "items": {
        "type": "object",
        "properties": {
            "claim_ref": {
                "type": "string",
                "description": (
                    "claim_id in THIS result that the calculation supports. "
                    "Must name a claim you emitted; anything else is dropped."
                ),
            },
            "operands": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "basis": {"type": "string", "enum": _enum(OperandBasis)},
                        "label": {"type": "string"},
                        "number_format": {
                            "type": "string",
                            "enum": _enum(NumberFormat),
                        },
                        "operand_id": {
                            "type": "string",
                            "description": "Unique within this proposal.",
                        },
                        "role": {
                            "type": "string",
                            "enum": _enum(OperandRole),
                            "description": (
                                "'stated_comparison' marks the applicant's own "
                                "claimed figure for a 'variance' operation."
                            ),
                        },
                        "source_hint": {
                            "type": "object",
                            "properties": {
                                "document_label": {"type": "string"},
                                "filename": {"type": "string"},
                                "quoted_text": {"type": "string"},
                                "retrieval_id_hint": {"type": "string"},
                            },
                        },
                        "unit": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": (
                                        "A registry code (kg, t, ha, m2, "
                                        "kg_per_m2, pct, pct_point, ...) or "
                                        "currency:<ISO> such as currency:TND."
                                    ),
                                }
                            },
                            "required": ["code"],
                        },
                        "value": {
                            "type": "string",
                            "description": "The number as a string, never a float.",
                        },
                    },
                    "required": ["operand_id", "label", "value", "unit", "basis"],
                },
            },
            "operation": {"type": "string", "enum": _enum(OperationId)},
            "purpose": {"type": "string"},
            "rounding": {"type": "string", "enum": _enum(RoundingMode)},
            "scale": {"type": "integer", "minimum": 0, "maximum": MAX_SCALE},
            "target_unit": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
        "required": ["operation", "operands", "purpose"],
    },
}
"""The wire shape of one ``calculation_requests`` entry, owned here beside the
model it mirrors. Property names are declared in sorted order so the serialized
tool block is byte-stable across processes (prompt caching keys on it).

Note what is absent: no ``request_id``, ``correlation``, result, status,
fingerprint, ``computed_at``, ``authority`` or evidence field. The model has no
slot in which to assert any of them, and ``extra="forbid"`` on the model above
rejects a payload that tries."""
