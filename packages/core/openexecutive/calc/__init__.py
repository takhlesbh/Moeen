"""Deterministic calculation contracts — types and authority, no arithmetic.

**Production reaches this package through exactly one door.** Only
``specialists/calculation_gateway.py`` imports the engine or the authority
module, and only ``agents/finance.py`` calls the gateway (Phase 3B2, behind a
default-off flag). Everything else imports calc *types* at most. The package
shipped as contracts + tests first so the engine was written against a boundary
that already existed and had already been reviewed.

What is here: a closed unit registry with exact conversion factors and real
dimensions; a numeric boundary that refuses floats, ambiguous separators, NaN,
Infinity, and out-of-range exponents; a model-proposed
:class:`~openexecutive.calc.contract.CalculationRequest`; an application-authored
:class:`~openexecutive.calc.contract.CalculationResult` that keeps arithmetic
status and evidence status on separate axes; and a single sanctioned channel for
issuing a result.

Phase 2 adds the engine: :mod:`openexecutive.calc.engine` executes a closed set
of twelve operations over ``Decimal`` with declared arity and dimensional
signatures, and :mod:`openexecutive.calc.fingerprint` derives each result's
identity. Arithmetic now happens here — deterministically, with no model in the
loop.

What is still deliberately **not** here: any expression evaluator, any tool
schema, any provider call, any filesystem or network access. This package imports only ``json``, ``re``, ``decimal``,
``types``, ``typing``, ``collections.abc``, ``hashlib`` (the fingerprint
digest), ``time`` (the engine's monotonic budget clock, never a wall clock and
never the source of a record's timestamp), ``pydantic``, and its own modules —
enforced by tests that walk the package recursively rather than naming modules,
plus a test asserting the package stays flat (no subdirectories), because
``pkgutil`` does not descend into a directory lacking ``__init__.py`` and a file
hidden there would otherwise be importable, executable, and unscanned.

Dependency rule: ``calc`` is a leaf. It must not import ``agents``,
``specialists``, ``providers``, ``orchestrator``, ``prompts``, or anything that
reaches a model, a database, or a network. That constraint is what lets the
engine be tested with no fixtures and no mocks.
"""
from openexecutive.calc.authority import (
    AUTHORITY_ID,
    AUTHORITY_VERSION,
    current_authority,
    issue_calculation_result,
)
from openexecutive.calc.contract import (
    FINGERPRINT_EXCLUDED_FIELDS,
    FINGERPRINT_INCLUDED_FIELDS,
    FINGERPRINT_OPTIONAL_FIELDS,
    KNOWN_AUTHORITY_IDS,
    KNOWN_AUTHORITY_VERSIONS,
    MAX_EXPRESSION_LEN,
    MAX_OPERANDS_PER_REQUEST,
    MAX_REQUESTS_PER_BATCH,
    NESTED_OPERATION_DEPTH,
    NON_COMMUTATIVE_OPERATIONS,
    SCHEMA_VERSION,
    ApplicationAuthority,
    ArithmeticStatus,
    CalculationBatch,
    CalculationError,
    CalculationRequest,
    CalculationResult,
    ConflictClass,
    Correlation,
    InputEvidenceStatus,
    InputEvidenceSummary,
    NormalizedOperand,
    Operand,
    OperandBasis,
    OperandRole,
    OperationId,
    RoundingMode,
    SourceHint,
    canonical_payload_json,
    fingerprint_payload,
)
from openexecutive.calc.engine import (
    LIMITS,
    EngineLimits,
    OperationSignature,
    TimeConversionPolicy,
    WeightPolicy,
    execute,
    execute_batch,
    signature_for,
)
from openexecutive.calc.fingerprint import (
    FAILURE_FINGERPRINT_RULE,
    FINGERPRINT_ALGORITHM,
    fingerprint_for,
)
from openexecutive.calc.numeric import (
    MAX_ADJUSTED_EXPONENT,
    MAX_NUMERIC_STRING_LEN,
    MAX_PRECISION_REQUEST,
    MAX_SCALE,
    NumberFormat,
    NumericPolicyError,
    canonical_numeric_string,
    parse_numeric,
)
from openexecutive.calc.units import (
    CURRENCY_PREFIX,
    MULTIPLICATIVE_COMPOSITIONS,
    ConversionPolicy,
    Dimension,
    Unit,
    UnitSpec,
    additively_compatible,
    composed_dimension,
    convertible,
    known_unit_codes,
    same_dimension,
    unit_spec,
)

__all__ = [
    "AUTHORITY_ID",
    "AUTHORITY_VERSION",
    "CURRENCY_PREFIX",
    "FINGERPRINT_EXCLUDED_FIELDS",
    "FAILURE_FINGERPRINT_RULE",
    "FINGERPRINT_ALGORITHM",
    "FINGERPRINT_INCLUDED_FIELDS",
    "FINGERPRINT_OPTIONAL_FIELDS",
    "LIMITS",
    "KNOWN_AUTHORITY_IDS",
    "KNOWN_AUTHORITY_VERSIONS",
    "MAX_ADJUSTED_EXPONENT",
    "MAX_EXPRESSION_LEN",
    "MAX_NUMERIC_STRING_LEN",
    "MAX_OPERANDS_PER_REQUEST",
    "MAX_PRECISION_REQUEST",
    "MAX_REQUESTS_PER_BATCH",
    "MAX_SCALE",
    "MULTIPLICATIVE_COMPOSITIONS",
    "NESTED_OPERATION_DEPTH",
    "NON_COMMUTATIVE_OPERATIONS",
    "SCHEMA_VERSION",
    "ApplicationAuthority",
    "ArithmeticStatus",
    "CalculationBatch",
    "CalculationError",
    "CalculationRequest",
    "CalculationResult",
    "ConflictClass",
    "ConversionPolicy",
    "Correlation",
    "Dimension",
    "EngineLimits",
    "InputEvidenceStatus",
    "InputEvidenceSummary",
    "NormalizedOperand",
    "NumberFormat",
    "NumericPolicyError",
    "Operand",
    "OperandBasis",
    "OperandRole",
    "OperationId",
    "OperationSignature",
    "RoundingMode",
    "SourceHint",
    "TimeConversionPolicy",
    "WeightPolicy",
    "Unit",
    "UnitSpec",
    "additively_compatible",
    "canonical_numeric_string",
    "canonical_payload_json",
    "composed_dimension",
    "convertible",
    "current_authority",
    "execute",
    "execute_batch",
    "fingerprint_for",
    "fingerprint_payload",
    "issue_calculation_result",
    "known_unit_codes",
    "parse_numeric",
    "same_dimension",
    "signature_for",
    "unit_spec",
]
