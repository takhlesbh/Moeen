"""Evidence foundation: identity, normalization and span verification.

A leaf package. It imports only the standard library and pydantic, and must not
acquire a dependency on ``calc``, ``specialists``, ``knowledge``, ``agents``,
routes or persistence. Nothing in the application imports it yet.

Read :mod:`openexecutive.evidence.contracts` for the two construction paths and
the offset semantics, and :mod:`openexecutive.evidence._authority` for the
honest scope of the trusted-construction guard.
"""
from openexecutive.evidence.contracts import (
    DocumentIngestProposal,
    DocumentVersion,
    Extraction,
    SourceSpan,
    SourceSpanProposal,
)
from openexecutive.evidence.extractor_config import (
    ExtractorConfigError,
    extractor_config_sha256,
)
from openexecutive.evidence.factory import (
    EvidenceFactoryError,
    NormalizationError,
    SpanVerificationError,
    mint_document_version,
    mint_extraction,
    mint_source_span,
)
from openexecutive.evidence.identity import (
    IdentityError,
    content_sha256,
    mint_id,
    text_sha256,
)

__all__ = [
    "DocumentIngestProposal",
    "DocumentVersion",
    "EvidenceFactoryError",
    "Extraction",
    "ExtractorConfigError",
    "IdentityError",
    "NormalizationError",
    "SourceSpan",
    "SourceSpanProposal",
    "SpanVerificationError",
    "content_sha256",
    "extractor_config_sha256",
    "mint_document_version",
    "mint_extraction",
    "mint_id",
    "mint_source_span",
    "text_sha256",
]
