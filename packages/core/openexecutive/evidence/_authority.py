"""Who may construct a canonical evidence record, and how strong that is.

**Against the wire — structural, and it holds.** Wire and model JSON parses only
into the *proposal* types in :mod:`openexecutive.evidence.contracts`, which have
no field for an id, hash, timestamp, scope, logical-source key, lineage, status,
quality, independence, applicability, verification or authority; ``extra=
"forbid"`` rejects a payload carrying one rather than stripping it silently.
There is no wire on which a model can assert its own provenance. That is the
real boundary.

**Against a same-process caller — conventional, and not absolute.** Python has
no private constructor. :func:`trusted_construction` is a ``ContextVar`` any
importer, including a test, can enter, and ``model_construct`` bypasses pydantic
for any model. This is **in-process misuse resistance, not a security
boundary**: accidental construction outside the factory fails loudly, and every
deliberate site is one greppable name review can ask about.

The ``ContextVar`` is not a model field, so it cannot arrive through
deserialization, and it is task-local, so a task that did not enter the context
does not inherit one that did.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class UntrustedConstructionError(RuntimeError):
    """A canonical evidence record was constructed outside the trusted path."""


_TRUSTED_CONSTRUCTION: ContextVar[bool] = ContextVar(
    "openexecutive.evidence.trusted_construction", default=False
)


@contextmanager
def trusted_construction() -> Iterator[None]:
    """Permit canonical construction for the duration of the block.

    Only :mod:`openexecutive.evidence.factory` enters this in production; an
    architectural test pins that. Token-based reset, so nesting and early
    exceptions restore the previous state exactly.
    """
    token = _TRUSTED_CONSTRUCTION.set(True)
    try:
        yield
    finally:
        _TRUSTED_CONSTRUCTION.reset(token)


def require_trusted_construction(model_name: str) -> None:
    """Raise unless the trusted construction context is currently active."""
    if not _TRUSTED_CONSTRUCTION.get():
        raise UntrustedConstructionError(
            f"{model_name} may only be constructed by openexecutive.evidence.factory; "
            "untrusted input parses into the proposal types instead"
        )
