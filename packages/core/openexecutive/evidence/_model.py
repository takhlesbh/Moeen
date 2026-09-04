"""Shared base model for the evidence contracts.

*Mirrors* :mod:`openexecutive.calc._model` rather than importing it: ``evidence``
is a leaf package and must not depend on ``calc``.

``extra="forbid"`` is load-bearing. It makes "an untrusted proposal cannot claim
authority" true of the *proposal* types: a payload carrying an id, a hash or a
verification field is rejected outright rather than silently dropped, so the
attempt is visible.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

EVIDENCE_CONFIG = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class EvidenceModel(BaseModel):
    """Frozen, revalidating base for every model in this package.

    ``frozen=True`` blocks ``__setattr__`` only; ``model_copy(update=...)``
    otherwise writes fields without validators, so it is routed back through
    ``model_validate``. Stated honestly: not a same-process security boundary —
    ``model_construct`` bypasses pydantic for any model. See ``_authority``.
    """

    model_config = EVIDENCE_CONFIG

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        if not update:
            return super().model_copy(deep=deep)
        return type(self).model_validate({**self.model_dump(), **update})
