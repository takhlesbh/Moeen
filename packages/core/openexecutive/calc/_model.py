"""Shared base model for the calculation contracts.

This deliberately *mirrors* the private base model in the specialist result
contract (``openexecutive.specialists``) rather than importing it. Two reasons,
both about blast radius:

* that name is module-private, and importing a private name across packages
  makes a refactor of that module a silent breaking change here;
* ``calc`` is a leaf package. It must not depend on ``specialists``,
  ``agents``, ``providers``, or anything that reaches a model or a network. A
  one-class duplication is the cheaper of the two couplings, and the
  architecture entry records it as a deliberate constraint rather than an
  oversight.

The semantics are identical and intentionally so: frozen, ``extra="forbid"``,
``revalidate_instances="always"``, and a ``model_copy`` override that routes
updates back through validation.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

CONTRACT_CONFIG = ConfigDict(
    extra="forbid", frozen=True, revalidate_instances="always"
)
"""``extra="forbid"`` is load-bearing, not tidiness.

It is what makes "the model cannot assign verification authority" true of the
*request* type: a payload carrying an unexpected ``verification_status`` or
``fingerprint`` key is rejected outright instead of being silently dropped, so
an attempt to claim authority surfaces as a validation error rather than
vanishing. ``revalidate_instances="always"`` re-checks a nested model when it is
placed inside a parent, so a forged child cannot ride in on a valid parent.
"""


class ContractModel(BaseModel):
    """Frozen, revalidating base for every model in this package.

    ``frozen=True`` blocks ``__setattr__`` only. Pydantic's
    ``model_copy(update=...)`` writes fields straight onto the new object
    *without* running validators, which would let a caller mint a copy carrying
    a unit, a status, or an authority stamp the constructor would have refused.
    Routing the update through ``model_validate`` applies the same rules a
    constructor would.

    Scope, stated honestly: this makes the documented invariants hold for the
    object's lifetime and catches a forged nested model on its way into a
    parent. It is **not** a same-process security boundary — a top-level
    ``model_construct`` or ``object.__setattr__`` bypasses pydantic entirely, as
    it does for any pydantic model. See ``authority.py`` for what is and is not
    enforceable here, and against whom.
    """

    model_config = CONTRACT_CONFIG

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if not update:
            return super().model_copy(deep=deep)
        # Round-trips through validation, so an update cannot write a field the
        # constructor would have rejected. Two knock-on differences from
        # BaseModel: the copy is always deep, and ``model_fields_set`` becomes
        # "all fields" rather than the caller's subset — do not rely on
        # ``exclude_unset`` after a copy-with-update.
        return type(self).model_validate({**self.model_dump(), **update})
