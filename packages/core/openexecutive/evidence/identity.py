"""Deterministic, domain-separated identity for evidence records.

Nothing here normalizes text: callers decide *which representation* they hash,
and separating ``raw_text_sha256`` from ``nfc_text_sha256`` only means something
if the hash functions stay dumb. SHA-256 is used for *identity*, not secrecy;
nothing here resists a forger, since a same-process caller can write any field
on any object.

**Encoding.** ``mint_id`` digests a fixed prefix, then the entity type tag and
each component, every one in a *typed frame*: a one-byte type discriminator,
a four-byte big-endian payload length, then the payload. Discriminators are
``b"b"`` for bytes (payload unchanged), ``b"s"`` for str (strict UTF-8) and
``b"i"`` for unsigned integers (eight-byte big-endian). The length makes the
encoding injective across component boundaries — without it ``(b"ab", b"c")``
and ``(b"a", b"bc")`` would digest identically — and the discriminator makes it
injective across types, so ``"a"`` and ``b"a"``, or ``0`` and ``b"\\x00" * 8``,
never collide at the same position. The tag is framed the same way, so two
entity kinds cannot mint the same id from the same parts.
"""
from __future__ import annotations

import hashlib

ID_PREFIX = b"oe.evidence.v1\x00"
"""Domain separation. Bump the version if the encoding above ever changes."""

MAX_ID_COMPONENTS = 16
MAX_COMPONENT_BYTES = 64 * 1024

TAG_LOGICAL_SOURCE = b"logical_source"
TAG_DOCUMENT_VERSION = b"document_version"
TAG_EXTRACTION = b"extraction"
TAG_SOURCE_SPAN = b"source_span"

IdComponent = bytes | str | int

DISCRIMINATOR_BYTES = b"b"
DISCRIMINATOR_STR = b"s"
DISCRIMINATOR_INT = b"i"


class IdentityError(ValueError):
    """An identity component was of an unsupported type or out of bounds."""


def content_sha256(raw: bytes) -> str:
    """Hex SHA-256 of immutable raw document bytes. No normalization."""
    if not isinstance(raw, bytes):
        raise IdentityError("content_sha256 requires bytes")
    return hashlib.sha256(raw).hexdigest()


def text_sha256(text: str) -> str:
    """Hex SHA-256 of the exact string as UTF-8. No normalization.

    Lone surrogates raise :class:`IdentityError`: they have no UTF-8 encoding,
    so such a string has no hash under this definition.
    """
    if not isinstance(text, str):
        raise IdentityError("text_sha256 requires str")
    return hashlib.sha256(_utf8(text, "text")).hexdigest()


def _utf8(text: str, label: str) -> bytes:
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise IdentityError(f"{label} contains a lone surrogate") from exc


def _frame(component: IdComponent) -> bytes:
    if isinstance(component, bool):
        raise IdentityError("bool is not a valid id component")
    if isinstance(component, int):
        if not 0 <= component < 2**64:
            raise IdentityError("int id component must fit in unsigned 64 bits")
        kind, raw = DISCRIMINATOR_INT, component.to_bytes(8, "big")
    elif isinstance(component, bytes):
        kind, raw = DISCRIMINATOR_BYTES, component
    elif isinstance(component, str):
        kind, raw = DISCRIMINATOR_STR, _utf8(component, "id component")
    else:
        raise IdentityError(f"unsupported id component type: {type(component).__name__}")
    if len(raw) > MAX_COMPONENT_BYTES:
        raise IdentityError("id component exceeds 64 KiB")
    return kind + len(raw).to_bytes(4, "big") + raw


def mint_id(type_tag: bytes, *components: IdComponent) -> str:
    """Digest of the prefix, the typed-framed tag and the typed-framed components.

    The tag counts toward the budget, so at most 15 components may follow it.
    """
    if not isinstance(type_tag, bytes):
        raise IdentityError("type_tag must be bytes")
    if 1 + len(components) > MAX_ID_COMPONENTS:
        raise IdentityError(f"at most {MAX_ID_COMPONENTS} id components including the type tag")
    digest = hashlib.sha256()
    digest.update(ID_PREFIX)
    digest.update(_frame(type_tag))
    for component in components:
        digest.update(_frame(component))
    return digest.hexdigest()
