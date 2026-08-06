"""Feature / transformation / compilation identity helpers (Sprint 2 + 5).

``FEAT_*`` / ``COMP_*`` are UUIDv7 event/object ids.
``TR_*`` is content-addressed from canonical TL text (hash-derived; not UUIDv7).
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import uuid

FEATURE_UUID_PATTERN = re.compile(r"^FEAT_[0-9A-F]{32}$")
TRANSFORM_UUID_PATTERN = re.compile(r"^TR_[0-9A-F]{32}$")
COMPILATION_UUID_PATTERN = re.compile(r"^COMP_[0-9A-F]{32}$")

_HYPHENATED = re.compile(
    r"^FEAT_([0-9a-fA-F]{8})-([0-9a-fA-F]{4})-([0-9a-fA-F]{4})-"
    r"([0-9a-fA-F]{4})-([0-9a-fA-F]{12})$"
)
_HEX32 = re.compile(r"^FEAT_([0-9a-fA-F]{32})$")
_TR_HEX32 = re.compile(r"^TR_([0-9a-fA-F]{32})$", re.IGNORECASE)
_TR_HYPHENATED = re.compile(
    r"^TR_([0-9a-fA-F]{8})-([0-9a-fA-F]{4})-([0-9a-fA-F]{4})-"
    r"([0-9a-fA-F]{4})-([0-9a-fA-F]{12})$",
    re.IGNORECASE,
)
_COMP_HEX32 = re.compile(r"^COMP_([0-9a-fA-F]{32})$", re.IGNORECASE)
_COMP_HYPHENATED = re.compile(
    r"^COMP_([0-9a-fA-F]{8})-([0-9a-fA-F]{4})-([0-9a-fA-F]{4})-"
    r"([0-9a-fA-F]{4})-([0-9a-fA-F]{12})$",
    re.IGNORECASE,
)


def _uuidv7_bytes() -> bytes:
    """Generate RFC 9562 UUIDv7 as 16 bytes."""
    if hasattr(uuid, "uuid7"):
        return uuid.uuid7().bytes  # type: ignore[attr-defined]

    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = int.from_bytes(os.urandom(10), "big")
    # 48-bit ts | ver(4)=0111 | rand_a(12) | var(2)=10 | rand_b(62)
    rand_a = (rand >> 62) & 0x0FFF
    rand_b = rand & ((1 << 62) - 1)
    value = (ts_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return value.to_bytes(16, "big")


def generate_feature_uuid() -> str:
    """Return a new canonical feature UUID: FEAT_ + 32 uppercase hex (UUIDv7)."""
    return "FEAT_" + _uuidv7_bytes().hex().upper()


def generate_compilation_uuid() -> str:
    """Return a new compilation event id: COMP_ + 32 uppercase hex (UUIDv7)."""
    return "COMP_" + _uuidv7_bytes().hex().upper()


def expression_hash(canonical_text: str) -> str:
    """Full SHA-256 hex (lowercase) of canonical TL UTF-8 bytes."""
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def derive_transformation_uuid(canonical_text: str) -> str:
    """Derive ``TR_*`` from canonical TL text only (Sprint 5 Decision A/D)."""
    digest = expression_hash(canonical_text)
    return "TR_" + digest[:32].upper()


def _parse_hex_body(hex32: str) -> uuid.UUID:
    return uuid.UUID(hex=hex32)


def _require_v7(u: uuid.UUID) -> None:
    if u.version != 7:
        raise ValueError(f"UUID version must be 7, got {u.version}")
    # variant check via RFC bits
    if (u.bytes[8] & 0xC0) != 0x80:
        raise ValueError("UUID variant must be RFC 9562 (10xx)")


def normalize_feature_uuid(raw: str) -> str:
    """Normalize to ``^FEAT_[0-9A-F]{32}$``; reject non-v7 / invalid forms."""
    text = (raw or "").strip()
    m = _HEX32.fullmatch(text)
    if m:
        hex32 = m.group(1).upper()
    else:
        m2 = _HYPHENATED.fullmatch(text)
        if not m2:
            raise ValueError(f"Invalid feature_uuid format: {raw!r}")
        hex32 = "".join(m2.groups()).upper()
    u = _parse_hex_body(hex32)
    _require_v7(u)
    return "FEAT_" + hex32


def is_valid_feature_uuid(value: str) -> bool:
    try:
        normalize_feature_uuid(value)
        return True
    except ValueError:
        return False


def normalize_transformation_uuid(raw: str | None) -> str | None:
    """Shape-normalize ``TR_*`` (32 uppercase hex). No UUIDv7 nibble check."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    m = _TR_HEX32.fullmatch(text)
    if m:
        hex32 = m.group(1).upper()
    else:
        m2 = _TR_HYPHENATED.fullmatch(text)
        if not m2:
            raise ValueError(f"Invalid transformation_uuid format: {raw!r}")
        hex32 = "".join(m2.groups()).upper()
    return "TR_" + hex32


def is_valid_transformation_uuid(value: str) -> bool:
    try:
        return normalize_transformation_uuid(value) is not None
    except ValueError:
        return False


def normalize_compilation_uuid(raw: str) -> str:
    """Normalize to ``^COMP_[0-9A-F]{32}$``; reject non-v7 / invalid forms."""
    text = (raw or "").strip()
    m = _COMP_HEX32.fullmatch(text)
    if m:
        hex32 = m.group(1).upper()
    else:
        m2 = _COMP_HYPHENATED.fullmatch(text)
        if not m2:
            raise ValueError(f"Invalid compilation_uuid format: {raw!r}")
        hex32 = "".join(m2.groups()).upper()
    u = _parse_hex_body(hex32)
    _require_v7(u)
    return "COMP_" + hex32


def is_valid_compilation_uuid(value: str) -> bool:
    try:
        normalize_compilation_uuid(value)
        return True
    except ValueError:
        return False
