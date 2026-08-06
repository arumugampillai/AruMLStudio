"""Grammar registry record and internal syntax-tree nodes (Sprint 4).

Internal nodes are package-private helpers for the validator/formatter.
They are not a public compiler AST API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class GrammarRegistryRecord:
    grammar_version: str
    grammar_pack_version: str
    token_pack_version: str
    formatter_version: str
    checksum: str
    ebnf_path: str
    compatibility_json: str
    notes: str | None = None
    created_at: str = ""


# --- Internal syntax tree (not exported as public AST) ---


@dataclass
class _PrimitiveRef:
    name: str  # full PR_* id


@dataclass
class _FeatureRef:
    name: str  # full FEAT_* id


@dataclass
class _OperatorRef:
    name: str  # full OP_* id


@dataclass
class _IntLit:
    text: str
    value: int


@dataclass
class _FloatLit:
    text: str
    value: float


@dataclass
class _BoolLit:
    text: str
    value: bool


@dataclass
class _StringLit:
    text: str  # raw quoted form including escapes as stored
    value: str


@dataclass
class _ListLit:
    items: list["_Value"]


_Literal = Union[_IntLit, _FloatLit, _BoolLit, _StringLit]
_Value = Union["_Call", _PrimitiveRef, _FeatureRef, _Literal, _ListLit]


@dataclass
class _Arg:
    name: str
    value: _Value


@dataclass
class _Call:
    operator: str  # OP_* id
    args: list[_Arg] = field(default_factory=list)
    depth: int = 1  # nesting depth of this call (1 = root)
