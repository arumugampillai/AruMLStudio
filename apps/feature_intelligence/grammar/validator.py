"""TL syntax validator — syntax_only and bound modes (Sprint 4).

Uses an internal recursive-descent parse tree. No public AST / compiler API.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from feature_intelligence.grammar.models import (
    _Arg,
    _BoolLit,
    _Call,
    _FeatureRef,
    _FloatLit,
    _IntLit,
    _ListLit,
    _PrimitiveRef,
    _StringLit,
    _Value,
)
from feature_intelligence.grammar.pack import (
    EXPECTED_GRAMMAR_CHECKSUM,
    GRAMMAR_VERSION,
    compute_grammar_pack_checksum,
)
from feature_intelligence.registry.models import ValidationReport

MAX_NESTING_DEPTH = 64

RE_OP = re.compile(r"^OP_[A-Z][A-Z0-9_]*$")
RE_PR = re.compile(r"^PR_[A-Z][A-Z0-9_]*$")
RE_FEAT = re.compile(r"^FEAT_[0-9A-F]{32}$")
RE_IDENT = re.compile(r"^[a-z][a-z0-9_]*$")
RE_INT = re.compile(r"^-?[0-9]+$")
RE_FLOAT = re.compile(r"^-?[0-9]+\.[0-9]+$")

# Series-like params allowed beyond schema when operator has input arity.
# Sprint 3 schemas often omit the series `source` (arity is separate); TL
# function form still binds series via named args like source=PR_SPOT.
_SERIES_SLOT_NAMES = frozenset(
    {"source", "left", "right", "input", "inputs", "x", "y", "a", "b"}
)


@dataclass
class _Token:
    kind: str
    value: str
    pos: int


class _LexError(Exception):
    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        self.message = message
        super().__init__(message or code)


class _ParseError(Exception):
    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        self.message = message
        super().__init__(message or code)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fail(code: str, detail: str = "") -> str:
    return f"{code}:{detail}" if detail else code


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch == "#":
            # Line comments (examples use `# expect: CODE`)
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        if ch == ":":
            raise _LexError("FORBIDDEN_COLON", f"at {i}")
        if ch in "(),=[]":
            tokens.append(_Token(ch, ch, i))
            i += 1
            continue
        if ch == '"':
            start = i
            i += 1
            buf: list[str] = []
            while i < n:
                c = text[i]
                if c == "\\":
                    if i + 1 >= n:
                        raise _LexError("BAD_LITERAL", "unterminated string escape")
                    nxt = text[i + 1]
                    if nxt not in {'"', "\\"}:
                        raise _LexError("BAD_LITERAL", f"bad escape \\{nxt}")
                    buf.append(nxt)
                    i += 2
                    continue
                if c == '"':
                    i += 1
                    tokens.append(_Token("STRING", "".join(buf), start))
                    break
                buf.append(c)
                i += 1
            else:
                raise _LexError("BAD_LITERAL", "unterminated string")
            continue
        # Number or identifier / ref
        if ch.isdigit() or (ch == "-" and i + 1 < n and text[i + 1].isdigit()):
            start = i
            if ch == "-":
                i += 1
            while i < n and text[i].isdigit():
                i += 1
            if i < n and text[i] == ".":
                i += 1
                if i >= n or not text[i].isdigit():
                    raise _LexError("BAD_LITERAL", "float needs digits after '.'")
                while i < n and text[i].isdigit():
                    i += 1
                tokens.append(_Token("FLOAT", text[start:i], start))
            else:
                tokens.append(_Token("INT", text[start:i], start))
            continue
        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (text[i].isalnum() or text[i] == "_"):
                i += 1
            word = text[start:i]
            if word in ("true", "false"):
                tokens.append(_Token("BOOL", word, start))
            else:
                tokens.append(_Token("IDENT", word, start))
            continue
        # Arrow or other junk
        if ch == "\u2192" or (ch == "-" and i + 1 < n and text[i + 1] == ">"):
            raise _LexError("UNEXPECTED_TOKEN", "arrow syntax not in Grammar 1.0")
        raise _LexError("UNEXPECTED_TOKEN", f"{ch!r} at {i}")
    tokens.append(_Token("EOF", "", n))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.i = 0
        self.max_depth_seen = 0

    def _cur(self) -> _Token:
        return self.tokens[self.i]

    def _eat(self, kind: str) -> _Token:
        tok = self._cur()
        if tok.kind != kind:
            if kind == ")" and tok.kind == "EOF":
                raise _ParseError("UNBALANCED_PAREN", "missing ')'")
            if kind == "(" and tok.kind != "(":
                raise _ParseError("UNEXPECTED_TOKEN", f"expected '(', got {tok.kind}")
            raise _ParseError("UNEXPECTED_TOKEN", f"expected {kind}, got {tok.kind}")
        self.i += 1
        return tok

    def parse_expression(self) -> _Call:
        call = self._parse_call(depth=1)
        if self._cur().kind != "EOF":
            raise _ParseError("UNEXPECTED_TOKEN", f"trailing {self._cur().kind}")
        return call

    def _parse_call(self, depth: int) -> _Call:
        if depth > MAX_NESTING_DEPTH:
            raise _ParseError("NESTING_DEPTH", str(depth))
        self.max_depth_seen = max(self.max_depth_seen, depth)
        tok = self._cur()
        if tok.kind != "IDENT" or not RE_OP.match(tok.value):
            raise _ParseError("UNEXPECTED_TOKEN", f"expected OP_* call, got {tok.value!r}")
        op = tok.value
        self.i += 1
        self._eat("(")
        args: list[_Arg] = []
        seen: set[str] = set()
        if self._cur().kind == ")":
            self.i += 1
            return _Call(operator=op, args=args, depth=depth)
        while True:
            # Trailing comma: ',' then ')'
            if self._cur().kind == ",":
                raise _ParseError("TRAILING_COMMA", "empty arg slot")
            name_tok = self._cur()
            if name_tok.kind != "IDENT":
                # Positional: value without name=
                if name_tok.kind in ("INT", "FLOAT", "BOOL", "STRING", "[", "(") or (
                    name_tok.kind == "IDENT"
                    and (
                        RE_OP.match(name_tok.value)
                        or RE_PR.match(name_tok.value)
                        or name_tok.value.startswith("FEAT_")
                    )
                ):
                    raise _ParseError("POSITIONAL_ARG", name_tok.value)
                raise _ParseError("UNEXPECTED_TOKEN", f"arg name, got {name_tok.kind}")
            # Distinguish positional OP_/PR_/FEAT_ (no '=') from param name
            peek = self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else None
            if peek is None or peek.kind != "=":
                if (
                    RE_OP.match(name_tok.value)
                    or RE_PR.match(name_tok.value)
                    or name_tok.value.startswith("FEAT_")
                    or RE_INT.match(name_tok.value)
                ):
                    raise _ParseError("POSITIONAL_ARG", name_tok.value)
                raise _ParseError("UNEXPECTED_TOKEN", f"expected '=' after {name_tok.value}")
            if not RE_IDENT.match(name_tok.value):
                # OP_EMA(...) as positional would have been caught; invalid param name
                raise _ParseError("UNEXPECTED_TOKEN", f"bad param name {name_tok.value!r}")
            self.i += 1
            self._eat("=")
            if name_tok.value in seen:
                raise _ParseError("DUPLICATE_PARAM", name_tok.value)
            seen.add(name_tok.value)
            value = self._parse_value(depth)
            args.append(_Arg(name=name_tok.value, value=value))
            if self._cur().kind == ",":
                self.i += 1
                if self._cur().kind == ")":
                    raise _ParseError("TRAILING_COMMA", "before ')'")
                continue
            if self._cur().kind == ")":
                self.i += 1
                break
            if self._cur().kind == "EOF":
                raise _ParseError("UNBALANCED_PAREN", "missing ')'")
            raise _ParseError("UNEXPECTED_TOKEN", self._cur().kind)
        return _Call(operator=op, args=args, depth=depth)

    def _parse_value(self, parent_depth: int) -> _Value:
        tok = self._cur()
        if tok.kind == "IDENT":
            if RE_OP.match(tok.value):
                return self._parse_call(depth=parent_depth + 1)
            if RE_PR.match(tok.value):
                self.i += 1
                return _PrimitiveRef(tok.value)
            if tok.value.startswith("FEAT_"):
                if not RE_FEAT.match(tok.value):
                    raise _ParseError("UNEXPECTED_TOKEN", f"bad FEAT id {tok.value}")
                self.i += 1
                return _FeatureRef(tok.value)
            if tok.value.startswith("TR_"):
                raise _ParseError("UNEXPECTED_TOKEN", "TR_* not issued in Sprint 4")
            raise _ParseError("UNEXPECTED_TOKEN", tok.value)
        if tok.kind == "INT":
            if not RE_INT.match(tok.value):
                raise _ParseError("BAD_LITERAL", tok.value)
            self.i += 1
            return _IntLit(tok.value, int(tok.value))
        if tok.kind == "FLOAT":
            if not RE_FLOAT.match(tok.value):
                raise _ParseError("BAD_LITERAL", tok.value)
            self.i += 1
            return _FloatLit(tok.value, float(tok.value))
        if tok.kind == "BOOL":
            self.i += 1
            return _BoolLit(tok.value, tok.value == "true")
        if tok.kind == "STRING":
            self.i += 1
            return _StringLit(f'"{_escape_str(tok.value)}"', tok.value)
        if tok.kind == "[":
            return self._parse_list(parent_depth)
        raise _ParseError("UNEXPECTED_TOKEN", tok.kind)

    def _parse_list(self, parent_depth: int) -> _ListLit:
        self._eat("[")
        items: list[_Value] = []
        if self._cur().kind == "]":
            self.i += 1
            return _ListLit(items)
        while True:
            if self._cur().kind == ",":
                raise _ParseError("TRAILING_COMMA", "in list")
            items.append(self._parse_value(parent_depth))
            if self._cur().kind == ",":
                self.i += 1
                if self._cur().kind == "]":
                    raise _ParseError("TRAILING_COMMA", "in list before ']'")
                continue
            if self._cur().kind == "]":
                self.i += 1
                break
            raise _ParseError("UNEXPECTED_TOKEN", self._cur().kind)
        return _ListLit(items)


def _escape_str(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def parse_internal(text: str) -> _Call:
    """Parse TL text into an internal call tree. Raises with .code attribute."""
    if "\u2192" in text or "->" in text:
        # Also catch arrow outside lexer when used as documentation sugar
        if re.search(r"->|\u2192", text):
            # Allow inside comments already stripped by lexer path; for raw check:
            stripped = re.sub(r"#.*", "", text)
            if "->" in stripped or "\u2192" in stripped:
                raise _ParseError("UNEXPECTED_TOKEN", "arrow syntax not in Grammar 1.0")
    try:
        tokens = _tokenize(text)
    except _LexError as exc:
        raise _ParseError(exc.code, exc.message) from exc
    # Pre-scan paren balance for clearer errors
    depth = 0
    for t in tokens:
        if t.kind == "(":
            depth += 1
        elif t.kind == ")":
            depth -= 1
            if depth < 0:
                raise _ParseError("UNBALANCED_PAREN", "extra ')'")
    if depth != 0:
        raise _ParseError("UNBALANCED_PAREN", "unclosed '('")
    return _Parser(tokens).parse_expression()


def _is_series_value(value: _Value) -> bool:
    return isinstance(value, (_Call, _PrimitiveRef, _FeatureRef))


def _bind_value(
    value: _Value,
    expected_type: str,
    *,
    path: str,
    failures: list[str],
    check_op: Callable[[str], bool],
    check_pr: Callable[[str], bool],
    check_feat: Callable[[str], bool],
    get_schema: Callable[[str], dict[str, Any] | None] | None = None,
    get_arity: Callable[[str], tuple[int, int | None]] | None = None,
    items_schema: dict[str, Any] | None = None,
) -> None:
    """Type-check a value against a schema parameter type (bound mode)."""
    # feature: FEAT_*, nested OP_*, and PR_* (series refs — common for source=PR_SPOT)
    if expected_type == "feature":
        if isinstance(value, _FeatureRef):
            if not check_feat(value.name):
                failures.append(_fail("UNKNOWN_FEATURE", value.name))
            return
        if isinstance(value, _Call):
            _bind_call(
                value,
                failures=failures,
                check_op=check_op,
                check_pr=check_pr,
                check_feat=check_feat,
                get_schema=get_schema,
                get_arity=get_arity,
            )
            return
        if isinstance(value, _PrimitiveRef):
            if not check_pr(value.name):
                failures.append(_fail("UNKNOWN_PRIMITIVE", value.name))
            return
        failures.append(_fail("BAD_LITERAL", f"{path}: expected feature/series"))
        return
    if expected_type == "primitive":
        if not isinstance(value, _PrimitiveRef):
            failures.append(_fail("BAD_LITERAL", f"{path}: expected PR_*"))
            return
        if not check_pr(value.name):
            failures.append(_fail("UNKNOWN_PRIMITIVE", value.name))
        return
    if expected_type in ("window", "integer"):
        if not isinstance(value, _IntLit):
            failures.append(_fail("BAD_LITERAL", f"{path}: expected integer"))
            return
        return
    if expected_type == "float":
        if not isinstance(value, _FloatLit):
            failures.append(_fail("BAD_LITERAL", f"{path}: expected float"))
        return
    if expected_type == "boolean":
        if not isinstance(value, _BoolLit):
            failures.append(_fail("BAD_LITERAL", f"{path}: expected boolean"))
        return
    if expected_type == "string":
        if not isinstance(value, _StringLit):
            failures.append(_fail("BAD_LITERAL", f"{path}: expected string"))
        return
    if expected_type == "list":
        if not isinstance(value, _ListLit):
            failures.append(_fail("BAD_LITERAL", f"{path}: expected list"))
            return
        item_type = None
        if items_schema and isinstance(items_schema, dict):
            item_type = items_schema.get("type")
        if item_type:
            for idx, item in enumerate(value.items):
                _bind_value(
                    item,
                    str(item_type),
                    path=f"{path}[{idx}]",
                    failures=failures,
                    check_op=check_op,
                    check_pr=check_pr,
                    check_feat=check_feat,
                    get_schema=get_schema,
                    get_arity=get_arity,
                    items_schema=None,
                )
        return
    failures.append(_fail("BAD_LITERAL", f"{path}: unknown type {expected_type}"))


def _bind_call(
    call: _Call,
    *,
    failures: list[str],
    check_op: Callable[[str], bool],
    check_pr: Callable[[str], bool],
    check_feat: Callable[[str], bool],
    get_schema: Callable[[str], dict[str, Any] | None] | None = None,
    get_arity: Callable[[str], tuple[int, int | None]] | None = None,
) -> None:
    if not check_op(call.operator):
        # Pattern already OP_*; unknown in registry
        if RE_OP.match(call.operator):
            failures.append(_fail("UNKNOWN_OPERATOR", call.operator))
        else:
            failures.append(_fail("UNEXPECTED_TOKEN", call.operator))
        return

    schema: dict[str, Any] = {}
    if get_schema is not None:
        loaded = get_schema(call.operator)
        if loaded is not None:
            schema = loaded

    props: dict[str, Any] = dict(schema.get("properties") or {})
    required: list[str] = list(schema.get("required") or [])
    amin, amax = (1, 1)
    if get_arity is not None:
        amin, amax = get_arity(call.operator)

    present = {a.name for a in call.args}
    for req in required:
        if req not in present:
            failures.append(_fail("MISSING_PARAM", f"{call.operator}.{req}"))

    series_extras = 0
    for arg in call.args:
        if arg.name in props:
            prop = props[arg.name]
            ptype = str(prop.get("type") or "")
            _bind_value(
                arg.value,
                ptype,
                path=f"{call.operator}.{arg.name}",
                failures=failures,
                check_op=check_op,
                check_pr=check_pr,
                check_feat=check_feat,
                get_schema=get_schema,
                get_arity=get_arity,
                items_schema=prop.get("items") if isinstance(prop.get("items"), dict) else None,
            )
            continue
        # Extra param: allow series slots when operator has input arity
        if (
            amin >= 1
            and arg.name in _SERIES_SLOT_NAMES
            and _is_series_value(arg.value)
        ):
            series_extras += 1
            if isinstance(arg.value, _PrimitiveRef):
                if not check_pr(arg.value.name):
                    failures.append(_fail("UNKNOWN_PRIMITIVE", arg.value.name))
            elif isinstance(arg.value, _FeatureRef):
                if not check_feat(arg.value.name):
                    failures.append(_fail("UNKNOWN_FEATURE", arg.value.name))
            elif isinstance(arg.value, _Call):
                _bind_call(
                    arg.value,
                    failures=failures,
                    check_op=check_op,
                    check_pr=check_pr,
                    check_feat=check_feat,
                    get_schema=get_schema,
                    get_arity=get_arity,
                )
            continue
        failures.append(_fail("UNKNOWN_PARAM", f"{call.operator}.{arg.name}"))

    # Also bind nested calls that appear as values of feature props — already done.
    # Recurse into nested calls under series extras — done.
    _ = (amax, series_extras)  # arity count soft; compiler owns wiring


def _syntax_walk_refs(call: _Call, failures: list[str]) -> None:
    """syntax_only: pattern-check refs; do not hit registries."""
    if not RE_OP.match(call.operator):
        failures.append(_fail("UNEXPECTED_TOKEN", call.operator))
    for arg in call.args:
        _syntax_walk_value(arg.value, failures)


def _syntax_walk_value(value: _Value, failures: list[str]) -> None:
    if isinstance(value, _Call):
        if value.depth > MAX_NESTING_DEPTH:
            failures.append(_fail("NESTING_DEPTH", str(value.depth)))
        _syntax_walk_refs(value, failures)
    elif isinstance(value, _PrimitiveRef):
        if not RE_PR.match(value.name):
            failures.append(_fail("UNEXPECTED_TOKEN", value.name))
    elif isinstance(value, _FeatureRef):
        if not RE_FEAT.match(value.name):
            failures.append(_fail("UNEXPECTED_TOKEN", value.name))
    elif isinstance(value, _ListLit):
        for item in value.items:
            _syntax_walk_value(item, failures)


def validate_expression(
    text: str,
    *,
    mode: str = "syntax_only",
    db_path: Path | None = None,
    expected_grammar_version: str = GRAMMAR_VERSION,
) -> tuple[ValidationReport, _Call | None]:
    """Validate a single TL expression. Returns (report, tree_or_None)."""
    failures: list[str] = []
    warnings: list[str] = []
    tree: _Call | None = None

    if expected_grammar_version != GRAMMAR_VERSION:
        failures.append(_fail("GRAMMAR_VERSION", expected_grammar_version))

    try:
        tree = parse_internal(text)
    except _ParseError as exc:
        failures.append(_fail(exc.code, exc.message))
        return (
            ValidationReport(
                passed=False,
                failed_rules=failures,
                warnings=warnings,
                seed_hash="",
                expected_seed_hash=EXPECTED_GRAMMAR_CHECKSUM,
                validated_objects="0 expressions",
                timestamp=_utc_now(),
            ),
            None,
        )

    _syntax_walk_refs(tree, failures)

    if mode == "bound":
        if db_path is None:
            failures.append(_fail("UNEXPECTED_TOKEN", "bound mode requires --db"))
        else:
            from feature_intelligence.operators.operator_store import OperatorStore
            from feature_intelligence.registry.feature_store import FeatureStore
            from feature_intelligence.registry.store import PrimitiveStore

            op_store = OperatorStore(db_path)
            pr_store = PrimitiveStore(db_path)
            feat_store = FeatureStore(db_path)

            op_cache: dict[str, Any] = {}

            def check_op(oid: str) -> bool:
                if not op_store.table_exists():
                    return False
                return op_store.get_by_id(oid) is not None

            def check_pr(pid: str) -> bool:
                if not pr_store.table_exists():
                    return False
                return pr_store.get_by_id(pid) is not None

            def check_feat(fid: str) -> bool:
                if not feat_store.table_exists():
                    return False
                return feat_store.get_by_uuid(fid) is not None

            def get_schema(oid: str) -> dict[str, Any] | None:
                if oid in op_cache:
                    return op_cache[oid]
                row = op_store.get_by_id(oid)
                if row is None:
                    op_cache[oid] = None
                    return None
                schema = json.loads(row.parameter_schema_json)
                op_cache[oid] = schema
                return schema

            def get_arity(oid: str) -> tuple[int, int | None]:
                row = op_store.get_by_id(oid)
                if row is None:
                    return 0, 0
                return row.input_arity_min, row.input_arity_max

            # Clear pattern-only failures for refs; re-check with registry
            # Keep structural failures already recorded.
            bind_failures: list[str] = []
            _bind_call(
                tree,
                failures=bind_failures,
                check_op=check_op,
                check_pr=check_pr,
                check_feat=check_feat,
                get_schema=get_schema,
                get_arity=get_arity,
            )
            failures.extend(bind_failures)
    elif mode != "syntax_only":
        failures.append(_fail("UNEXPECTED_TOKEN", f"unknown mode {mode}"))

    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for f in failures:
        if f not in seen:
            seen.add(f)
            uniq.append(f)

    passed = len(uniq) == 0
    return (
        ValidationReport(
            passed=passed,
            failed_rules=uniq,
            warnings=warnings,
            seed_hash="",
            expected_seed_hash=EXPECTED_GRAMMAR_CHECKSUM,
            validated_objects="1 expression" if tree is not None else "0 expressions",
            timestamp=_utc_now(),
        ),
        tree,
    )


def validate_text(
    text: str,
    *,
    mode: str = "syntax_only",
    db_path: Path | None = None,
) -> ValidationReport:
    report, _ = validate_expression(text, mode=mode, db_path=db_path)
    return report


def validate_file(
    path: Path,
    *,
    mode: str = "syntax_only",
    db_path: Path | None = None,
) -> ValidationReport:
    text = Path(path).read_text(encoding="utf-8")
    return validate_text(text, mode=mode, db_path=db_path)


def primary_error_code(report: ValidationReport) -> str | None:
    """Extract primary grammar error code from failed_rules."""
    if not report.failed_rules:
        return None
    first = report.failed_rules[0]
    return first.split(":", 1)[0]


def grammar_pack_report() -> ValidationReport:
    """Validate grammar pack checksum against the locked expected value."""
    current = compute_grammar_pack_checksum()
    failed: list[str] = []
    if current != EXPECTED_GRAMMAR_CHECKSUM:
        failed.append(_fail("GRAMMAR_VERSION", "checksum mismatch"))
    return ValidationReport(
        passed=not failed,
        failed_rules=failed,
        warnings=[],
        seed_hash=current,
        expected_seed_hash=EXPECTED_GRAMMAR_CHECKSUM,
        validated_objects="1 grammar pack",
        timestamp=_utc_now(),
    )
