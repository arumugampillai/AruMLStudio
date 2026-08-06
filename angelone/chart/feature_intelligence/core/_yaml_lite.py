"""Minimal YAML subset loader (stdlib-only) for Sprint 0 config files.

Supports nested mappings, lists, scalars (str/int/float/bool/null).
Prefer PyYAML when available; this fallback keeps the foundation runnable
without adding a hard dependency for local smoke tests.
"""

from __future__ import annotations

from typing import Any


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if not text or text in {"null", "Null", "NULL", "~"}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def loads(text: str) -> Any:
    """Parse a restricted YAML document into Python objects."""
    lines = text.splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        content = raw.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"List item without list parent: {raw!r}")
            item_raw = content[2:].strip()
            if item_raw.endswith(":") and ":" == item_raw[-1]:
                key = item_raw[:-1].strip()
                child: dict[str, Any] = {}
                parent.append({key: child} if key else child)
                # Simplified: treat "- key:" as starting a mapping item
                if key:
                    stack.append((indent, child))
                else:
                    stack.append((indent, child))
            elif ":" in item_raw and not item_raw.startswith("{"):
                key, _, val = item_raw.partition(":")
                parent.append({key.strip(): _parse_scalar(val)})
            else:
                parent.append(_parse_scalar(item_raw))
            continue

        if ":" not in content:
            raise ValueError(f"Unsupported YAML line: {raw!r}")

        key, _, rest = content.partition(":")
        key = key.strip()
        rest = rest.strip()

        if not isinstance(parent, dict):
            raise ValueError(f"Mapping entry without dict parent: {raw!r}")

        if rest == "":
            # Peek next non-empty line to decide list vs mapping
            next_indent = None
            next_is_list = False
            for peek in lines[i:]:
                if not peek.strip() or peek.lstrip().startswith("#"):
                    continue
                next_indent = len(peek) - len(peek.lstrip(" "))
                next_is_list = peek.lstrip().startswith("- ")
                break
            if next_indent is not None and next_indent > indent and next_is_list:
                child_list: list[Any] = []
                parent[key] = child_list
                stack.append((indent, child_list))
            else:
                child_map: dict[str, Any] = {}
                parent[key] = child_map
                stack.append((indent, child_map))
        else:
            parent[key] = _parse_scalar(rest)

    return root


def load(path: str | bytes) -> Any:
    """Load a YAML file from a filesystem path."""
    from pathlib import Path

    data = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(data)
    except ImportError:
        return loads(data)
