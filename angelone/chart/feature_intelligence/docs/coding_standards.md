# Coding Standards (FIC)

Aligned with AruNeo chart Python style.

## Typing

- Use `from __future__ import annotations`
- Annotate public functions and dataclasses
- Prefer `Path` over raw strings for filesystem paths

## Naming

- Packages: short lowercase nouns (`registry`, `grammar`)
- Modules: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE
- Migration files: `NNNN_description.py` with zero-padded version

## Documentation

- Module docstring states purpose and sprint ownership
- Public APIs get a one-line docstring minimum
- Do not document unimplemented business APIs as if they exist

## Formatting / linting

- Follow existing chart conventions (4-space indent, no unused imports)
- Keep Sprint stubs empty rather than filling with placeholder business APIs

## Dependencies

- Prefer stdlib for infrastructure
- New third-party deps must be added to root `requirements.txt` and justified in the PR/sprint notes
- `PyYAML` is recommended for config; lite fallback ships in `core/_yaml_lite.py`
