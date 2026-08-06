# Transformation Language Grammar (Sprint 4)

TL **1.0** — function-form feature construction language.

| Field | Value |
|-------|-------|
| `grammar_version` | `1.0` |
| `grammar_pack_version` | `1.0.0` |
| `token_pack_version` | `1.0.0` |
| `formatter_version` | `1.0.0` |

Sprint 4 ships a **syntax validator** + **canonical formatter**. No compiler, no public AST, no execution.

```python
from feature_intelligence.grammar import validate_text, format_expression

validate_text("OP_EMA(period=20)", mode="syntax_only")
format_expression("OP_EMA(period=20)")
```

Artifacts: `grammar/ebnf/tl_v1.ebnf`, `grammar/tokens.json`, `grammar/grammar_compatibility.json`.

Checksum: SHA-256 of `ebnf || compatibility || tokens` (see `compute_grammar_pack_checksum()`).

CLI: `grammar version|validate|format|export|import`.

Freeze: `docs/antigravity-doc/roadmap_1_feature_intelligence/sprint_4_transformation_language.md`
