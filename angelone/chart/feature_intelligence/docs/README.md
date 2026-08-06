# Feature Intelligence Core (FIC)

Sprint 0–9 foundation + consumption package for AruNeo Feature Intelligence.

**Location:** `angelone/chart/feature_intelligence/`

## What this is

- Package modules for FIC registries, ontology, lineage, research shells
- SQLite database init (`feature_intelligence.db`) + migrations **0001–0009**
- YAML configuration, logging, test harness
- **Sprint 9:** Semantic Query Engine, Public API, Feature Inspector, Model Builder → Feature Intelligence page

**Phase 1 complete** — FIC is publicly consumable via Semantic Query / Inspector / API / CLI.
Later phases may add AI, scoring, Experiment Intelligence, Knowledge Objects, REST — **not** in Phase 1.


## Quick start

From a shell with `angelone/chart` on `PYTHONPATH` (same as other chart modules):

```bash
# Initialize DB + apply migrations
python -m feature_intelligence init-db

# Migration status
python -m feature_intelligence status

# Semantic Query (read-only)
python -m feature_intelligence query capabilities
python -m feature_intelligence query search --query "status:EMPTY"
python -m feature_intelligence query inspect --name <canonical_name>
```

Default DB path:

- Windows: `%APPDATA%/AruNeo/feature_intelligence/feature_intelligence.db`
- Other: `~/.aruneo/feature_intelligence/feature_intelligence.db`

## Tests

```bash
cd angelone/chart
python -m unittest discover -s feature_intelligence/tests -v
```

## Docs

| Doc | Purpose |
|-----|---------|
| [architecture.md](./architecture.md) | High-level layout |
| [module_overview.md](./module_overview.md) | Package map |
| [developer_guide.md](./developer_guide.md) | Day-to-day workflow |
| [coding_standards.md](./coding_standards.md) | Typing / naming / style |
| [research_record.md](./research_record.md) | FRR overview / CLI / policy |
| [research_schema.md](./research_schema.md) | FRR tables + checksum |
| [research_examples.md](./research_examples.md) | Worked FRR examples |
| [semantic_query.md](./semantic_query.md) | Structured query language + CLI |
| [feature_inspector.md](./feature_inspector.md) | Inspector sections / Model Builder FI page |
| [public_api.md](./public_api.md) | Python public API envelopes |

## Governance

- Sprint contract: `docs/antigravity-doc/roadmap_1_feature_intelligence/sprint_9_semantic_query.md`
- Phase 1 FIC: `docs/antigravity-doc/roadmap_1_feature_intelligence/phase_1_feature_intelligence_foundation.md`
