# Module Overview

| Package | Sprint ownership | Sprint 0 state |
|---------|------------------|----------------|
| `core` | Platform | Implemented (config/logging/db/benchmarks) |
| `migrations` | Platform | Implemented (runner + baseline) |
| `config` | Platform | YAML files present |
| `registry` | Sprint 1–2 | Primitive Catalog + Feature Registry / adapter |
| `operators` | Sprint 3 | Operator Registry / pack 1.0.0 (31 ops) |
| `grammar` | Sprint 4 | Stub |
| `compiler` / `ast` | Sprint 5 | Stub |
| `ontology` | Sprint 6 | Feature Ontology / vocab + classification |
| `lineage` | Sprint 7 | Relationship DAG (`LINEAGE_*`, derive from AST) |
| `research` / `query` | Sprint 8 | Stub |
| `api` / `ui` / `validation` | Sprint 9 | Stub |
| `tests` | Platform | Infra smoke tests |
| `docs` | Platform | This documentation set |

## Core entry points

| API | Role |
|-----|------|
| `feature_intelligence.core.load_config` | Load YAML config |
| `feature_intelligence.core.setup_logging` | Configure logger |
| `feature_intelligence.core.init_database` | Create DB + migrate |
| `feature_intelligence.migrations.MigrationRunner` | Upgrade / downgrade / history |
| `python -m feature_intelligence` | CLI |
