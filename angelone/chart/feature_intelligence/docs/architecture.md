# FIC Architecture Overview (Sprint 0)

```
feature_intelligence/
├── core/           # config, logging, database, paths, benchmarks
├── config/         # YAML: feature_intelligence, database, logging
├── migrations/     # runner + versions/
├── registry/       # Sprint 1+ (empty)
├── operators/      # Sprint 3+ (empty)
├── grammar/        # Sprint 4+ (empty)
├── compiler/       # Sprint 5+ (empty)
├── ast/            # Sprint 5+ (empty)
├── ontology/       # Sprint 6+ (empty)
├── lineage/        # Sprint 7+ (empty)
├── research/       # Sprint 8+ (empty)
├── query/          # Sprint 8+ (empty)
├── api/            # Sprint 9+ (empty)
├── ui/             # Sprint 9+ (empty)
├── validation/     # Sprint 9+ (empty)
├── tests/          # unit / integration / validation
└── docs/
```

## Principles

1. **Adapter, don't replace** — existing feature generation / registry stays; FIC will adapt later.
2. **Migrations-first schema** — all DB changes go through `migrations/versions`.
3. **Config outside code** — runtime knobs live under `config/*.yaml`.
4. **No business logic in Sprint 0** — packages exist as importable stubs only.
