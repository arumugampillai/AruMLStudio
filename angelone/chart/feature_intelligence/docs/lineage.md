# Feature Lineage (Sprint 7)

Deterministic **relationship DAG** for Feature Intelligence Core. Lineage records
how FIC objects are connected — not how they were built (compiler) or what kind
they are (ontology).

> **Invariant: Lineage stores only relationships.**  
> No AST/manifest duplication, no KG reasoning, AI, scoring, semantic search,
> ranking, execution graphs, cycle optimization, or graph analytics.

## Identity

```text
lineage_uuid = LINEAGE_ + SHA256("{parent}|{child}|{relationship_id}")[:32].upper()
```

| Version | Value |
|---------|-------|
| `lineage_version` | `1.0.0` |
| `graph_schema_version` | `1.0` (≠ pack version) |
| `graph_export_version` | `1.0` (on export envelopes) |

## Materialization

| Path | Role |
|------|------|
| **Derive** from Sprint 5 (`ast_nodes` + `feature_ast`) | Primary — CLI `lineage derive` |
| **Import** JSON/YAML/CSV | Secondary |
| CLI freeform edit | Forbidden |

Derive upserts edges only — it never copies AST JSON or manifests into lineage tables.

## CLI

```bash
python -m feature_intelligence lineage list [--relationship REL_*]
python -m feature_intelligence lineage get --id LINEAGE_* 
python -m feature_intelligence lineage get --parent ID --child ID --relationship REL_*
python -m feature_intelligence lineage parents|children|ancestors|descendants --id OBJECT_ID
python -m feature_intelligence lineage validate [--mode strict|present] [--strict-refs]
python -m feature_intelligence lineage derive [--transformation TR_*] [--feature FEAT_*] [--no-closure]
python -m feature_intelligence lineage stats
python -m feature_intelligence lineage export --format json|yaml|csv --out PATH
python -m feature_intelligence lineage import --format json|yaml|csv --path PATH
```

### Snapshot write policy

| Command | `lineage_statistics` + `relationship_statistics` |
|---------|---------------------------------------------------|
| `validate` | **Always** appends snapshot |
| `stats` | Reads latest; writes **only if empty** |
| `import` / `derive` | **Never** write stats (checksum may refresh) |

## Related docs

- [graph_model.md](./graph_model.md)
- [relationship_types.md](./relationship_types.md)

Freeze: `docs/antigravity-doc/roadmap_1_feature_intelligence/sprint_7_lineage_intelligence.md` (v1.1)
