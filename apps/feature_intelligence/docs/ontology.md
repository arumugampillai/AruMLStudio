# Feature Ontology (Sprint 6)

Controlled-vocabulary **classification metadata** for FIC objects. Ontology describes
**what kind** of object something is — not how it was built (compiler) or its ancestry
(lineage / Sprint 7).

**No AI, inference, knowledge graph, semantic search, scoring, or research** in this sprint.

## Identity

```text
ontology_uuid = ONT_ + SHA256("{object_type}:{object_id}")[:32].upper()
```

Re-classification updates the same `ONT_*` in place.

| `object_type` | Table | Assignment |
|---------------|-------|------------|
| `PRIMITIVE` | `primitive_ontology` | Required (14 seed) |
| `OPERATOR` | `operator_ontology` | Required (31 pack) |
| `TRANSFORMATION` | `transformation_ontology` | Optional |
| `FEATURE` | `feature_ontology` | Optional |

## Package

```python
from feature_intelligence.ontology import OntologyService, derive_ontology_uuid

svc = OntologyService(db_path)
svc.get_ontology("PRIMITIVE", "PR_SPOT")
svc.validate_ontology(mode="strict")  # always writes ontology_statistics snapshot
svc.coverage_ontology()               # reads latest; writes only if none
```

One shared `OntologyStore` parameterized by `OBJECT_TYPE_TABLE`. Classification arrives via
**seed migration + import only** — no CLI edit/set/infer.

## CLI

```bash
python -m feature_intelligence ontology list [--type PRIMITIVE|OPERATOR|TRANSFORMATION|FEATURE]
python -m feature_intelligence ontology get --type PRIMITIVE --id PR_SPOT
python -m feature_intelligence ontology validate [--mode strict|present]
python -m feature_intelligence ontology coverage
python -m feature_intelligence ontology export --format json|yaml|csv --out PATH
python -m feature_intelligence ontology import --format json|yaml|csv --path PATH
```

### Snapshot write policy

| Command | `ontology_statistics` |
|---------|------------------------|
| `validate` | **Always** appends snapshot |
| `coverage` | Reads latest; writes **only if empty** |
| `import` | **Never** writes snapshot |

## Related docs

- [controlled_vocabularies.md](./controlled_vocabularies.md)
- [classification_examples.md](./classification_examples.md)

Freeze: `docs/antigravity-doc/roadmap_1_feature_intelligence/sprint_6_feature_ontology.md` (v1.3)
