# Feature Research Record — Examples (Sprint 8)

## Sync creates EMPTY shell

Given a registered feature:

```text
feature_uuid = FEAT_0123456789ABCDEF0123456789ABCDEF
```

After `research sync`:

```text
research_uuid     = FRR_ + SHA256("FEAT_0123456789ABCDEF0123456789ABCDEF")[:32].upper()
research_status   = EMPTY
validation_status = pending
evidence_json     = NULL
record_source     = SYNC
```

CLI output:

```text
created=1 updated=0 unchanged=0 skipped=0
```

Re-running sync with no changes:

```text
created=0 updated=0 unchanged=1 skipped=0
```

## Import with links and evidence

Import may set nullable links and evidence placeholders (sync must not fabricate evidence):

```json
{
  "research_version": "1.0.0",
  "schema_version": "1.0",
  "research_export_version": "1.0",
  "records": [
    {
      "research_uuid": "FRR_…",
      "feature_uuid": "FEAT_…",
      "ontology_uuid": "ONT_…",
      "transformation_uuid": "TR_…",
      "research_status": "ACTIVE",
      "validation_status": "pending",
      "evidence_json": "{\"note\":\"imported only\"}",
      "record_source": "IMPORT"
    }
  ]
}
```

Import refreshes checksum but does **not** write `research_statistics`.

## Coverage fail (strict validate)

If a `FEAT_*` lacks an FRR:

```text
failed_rules: ["MISSING_FRR"]
```

Validate never creates the missing row — run `research sync` (or import) first.

## Completeness gaps (reporting only)

```bash
python -m feature_intelligence research completeness
```

Example:

```json
{
  "total_frr": 1,
  "complete": 0,
  "incomplete": 1,
  "gaps": [
    {
      "research_uuid": "FRR_…",
      "feature_uuid": "FEAT_…",
      "missing_fields": [
        "ontology_uuid",
        "transformation_uuid",
        "compiler_version",
        "grammar_version",
        "lineage_version"
      ]
    }
  ]
}
```

Missing links do **not** fail `research validate`.

## Migration ≠ Business rows

After `migrate` / `init-db` through `0009`:

- `research_registry` has pack `1.0.0`
- `feature_research_record` has **0** rows
- FRRs appear only after sync or import
