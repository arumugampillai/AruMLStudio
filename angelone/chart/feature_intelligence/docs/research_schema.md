# Feature Research Record — Schema (Sprint 8)

## Tables

### `research_registry` (pack metadata)

| Column | Notes |
|--------|-------|
| `research_version` PK | e.g. `1.0.0` |
| `schema_version` | e.g. `1.0` — row shape (≠ pack version) |
| `checksum` | SHA-256 of canonical FRR identity/link/status set |
| `description` | Optional |
| `created_at` / `updated_at` | ISO-8601 |

Seed: one pack row for `1.0.0` with empty-set checksum. **No** FRR business rows.

### `feature_research_record`

| Column | Nullability | Notes |
|--------|-------------|-------|
| `research_uuid` PK | NOT NULL | `FRR_*` |
| `feature_uuid` UNIQUE FK | NOT NULL | `FEAT_*` |
| `ontology_uuid` | NULL | `ONT_*` when set must exist |
| `transformation_uuid` | NULL | `TR_*` when set must exist |
| `lineage_version` | NULL | Version pointer |
| `compiler_version` | NULL | Version pointer |
| `grammar_version` | NULL | Version pointer |
| `research_status` | NOT NULL | `EMPTY\|ACTIVE\|ARCHIVED` |
| `validation_status` | NOT NULL | `validated\|pending\|failed` |
| `evidence_json` | NULL | Phase 1 NULL unless import |
| `strengths_json` / `weaknesses_json` / `regimes_json` / `failure_modes_json` | NULL | Placeholders |
| `experiment_ids` | NULL | JSON array of opaque refs |
| `notes` | NULL | Free text |
| `record_source` | NULL | `SYNC\|IMPORT\|MIGRATION` (CHECK; full registry reserved) |
| `created_at` / `updated_at` | NOT NULL | |

### `research_statistics`

Append-only reporting snapshots:

| Column | Meaning |
|--------|---------|
| `total_frr` | COUNT of FRR rows |
| `expected_features` | COUNT of `FEAT_*` in feature registry |
| `coverage_pct` | `100.0 * total_frr / expected` (0 if expected=0) |
| `status_empty` / `status_active` / `status_archived` | Status counts |
| `last_sync_at` | Last successful sync time (nullable until first sync) |
| `research_version` / `schema_version` | Pack + schema at snapshot |
| `created_at` | Snapshot write time |

## Canonical checksum

1. Collect all FRR rows; sort ascending by `research_uuid` (ASCII).
2. For each row, UTF-8 line:

```text
{research_uuid}\t{feature_uuid}\t{ontology_uuid}\t{transformation_uuid}\t{research_status}\t{validation_status}\n
```

(`ontology_uuid` / `transformation_uuid` empty string if null.)

3. SHA-256 hex of concatenated lines.

Does **not** hash evidence body, notes, or timestamps.

## Write policy (stats)

| Caller | Behavior |
|--------|----------|
| validate | Always append |
| sync (success) | Append + set `last_sync_at` |
| stats | Read latest; write only if none |
| import | Never |

Freeze: Sprint 8 v1.1 §8.
