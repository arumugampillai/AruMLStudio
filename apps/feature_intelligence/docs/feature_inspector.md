# Feature Inspector

Sprint 9 — unified **metadata** inspect view over one FRR subject.

## Studio UX (Feature Explorer)

Model Builder → **Feature Intelligence** (sidebar page, route `builder.feature_intelligence`) opens a read-only explorer:

1. **Search** — simple text or advanced `field:value` tokens (alias `feat:` → `feature:`)  
2. **Scope** checkboxes — Name · FEAT · Primitive · Operator · Ontology  
3. **Sync from Feature Registry** — admin pull of the legacy catalog into FIC `feature_registry` (idempotent; optional research FRR sync). Shows a summary dialog.  
4. **Results table** — Feature Name, FEAT ID, Research Status, Domain, Primary Operator, Primary Primitive, Compiler Version, Ontology Version; click headers to sort; select / double-click to inspect  
5. **Empty state** — Overview shows platform summary from `get_platform_summary()` (registry counts + pack versions) until a feature is selected  

### Search → query engine mapping

| Input | Behavior |
|-------|----------|
| Empty + Search / List all | `search_features(match_all=True)` |
| All tokens are `field:value` | Pass-through structured query (scopes ignored) |
| `FEAT_*` / `PR_*` / `OP_*` / `TR_*` / `DOM_*` / `SIG_*` | Mapped to the matching engine field |
| `FRR_*` | `match_all` + client filter on `research_uuid` |
| Scope **Name** | `match_all` + case-insensitive substring on `canonical_name` (engine `feature:` is exact-only) |
| Scope **FEAT** | `match_all` + substring on `feature_uuid` |
| Scope **Primitive** / **Operator** | `primitive:<token>` / `operator:<token>` |
| Scope **Ontology** | `domain:<token>` ∪ `signal:<token>` |

Multiple scopes are **OR**-united. Unresolved primitive/operator/ontology tokens soft-fail that branch (empty hits).

Search hits are **enriched** in the query service (domain / primary operator / primary primitive / versions) so the UI never queries SQLite.

### FRR-mandatory (why List All can show 0)

Sprint 9 Semantic Query **always** starts from `feature_research_record` (FRR), never raw `feature_registry`.  
`List all` runs `SELECT * FROM feature_research_record …` (`match_all=True`).

If features exist in `feature_registry` but research sync never ran, FRR count is 0 and Explorer correctly returns **0 hits**. The Results status label and Overview dashboard note explain this and point to:

```bash
# from angelone/chart (package root on PYTHONPATH)
python -m feature_intelligence init-db          # if schema missing
python -m feature_intelligence features sync   # import legacy + research FRR shells
python -m feature_intelligence research sync   # materialize EMPTY FRR shells only
```

Or use **Sync from Feature Registry** in the Feature Explorer toolbar (runs feature sync + research sync).

Default DB path: `%APPDATA%/AruNeo/feature_intelligence/feature_intelligence.db` (Windows).

## Sections (card layout)

| Section | Content |
|---------|---------|
| Overview | Feature name, FEAT/FRR/TR, status, compiler/grammar/ontology/lineage versions, created/updated, templated `overview_summary`, architecture strip `FEAT → TR → ONT → FRR`, `sections_present` chips |
| Identity | UUID, canonical name, def/impl versions, controller owner, warmup, gap policy, memory model, primitive dependencies |
| Compiler | Visual stack: Transformation → Grammar → AST summary → Manifest; optional Raw JSON expander |
| Ontology | Chips for Domain / Signal / Math family / Output / Horizon / Frequency / Stability |
| Lineage | Parents, children, primitive inputs, operators used, transformation chain text tree |
| Research | FRR statuses, source, experiment refs, notes, evidence — stored only |
| References | Models / Datasets / Experiments / Research Programs; Phase 1 empty → **"No references found."** (`get_references` stub) |

**Must NOT show:** importance scores, distribution/drift charts, model metrics, recommendations.

Header feel when loaded: `Feature : <display name>` plus FEAT/FRR/TR/Status/Primitive/Operator/Ontology/Research chips.

Inspector load is **async** (worker thread + UI marshal) so search stays snappy.

## Public API helpers (read-only)

| Callable | Purpose |
|----------|---------|
| `search_features` | Enriched FRR search hits for the results grid |
| `inspect_feature` | Full inspector envelope |
| `get_platform_summary` | Dashboard counts + pack versions |
| `get_references` | Phase 1 stub — empty linkage lists |

## `sections_present`

`inspect_feature` always includes a boolean completeness map (reporting only):

```json
{
  "sections_present": {
    "identity": true,
    "compiler": false,
    "ast": false,
    "ontology": false,
    "lineage": false,
    "research": true,
    "references": true
  }
}
```

Missing optional links do **not** fail inspect. Absent sections render grey “(absent)” chips/rows.

## Deep links (reserved)

Click-through from OP_/TR_/ONT_/FRR_ ids to registry browsers is **reserved** post Phase 1. Phase 1 displays ids as plain text only.

