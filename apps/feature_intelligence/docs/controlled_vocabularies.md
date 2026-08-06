# Controlled Vocabularies (Sprint 6)

Persisted in `vocabulary_registry`. Semantic PK is `vocabulary_id`. Internal
`vocabulary_pk` (AUTOINCREMENT) is **never** exposed in CLI, export, or API.

Fields: `vocabulary_id`, `vocabulary_type`, `canonical_name`, `display_name`,
`description`, `ontology_version`, `active`, optional `retired_reason` / `sort_order`.

Ontology rows store **vocabulary_id** values only (e.g. `DOM_PRICE`), never display names.

Pack: `ontology_version` / `vocab_pack_version` = **1.0.0** (64 entries).

## Domain — `DOM_*`

| vocabulary_id | canonical_name | display_name |
|---------------|----------------|--------------|
| DOM_PRICE | price | Price |
| DOM_VOLUME | volume | Volume |
| DOM_OPEN_INTEREST | open_interest | Open Interest |
| DOM_VOLATILITY | volatility | Volatility |
| DOM_TIME | time | Time |
| DOM_CALENDAR | calendar | Calendar |
| DOM_ORDER_FLOW | order_flow | Order Flow |
| DOM_GREEK | greek | Greek |
| DOM_CONTRACT | contract | Contract |
| DOM_QUOTE | quote | Quote |
| DOM_DERIVED | derived | Derived |

## Signal Type — `SIG_*`

`SIG_RAW`, `SIG_LEVEL`, `SIG_TREND`, `SIG_MOMENTUM`, `SIG_MEAN_REVERSION`,
`SIG_VOLATILITY`, `SIG_LIQUIDITY`, `SIG_PARTICIPATION`, `SIG_SPREAD`, `SIG_RATIO`,
`SIG_DIFFERENCE`, `SIG_STATISTICAL`, `SIG_STRUCTURE`, `SIG_INTERACTION`, `SIG_TRANSFORM`

## Mathematical Family — `MATH_*`

`MATH_IDENTITY`, `MATH_MOVING_AVERAGE`, `MATH_ROLLING_WINDOW`, `MATH_NORMALIZATION`,
`MATH_ARITHMETIC`, `MATH_STATISTICAL`, `MATH_COMPARISON`, `MATH_RANKING`,
`MATH_TRANSFORMATION`, `MATH_AGGREGATION`, `MATH_TIME_SHIFT`, `MATH_SLOPE`, `MATH_INTERACTION`

## Horizon — `HOR_*`

`HOR_TICK`, `HOR_INTRADAY`, `HOR_SHORT`, `HOR_MEDIUM`, `HOR_LONG`, `HOR_MULTI_SCALE`,
`HOR_CONTRACT`, `HOR_STATIC`

## Output Type — `OUT_*`

`OUT_NUMERIC`, `OUT_BOOLEAN`, `OUT_CATEGORY`, `OUT_RANKING`, `OUT_PROBABILITY`

## Frequency — `FREQ_*`

`FREQ_TICK`, `FREQ_1S`, `FREQ_3S`, `FREQ_1M`, `FREQ_5M`, `FREQ_15M`, `FREQ_DAILY`,
`FREQ_ANY`, `FREQ_EVENT`

## Stability — `STAB_*`

`STAB_STABLE`, `STAB_EXPERIMENTAL`, `STAB_DEPRECATED`

Stability is **ontology metadata only** — not the operator lifecycle engine.

## Soft-retire

Set `active=0` and optionally `retired_reason`. Existing ontology rows may keep the id;
**new** assignments reject inactive ids.
