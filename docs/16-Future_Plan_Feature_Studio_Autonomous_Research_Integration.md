# Future Plan --- Feature Studio ↔ Autonomous Model Research Integration

**Project:** AruMLStudio\
**Status:** Future Plan --- not yet implemented

## 1. Purpose

Feature Studio and Autonomous Model Research should become a closed-loop
research system.

-   **Feature Studio** = feature evidence, health, drift, stability,
    generalization, consensus, freshness, governance and research
    memory.
-   **Autonomous Model Research** = experimentation engine that tests
    feature-set hypotheses using model training, walk-forward
    validation, OOS evidence and trading evidence.

The key principle is:

> Feature Studio tells research **what deserves investigation**.
> Autonomous Research tests those ideas against real model/OOS evidence.
> The resulting evidence flows back into Feature Studio.

## 2. Target Architecture

``` text
Selected Dataset
      ↓
Complete Eligible Feature Universe
      ↓
Feature Studio
  • Production validation
  • Drift
  • Stability
  • Generalization
  • Model consensus
  • Freshness
  • Historical research evidence
  • Governance
      ↓
Feature Evidence + Governance
      ↓
Research Hypotheses
      ↓
Autonomous Model Research
  • Generation 0 full baseline
  • Feature-set mutations
  • RFE / SHAP / Permutation / Ablation
  • Walk-forward
  • OOS validation
  • Trading replay
      ↓
Candidate Results
      ↓
Feature Usage / Contribution Evidence
      ↓
Feature Studio
      ↓
Next Research Cycle
```

## 3. Feature Studio Is Not Just Feature Elimination

Traditional elimination asks:

> Which features are unnecessary for this particular model?

Feature Studio should answer:

> Which features have demonstrated reliable value across production
> validation, time, regimes, models, drift and accumulated research
> evidence?

Therefore:

-   RFE = model-specific evidence.
-   SHAP = model-specific evidence.
-   Permutation importance = model-specific evidence.
-   Feature Studio = broader feature-level evidence and governance.

All can work together.

## 4. Authoritative Dataset

Research must be anchored to the selected Analysis Dataset.

The dataset provides:

-   Dataset ID/name
-   Snapshot hash
-   Market
-   Sampling interval
-   Rows
-   Trading days
-   Feature schema
-   Target schema
-   Eligible feature columns

The research engine must derive its feature universe from the selected
dataset and must not invent a separate feature universe.

## 5. Generation 0 Full Baseline

If the selected dataset has 382 eligible features:

``` text
Generation 0
    ↓
382 features
    ↓
Baseline candidates
```

Generation 0 must remain the full-feature reference point.

Feature Studio must not silently remove features before the baseline is
measured.

Example:

``` text
FULL FEATURE BASELINE
382 features
Composite = 75.80
```

All later feature-set experiments are compared with this baseline.

## 6. Feature Evidence

Feature Studio should maintain an evidence profile for each feature:

-   Evidence score
-   Confidence
-   Stability
-   Generalization
-   Model consensus
-   Freshness
-   Drift
-   Production validation history
-   Research usage
-   Health status
-   Governance recommendation
-   Feature interactions/synergies

Example:

``` text
iv_zscore_15m
Evidence Score: +82
Confidence: 91%
Generalization: STRONG
Model Consensus: 4/5
Drift: LOW
Freshness: RECENT
Recommendation: KEEP
```

Another feature could be:

``` text
current_iv
Evidence Score: -60
Confidence: 74%
Generalization: WEAK
Model Consensus: REMOVE
Drift: HIGH
Recommendation: REMOVE
```

The exact scoring weights and governance thresholds should remain
controlled by the existing Feature Studio policy.

## 7. Feature Governance States

The system should preserve meaningful feature states such as:

-   STRONG
-   PROMISING
-   ACTIVE
-   WATCH
-   HELD
-   REMOVE / REJECTED / HARMFUL
-   EXPERIMENTAL
-   RETIRED

Important rule:

**WATCH does not mean delete.**

It means continue collecting evidence.

## 8. Feature Studio → Autonomous Research

Feature Studio should provide research with:

``` text
feature_name
health_status
evidence_score
confidence
stability
generalization
model_consensus
freshness
drift status
governance recommendation
reason
```

Research uses this information to construct explicit hypotheses.

## 9. Research Hypotheses

Instead of simply selecting one elimination algorithm, Autonomous
Research should test different evidence-driven hypotheses.

### Hypothesis A --- High-Confidence Remove

``` text
Full universe
    ↓
remove only high-confidence REMOVE features
    ↓
train
    ↓
OOS/trading evaluation
```

### Hypothesis B --- Strong + Promising

``` text
Full universe
    ↓
STRONG + PROMISING
    ↓
train
    ↓
evaluate
```

### Hypothesis C --- Drift-Based

Remove or isolate features identified by Feature Studio as having
unacceptable drift according to its governance rules.

### Hypothesis D --- Generalization-Based

Test the effect of removing features with poor cross-context or
cross-regime generalization.

### Hypothesis E --- Model Consensus

Test feature sets based on agreement across multiple model families.

### Hypothesis F --- Feature Studio Recommended Set

Create a candidate from the current Feature Studio governance
recommendation.

Every hypothesis must be recorded explicitly so the result is
explainable.

## 10. RFE / SHAP / Permutation Importance

These techniques should remain available, but they should be treated as
**research tools**, not as the entire feature-governance system.

Examples:

``` text
Feature Studio Evidence + RFE
Feature Studio Evidence + SHAP
Feature Studio Evidence + Permutation
Feature Studio Evidence + Ablation
```

This lets research answer:

> Does model-specific feature importance agree with accumulated Feature
> Studio evidence?

That is more valuable than blindly running RFE for every candidate.

## 11. Feature Research Policy

The future Autonomous Research UI should preferably expose a broader
policy rather than making one elimination algorithm the main control.

Possible policies:

``` text
○ Full Universe Baseline
○ Feature Studio Evidence-Driven
○ Hybrid Evidence + ML Elimination
```

### Full Universe Baseline

Use all eligible dataset features.

### Feature Studio Evidence-Driven

Use Feature Studio evidence and governance recommendations to create
feature-set hypotheses.

### Hybrid Evidence + ML Elimination

Combine Feature Studio evidence with RFE, SHAP, permutation importance
and ablation.

The selected policy must be persisted with the campaign.

## 12. Candidate Feature Lineage

Every candidate must have an exact feature lineage.

Persist:

``` text
candidate_id
dataset_id
dataset_snapshot_hash
parent_candidate_id
generation_number
feature_policy
elimination_method
feature_count
features[]
```

Example:

``` text
Candidate:
CAND_NIFTY_RAN_202d9fe6

Generation:
0

Feature Policy:
FULL_FEATURE_BASELINE

Elimination:
NONE

Features:
382
```

Later:

``` text
Candidate:
CAND_NIFTY_XGB_abc123

Generation:
2

Parent:
CAND_NIFTY_RAN_202d9fe6

Feature Policy:
FEATURE_STUDIO_EVIDENCE_DRIVEN

Elimination:
HIGH_CONFIDENCE_REMOVE

Features:
310
```

## 13. Training Matrix Integrity

Before training, verify:

``` text
CandidateSpec.features == Training X columns
```

Also verify:

-   No target columns are present.
-   No duplicate features exist.
-   Every requested feature exists in the selected dataset.
-   Dataset snapshot matches the candidate.
-   Feature count matches.
-   Feature ordering is deterministic.

Any mismatch should produce a hard **FEATURE LINEAGE ERROR** instead of
silently changing the feature set.

## 14. Candidate → Feature Studio Feedback

Autonomous Research should send back:

``` text
candidate_id
dataset_id
generation
algorithm
feature_set
features_added
features_removed
model_score
trading_score
OOS result
regime result
feature importance
ablation evidence
candidate outcome
```

This creates new feature evidence.

## 15. Feature Usage Analytics

Feature Studio should eventually show research usage for every feature.

Example:

``` text
Feature: iv_zscore_15m

Candidates Tested: 18
Candidates Using Feature: 15
Candidates Excluding Feature: 3
Winning Candidates: 4

Best Composite With Feature: 78.42
Average Score With Feature: 76.8
Average Score Without Feature: 73.9

Observed Difference: +2.9
```

This creates a distinction between:

-   A feature that merely looks strong.
-   A feature that repeatedly improves real OOS candidates.

The second becomes stronger evidence.

## 16. Candidate Feature Composition View

The Research Leaderboard should provide a Feature Composition view for
each candidate.

Example:

``` text
CAND_NIFTY_RAN_202d9fe6

Total Features: 382
Feature Policy: FULL_FEATURE_BASELINE

Registry Features: ...
Baseline Pipeline Features: ...
Experimental Features: ...

Strong: ...
Promising: ...
Held: ...
Remove: ...
```

The counts must be calculated from the actual candidate feature list and
the Feature Studio classification snapshot.

The user should be able to filter:

``` text
ALL
STRONG
PROMISING
ACTIVE
HELD
REMOVE
EXPERIMENTAL
```

Clicking a feature should expose its Feature Studio evidence.

## 17. Feature → Candidate Reverse Lookup

Feature Studio should eventually show:

``` text
Feature: iv_zscore_15m

Candidates Tested: 18
Candidates Using: 15
Candidates Excluding: 3
Winning Candidates: 4
Best Composite: 78.42
Average Score With Feature: 76.8
Average Score Without Feature: 73.9
```

This creates two-way feature/model lineage.

## 18. Separate Feature Score and Model Score

These scores must never be treated as one combined numeric score.

### Feature Evidence Score

Answers:

> How strong and reliable is this feature based on accumulated evidence?

### Model Composite Score

Answers:

> How good is this particular feature set + algorithm + configuration on
> the research evaluation?

Correct relationship:

``` text
Feature Evidence
      ↓
Feature Set
      ↓
Model
      ↓
OOS Model Evidence
      ↓
Trading Evidence
      ↓
Composite Model Score
      ↓
New Feature Evidence
```

## 19. Closed-Loop Research

The long-term system should continuously cycle:

``` text
Feature Studio
      ↓
Evidence / Governance
      ↓
Research Agenda
      ↓
Autonomous Research
      ↓
Model + OOS + Trading Evidence
      ↓
Feature Usage / Contribution
      ↓
Feature Studio Evidence Update
      ↓
Next Research Cycle
```

This is the intended continuous research loop.

## 20. Research Generation Example

Suppose:

``` text
Dataset = analysis_198r_171b_6s_20260820_223630
Eligible Features = 382
```

### G0

``` text
382 features
Random Forest
Composite = 75.80
```

### Feature Studio Evidence

Suppose Feature Studio identifies 70 high-confidence REMOVE features.

### G1

``` text
382 → 312 features
Policy = FEATURE_STUDIO_EVIDENCE_DRIVEN
Composite = 76.40
```

### G2 --- RFE

``` text
312 → 180 features
Policy = HYBRID
Method = RFE
Composite = 77.10
```

### G2 --- SHAP

``` text
312 → 150 features
Policy = HYBRID
Method = SHAP
Composite = 76.60
```

The better result becomes new research evidence.

Feature Studio then receives the observed feature usage and candidate
outcomes.

The next research cycle starts with the updated evidence.

## 21. Governance

Feature Studio recommendations remain advisory.

The existing governance principle should remain:

``` text
Research Memory is advisory.
Candidate promotions require human governance approval.
```

Therefore:

``` text
Feature Studio
    ↓
Recommendation

Autonomous Research
    ↓
Evidence

Research Leaderboard
    ↓
Candidate ranking

Human Governance
    ↓
Promotion decision
```

No feature should automatically become a production requirement merely
because Feature Studio marks it Strong.

No feature should be permanently removed merely because one candidate
did not use it.

## 22. Audit Trail

Every candidate should record:

``` text
dataset_name
dataset_snapshot_hash
feature_policy
elimination_method
feature_count
feature_list
feature_studio_snapshot
feature evidence references
parent_candidate_id
generation_number
algorithm
hyperparameters
model metrics
trading metrics
OOS metrics
candidate verdict
```

Recommended future event types:

``` text
FEATURE_POLICY_SELECTED
FEATURE_UNIVERSE_RESOLVED
CANDIDATE_CREATED
FEATURES_ADDED
FEATURES_REMOVED
CANDIDATE_EVAL_START
CANDIDATE_EVAL_DONE
CANDIDATE_VERDICT
FEATURE_EVIDENCE_CONSUMED
FEATURE_EVIDENCE_UPDATED
```

## 23. Feature Studio Snapshot

A candidate must not depend on mutable Feature Studio state without
recording what state existed when it was generated.

Persist references such as:

``` text
feature_studio_snapshot_id
feature_policy_version
feature_evidence_version
governance_policy_hash
```

This allows old research candidates to remain reproducible even after
Feature Studio changes.

## 24. Feature Categories

The future architecture should preserve:

### Registry Features

Already registered and governed.

### Baseline Pipeline Features

Approved baseline pipeline features.

### Experimental Pipeline Features

Auto-generated or experimental features under research.

### Retired Features

No longer eligible.

Research must not silently merge these categories into one
undifferentiated feature pool.

## 25. Future Auto-Generated Feature Pipeline

Auto-generated features should be added only after the evidence-driven
research loop is stable.

Target flow:

``` text
Existing Feature Universe
        ↓
Feature Studio Evidence
        ↓
Research discovers gaps
        ↓
Auto Feature Generator
        ↓
New Experimental Features
        ↓
Analysis Dataset
        ↓
Feature Validation
        ↓
Feature Studio
        ↓
Experimental → Promising / Strong
        ↓
Autonomous Research
```

Newly generated features should initially be classified as:

``` text
EXPERIMENTAL
```

They should earn stronger governance status through evidence.

## 26. Recommended Implementation Order

### Phase A --- Feature Lineage

Implement:

-   Candidate feature list persistence
-   Dataset snapshot binding
-   Feature policy persistence
-   Training-matrix verification
-   Candidate Feature Composition UI

### Phase B --- Feature Studio Research Integration

Implement:

-   Feature evidence snapshot
-   Feature governance snapshot
-   Feature Studio → Research interface
-   Research → Feature Studio evidence interface

### Phase C --- Evidence-Driven Research Policies

Implement:

-   Full Universe Baseline
-   Feature Studio Recommended Set
-   High-confidence Remove
-   Strong + Promising
-   Drift-based research
-   Generalization-based research

### Phase D --- Hybrid ML Feature Analysis

Implement:

-   RFE
-   SHAP
-   Permutation Importance
-   Ablation
-   Combined experiments

### Phase E --- Closed-Loop Evidence

Implement:

-   Candidate usage statistics
-   Feature win-rate across candidates
-   With-feature vs without-feature analysis
-   Feature contribution history
-   Cross-model feature consensus

### Phase F --- Auto-Generated Features

Only after the above is stable:

-   Automatic feature generation
-   Experimental feature registration
-   Experimental validation
-   Feature discovery feedback
-   Automatic research agenda generation

## 27. Final Target

The final system should behave as a **Feature Evidence ↔ Autonomous
Research closed loop**.

Feature Studio should answer:

> What features deserve attention, and why?

Autonomous Research should answer:

> Does changing the feature set actually improve robust
> OOS/model/trading performance?

The combined system should answer:

> Which features have accumulated enough evidence to keep, which should
> be watched, which should be removed from future experiments, and which
> new features should be investigated next?

## 28. Scope Status

This document is a **future-plan specification only**.

It does not authorize:

-   Immediate implementation
-   Schema changes
-   Automatic production promotion
-   Automatic feature retirement
-   Automatic governance changes
-   Automatic model deployment

Implementation should begin only after the current Feature Studio and
Autonomous Research architecture is considered stable.
