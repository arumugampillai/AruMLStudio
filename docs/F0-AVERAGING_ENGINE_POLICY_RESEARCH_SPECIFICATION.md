# AVERAGING ENGINE POLICY RESEARCH SPECIFICATION
Document ID: F0-AVERAGING_ENGINE_POLICY_RESEARCH_SPECIFICATION
Status: DESIGN / RESEARCH SPECIFICATION

## 1. Executive Decision & System Purpose

The final production trading system consists of **TWO separate production trading engines**:
1. **Strategy Allocation Engine** (Pre-Entry Decision Engine)
2. **Averaging Engine** (Post-Entry Position Management Engine)

**AruMLStudio** is the **research, validation, fine-tuning, evidence, and governance platform** responsible for continuously improving the decision models and policies used by these two engines. AruMLStudio is **NOT** the live execution engine and does **NOT** place broker orders.

The Averaging Engine must not assume a fixed averaging sequence, fixed adverse-move spacing, fixed basket target, or fixed maximum number of additions. These are **empirical research variables** discovered, validated, and fine-tuned inside AruMLStudio.

The final production engine will separate Entry Decision, Target Classification, Meta-Confidence Gate, Averaging Trigger Policy, Position-Size/Lot Ladder Policy, Basket-Level Target and Exit Policy, Hard Risk Guard, and Intraday/Expiry Safety Rules.
The ML system may recommend an action, but the Risk Engine has final authority. A high model confidence score can never override a hard exposure or loss limit.

## 2. Strategic Objective: Classification-First Modeling

AruMLStudio is designed to discover and fine-tune decision policies for an intraday current-expiry option-buying system with controlled averaging into adverse movement.

> [!IMPORTANT]
> **Classification-First Production Principle**:  
> The production decision layer is strictly **classification-first**. Regression prediction of exact future option LTP or rupee P&L is **NOT** a production dependency. Classification models (action classes, recovery probability classes, target classes) and meta-confidence models provide the verified decision evidence.

The engine must continuously calculate live multi-dimensional basket state.

## 3. Core Principle: Basket, Not Individual Entry

An averaging trade is one basket position. The engine must maintain total quantity, weighted average entry, current option LTP, basket unrealized P&L, basket return percentage, current averaging level, premium exposure, remaining risk budget, target price, distance to target, adverse excursion, and time state.
The normal profit objective applies to the whole basket rather than assigning independent targets to every entry.

## 4. Classification Model Layer

Initial target classifiers may estimate the probability of reaching +2%, +3%, and +4% within a defined horizon. The model is not asked to predict an exact future option LTP.
These probabilities are inputs to the decision engine, not direct trading commands.

## 5. Meta / Confidence Model

The meta model answers: How trustworthy is the current classification output for this specific market state?
It may use classifier probabilities, agreement between target classifiers, market regime, time to expiry, averaging level, basket exposure, recent calibration, model stability, feature validity, and historical context performance.
Confidence is a gate. Low confidence can block an additional position even when the raw classifier is positive.

## 6. Target Selection

The basket target must be researched rather than permanently fixed. Initial research candidates may include +1%, +1.5%, +2%, +2.5%, +3%, +3.5%, +4%, and +5%.
The final target may eventually be conditional on classifier probabilities, meta confidence, basket state, averaging level, exposure, time, regime, and volatility.
Any adaptive target policy must be validated out-of-sample.

## 7. Averaging Trigger Research

The project must discover how much adverse movement should occur before another position can be added. Initial candidate spacings may include 0.5%, 1%, 1.5%, 2%, 2.5%, 3%, 4%, and 5%.
The reference price must be defined unambiguously: previous entry, last addition, first entry, basket average, or another canonical reference. Production must use one definition.

## 8. Position Sequence Research

Candidate quantity ladders can include 1,1,1,1,1,1; 1,1,1,1,2,2; 1,1,1,1,2,2,2,2; 1,1,1,1,2,2,4,4; 1,1,2,2,4,4,8,8; and conservative custom ladders.
These are research candidates only. No sequence is approved by this document.

## 9. Position Size Must Be Evaluated With Risk

Every ladder must be evaluated for maximum lots, premium deployed, capital exposure, maximum basket size, target hit rate, recovery probability, recovery time, maximum adverse excursion, failure after each level, loss after maximum level, consecutive failed baskets, expectancy, profit factor, drawdown, and tail loss.
A sequence with high average profit but dangerous tail exposure must be rejected.

## 10. Maximum Averaging Level

The engine must have a hard maximum averaging level. After the final permitted level, no additional averaging is allowed.
The classifier cannot authorize a position beyond the risk policy.

## 11. Risk Engine Has Absolute Authority

The hierarchy is: Model → Target/Add Recommendation → Risk Engine → Allow/Reduce/Block → Order Engine.
Hard controls include maximum basket size, maximum lots, maximum premium exposure, maximum loss, maximum averaging level, daily loss limit, session stop, expiry safety, liquidity/spread checks, stale-data checks, and model-health checks.

## 12. Basket Target Calculation

The default design should calculate the target from the basket weighted average rather than the first entry.
Weighted average = sum(entry price × quantity) / total quantity. The selected basket target percentage is then applied to that basket average, with execution-cost assumptions handled consistently.

## 13. No Independent Entry Targets by Default

The default structure is: multiple entries → one basket → one target/exit policy.
An individual-entry exit should be introduced only if empirical research proves that it improves the complete basket policy without creating unacceptable risk.

## 14. Time-to-Target

Because the system is intraday and short-duration, target probability alone is insufficient. Research should evaluate probabilities such as +2% within 15/30/60/120 seconds and similarly for other targets.
The exact horizons should be selected from the actual trade-duration distribution.

## 15. Averaging Decision State Machine

NO POSITION → ENTRY SIGNAL → INITIAL LOT → CLASSIFIERS → META CONFIDENCE → TARGET SELECTION → WAIT FOR PRICE STATE → AVERAGING TRIGGER → RISK ENGINE → ADD or BLOCK → RECALCULATE BASKET → TARGET/EXIT.
After every addition, the model state should be re-evaluated because weighted average, exposure, target distance, downside, and risk concentration have changed.

## 16. Required Research Experiments

Group A — Entry: entry thresholds, model confidence thresholds, regime-specific entry rules.
Group B — Spacing: fixed percentage spacing, adaptive spacing, regime-dependent spacing.
Group C — Quantity Ladder: linear, stepped, geometric, capped-geometric, and conservative custom ladders.
Group D — Target: fixed targets versus classifier-selected targets versus confidence-conditioned targets.
Group E — Exposure: maximum levels, maximum lots, maximum premium, maximum basket risk.
Group F — Time: target time limits, stale-basket handling, and end-of-session forced exit.

## 17. Research Objective

The policy must not be optimized on raw P&L alone. Evaluation should jointly consider expectancy, target hit rate, recovery rate, profit factor, time efficiency, maximum drawdown, tail loss, maximum exposure, failed basket rate, averaging depth, and model instability.
The exact objective weights should be established as a separate governance decision.

## 18. Anti-Overfitting Rules

Walk-forward validation is mandatory. Multiple market regimes, expiry/non-expiry sessions, volatility conditions, time-of-day segments, realistic transaction costs, and unseen periods must be evaluated.
Parameter sensitivity must be tested. A policy that works only at one exact spacing, target, and ladder is suspicious.

## 19. Required Policy Dossier

The eventual research output should specify context, entry policy, classifier signatures, meta-model signature, averaging spacing by level, quantity ladder, maximum level, maximum lots, target policy, target selector, maximum exposure, maximum loss, maximum holding time, evidence confidence, walk-forward results, worst regime, worst basket, and tail-risk status.

## 20. Relationship With Existing AruMLStudio Architecture

The Averaging Engine should reuse the existing Model Taxonomy, Experiment Signatures, Research Memory, Benchmarks, Regime Evaluation, Feature Composition, Robustness Ranking, Negative Evidence, Research Priority, and Recommendation Dossier infrastructure.
It must not create a parallel research identity or duplicate existing business logic.

## 21. Critical Architectural Boundary: Model Quality vs Complete Trading Engine Quality

Model quality is **not** trading engine quality. A model champion is **not** automatically an averaging-policy champion.

$$\boxed{\text{MODEL} \longrightarrow \text{DECISION POLICY} \longrightarrow \text{POSITION / CAPITAL POLICY} \longrightarrow \text{RISK POLICY} \longrightarrow \text{TRADING ENGINE} \longrightarrow \text{REAL TRADE OUTCOME}}$$

The model estimates recovery probabilities; the Averaging Engine evaluates complete trading and risk policies across multiple market regimes.

## 22. Cooperative Capital Contract with Strategy Allocation Engine

The Averaging Engine operates under a strict shared capital contract with the Strategy Allocation Engine:

$$\text{Initial Strategy Allocation} + \text{Maximum Averaging Reserve} + \text{Safety Reserve} \le \text{Total Available Capital}$$

1. The Strategy Allocation Engine must never consume capital designated for the Averaging Reserve.
2. The Averaging Engine defines its required reserve policy based on its active lot progression ladder.

## 23. Continuous Research-to-Production Fine-Tuning Loop

Live trade outcomes feed telemetry back into AruMLStudio's Research Memory (`analysis.db`), closing the research loop:

$$\text{Live Trading Outcome} \longrightarrow \text{Telemetry} \longrightarrow \text{AruMLStudio Research} \longrightarrow \text{Walk-Forward Validation} \longrightarrow \text{Governance Approval} \longrightarrow \text{Fine-Tuned Production Model / Policy}$$

Live losses **never** automatically alter production models; all fine-tuning follows the strict 10-step governance lifecycle.

## 24. Strategy Policy Identity

Every complete averaging policy should have a canonical identity containing context, entry policy, classifier signatures, meta-model signature, target policy, spacing policy, quantity ladder, maximum levels, maximum lots, risk limits, time limits, and execution assumptions.
Any material policy change should create a new policy identity so historical results remain reconstructable.

## 25. Production Governance

Research Candidate → Policy Backtest → Walk-Forward Validation → Regime Stress → Tail-Risk Review → Human Approval → Paper/Simulation → Production Approval → Live Policy.
No research result should automatically become a live trading policy without human sign-off.

## 26. What Is Not Decided Yet

This document intentionally does not decide the final averaging percentage, lot sequence, maximum lots, basket target, confidence threshold, maximum holding time, or final kill/stop policy.
These values must come from empirical research and governance inside AruMLStudio.

## 27. Final Architecture

$$\text{Market/Option Ticks} \longrightarrow \text{Feature Pipeline} \longrightarrow \text{Classification Lab} \longrightarrow \text{Meta Confidence} \longrightarrow \text{Target Selection} \longrightarrow \text{Averaging Policy} \longrightarrow \text{Basket Manager} \longrightarrow \text{Risk Engine} \longrightarrow \text{Order Execution}$$

The classification models estimate recovery opportunities. The meta model estimates confidence. The Averaging Policy determines spacing and quantity. The Basket Manager manages the current basket. The Risk Engine has absolute authority over exposure and loss.

## 28. Status

DOCUMENT STATUS: RESEARCH SPECIFICATION.  
Production Averaging Policy: NOT YET DEFINED.  
Research implementation: NOT YET STARTED.  
No production model, active model, base pipeline, or live trading configuration should be changed by this specification.