"""Configurable recommendation policy, thresholds, scoring model, and Phase 2A intelligence helpers."""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

POLICY_FILE_NAME = "recommendation_policy.json"
POLICIES_STORE_FILE_NAME = "recommendation_policies.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ScoringPolicy:
    weight_keep: float = 25.0
    weight_remove: float = -35.0
    weight_watch: float = -10.0
    bonus_consecutive_keep: float = 15.0
    penalty_consecutive_remove: float = -25.0
    min_score: float = -100.0
    max_score: float = 100.0
    confidence_runs_saturation: float = 3.0
    confidence_models_saturation: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentalLifecyclePolicy:
    remove_block_consecutive_threshold: int = 2
    remove_block_total_threshold: int = 4
    promotion_candidate_consecutive_keep: int = 3
    promotion_candidate_min_score: float = 75.0
    experimental_promotion_min_unique_models: int = 2

    @property
    def min_unique_models(self) -> int:
        """Backward-compatible alias for experimental_promotion_min_unique_models."""
        return self.experimental_promotion_min_unique_models

    @min_unique_models.setter
    def min_unique_models(self, value: int) -> None:
        self.experimental_promotion_min_unique_models = int(value)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["min_unique_models"] = self.experimental_promotion_min_unique_models
        return d


@dataclass
class BasePipelinePolicy:
    negative_alert_score_threshold: float = -40.0
    strong_keep_min_score: float = 50.0
    min_validation_runs_for_ranking: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureRegistryPolicy:
    remove_audit_alert_threshold: int = 3
    registry_alert_min_unique_models: int = 2

    @property
    def min_unique_models(self) -> int:
        """Backward-compatible alias for registry_alert_min_unique_models."""
        return self.registry_alert_min_unique_models

    @min_unique_models.setter
    def min_unique_models(self, value: int) -> None:
        self.registry_alert_min_unique_models = int(value)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["min_unique_models"] = self.registry_alert_min_unique_models
        return d


@dataclass
class TrainingDecisionPolicy:
    train_candidate_min_score: float = 20.0
    train_candidate_min_confidence: float = 0.30
    require_zero_negative_votes: bool = False
    allow_stale_in_review: bool = True
    max_volatility_for_candidate: float = 35.0
    min_level1_generalization: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationPolicy:
    policy_id: str = "pol_global_v1"
    policy_version: int = 1
    context_id: str | None = None
    is_active: bool = True
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    created_by: str = "system"
    description: str = ""
    restored_from_version: int | None = None
    scoring: ScoringPolicy = field(default_factory=ScoringPolicy)
    experimental_lifecycle: ExperimentalLifecyclePolicy = field(
        default_factory=ExperimentalLifecyclePolicy
    )
    base_pipeline: BasePipelinePolicy = field(default_factory=BasePipelinePolicy)
    feature_registry: FeatureRegistryPolicy = field(
        default_factory=FeatureRegistryPolicy
    )
    training_decision: TrainingDecisionPolicy = field(
        default_factory=TrainingDecisionPolicy
    )

    @property
    def version(self) -> int:
        """Backward-compatible alias for policy_version."""
        return self.policy_version

    @version.setter
    def version(self, val: int) -> None:
        self.policy_version = int(val)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "version": self.policy_version,
            "context_id": self.context_id,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "description": self.description,
            "restored_from_version": self.restored_from_version,
            "scoring": self.scoring.to_dict(),
            "experimental_lifecycle": self.experimental_lifecycle.to_dict(),
            "base_pipeline": self.base_pipeline.to_dict(),
            "feature_registry": self.feature_registry.to_dict(),
            "training_decision": self.training_decision.to_dict(),
        }

    def parameters_equal(self, other: RecommendationPolicy) -> bool:
        """Compare purely functional threshold & scoring parameters, ignoring timestamps/IDs."""
        if not isinstance(other, RecommendationPolicy):
            return False
        return (
            self.scoring == other.scoring
            and self.experimental_lifecycle.remove_block_consecutive_threshold
            == other.experimental_lifecycle.remove_block_consecutive_threshold
            and self.experimental_lifecycle.remove_block_total_threshold
            == other.experimental_lifecycle.remove_block_total_threshold
            and self.experimental_lifecycle.promotion_candidate_consecutive_keep
            == other.experimental_lifecycle.promotion_candidate_consecutive_keep
            and self.experimental_lifecycle.promotion_candidate_min_score
            == other.experimental_lifecycle.promotion_candidate_min_score
            and self.experimental_lifecycle.experimental_promotion_min_unique_models
            == other.experimental_lifecycle.experimental_promotion_min_unique_models
            and self.base_pipeline == other.base_pipeline
            and self.feature_registry.remove_audit_alert_threshold
            == other.feature_registry.remove_audit_alert_threshold
            and self.feature_registry.registry_alert_min_unique_models
            == other.feature_registry.registry_alert_min_unique_models
            and getattr(self, "training_decision", TrainingDecisionPolicy())
            == getattr(other, "training_decision", TrainingDecisionPolicy())
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RecommendationPolicy:
        if not data or not isinstance(data, dict):
            return cls()

        scoring_data = data.get("scoring") or {}
        exp_data = dict(data.get("experimental_lifecycle") or {})
        base_data = data.get("base_pipeline") or {}
        reg_data = dict(data.get("feature_registry") or {})
        td_data = data.get("training_decision") or {}

        # Handle backward-compatible alias for experimental_promotion_min_unique_models
        if "min_unique_models" in exp_data and "experimental_promotion_min_unique_models" not in exp_data:
            exp_data["experimental_promotion_min_unique_models"] = exp_data["min_unique_models"]
        # Handle backward-compatible alias for registry_alert_min_unique_models
        if "min_unique_models" in reg_data and "registry_alert_min_unique_models" not in reg_data:
            reg_data["registry_alert_min_unique_models"] = reg_data["min_unique_models"]

        raw_ver = data.get("policy_version") or data.get("version") or 1
        p_version = int(raw_ver)
        ctx_id = data.get("context_id")
        p_id = str(data.get("policy_id") or f"pol_{ctx_id or 'global'}_v{p_version}")

        return cls(
            policy_id=p_id,
            policy_version=p_version,
            context_id=str(ctx_id) if ctx_id else None,
            is_active=bool(data.get("is_active", True)),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            created_by=str(data.get("created_by") or "system"),
            description=str(data.get("description") or ""),
            restored_from_version=(
                int(data["restored_from_version"])
                if data.get("restored_from_version") is not None
                else None
            ),
            scoring=ScoringPolicy(
                **{k: v for k, v in scoring_data.items() if k in ScoringPolicy.__dataclass_fields__}
            ),
            experimental_lifecycle=ExperimentalLifecyclePolicy(
                **{
                    k: v
                    for k, v in exp_data.items()
                    if k in ExperimentalLifecyclePolicy.__dataclass_fields__
                }
            ),
            base_pipeline=BasePipelinePolicy(
                **{k: v for k, v in base_data.items() if k in BasePipelinePolicy.__dataclass_fields__}
            ),
            feature_registry=FeatureRegistryPolicy(
                **{
                    k: v
                    for k, v in reg_data.items()
                    if k in FeatureRegistryPolicy.__dataclass_fields__
                }
            ),
            training_decision=TrainingDecisionPolicy(
                **{k: v for k, v in td_data.items() if k in TrainingDecisionPolicy.__dataclass_fields__}
            ),
        )


def validate_recommendation_policy(policy: RecommendationPolicy) -> list[str]:
    """Validate policy rules, thresholds, and bounds. Returns list of error messages (empty if valid)."""
    errors: list[str] = []
    s = policy.scoring
    exp = policy.experimental_lifecycle
    base = policy.base_pipeline
    reg = policy.feature_registry
    td = getattr(policy, "training_decision", None)

    if s.min_score >= s.max_score:
        errors.append(f"Score minimum ({s.min_score}) must be less than maximum ({s.max_score})")
    if s.weight_keep <= 0:
        errors.append(f"KEEP weight must be positive (got {s.weight_keep})")
    if s.weight_remove >= 0:
        errors.append(f"REMOVE weight must be negative (got {s.weight_remove})")
    if s.bonus_consecutive_keep < 0:
        errors.append(f"KEEP streak bonus must be >= 0 (got {s.bonus_consecutive_keep})")
    if s.penalty_consecutive_remove > 0:
        errors.append(f"REMOVE streak penalty must be <= 0 (got {s.penalty_consecutive_remove})")
    if s.confidence_runs_saturation <= 0:
        errors.append(f"Confidence runs saturation must be > 0 (got {s.confidence_runs_saturation})")
    if s.confidence_models_saturation <= 0:
        errors.append(f"Confidence models saturation must be > 0 (got {s.confidence_models_saturation})")

    if exp.remove_block_consecutive_threshold < 1:
        errors.append(f"Consecutive REMOVE block threshold must be >= 1 (got {exp.remove_block_consecutive_threshold})")
    if exp.remove_block_total_threshold < exp.remove_block_consecutive_threshold:
        errors.append(
            f"Total REMOVE block threshold ({exp.remove_block_total_threshold}) cannot be less than consecutive threshold ({exp.remove_block_consecutive_threshold})"
        )
    if exp.promotion_candidate_consecutive_keep < 1:
        errors.append(f"Promotion consecutive KEEP streak must be >= 1 (got {exp.promotion_candidate_consecutive_keep})")
    if exp.experimental_promotion_min_unique_models < 1:
        errors.append(
            f"Experimental promotion unique models count must be >= 1 (got {exp.experimental_promotion_min_unique_models})"
        )

    if base.min_validation_runs_for_ranking < 1:
        errors.append(f"Base pipeline min validation runs for ranking must be >= 1 (got {base.min_validation_runs_for_ranking})")

    if reg.remove_audit_alert_threshold < 1:
        errors.append(f"Registry remove audit alert threshold must be >= 1 (got {reg.remove_audit_alert_threshold})")
    if reg.registry_alert_min_unique_models < 1:
        errors.append(f"Registry alert unique models count must be >= 1 (got {reg.registry_alert_min_unique_models})")

    if td is not None:
        if td.train_candidate_min_confidence < 0.0 or td.train_candidate_min_confidence > 1.0:
            errors.append(
                f"Training candidate min confidence must be in [0, 1] (got {td.train_candidate_min_confidence})"
            )
        if td.max_volatility_for_candidate <= 0.0:
            errors.append(
                f"Max volatility for candidate must be > 0 (got {td.max_volatility_for_candidate})"
            )
        if td.min_level1_generalization < 0.0 or td.min_level1_generalization > 1.0:
            errors.append(
                f"Min level-1 generalization must be in [0, 1] (got {td.min_level1_generalization})"
            )

    return errors


def policy_file_path(data_dir: str) -> str:
    return os.path.join(data_dir, POLICY_FILE_NAME)


def policies_store_path(data_dir: str) -> str:
    return os.path.join(data_dir, POLICIES_STORE_FILE_NAME)


def load_policy_store(data_dir: str) -> dict[str, Any]:
    """Load the full multi-context policy store or create default structure."""
    path = policies_store_path(data_dir)
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass

    # Backward compatibility: check legacy single policy file
    legacy_path = policy_file_path(data_dir)
    if os.path.isfile(legacy_path):
        try:
            with open(legacy_path, encoding="utf-8") as fh:
                legacy_dict = json.load(fh)
                if isinstance(legacy_dict, dict):
                    pol = RecommendationPolicy.from_dict(legacy_dict)
                    return {
                        "version": 1,
                        "global": pol.to_dict(),
                        "contexts": {},
                        "history": [pol.to_dict()],
                    }
        except Exception:
            pass

    # Fresh default store
    default_pol = RecommendationPolicy()
    return {
        "version": 1,
        "global": default_pol.to_dict(),
        "contexts": {},
        "history": [],
    }


def save_policy_store(data_dir: str, store: dict[str, Any]) -> None:
    """Save the multi-context policy store and mirror global default to recommendation_policy.json."""
    os.makedirs(data_dir, exist_ok=True)
    path = policies_store_path(data_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)

    # Mirror global default to legacy path for backward compatibility
    glob_dict = store.get("global")
    if glob_dict and isinstance(glob_dict, dict):
        legacy_path = policy_file_path(data_dir)
        try:
            with open(legacy_path, "w", encoding="utf-8") as fh:
                json.dump(glob_dict, fh, indent=2)
        except Exception:
            pass


def load_recommendation_policy(
    data_dir: str,
    context_id: str | None = None,
) -> RecommendationPolicy:
    """Load active recommendation policy for context_id (or global default). Read-only: never changes version."""
    store = load_policy_store(data_dir)
    cid = str(context_id or "").strip() or None

    if cid:
        contexts = store.get("contexts") or {}
        if isinstance(contexts, dict) and cid in contexts:
            return RecommendationPolicy.from_dict(contexts[cid])

    # Fallback to global
    glob = store.get("global") or {}
    if glob and isinstance(glob, dict):
        return RecommendationPolicy.from_dict(glob)

    return RecommendationPolicy()


def save_recommendation_policy(
    data_dir: str,
    policy: RecommendationPolicy,
    context_id: str | None = None,
    force_new_version: bool = False,
) -> RecommendationPolicy:
    """Save policy for context_id (or global default). Increments version and records history if parameters changed."""
    validation_errs = validate_recommendation_policy(policy)
    if validation_errs:
        raise ValueError(f"Invalid recommendation policy: {'; '.join(validation_errs)}")

    cid = str(context_id or "").strip() or None
    store = load_policy_store(data_dir)
    current = load_recommendation_policy(data_dir, context_id=cid)

    # Check if modified
    if not force_new_version and policy.parameters_equal(current) and (cid == current.context_id or cid is None):
        return current

    # Archive previous version to history if it has valid policy_version
    history = store.setdefault("history", [])
    if isinstance(history, list) and current.policy_version >= 1:
        hist_entry = current.to_dict()
        hist_entry["context_id"] = cid  # Record for this context's version history
        hist_entry["archived_at"] = _utc_now()
        history.append(hist_entry)

    # Calculate next version
    next_ver = current.policy_version + 1 if policy.policy_version <= current.policy_version else policy.policy_version
    now_str = _utc_now()

    new_policy = copy.deepcopy(policy)
    new_policy.context_id = cid
    new_policy.policy_version = next_ver
    prefix = cid if cid else "global"
    new_policy.policy_id = f"pol_{prefix}_v{next_ver}"
    new_policy.updated_at = now_str
    new_policy.is_active = True

    policy_dict = new_policy.to_dict()
    if cid:
        contexts = store.setdefault("contexts", {})
        if not isinstance(contexts, dict):
            contexts = {}
            store["contexts"] = contexts
        contexts[cid] = policy_dict
    else:
        store["global"] = policy_dict

    save_policy_store(data_dir, store)
    return new_policy


def restore_policy_version(
    data_dir: str,
    target_version: int,
    context_id: str | None = None,
) -> RecommendationPolicy:
    """Restore an older policy version as a NEW policy version containing target's settings."""
    cid = str(context_id or "").strip() or None
    store = load_policy_store(data_dir)
    history = store.get("history") or []

    # Search history for target version
    found_dict: dict[str, Any] | None = None
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        entry_cid = str(entry.get("context_id") or "").strip() or None
        entry_ver = int(entry.get("policy_version") or entry.get("version") or 0)
        if entry_ver == target_version and (entry_cid == cid or (cid and entry_cid is None)):
            found_dict = entry
            break

    if not found_dict:
        # Check current active as well
        cur = load_recommendation_policy(data_dir, context_id=cid)
        if cur.policy_version == target_version:
            found_dict = cur.to_dict()

    if not found_dict:
        # Check global default
        glob = load_recommendation_policy(data_dir, context_id=None)
        if glob.policy_version == target_version:
            found_dict = glob.to_dict()

    if not found_dict:
        raise ValueError(
            f"Policy version {target_version} not found in history for context '{cid or 'global'}'"
        )

    restored = RecommendationPolicy.from_dict(found_dict)
    restored.restored_from_version = target_version
    restored.description = f"Restored from version {target_version} on {_utc_now()[:19]}"
    return save_recommendation_policy(
        data_dir,
        restored,
        context_id=cid,
        force_new_version=True,
    )


def list_policy_history(
    data_dir: str,
    context_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return chronological history of policy versions for context/global, including active."""
    cid = str(context_id or "").strip() or None
    store = load_policy_store(data_dir)
    raw_hist = store.get("history") or []
    out: list[dict[str, Any]] = []

    for entry in raw_hist:
        if not isinstance(entry, dict):
            continue
        entry_cid = str(entry.get("context_id") or "").strip() or None
        if entry_cid == cid or (cid and entry_cid is None and not any(str(e.get("context_id") or "") == cid for e in raw_hist)):
            out.append(dict(entry))

    # Append current active version
    cur = load_recommendation_policy(data_dir, context_id=cid)
    cur_dict = cur.to_dict()
    cur_dict["is_active"] = True

    seen_versions = set()
    final_out = [cur_dict]
    seen_versions.add(cur.policy_version)

    for e in out:
        v = int(e.get("policy_version") or e.get("version") or 0)
        if v not in seen_versions:
            seen_versions.add(v)
            final_out.append(e)

    final_out.sort(key=lambda e: int(e.get("policy_version") or e.get("version") or 0), reverse=True)
    return final_out


def compute_evidence_score(
    *,
    keep_models: int,
    remove_models: int,
    watch_models: int,
    consecutive_keeps: int,
    consecutive_removes: int,
    policy: ScoringPolicy | None = None,
) -> float:
    """Compute evidence score bounded strictly within [min_score, max_score]."""
    p = policy or ScoringPolicy()
    raw = (
        p.weight_keep * float(keep_models)
        + p.weight_remove * float(remove_models)
        + p.weight_watch * float(watch_models)
        + p.bonus_consecutive_keep * float(consecutive_keeps)
        + p.penalty_consecutive_remove * float(consecutive_removes)
    )
    return round(max(p.min_score, min(p.max_score, raw)), 2)


# ==============================================================================
# Phase 2A: Evidence Intelligence Helper Functions (Advisory & Descriptive)
# ==============================================================================

def compute_evidence_confidence(
    runs: int,
    models: int,
    policy: ScoringPolicy | RecommendationPolicy | None = None,
) -> float:
    """Calculate Evidence Confidence score C in [0.0, 1.0] using multiplicative saturation curves."""
    k_runs = 3.0
    k_models = 2.0
    if policy is not None:
        scoring = policy.scoring if isinstance(policy, RecommendationPolicy) else policy
        k_runs = float(getattr(scoring, "confidence_runs_saturation", 3.0) or 3.0)
        k_models = float(getattr(scoring, "confidence_models_saturation", 2.0) or 2.0)
    if k_runs <= 0:
        k_runs = 3.0
    if k_models <= 0:
        k_models = 2.0

    n = max(0, int(runs or 0))
    m = max(0, int(models or 0))
    if n == 0 or m == 0:
        return 0.0

    c_runs = 1.0 - math.exp(-n / k_runs)
    c_models = 1.0 - math.exp(-m / k_models)
    conf = math.sqrt(max(0.0, c_runs * c_models))
    return round(min(1.0, max(0.0, conf)), 4)


def compute_recency_staleness(
    last_validated_at: str | None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Calculate elapsed time and descriptive freshness band from ISO UTC timestamp."""
    if not last_validated_at or not str(last_validated_at).strip():
        return {
            "staleness_seconds": None,
            "staleness_days": None,
            "freshness_label": "Unvalidated",
            "display_text": "⚪ Never Validated",
        }

    now = now_utc or datetime.now(timezone.utc)
    ts_str = str(last_validated_at).strip()
    try:
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta_sec = max(0.0, (now - dt).total_seconds())
        delta_days = delta_sec / 86400.0
    except Exception:
        return {
            "staleness_seconds": None,
            "staleness_days": None,
            "freshness_label": "Unvalidated",
            "display_text": "⚪ Unvalidated",
        }

    if delta_sec < 86400.0:
        hours = int(delta_sec // 3600)
        label = "Fresh"
        disp = f"🟢 Fresh (< 24h)" if hours <= 1 else f"🟢 Fresh ({hours}h ago)"
    elif delta_days < 7.0:
        days = max(1, int(delta_days))
        label = "Recent"
        disp = f"🟢 Recent ({days}d ago)"
    elif delta_days < 30.0:
        days = int(delta_days)
        label = "Aging"
        disp = f"🟡 Aging ({days}d ago)"
    else:
        days = int(delta_days)
        label = "Stale"
        disp = f"🔴 Stale ({days}d ago)"

    return {
        "staleness_seconds": round(delta_sec, 1),
        "staleness_days": round(delta_days, 2),
        "freshness_label": label,
        "display_text": disp,
    }


def compute_model_consensus(
    evidence_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate model consensus based on latest recommendation per unique model with strict tie handling."""
    if not evidence_rows:
        return {
            "total_models": 0,
            "dominant_recommendation": "NONE",
            "consensus_ratio": 0.0,
            "is_tie": False,
            "tied_recommendations": [],
            "vote_distribution": {"KEEP": 0, "WATCH": 0, "REMOVE": 0},
            "display_text": "—",
        }

    # Group by model_name, take latest run_timestamp
    latest_by_model: dict[str, dict[str, Any]] = {}
    for r in sorted(evidence_rows, key=lambda x: str(x.get("run_timestamp") or "")):
        m = str(r.get("model_name") or "").strip()
        if not m:
            continue
        rec = str(r.get("recommendation") or "").strip().upper()
        if rec in ("KEEP", "WATCH", "REMOVE"):
            latest_by_model[m] = {"model_name": m, "recommendation": rec}

    total_models = len(latest_by_model)
    if total_models == 0:
        return {
            "total_models": 0,
            "dominant_recommendation": "NONE",
            "consensus_ratio": 0.0,
            "is_tie": False,
            "tied_recommendations": [],
            "vote_distribution": {"KEEP": 0, "WATCH": 0, "REMOVE": 0},
            "display_text": "—",
        }

    votes = {"KEEP": 0, "WATCH": 0, "REMOVE": 0}
    for m_info in latest_by_model.values():
        votes[m_info["recommendation"]] += 1

    v_max = max(votes.values())
    winners = [cat for cat in ("KEEP", "WATCH", "REMOVE") if votes[cat] == v_max]

    if total_models == 1:
        winner = winners[0]
        return {
            "total_models": 1,
            "dominant_recommendation": winner,
            "consensus_ratio": 1.0,
            "is_tie": False,
            "tied_recommendations": [],
            "vote_distribution": votes,
            "display_text": f"100.0% {winner} (1/1 model - Single)",
        }

    ratio = round(v_max / total_models, 4)

    if len(winners) == 1:
        winner = winners[0]
        return {
            "total_models": total_models,
            "dominant_recommendation": winner,
            "consensus_ratio": ratio,
            "is_tie": False,
            "tied_recommendations": [],
            "vote_distribution": votes,
            "display_text": f"{ratio * 100:.1f}% {winner} ({v_max}/{total_models})",
        }
    elif len(winners) == 2:
        split_desc = f"SPLIT ({winners[0]}/{winners[1]})"
        return {
            "total_models": total_models,
            "dominant_recommendation": split_desc,
            "consensus_ratio": ratio,
            "is_tie": True,
            "tied_recommendations": winners,
            "vote_distribution": votes,
            "display_text": f"{ratio * 100:.1f}% SPLIT ({v_max} {winners[0]} / {v_max} {winners[1]})",
        }
    else:  # len(winners) == 3
        return {
            "total_models": total_models,
            "dominant_recommendation": "SPLIT (3-WAY)",
            "consensus_ratio": ratio,
            "is_tie": True,
            "tied_recommendations": winners,
            "vote_distribution": votes,
            "display_text": f"{ratio * 100:.1f}% SPLIT ({v_max} KEEP / {v_max} WATCH / {v_max} REMOVE)",
        }


# ==============================================================================
# Phase 2B: Score Stability, Volatility & Cross-Context Generalization Helpers
# ==============================================================================

def compute_score_volatility(
    evidence_rows: list[dict[str, Any]],
    policy: ScoringPolicy | RecommendationPolicy | None = None,
) -> dict[str, Any]:
    """Calculate sample standard deviation, range, and direction flips of cumulative score trajectory for N >= 3."""
    if not evidence_rows or len(evidence_rows) < 3:
        return {
            "total_observations": len(evidence_rows) if evidence_rows else 0,
            "volatility_score": None,
            "score_range": None,
            "direction_flips": None,
            "stability_label": "Insufficient Data",
            "display_text": "⚪ N/A (< 3 runs)",
            "score_trajectory": [],
        }

    scoring = policy.scoring if isinstance(policy, RecommendationPolicy) else (policy or ScoringPolicy())
    sorted_rows = sorted(
        evidence_rows,
        key=lambda x: (str(x.get("run_timestamp") or ""), str(x.get("evidence_id") or "")),
    )

    trajectory: list[float] = []
    k_count = 0
    w_count = 0
    r_count = 0
    consec_k = 0
    consec_r = 0

    for row in sorted_rows:
        rec = str(row.get("recommendation") or "").strip().upper()
        if rec == "KEEP":
            k_count += 1
            consec_k += 1
            consec_r = 0
        elif rec == "REMOVE":
            r_count += 1
            consec_r += 1
            consec_k = 0
        elif rec == "WATCH":
            w_count += 1
            consec_k = 0
            consec_r = 0

        s_t = compute_evidence_score(
            keep_models=k_count,
            remove_models=r_count,
            watch_models=w_count,
            consecutive_keeps=consec_k,
            consecutive_removes=consec_r,
            policy=scoring,
        )
        trajectory.append(s_t)

    n = len(trajectory)
    mean_s = sum(trajectory) / n
    var_s = sum((x - mean_s) ** 2 for x in trajectory) / (n - 1)
    sigma_s = round(math.sqrt(max(0.0, var_s)), 2)
    s_min = min(trajectory)
    s_max = max(trajectory)
    s_range = round(s_max - s_min, 2)

    flips = 0
    for i in range(1, n - 1):
        d1 = trajectory[i] - trajectory[i - 1]
        d2 = trajectory[i + 1] - trajectory[i]
        if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0):
            flips += 1

    if sigma_s < 15.0:
        label = "Stable"
        disp = f"🟢 Stable (σ={sigma_s:.1f})"
    elif sigma_s < 35.0:
        label = "Moderate"
        disp = f"🟡 Moderate (σ={sigma_s:.1f})"
    else:
        label = "Volatile"
        disp = f"🔴 Volatile (σ={sigma_s:.1f})"

    return {
        "total_observations": n,
        "volatility_score": sigma_s,
        "score_range": s_range,
        "direction_flips": flips,
        "stability_label": label,
        "display_text": disp,
        "score_trajectory": trajectory,
    }


def compute_context_generalization(
    primary_context_id: str,
    feature_name: str,
    context_summaries_by_cid: dict[str, dict[str, Any]],
    level1_comparable_context_ids: list[str],
) -> dict[str, Any]:
    """Calculate Level 1 cross-context generalization score G in [0.0, 1.0] across timeframes."""
    # Find which comparable contexts have data for this feature
    matched_contexts = [
        cid for cid in level1_comparable_context_ids
        if cid in context_summaries_by_cid and feature_name in context_summaries_by_cid[cid]
    ]

    # Ensure primary_context is included if it has data
    if primary_context_id in context_summaries_by_cid and feature_name in context_summaries_by_cid[primary_context_id]:
        if primary_context_id not in matched_contexts:
            matched_contexts.insert(0, primary_context_id)

    k = len(matched_contexts)
    if k < 2:
        return {
            "comparable_context_count": k,
            "generalization_score": None,
            "agreement_ratio": None,
            "score_spread": None,
            "generalization_label": "Single Context",
            "display_text": "⚪ Single Context",
        }

    prim_feat = context_summaries_by_cid[primary_context_id][feature_name]
    prim_rec = str(prim_feat.get("dominant_recommendation") or prim_feat.get("last_recommendation") or "NONE").upper()

    matching_recs = 0
    scores: list[float] = []

    for cid in matched_contexts:
        feat_data = context_summaries_by_cid[cid][feature_name]
        c_rec = str(feat_data.get("dominant_recommendation") or feat_data.get("last_recommendation") or "NONE").upper()
        c_score = float(feat_data.get("evidence_score") or feat_data.get("lineage_evidence_score") or 0.0)
        scores.append(c_score)
        if c_rec == prim_rec and prim_rec != "NONE":
            matching_recs += 1

    a_context = matching_recs / k
    delta_s = max(scores) - min(scores)
    g = round(a_context * (1.0 - min(1.0, delta_s / 100.0)), 4)

    if g >= 0.75:
        label = "Universal"
        disp = f"🟢 Universal (G={g:.2f})"
    elif g >= 0.50:
        label = "Scale-Robust"
        disp = f"🟢 Robust (G={g:.2f})"
    elif g >= 0.25:
        label = "Scale-Sensitive"
        disp = f"🟡 Sensitive (G={g:.2f})"
    else:
        label = "Scale-Specific"
        disp = f"🔴 Specific (G={g:.2f})"

    return {
        "comparable_context_count": k,
        "generalization_score": g,
        "agreement_ratio": round(a_context, 4),
        "score_spread": round(delta_s, 2),
        "generalization_label": label,
        "display_text": disp,
    }


def derive_risk_badges(
    evidence_score: float,
    is_consensus_tie: bool,
    freshness_label: str,
    stability_label: str,
) -> list[str]:
    """Expose explicit existing-condition risk badges without inventing an arbitrary composite scalar."""
    badges: list[str] = []
    if float(evidence_score or 0.0) <= -40.0:
        badges.append("DEGRADED")
    if is_consensus_tie:
        badges.append("SPLIT")
    if str(freshness_label or "") == "Stale":
        badges.append("STALE")
    if str(stability_label or "") == "Volatile":
        badges.append("UNSTABLE")
    return badges


__all__ = [
    "BasePipelinePolicy",
    "ExperimentalLifecyclePolicy",
    "FeatureRegistryPolicy",
    "POLICY_FILE_NAME",
    "POLICIES_STORE_FILE_NAME",
    "RecommendationPolicy",
    "ScoringPolicy",
    "compute_context_generalization",
    "compute_evidence_confidence",
    "compute_evidence_score",
    "compute_model_consensus",
    "compute_recency_staleness",
    "compute_score_volatility",
    "derive_risk_badges",
    "list_policy_history",
    "load_policy_store",
    "load_recommendation_policy",
    "policies_store_path",
    "policy_file_path",
    "restore_policy_version",
    "save_policy_store",
    "save_recommendation_policy",
    "validate_recommendation_policy",
]

