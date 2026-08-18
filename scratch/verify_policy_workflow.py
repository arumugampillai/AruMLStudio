"""Comprehensive manual/runtime verification script for Feature Recommendation Policy Settings."""

import os
import sys
import tempfile
import sqlite3
import json

# Ensure apps directory is on python path
sys.path.insert(0, os.path.abspath("apps"))

from chain_replay_ml.production_validation.api import (
    BasePipelinePolicy,
    ExperimentalLifecyclePolicy,
    FeatureRegistryPolicy,
    RecommendationPolicy,
    ScoringPolicy,
    build_dataset_context,
    compute_evidence_score,
    list_policy_history,
    load_policy_store,
    load_recommendation_policy,
    preview_policy_impact,
    query_blocked_candidates,
    rebuild_all_projections,
    restore_policy_version,
    save_recommendation_policy,
    validate_recommendation_policy,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    get_connection,
    get_experimental_lineage_summaries,
    get_feature_context_summaries,
)
from master_dataset_tk.feature_recommendation_viewer import open_feature_recommendation_viewer
import tkinter as tk

def run_verification():
    print("=================================================================")
    print("FEATURE RECOMMENDATION POLICY SETTINGS — RUNTIME VERIFICATION")
    print("=================================================================")

    tmp_dir = tempfile.mkdtemp(prefix="aruml_verify_")
    chart_data = os.path.join(tmp_dir, "data")
    os.makedirs(chart_data, exist_ok=True)
    
    ctx = build_dataset_context(
        market="NIFTY",
        sampling_interval_sec=3,
        sliding_window="standard",
        feature_project_id="all",
    )
    print(f"\n[Context] Market={ctx.market}, Interval={ctx.sampling_interval_sec}s, Window={ctx.sliding_window}, Project={ctx.feature_project_id}")
    print(f"[Context ID] {ctx.context_id}")

    # Seed initial Production Evidence
    conn = get_connection(chart_data)
    initial_rows = [
        # Experimental Stellar: 3 KEEPs on unique models -> PROMOTION_CANDIDATE under default (keep>=3, models>=2, score>=75)
        {"evidence_id": "ev_s1", "feature_name": "feat_exp_stellar", "feature_source": "experimental", "pipeline_id": "PL_0001", "pipeline_snapshot_id": "snap_v1", "recommendation": "KEEP", "validation_run_id": "run_1", "model_name": "Model_A", "run_timestamp": "2026-08-16T10:00:00Z"},
        {"evidence_id": "ev_s2", "feature_name": "feat_exp_stellar", "feature_source": "experimental", "pipeline_id": "PL_0001", "pipeline_snapshot_id": "snap_v1", "recommendation": "KEEP", "validation_run_id": "run_2", "model_name": "Model_B", "run_timestamp": "2026-08-16T11:00:00Z"},
        {"evidence_id": "ev_s3", "feature_name": "feat_exp_stellar", "feature_source": "experimental", "pipeline_id": "PL_0001", "pipeline_snapshot_id": "snap_v1", "recommendation": "KEEP", "validation_run_id": "run_3", "model_name": "Model_C", "run_timestamp": "2026-08-16T12:00:00Z"},
        # Experimental Degraded: 2 REMOVEs on unique models -> BLOCKED under default (consec>=2)
        {"evidence_id": "ev_d1", "feature_name": "feat_exp_degraded", "feature_source": "experimental", "pipeline_id": "PL_0001", "pipeline_snapshot_id": "snap_v1", "recommendation": "REMOVE", "validation_run_id": "run_1", "model_name": "Model_A", "run_timestamp": "2026-08-16T10:00:00Z"},
        {"evidence_id": "ev_d2", "feature_name": "feat_exp_degraded", "feature_source": "experimental", "pipeline_id": "PL_0001", "pipeline_snapshot_id": "snap_v1", "recommendation": "REMOVE", "validation_run_id": "run_2", "model_name": "Model_B", "run_timestamp": "2026-08-16T11:00:00Z"},
        # Base Pipeline: 2 REMOVEs -> ALERT state (score <= -40.0), but NEVER blocked
        {"evidence_id": "ev_b1", "feature_name": "base_lag_6s", "feature_source": "base_pipeline", "recommendation": "REMOVE", "validation_run_id": "run_1", "model_name": "Model_A", "run_timestamp": "2026-08-16T10:00:00Z"},
        {"evidence_id": "ev_b2", "feature_name": "base_lag_6s", "feature_source": "base_pipeline", "recommendation": "REMOVE", "validation_run_id": "run_2", "model_name": "Model_B", "run_timestamp": "2026-08-16T11:00:00Z"},
        # Registry: 3 REMOVEs on 2 models -> ALERT state, but NEVER blocked
        {"evidence_id": "ev_r1", "feature_name": "reg_vol_ratio", "feature_source": "registry", "recommendation": "REMOVE", "validation_run_id": "run_1", "model_name": "Model_A", "run_timestamp": "2026-08-16T10:00:00Z"},
        {"evidence_id": "ev_r2", "feature_name": "reg_vol_ratio", "feature_source": "registry", "recommendation": "REMOVE", "validation_run_id": "run_2", "model_name": "Model_B", "run_timestamp": "2026-08-16T11:00:00Z"},
        {"evidence_id": "ev_r3", "feature_name": "reg_vol_ratio", "feature_source": "registry", "recommendation": "REMOVE", "validation_run_id": "run_3", "model_name": "Model_B", "run_timestamp": "2026-08-16T12:00:00Z"},
    ]
    append_validation_evidence(conn, context=ctx, evidence_rows=initial_rows)
    ev_count_initial = conn.execute("SELECT count(*) FROM recommendation_evidence;").fetchone()[0]
    print(f"\n[Step 1-3] Seeded initial evidence rows: {ev_count_initial}")
    assert ev_count_initial == 10, f"Expected 10 rows, got {ev_count_initial}"

    # Step 1-3: Verify Tkinter UI instantiation & Global Default loading
    root = tk.Tk()
    root.withdraw()
    dlg = open_feature_recommendation_viewer(
        root,
        chart_dir=tmp_dir,
        initial_market="NIFTY",
        initial_interval_sec=3,
        initial_sliding_window="standard",
        initial_feature_project_id="all",
    )
    tab_count = dlg._notebook.index("end")
    tab_titles = [dlg._notebook.tab(i, "text") for i in range(tab_count)]
    print(f"[Step 1-3] UI Initialized with {tab_count} Tabs: {tab_titles}")
    assert "5. Policy Settings" in tab_titles, "Tab 5 missing from UI"
    
    # Verify Global Default loaded
    glob_pol = load_recommendation_policy(chart_data)
    print(f"[Step 3] Global Policy loaded: ID={glob_pol.policy_id}, Version={glob_pol.policy_version}, KEEP_Weight={glob_pol.scoring.weight_keep}")
    assert glob_pol.policy_version == 1

    # Step 4-5: Context Override Resolution
    ctx_pol_before = load_recommendation_policy(chart_data, context_id=ctx.context_id)
    print(f"[Step 4-5] Context resolution before override: ID={ctx_pol_before.policy_id}, Scope={'Global Default' if not ctx_pol_before.context_id else 'Context'}")
    assert ctx_pol_before.context_id is None  # Falls back to global

    # Step 6-9: Preview Policy Impact (Read-Only) with modified promotion threshold
    proposed = RecommendationPolicy(
        scoring=ScoringPolicy(weight_keep=25.0),
        experimental_lifecycle=ExperimentalLifecyclePolicy(
            promotion_candidate_consecutive_keep=4,  # Changed from 3 to 4 (stellar loses promo)
            remove_block_consecutive_threshold=3,    # Changed from 2 to 3 (degraded unblocked)
            remove_block_total_threshold=5,
        ),
    )
    preview = preview_policy_impact(conn, context_id=ctx.context_id, proposed_policy=proposed)
    print("\n[Step 7-8] Preview Policy Impact computed in-memory:")
    print(f"  Current Promo: {preview['current_counts']['promotion_candidates']} -> Proposed Promo: {preview['proposed_counts']['promotion_candidates']} (Delta: {preview['deltas']['promotion_candidates']})")
    print(f"  Current Blocked: {preview['current_counts']['blocked']} -> Proposed Blocked: {preview['proposed_counts']['blocked']} (Delta: {preview['deltas']['blocked']})")
    print(f"  Lost Promotion: {preview['lost_promotion']}")
    print(f"  Unblocked: {preview['unblocked']}")
    
    assert preview["lost_promotion"] == ["feat_exp_stellar"]
    assert preview["unblocked"] == ["feat_exp_degraded"]

    # Step 9: Confirm preview did NOT modify recommendation_evidence
    ev_count_after_preview = conn.execute("SELECT count(*) FROM recommendation_evidence;").fetchone()[0]
    print(f"[Step 9] Evidence row count after preview: {ev_count_after_preview} (matches initial: {ev_count_initial})")
    assert ev_count_after_preview == ev_count_initial

    # Step 10-11: Save modified context policy and confirm version increment
    saved_v2 = save_recommendation_policy(chart_data, proposed, context_id=ctx.context_id)
    print(f"\n[Step 10-11] Saved modified policy: ID={saved_v2.policy_id}, Version={saved_v2.policy_version}, Context={saved_v2.context_id}")
    assert saved_v2.policy_version == 2
    assert saved_v2.context_id == ctx.context_id

    # Step 12: Reload and confirm persistence
    reloaded_v2 = load_recommendation_policy(chart_data, context_id=ctx.context_id)
    print(f"[Step 12] Reloaded policy: ID={reloaded_v2.policy_id}, Version={reloaded_v2.policy_version}, ExpPromoStreak={reloaded_v2.experimental_lifecycle.promotion_candidate_consecutive_keep}")
    assert reloaded_v2.policy_version == 2
    assert reloaded_v2.experimental_lifecycle.promotion_candidate_consecutive_keep == 4

    # Step 13: Save same values again and confirm NO version increment
    saved_again = save_recommendation_policy(chart_data, reloaded_v2, context_id=ctx.context_id)
    print(f"[Step 13] Saved unchanged policy: Version={saved_again.policy_version} (No version bump!)")
    assert saved_again.policy_version == 2

    # Step 14-15: Restore older policy version (v1)
    restored_v3 = restore_policy_version(chart_data, target_version=1, context_id=ctx.context_id)
    print(f"\n[Step 14-15] Restored Version 1 -> Created NEW Version={restored_v3.policy_version}, RestoredFrom={restored_v3.restored_from_version}")
    print(f"  Description: '{restored_v3.description}'")
    print(f"  ExpPromoStreak restored to: {restored_v3.experimental_lifecycle.promotion_candidate_consecutive_keep}")
    assert restored_v3.policy_version == 3
    assert restored_v3.restored_from_version == 1
    assert restored_v3.experimental_lifecycle.promotion_candidate_consecutive_keep == 3

    history = list_policy_history(chart_data, context_id=ctx.context_id)
    hist_versions = [h["policy_version"] for h in history]
    print(f"  Policy History preserved versions: {hist_versions}")
    assert hist_versions == [3, 2, 1]

    # Step 16-17: Rebuild projections and verify projection metadata
    rebuild_res = rebuild_all_projections(conn, policy=restored_v3, context_id=ctx.context_id)
    print(f"\n[Step 16-17] Rebuilt projections: {rebuild_res}")
    ctx_sums = get_feature_context_summaries(conn, ctx.context_id)
    lin_sums = get_experimental_lineage_summaries(conn, ctx.context_id)
    
    print(f"  Context Summary [0] policy_id: {ctx_sums[0]['projection_policy_id']}, version: {ctx_sums[0]['projection_policy_version']}, rebuilt_at: {ctx_sums[0]['projection_rebuilt_at']}")
    print(f"  Lineage Summary [0] policy_id: {lin_sums[0]['projection_policy_id']}, version: {lin_sums[0]['projection_policy_version']}, rebuilt_at: {lin_sums[0]['projection_rebuilt_at']}")
    assert ctx_sums[0]["projection_policy_version"] == 3
    assert lin_sums[0]["projection_policy_version"] == 3

    # Step 18: Evidence row count unchanged throughout
    ev_count_final = conn.execute("SELECT count(*) FROM recommendation_evidence;").fetchone()[0]
    print(f"\n[Step 18] Final evidence row count: {ev_count_final} (matches initial: {ev_count_initial})")
    assert ev_count_final == ev_count_initial

    # Step 20: Auto Candidate Generation respects Experimental blocking
    blocked_candidates = query_blocked_candidates(
        conn,
        context_id=ctx.context_id,
        candidate_names=["feat_exp_stellar", "feat_exp_degraded", "other_candidate"],
    )
    print(f"\n[Step 20] Pre-Training Elimination Gate query: blocked={blocked_candidates}")
    assert blocked_candidates == {"feat_exp_degraded"}

    # Step 21: Registry and Base Pipeline immunity
    by_name = {s["feature_name"]: s for s in ctx_sums}
    print(f"[Step 21] Population Immunity Check:")
    print(f"  Base Pipeline feature 'base_lag_6s' status: {by_name['base_lag_6s']['lifecycle_status']} (Immune from BLOCKED)")
    print(f"  Registry feature 'reg_vol_ratio' status: {by_name['reg_vol_ratio']['lifecycle_status']} (Immune from BLOCKED)")
    assert by_name["base_lag_6s"]["lifecycle_status"] == "alert"
    assert by_name["reg_vol_ratio"]["lifecycle_status"] == "alert"

    dlg.destroy()
    root.destroy()
    conn.close()

    print("\n=================================================================")
    print("ALL 21 MANUAL & RUNTIME VERIFICATION STEPS PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    run_verification()
