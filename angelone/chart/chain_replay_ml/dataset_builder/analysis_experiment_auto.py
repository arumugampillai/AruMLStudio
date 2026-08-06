"""Auto Research — iterative greedy feature-set optimisation.

Round 1: train baseline, try up to N one-family swaps on the current winner.
Accepted improvements accumulate (Price→B then IV→Y on that winner = both).
Round 2+: make the round's best the new baseline, propose the next batch of
remaining families, repeat until a round yields no meaningful Holdout gain
(or max_rounds is hit).

Prefers fewer Families Changed when Holdout/WF are near-ties (simpler sets).
"""
from __future__ import annotations

from typing import Any, Callable

from .analysis_artifacts import (
    KIND_DISCOVERY_BUNDLE,
    latest_artifact,
    publish_discovery_bundle,
    require_artifact,
)
from .analysis_experiments import (
    clone_experiment_variant,
    create_experiment,
    list_experiments,
    load_experiment,
    promote_champion,
    request_train_experiment,
)

# on_progress(fraction, message, dashboard_dict | None)
ProgressCb = Callable[..., None]

DEFAULT_MIN_IMPROVEMENT = 0.001
DEFAULT_WF_TOLERANCE = 0.005
DEFAULT_MAX_ROUNDS = 8
DEFAULT_PER_ROUND = 10
PER_ROUND_CHOICES = (10, 25, 75, 100, 150)
NEAR_TIE_EPS = 0.0005  # prefer fewer family changes within this Holdout/WF band

STRATEGY_SINGLE_SWAP = "single_swap"
STRATEGY_GREEDY = "greedy"
STRATEGY_HILL = "hill_climbing"
STRATEGY_BEAM = "beam"
STRATEGY_GENETIC = "genetic"
STRATEGY_LABELS = {
    STRATEGY_SINGLE_SWAP: "Single Swap",
    STRATEGY_GREEDY: "Greedy Search",
    STRATEGY_HILL: "Hill Climbing",
    STRATEGY_BEAM: "Beam Search",
    STRATEGY_GENETIC: "Genetic Search",
}
IMPLEMENTED_STRATEGIES = frozenset(
    {STRATEGY_SINGLE_SWAP, STRATEGY_GREEDY, STRATEGY_HILL}
)
CONTINUABLE_STRATEGIES = frozenset({STRATEGY_GREEDY, STRATEGY_HILL})

# Intent-first UI — maps to internal strategies
LEVEL_QUICK = "quick"
LEVEL_BALANCED = "balanced"
LEVEL_DEEP = "deep"
LEVEL_LABELS = {
    LEVEL_QUICK: "Quick",
    LEVEL_BALANCED: "Balanced",
    LEVEL_DEEP: "Deep",
}
LEVEL_HINTS = {
    LEVEL_QUICK: "Fast · one pass of neighbours vs baseline",
    LEVEL_BALANCED: "Standard · accepted swaps stack across rounds",
    LEVEL_DEEP: "Thorough · climb neighbour-by-neighbour until converged",
}
LEVEL_TO_STRATEGY = {
    LEVEL_QUICK: STRATEGY_SINGLE_SWAP,
    LEVEL_BALANCED: STRATEGY_GREEDY,
    LEVEL_DEEP: STRATEGY_HILL,
}
STRATEGY_TO_LEVEL = {
    STRATEGY_SINGLE_SWAP: LEVEL_QUICK,
    STRATEGY_GREEDY: LEVEL_BALANCED,
    STRATEGY_HILL: LEVEL_DEEP,
}
DEFAULT_RESEARCH_LEVEL = LEVEL_BALANCED


def resolve_research_strategy(
    *,
    level: str | None = None,
    strategy: str | None = None,
    advanced: bool = False,
) -> str:
    """Resolve user intent (level) or advanced strategy pick to an engine id."""
    if advanced and strategy:
        s = str(strategy).strip().lower()
        if s in ("single", "swap"):
            s = STRATEGY_SINGLE_SWAP
        if s in ("hill", "hillclimb", "hill_climb"):
            s = STRATEGY_HILL
        if s in STRATEGY_LABELS:
            return s
    lvl = str(level or DEFAULT_RESEARCH_LEVEL).strip().lower()
    if lvl not in LEVEL_TO_STRATEGY:
        lvl = DEFAULT_RESEARCH_LEVEL
    return LEVEL_TO_STRATEGY[lvl]


def _scores_from_bundle(payload: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for r in payload.get("discovery_ratings") or []:
        name = str(r.get("feature_name") or "").strip()
        if not name:
            continue
        sc = r.get("feature_score")
        if sc is None:
            sc = r.get("rating_score")
        try:
            if sc is not None:
                scores[name] = float(sc)
        except (TypeError, ValueError):
            continue
    return scores


def _reps_map(exp: dict[str, Any] | None) -> dict[str, str]:
    if not exp:
        return {}
    return {
        str(r["family_id"]): str(r["representative"])
        for r in (exp.get("family_reps") or [])
        if r.get("family_id") and r.get("representative")
    }


def families_changed_count(
    exp: dict[str, Any] | None,
    baseline_reps: dict[str, str] | None,
) -> int:
    """How many HCA families differ from the original Auto-baseline."""
    if not exp or not baseline_reps:
        changes = exp.get("variant_changes_list") if exp else None
        return len(changes or [])
    cur = _reps_map(exp)
    n = 0
    for fid, rep in cur.items():
        if str(baseline_reps.get(fid) or "") != str(rep):
            n += 1
    return n


def families_changed_labels(
    exp: dict[str, Any] | None,
    baseline_reps: dict[str, str] | None,
    *,
    fam_labels: dict[str, str] | None = None,
) -> list[str]:
    if not exp or not baseline_reps:
        return []
    labels = fam_labels or {}
    cur = _reps_map(exp)
    out: list[str] = []
    for fid, rep in cur.items():
        if str(baseline_reps.get(fid) or "") != str(rep):
            out.append(str(labels.get(fid) or fid))
    return out


def propose_intelligent_variants(
    data_dir: str,
    run_id: str,
    *,
    max_variants: int = 9,
    discovery_bundle_id: str | None = None,
    baseline_reps: dict[str, str] | None = None,
    exclude_family_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Propose one-family swaps: current → next highest Discovery-ranked member."""
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    bid = str(discovery_bundle_id or "").strip()
    if not bid:
        art = latest_artifact(data_dir, rid, KIND_DISCOVERY_BUNDLE)
        if not art:
            raise ValueError(
                "No Discovery Bundle — freeze Discovery Complete first."
            )
        bid = str(art["artifact_id"])
    bundle = require_artifact(
        data_dir, bid, expected_kind=KIND_DISCOVERY_BUNDLE
    )
    payload = dict(bundle.get("payload") or {})
    scores = _scores_from_bundle(payload)
    fam_rows = list(payload.get("families") or [])
    bundle_reps = {
        str(k): str(v)
        for k, v in dict(payload.get("family_reps") or {}).items()
        if k and v
    }
    reps = dict(baseline_reps or bundle_reps)
    if not reps:
        raise ValueError("Discovery Bundle has no family representatives")
    skip = {str(x) for x in (exclude_family_ids or set())}

    proposals: list[dict[str, Any]] = []
    seen_snapshots: set[tuple[tuple[str, str], ...]] = set()
    seen_snapshots.add(tuple(sorted(reps.items())))

    changeable_families = 0
    for fam in fam_rows:
        fid = str(fam.get("family_id") or "").strip()
        if not fid or fid not in reps:
            continue
        members = [str(m) for m in list(fam.get("members") or []) if m]
        if len(members) < 2:
            continue
        changeable_families += 1
        if fid in skip:
            continue
        cur = str(reps[fid])
        ranked = sorted(
            members,
            key=lambda m: scores.get(m, float("-inf")),
            reverse=True,
        )
        if cur not in ranked:
            ranked = [cur] + [m for m in ranked if m != cur]
        idx = ranked.index(cur)
        if idx + 1 >= len(ranked):
            continue
        nxt = ranked[idx + 1]
        if nxt == cur:
            continue
        trial = dict(reps)
        trial[fid] = nxt
        key = tuple(sorted(trial.items()))
        if key in seen_snapshots:
            continue
        seen_snapshots.add(key)
        cur_sc = float(scores.get(cur, 0.0))
        nxt_sc = float(scores.get(nxt, 0.0))
        proposals.append(
            {
                "family_id": fid,
                "family_label": str(fam.get("family_label") or fid),
                "old_representative": cur,
                "new_representative": nxt,
                "old_score": cur_sc,
                "new_score": nxt_sc,
                "score_gap": cur_sc - nxt_sc,
                "rank_from": idx + 1,
                "rank_to": idx + 2,
                "discovery_bundle_id": bid,
            }
        )

    proposals.sort(key=lambda p: (float(p["score_gap"]), -float(p["new_score"])))
    capped = proposals[: max(0, int(max_variants))]
    for p in capped:
        p["n_changeable_families"] = changeable_families
        p["n_remaining_after_exclude"] = len(proposals)
    return capped


def propose_neighbour_swaps(
    data_dir: str,
    run_id: str,
    *,
    max_variants: int = 150,
    discovery_bundle_id: str | None = None,
    baseline_reps: dict[str, str] | None = None,
    exclude_snapshots: set[tuple[tuple[str, str], ...]] | None = None,
) -> list[dict[str, Any]]:
    """All one-family neighbours of the current search state (Hill Climbing)."""
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    bid = str(discovery_bundle_id or "").strip()
    if not bid:
        art = latest_artifact(data_dir, rid, KIND_DISCOVERY_BUNDLE)
        if not art:
            raise ValueError(
                "No Discovery Bundle — freeze Discovery Complete first."
            )
        bid = str(art["artifact_id"])
    bundle = require_artifact(
        data_dir, bid, expected_kind=KIND_DISCOVERY_BUNDLE
    )
    payload = dict(bundle.get("payload") or {})
    scores = _scores_from_bundle(payload)
    fam_rows = list(payload.get("families") or [])
    bundle_reps = {
        str(k): str(v)
        for k, v in dict(payload.get("family_reps") or {}).items()
        if k and v
    }
    reps = dict(baseline_reps or bundle_reps)
    if not reps:
        raise ValueError("Discovery Bundle has no family representatives")
    seen = set(exclude_snapshots or set())
    seen.add(tuple(sorted(reps.items())))

    proposals: list[dict[str, Any]] = []
    for fam in fam_rows:
        fid = str(fam.get("family_id") or "").strip()
        if not fid or fid not in reps:
            continue
        members = [str(m) for m in list(fam.get("members") or []) if m]
        if len(members) < 2:
            continue
        cur = str(reps[fid])
        for nxt in members:
            if nxt == cur:
                continue
            trial = dict(reps)
            trial[fid] = nxt
            key = tuple(sorted(trial.items()))
            if key in seen:
                continue
            seen.add(key)
            cur_sc = float(scores.get(cur, 0.0))
            nxt_sc = float(scores.get(nxt, 0.0))
            proposals.append(
                {
                    "family_id": fid,
                    "family_label": str(fam.get("family_label") or fid),
                    "old_representative": cur,
                    "new_representative": nxt,
                    "old_score": cur_sc,
                    "new_score": nxt_sc,
                    "score_gap": cur_sc - nxt_sc,
                    "discovery_bundle_id": bid,
                    "snapshot_key": key,
                }
            )

    # Prefer higher Discovery-ranked replacements first when capping
    proposals.sort(key=lambda p: (-float(p["new_score"]), float(p["score_gap"])))
    return proposals[: max(0, int(max_variants))]


def estimate_search_space(payload: dict[str, Any]) -> dict[str, Any]:
    """Approximate combinatorial size of one-rep-per-family feature sets."""
    import math

    sizes: list[int] = []
    for fam in payload.get("families") or []:
        members = [m for m in list(fam.get("members") or []) if m]
        if len(members) >= 2:
            sizes.append(len(members))
        elif len(members) == 1:
            sizes.append(1)
    n_families = len(sizes)
    avg = (sum(sizes) / n_families) if n_families else 0.0
    log10_possible = 0.0
    for s in sizes:
        if s > 0:
            log10_possible += math.log10(float(s))
    # Display helper: ≈ avg^n when avg>=2
    if n_families and avg >= 2:
        possible_txt = f"≈ {avg:.1f}^{n_families}"
        if log10_possible >= 3:
            possible_txt += f"  (~1e{log10_possible:.0f})"
    elif n_families:
        possible_txt = f"≈ 10^{log10_possible:.1f}"
    else:
        possible_txt = "—"
    return {
        "n_families": n_families,
        "avg_candidates": round(avg, 1) if n_families else None,
        "member_sizes": sizes,
        "log10_possible": log10_possible,
        "possible_txt": possible_txt,
    }


def format_research_history(history: list[dict[str, Any]]) -> str:
    """Round → Champion → Score → Improvement table."""
    lines = [
        "Research History",
        f"{'Round':<6} {'Champion':<14} {'Score':<12} {'Improvement'}",
        f"{'-'*6:<6} {'-'*14:<14} {'-'*12:<12} {'-'*12}",
    ]
    if not history:
        lines.append("(none)")
        return "\n".join(lines)
    for row in history:
        rnd = row.get("round")
        champ = str(row.get("champion_id") or "—")
        score = row.get("score")
        score_txt = f"{float(score):.5f}" if score is not None else "—"
        if row.get("stop"):
            lines.append(
                f"{str(rnd):<6} {'Stop':<14} {'—':<12} "
                f"{row.get('note') or 'Below threshold'}"
            )
            continue
        imp = row.get("last_improvement")
        if row.get("round") == 0 or imp is None:
            imp_txt = "—"
        else:
            imp_txt = f"{float(imp):+.5f}"
        lines.append(
            f"{str(rnd):<6} {champ:<14} {score_txt:<12} {imp_txt}"
        )
    return "\n".join(lines)


def _composite_score(exp: dict[str, Any]) -> float:
    hold = float(exp.get("holdout_score") or 0.0)
    wf = float(exp.get("walk_forward_score") or 0.0)
    label = str(exp.get("validation_label") or "")
    hold_pts = max(0.0, min(50.0, hold * 50.0))
    wf_pts = max(0.0, min(35.0, wf * 35.0))
    if label == "Excellent":
        val_pts = 15.0
    elif label in ("Good", "Best"):
        val_pts = 10.0
    elif label == "Unstable":
        val_pts = 0.0
    else:
        val_pts = 5.0
    return float(hold_pts + wf_pts + val_pts)


def recommend_champion(
    experiments: list[dict[str, Any]],
    *,
    baseline_reps: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Best by Holdout/WF; within near-tie band prefer fewer Families Changed."""
    scored = [e for e in experiments if e.get("holdout_score") is not None]
    if not scored:
        return None
    primary = max(
        scored,
        key=lambda e: (
            float(e.get("holdout_score") or -1e9),
            float(e.get("walk_forward_score") or -1e9),
        ),
    )
    ph = float(primary.get("holdout_score") or 0.0)
    pw = float(primary.get("walk_forward_score") or 0.0)
    near = [
        e
        for e in scored
        if abs(float(e.get("holdout_score") or 0.0) - ph) <= NEAR_TIE_EPS
        and abs(float(e.get("walk_forward_score") or 0.0) - pw) <= NEAR_TIE_EPS
    ]
    return min(
        near,
        key=lambda e: (
            families_changed_count(e, baseline_reps),
            -float(e.get("holdout_score") or 0.0),
            -float(e.get("walk_forward_score") or 0.0),
            -_composite_score(e),
        ),
    )


def format_dashboard(dash: dict[str, Any]) -> str:
    """Live Search State card while Auto Research is running."""
    strat = str(dash.get("strategy") or "")
    if not strat:
        # Accept human strategy_label (e.g. "Single Swap") when id omitted
        label = str(dash.get("strategy_label") or "").strip().lower()
        for sid, slabel in STRATEGY_LABELS.items():
            if label == str(slabel).strip().lower() or label == sid:
                strat = sid
                break
    level = str(
        dash.get("research_level_label")
        or LEVEL_LABELS.get(STRATEGY_TO_LEVEL.get(strat, ""), "")
        or dash.get("strategy_label")
        or "—"
    )
    hold = dash.get("current_score")
    hold_txt = (
        f"{float(hold):.5f}"
        if hold is not None
        else str(dash.get("current_score_txt") or "—")
    )
    best_imp = dash.get("best_improvement")
    last_imp = dash.get("last_improvement")
    best_txt = f"{float(best_imp):+.5f}" if best_imp is not None else "—"
    last_txt = f"{float(last_imp):+.5f}" if last_imp is not None else "—"
    converged = dash.get("converged")
    if converged is True:
        conv_txt = "Yes"
    elif converged is False:
        conv_txt = "No"
    else:
        conv_txt = "—"
    lines = [
        "Search State",
        f"Research Level           {level}",
        f"Iteration                {dash.get('round', '—')} / {dash.get('max_rounds', '—')}",
        f"Current Champion         {dash.get('current_baseline_id') or dash.get('best_ever_id') or '—'}",
        f"Champion Score           {hold_txt}",
        f"Neighbours Evaluated     {dash.get('neighbours_evaluated', dash.get('n_experiments_so_far', '—'))}",
        f"Best Improvement         {best_txt}",
        f"Last Improvement         {last_txt}",
        f"Converged?               {conv_txt}",
    ]
    if dash.get("status"):
        lines.append(f"Status                   {dash.get('status')}")
    # Search space
    space = dash.get("search_space") or {}
    if space:
        lines.extend(
            [
                "",
                "Estimated Search Space",
                f"Families                 {space.get('n_families', '—')}",
                f"Avg candidates / family  {space.get('avg_candidates', '—')}",
                f"Possible feature sets    {space.get('possible_txt', '—')}",
                f"Explored                 {dash.get('neighbours_evaluated', space.get('explored', '—'))}",
                f"Exploration rate         {dash.get('exploration_rate_txt', '~0%')}",
            ]
        )
    tested = dash.get("families_tested")
    total = dash.get("n_families_total")
    if total:
        pct = 100.0 * int(tested or 0) / int(total)
        filled = int(round(16 * int(tested or 0) / int(total)))
        bar = "█" * filled + "░" * (16 - filled)
        lines.extend(
            [
                "",
                "Research Coverage",
                f"Families touched         {int(tested or 0)} / {int(total)}   {pct:.0f}%",
                bar,
            ]
        )
    if dash.get("stop_reason"):
        lines.append(f"Stop                     {dash.get('stop_reason')}")
    return "\n".join(lines)


def build_research_statistics(
    *,
    run_id: str,
    dataset: str = "",
    n_features: int | None = None,
    n_families: int | None = None,
    n_experiments: int = 0,
    n_models: int = 0,
    rounds: int = 0,
    strategy: str = "",
    strategy_label: str = "",
    champion_id: str | None = None,
    champion_holdout: float | None = None,
    champion_walk_forward: float | None = None,
    baseline_id: str | None = None,
    baseline_holdout: float | None = None,
    research_complete: bool = False,
    stop_reason: str = "",
    families_tested: int | None = None,
    neighbours_evaluated: int | None = None,
    best_improvement: float | None = None,
    last_improvement: float | None = None,
    converged: bool | None = None,
    search_space: dict[str, Any] | None = None,
    research_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Outcome card fields — readable in seconds after Auto Research."""
    improvement = None
    if champion_holdout is not None and baseline_holdout is not None:
        improvement = float(champion_holdout) - float(baseline_holdout)
    if best_improvement is None:
        best_improvement = improvement
    status = "Research Complete" if research_complete else "In Progress"
    space = dict(search_space or {})
    explored = int(
        neighbours_evaluated
        if neighbours_evaluated is not None
        else n_experiments
    )
    rate_txt = "~0%"
    log10_p = float(space.get("log10_possible") or 0.0)
    if log10_p > 0 and explored > 0:
        # explored / 10^log10_possible — almost always ~0%
        import math

        try:
            rate = explored / (10 ** log10_p)
            if rate < 0.0001:
                rate_txt = "~0%"
            else:
                rate_txt = f"{100.0 * rate:.4f}%"
        except OverflowError:
            rate_txt = "~0%"
    level_id = STRATEGY_TO_LEVEL.get(str(strategy or ""), "")
    level_label = LEVEL_LABELS.get(level_id, strategy_label or strategy)
    return {
        "run_id": run_id,
        "dataset": dataset or run_id,
        "n_features": n_features,
        "n_families": n_families,
        "n_experiments": int(n_experiments),
        "n_models": int(n_models),
        "rounds": int(rounds),
        "strategy": strategy,
        "strategy_label": strategy_label or strategy,
        "research_level": level_id,
        "research_level_label": level_label,
        "champion_id": champion_id,
        "champion_holdout": champion_holdout,
        "champion_walk_forward": champion_walk_forward,
        "baseline_id": baseline_id,
        "baseline_holdout": baseline_holdout,
        "improvement_over_baseline": improvement,
        "families_tested": families_tested,
        "neighbours_evaluated": explored,
        "best_improvement": best_improvement,
        "last_improvement": last_improvement,
        "converged": converged if converged is not None else research_complete,
        "search_space": space,
        "exploration_rate_txt": rate_txt,
        "research_history": list(research_history or []),
        "research_complete": bool(research_complete),
        "status": status,
        "stop_reason": stop_reason,
    }


def format_research_statistics(stats: dict[str, Any]) -> str:
    """Human-readable Research Statistics + Search State outcome card."""
    hold = stats.get("champion_holdout")
    wf = stats.get("champion_walk_forward")
    imp = stats.get("improvement_over_baseline")
    best_imp = stats.get("best_improvement")
    last_imp = stats.get("last_improvement")
    if imp is None:
        imp_txt = "—"
    else:
        imp_txt = f"{float(imp):+.6f}"
    best_txt = f"{float(best_imp):+.5f}" if best_imp is not None else "—"
    last_txt = f"{float(last_imp):+.5f}" if last_imp is not None else "—"
    conv = stats.get("converged")
    conv_txt = "Yes" if conv else "No"
    space = stats.get("search_space") or {}
    lines = [
        "Auto Research",
        f"Dataset                    {stats.get('dataset') or '—'}",
        f"Features                   {stats.get('n_features') if stats.get('n_features') is not None else '—'}",
        f"Families                   {stats.get('n_families') if stats.get('n_families') is not None else '—'}",
        f"Experiments                {stats.get('n_experiments', '—')}",
        f"Models                     {stats.get('n_models', '—')}",
        f"Iterations                 {stats.get('rounds', '—')}",
        f"Research Level             {stats.get('research_level_label') or stats.get('strategy_label') or stats.get('strategy') or '—'}",
        "",
        "Search State",
        f"Current Champion           {stats.get('champion_id') or '—'}",
        (
            f"Champion Score             {float(hold):.5f}"
            if hold is not None
            else "Champion Score             —"
        ),
        (
            f"Champion Walk-forward      {float(wf):.5f}"
            if wf is not None
            else "Champion Walk-forward      —"
        ),
        f"Neighbours Evaluated       {stats.get('neighbours_evaluated', stats.get('n_experiments', '—'))}",
        f"Best Improvement           {best_txt}",
        f"Last Improvement           {last_txt}",
        f"Improvement over Baseline  {imp_txt}",
        f"Converged?                 {conv_txt}",
        f"Status                     {stats.get('status') or '—'}",
    ]
    if stats.get("stop_reason"):
        lines.append(f"Stop                       {stats.get('stop_reason')}")
    if space:
        lines.extend(
            [
                "",
                "Estimated Search Space",
                f"Families                   {space.get('n_families', '—')}",
                f"Avg candidates / family    {space.get('avg_candidates', '—')}",
                f"Possible feature sets      {space.get('possible_txt', '—')}",
                f"Explored                   {stats.get('neighbours_evaluated', '—')}",
                f"Exploration rate           {stats.get('exploration_rate_txt', '~0%')}",
            ]
        )
    hist = stats.get("research_history") or []
    if hist:
        lines.append("")
        lines.append(format_research_history(hist))
    return "\n".join(lines)


def format_champion_card(
    *,
    experiment: dict[str, Any] | None,
    baseline_reps: dict[str, str] | None = None,
    fam_labels: dict[str, str] | None = None,
    stop_reason: str = "",
    overall_score: float | None = None,
    research_complete: bool = True,
) -> str:
    """Final destination card for the Champion page."""
    if not experiment:
        return "No champion yet — run Auto Research first."
    eid = str(experiment.get("experiment_id") or "—")
    hold = experiment.get("holdout_score")
    wf = experiment.get("walk_forward_score")
    label = str(experiment.get("validation_label") or "—")
    n_chg = families_changed_count(experiment, baseline_reps)
    improved = families_changed_labels(
        experiment, baseline_reps, fam_labels=fam_labels or {}
    )
    score = overall_score
    if score is None:
        score = _composite_score(experiment)
    ready = (
        research_complete
        and hold is not None
        and str(label) not in ("Unstable", "Pending", "")
    )
    lines = [
        "Champion Feature Set",
        "",
        f"Experiment         {eid}",
        f"Overall Score      {float(score):.0f}/100",
        f"Holdout            {float(hold):.5f}" if hold is not None else "Holdout            —",
        f"Walk-forward       {float(wf):.5f}" if wf is not None else "Walk-forward       —",
        f"Validation         {label}",
        f"Train device       {experiment.get('device_label') or experiment.get('train_device') or '—'}",
        f"Families Changed   {n_chg}",
        "",
        "Improvements",
    ]
    if improved:
        for name in improved:
            lines.append(f"✓ {name}")
    else:
        lines.append("(none vs Auto-baseline)")
    if stop_reason:
        lines.append("")
        lines.append(stop_reason)
    lines.append("")
    lines.append(f"Status             {'READY FOR PRODUCTION' if ready else 'IN PROGRESS'}")
    return "\n".join(lines)


def explain_recommendation(
    experiments: list[dict[str, Any]],
    *,
    baseline_id: str | None = None,
    recommended_id: str | None = None,
    baseline_reps: dict[str, str] | None = None,
    research_complete: bool = False,
    can_continue: bool = False,
) -> dict[str, Any]:
    """Current Best + reasons + Next Action."""
    scored = [e for e in experiments if e.get("holdout_score") is not None]
    if not scored:
        return {
            "recommended_id": None,
            "overall_score": None,
            "reasons": ["No trained experiments with Holdout scores."],
            "warnings": [],
            "next_action": "Start Auto Research",
            "text": "No recommendation — no trained experiments.",
        }

    champ = None
    if recommended_id:
        champ = next(
            (
                e
                for e in scored
                if str(e.get("experiment_id")) == str(recommended_id)
            ),
            None,
        )
    if champ is None:
        champ = recommend_champion(scored, baseline_reps=baseline_reps)
    assert champ is not None
    cid = str(champ.get("experiment_id"))
    hold = float(champ.get("holdout_score") or 0.0)
    wf = float(champ.get("walk_forward_score") or 0.0)
    label = str(champ.get("validation_label") or "—")
    n_chg = families_changed_count(champ, baseline_reps)

    best_hold = max(float(e.get("holdout_score") or -1e9) for e in scored)
    best_wf = max(float(e.get("walk_forward_score") or -1e9) for e in scored)
    reasons: list[str] = []
    warnings: list[str] = []

    if abs(hold - best_hold) <= NEAR_TIE_EPS:
        reasons.append(f"✓ Highest Holdout ({hold:.6f})")
    else:
        reasons.append(f"Holdout {hold:.6f} (best is {best_hold:.6f})")
    if abs(wf - best_wf) <= NEAR_TIE_EPS:
        reasons.append(f"✓ Highest Walk-forward ({wf:.6f})")
    else:
        reasons.append(f"Walk-forward {wf:.6f} (best is {best_wf:.6f})")
    if label in ("Excellent", "Good", "Best"):
        reasons.append(f"✓ Validation = {label}")
    else:
        reasons.append(f"Validation = {label}")
    reasons.append(f"Families Changed  {n_chg}")

    # Prefer-simpler note when a near-tie with more changes exists
    for e in scored:
        if str(e.get("experiment_id")) == cid:
            continue
        if abs(float(e.get("holdout_score") or 0) - hold) > NEAR_TIE_EPS:
            continue
        other_n = families_changed_count(e, baseline_reps)
        if other_n > n_chg:
            reasons.append(
                f"✓ Simpler than {e.get('experiment_id')} "
                f"({other_n} families changed vs {n_chg})"
            )
            break

    holds = [float(e.get("holdout_score") or 0.0) for e in scored]
    if len(holds) > 1 and (max(holds) - min(holds)) < 1e-4:
        warnings.append(
            "⚠ Holdout scores nearly identical — preferred fewer Families Changed."
        )

    baseline = None
    if baseline_id:
        baseline = next(
            (
                e
                for e in scored
                if str(e.get("experiment_id")) == str(baseline_id)
            ),
            None,
        )
    if baseline and cid != str(baseline.get("experiment_id")):
        b_hold = float(baseline.get("holdout_score") or 0.0)
        delta = hold - b_hold
        if delta > 0:
            reasons.append(f"✓ Improved baseline by {delta:+.6f}")
        elif delta < 0:
            reasons.append(f"Below original baseline by {delta:.6f}")
        else:
            reasons.append("No Holdout change vs original baseline")

    if research_complete:
        next_action = (
            f"Research Complete\n"
            f"No experiment improved the baseline further.\n"
            f"Champion remains {cid}."
        )
    elif can_continue:
        next_action = "Continue Auto Research"
    else:
        next_action = "Start Auto Research"

    overall = _composite_score(champ)
    lines = [
        f"Current Best       {cid}",
        f"Overall Score      {overall:.0f}/100",
        "",
        "Reason",
        *reasons,
        "",
        "Next Action",
        next_action,
    ]
    if warnings:
        lines.extend(["", *warnings])
    return {
        "recommended_id": cid,
        "overall_score": overall,
        "reasons": reasons,
        "warnings": warnings,
        "holdout": hold,
        "walk_forward": wf,
        "validation_label": label,
        "families_changed": n_chg,
        "next_action": next_action,
        "research_complete": research_complete,
        "text": "\n".join(lines),
        "experiment": champ,
    }


def _empty_dashboard(**kwargs: Any) -> dict[str, Any]:
    d = {
        "round": 0,
        "max_rounds": DEFAULT_MAX_ROUNDS,
        "strategy": STRATEGY_SINGLE_SWAP,
        "strategy_label": STRATEGY_LABELS[STRATEGY_SINGLE_SWAP],
        "current_baseline_id": None,
        "current_score": None,
        "current_score_txt": "—",
        "best_ever_id": None,
        "best_ever_score": None,
        "best_ever_score_txt": "—",
        "families_improved": [],
        "families_improved_txt": "—",
        "families_changed": 0,
        "families_tested": 0,
        "remaining_families": None,
        "n_families_total": None,
        "neighbours_evaluated": 0,
        "best_improvement": None,
        "last_improvement": None,
        "converged": False,
        "search_space": {},
        "exploration_rate_txt": "~0%",
        "status": "idle",
        "stop_reason": "",
    }
    d.update(kwargs)
    return d


def auto_create_and_train(
    data_dir: str,
    run_id: str,
    *,
    strategy: str = STRATEGY_SINGLE_SWAP,
    max_variants: int = DEFAULT_PER_ROUND,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
    wf_tolerance: float = DEFAULT_WF_TOLERANCE,
    target: str = "",
    promote: bool = True,
    freeze_if_missing: bool = True,
    resume_from_experiment_id: str | None = None,
    on_progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Auto Research with selectable strategy.

    * ``single_swap`` — one round of variants off the original baseline;
      pick the best (no stacking).
    * ``greedy`` — iterative: accepted swaps stack; next round from winner.
    * ``hill_climbing`` — each iteration evaluate neighbours of the current
      champion, move to the best improving neighbour, else converge.
    * ``beam`` / ``genetic`` — reserved (not implemented yet).
    """
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")

    strat = str(strategy or STRATEGY_SINGLE_SWAP).strip().lower()
    if strat in ("single", "swap"):
        strat = STRATEGY_SINGLE_SWAP
    if strat in ("hill", "hillclimb", "hill_climb"):
        strat = STRATEGY_HILL
    if strat not in STRATEGY_LABELS:
        raise ValueError(f"Unknown research strategy {strategy!r}")
    if strat not in IMPLEMENTED_STRATEGIES:
        raise ValueError(
            f"Strategy {STRATEGY_LABELS[strat]} is not implemented yet. "
            f"Use Single Swap, Greedy Search, or Hill Climbing."
        )

    # Single Swap = one round, always branch from original baseline
    if strat == STRATEGY_SINGLE_SWAP:
        max_rounds = 1

    dashboard = _empty_dashboard(
        max_rounds=int(max_rounds),
        status="running",
        strategy=strat,
        strategy_label=STRATEGY_LABELS[strat],
        research_level=STRATEGY_TO_LEVEL.get(strat),
        research_level_label=LEVEL_LABELS.get(
            STRATEGY_TO_LEVEL.get(strat, ""), STRATEGY_LABELS[strat]
        ),
        converged=False,
    )

    def _prog(p: float, msg: str) -> None:
        if on_progress:
            try:
                on_progress(
                    max(0.0, min(1.0, float(p))),
                    str(msg),
                    dict(dashboard),
                )
            except TypeError:
                on_progress(max(0.0, min(1.0, float(p))), str(msg))

    def _set_scores(current_id: str, hold: float, *, baseline_reps: dict[str, str],
                    fam_labels: dict[str, str], current_exp: dict[str, Any]) -> None:
        dashboard["current_baseline_id"] = current_id
        dashboard["current_score"] = hold
        dashboard["current_score_txt"] = f"{hold:.5f}"
        best = dashboard.get("best_ever_score")
        if best is None or hold >= float(best):
            dashboard["best_ever_id"] = current_id
            dashboard["best_ever_score"] = hold
            dashboard["best_ever_score_txt"] = f"{hold:.5f}"
        improved = families_changed_labels(
            current_exp, baseline_reps, fam_labels=fam_labels
        )
        dashboard["families_improved"] = improved
        dashboard["families_improved_txt"] = ", ".join(improved) if improved else "—"
        dashboard["families_changed"] = len(improved)

    def _sync_search_state(*, origin_hold: float) -> None:
        explored = int(dashboard.get("neighbours_evaluated") or 0)
        dashboard["neighbours_evaluated"] = explored
        space = dashboard.get("search_space") or {}
        log10_p = float(space.get("log10_possible") or 0.0)
        if log10_p > 0 and explored > 0:
            try:
                rate = explored / (10 ** log10_p)
                dashboard["exploration_rate_txt"] = (
                    "~0%" if rate < 0.0001 else f"{100.0 * rate:.4f}%"
                )
            except OverflowError:
                dashboard["exploration_rate_txt"] = "~0%"
        else:
            dashboard["exploration_rate_txt"] = "~0%"
        if origin_hold is not None and dashboard.get("current_score") is not None:
            total_imp = float(dashboard["current_score"]) - float(origin_hold)
            best = dashboard.get("best_improvement")
            if best is None or total_imp > float(best):
                dashboard["best_improvement"] = total_imp

    _prog(0.02, "Loading Discovery Bundle…")
    art = latest_artifact(data_dir, rid, KIND_DISCOVERY_BUNDLE)
    if not art:
        if not freeze_if_missing:
            raise ValueError(
                "No Discovery Bundle — freeze Discovery Complete first."
            )
        _prog(0.04, "Freezing Discovery Bundle…")
        art = publish_discovery_bundle(data_dir, rid)
    bid = str(art["artifact_id"])
    payload = dict(require_artifact(data_dir, bid, expected_kind=KIND_DISCOVERY_BUNDLE).get("payload") or {})
    fam_labels = {
        str(f.get("family_id")): str(f.get("family_label") or f.get("family_id"))
        for f in (payload.get("families") or [])
        if f.get("family_id")
    }
    n_changeable = sum(
        1
        for f in (payload.get("families") or [])
        if len(list(f.get("members") or [])) >= 2
    )
    search_space = estimate_search_space(payload)
    dashboard["n_families_total"] = n_changeable
    dashboard["search_space"] = search_space

    resume_id = str(resume_from_experiment_id or "").strip()
    trial_log: list[dict[str, Any]] = []
    train_errors: list[str] = []
    create_errors: list[str] = []
    variant_ids: list[str] = []
    fingerprints: list[str] = []
    tried_families: set[str] = set()
    proposals_all: list[dict[str, Any]] = []

    def _train_one(eid: str, frac0: float, frac1: float, label: str) -> dict[str, Any]:
        span = max(frac1 - frac0, 0.01)

        def _tp(frac: float, msg: str) -> None:
            _prog(frac0 + span * float(frac), f"{label}: {msg}")

        _prog(frac0, f"{label}: Training {eid}…")
        result = request_train_experiment(
            data_dir, eid, target=target, on_progress=_tp
        )
        fp = str(result.get("features_fingerprint") or "")
        if fp:
            fingerprints.append(fp)
        return result

    if resume_id:
        current = load_experiment(data_dir, resume_id)
        if not current:
            raise ValueError(f"Cannot resume — unknown experiment {resume_id!r}")
        baseline_id = str(
            current.get("parent_experiment_id")
            or next(
                (
                    e["experiment_id"]
                    for e in list_experiments(data_dir, rid)
                    if str(e.get("name") or "") == "Auto-baseline"
                ),
                resume_id,
            )
        )
        # Prefer true Auto-baseline for families_changed baseline_reps
        root = next(
            (
                e
                for e in list_experiments(data_dir, rid)
                if str(e.get("name") or "") == "Auto-baseline"
            ),
            None,
        )
        if root:
            baseline_id = str(root["experiment_id"])
            baseline_reps = _reps_map(root)
        else:
            baseline_reps = _reps_map(current)
        current_id = resume_id
        all_train_ids = [baseline_id, current_id]
        current_hold = float(current.get("holdout_score") or 0.0)
        current_wf = float(current.get("walk_forward_score") or 0.0)
        if current.get("holdout_score") is None:
            tres = _train_one(current_id, 0.06, 0.12, "[resume]")
            current = load_experiment(data_dir, current_id) or current
            current_hold = float(tres.get("holdout_r2") or 0.0)
            current_wf = float(tres.get("walk_forward_r2") or 0.0)
        # Families already different from root count as tried/improved
        for fid, rep in _reps_map(current).items():
            if baseline_reps.get(fid) != rep:
                tried_families.add(fid)
        _set_scores(
            current_id,
            current_hold,
            baseline_reps=baseline_reps,
            fam_labels=fam_labels,
            current_exp=current,
        )
        _prog(0.12, f"Resuming Auto Research from {current_id}")
    else:
        _prog(0.06, "Creating Auto-baseline…")
        baseline = create_experiment(
            data_dir,
            rid,
            name="Auto-baseline",
            notes="Auto Research baseline from Discovery Bundle",
            discovery_bundle_id=bid,
            freeze_discovery=False,
        )
        baseline_id = str(baseline["experiment_id"])
        baseline_reps = _reps_map(baseline)
        all_train_ids = [baseline_id]
        base_train = _train_one(baseline_id, 0.08, 0.16, "[baseline]")
        current = load_experiment(data_dir, baseline_id) or baseline
        current_id = baseline_id
        current_hold = float(
            current.get("holdout_score") or base_train.get("holdout_r2") or 0
        )
        current_wf = float(
            current.get("walk_forward_score")
            or base_train.get("walk_forward_r2")
            or 0
        )
        _set_scores(
            current_id,
            current_hold,
            baseline_reps=baseline_reps,
            fam_labels=fam_labels,
            current_exp=current,
        )

    stop_reason = ""
    rounds_done = 0
    research_complete = False
    origin_hold = float(current_hold)
    research_history: list[dict[str, Any]] = [
        {
            "round": 0,
            "champion_id": current_id,
            "score": current_hold,
            "last_improvement": None,
            "improvement": 0.0,
        }
    ]
    evaluated_snapshots: set[tuple[tuple[str, str], ...]] = set()
    evaluated_snapshots.add(tuple(sorted(_reps_map(current).items())))
    neighbours_evaluated = 0
    last_improvement: float | None = None
    best_improvement: float | None = 0.0

    def _sync_coverage() -> None:
        dashboard["families_tested"] = len(tried_families)
        dashboard["remaining_families"] = max(
            0, n_changeable - len(tried_families)
        )
        dashboard["neighbours_evaluated"] = neighbours_evaluated
        dashboard["last_improvement"] = last_improvement
        dashboard["best_improvement"] = best_improvement
        _sync_search_state(origin_hold=origin_hold)

    _sync_coverage()
    _prog(0.14, f"Search state ready · champion {current_id}")

    for round_i in range(1, max(1, int(max_rounds)) + 1):
        rounds_done = round_i
        dashboard["round"] = round_i
        dashboard["status"] = f"Iteration {round_i}"
        dashboard["converged"] = False

        if strat == STRATEGY_SINGLE_SWAP:
            cur_reps = dict(baseline_reps)
            propose_on_id = baseline_id
            proposals = propose_intelligent_variants(
                data_dir,
                rid,
                max_variants=max_variants,
                discovery_bundle_id=bid,
                baseline_reps=cur_reps,
                exclude_family_ids=tried_families,
            )
        elif strat == STRATEGY_HILL:
            cur_reps = _reps_map(current)
            propose_on_id = current_id
            proposals = propose_neighbour_swaps(
                data_dir,
                rid,
                max_variants=max_variants,
                discovery_bundle_id=bid,
                baseline_reps=cur_reps,
                exclude_snapshots=evaluated_snapshots,
            )
        else:
            cur_reps = _reps_map(current)
            propose_on_id = current_id
            proposals = propose_intelligent_variants(
                data_dir,
                rid,
                max_variants=max_variants,
                discovery_bundle_id=bid,
                baseline_reps=cur_reps,
                exclude_family_ids=tried_families,
            )

        proposals_all.extend(proposals)
        _sync_coverage()
        if not proposals:
            stop_reason = (
                f"Iteration {round_i}: no remaining neighbours — converged."
            )
            research_complete = True
            dashboard["converged"] = True
            research_history.append(
                {
                    "round": round_i,
                    "stop": True,
                    "note": "No neighbours left",
                }
            )
            break

        _prog(
            0.16 + 0.02 * (round_i - 1),
            f"Iteration {round_i}/{max_rounds}: "
            f"{len(proposals)} neighbours on {propose_on_id}…",
        )

        round_start_id = current_id
        round_start_hold = current_hold
        round_start_wf = current_wf
        n_prop = len(proposals)
        round_best_delta = 0.0

        # Hill climbing: evaluate all neighbours, then move once to best.
        # Greedy / Single Swap: may accept mid-batch (running champion).
        hill_best: dict[str, Any] | None = None

        for i, prop in enumerate(proposals):
            frac0 = 0.18 + 0.75 * (
                ((round_i - 1) * max_variants + i)
                / max(max_rounds * max_variants, 1)
            )
            frac1 = 0.18 + 0.75 * (
                ((round_i - 1) * max_variants + i + 1)
                / max(max_rounds * max_variants, 1)
            )
            fid = str(prop["family_id"])
            dashboard["status"] = (
                f"Iteration {round_i}/{max_rounds} · "
                f"neighbour {i + 1}/{n_prop}"
            )
            parent_for_clone = (
                baseline_id if strat == STRATEGY_SINGLE_SWAP else current_id
            )
            # Hill: always clone from iteration-start champion (steepest ascent)
            if strat == STRATEGY_HILL:
                parent_for_clone = round_start_id
            _prog(
                frac0,
                f"Iter {round_i} [{i + 1}/{n_prop}] "
                f"{prop['family_label']}: {prop['old_representative']} → "
                f"{prop['new_representative']}",
            )
            parent_before = parent_for_clone
            try:
                var = clone_experiment_variant(
                    data_dir,
                    parent_for_clone,
                    changes={fid: str(prop["new_representative"])},
                    name=(
                        f"Auto-r{round_i}-{fid}-"
                        f"{prop['new_representative']}"[:48]
                    ),
                    notes=(
                        f"{STRATEGY_LABELS[strat]} r{round_i} on {parent_for_clone}: "
                        f"{prop['family_label']} "
                        f"{prop['old_representative']} → "
                        f"{prop['new_representative']}"
                    ),
                )
            except Exception as exc:
                create_errors.append(f"{fid}: {exc}")
                continue
            vid = str(var["experiment_id"])
            variant_ids.append(vid)
            all_train_ids.append(vid)
            tried_families.add(fid)
            snap = prop.get("snapshot_key")
            if snap:
                evaluated_snapshots.add(snap)
            else:
                evaluated_snapshots.add(
                    tuple(sorted(_reps_map(var).items()))
                )
            neighbours_evaluated += 1
            _sync_coverage()
            try:
                tres = _train_one(
                    vid, frac0, frac1, f"[r{round_i} {i + 1}/{n_prop}]"
                )
            except Exception as exc:
                train_errors.append(f"{vid}: {exc}")
                trial_log.append(
                    {
                        "experiment_id": vid,
                        "parent_before": parent_before,
                        "accepted": False,
                        "reason": f"Train failed: {exc}",
                        "round": round_i,
                        **{
                            k: prop.get(k)
                            for k in (
                                "family_id",
                                "family_label",
                                "old_representative",
                                "new_representative",
                            )
                        },
                    }
                )
                continue

            v_hold = float(tres.get("holdout_r2") or 0.0)
            v_wf = float(tres.get("walk_forward_r2") or 0.0)

            if strat == STRATEGY_HILL:
                d_hold = v_hold - round_start_hold
                d_wf = v_wf - round_start_wf
                accepted = False
                improves = (
                    d_hold >= float(min_improvement)
                    and v_wf >= (round_start_wf - float(wf_tolerance))
                )
                if improves and (
                    hill_best is None
                    or v_hold > float(hill_best["holdout"]) + 1e-12
                    or (
                        abs(v_hold - float(hill_best["holdout"])) <= NEAR_TIE_EPS
                        and v_wf > float(hill_best["walk_forward"]) + 1e-12
                    )
                ):
                    hill_best = {
                        "experiment_id": vid,
                        "holdout": v_hold,
                        "walk_forward": v_wf,
                        "delta_holdout": d_hold,
                        "exp": load_experiment(data_dir, vid) or var,
                        "prop": prop,
                    }
                reason = (
                    f"Neighbour Holdout {v_hold:.6f} "
                    f"(Δ {d_hold:+.6f} vs {round_start_id})"
                )
            else:
                d_hold = v_hold - current_hold
                d_wf = v_wf - current_wf
                if strat == STRATEGY_SINGLE_SWAP:
                    accepted = (
                        v_hold > current_hold + 1e-12
                        and v_wf >= (current_wf - float(wf_tolerance))
                    ) or (
                        abs(v_hold - current_hold) <= NEAR_TIE_EPS
                        and v_wf > current_wf + 1e-12
                    )
                else:
                    accepted = (
                        d_hold >= float(min_improvement)
                        and v_wf >= (current_wf - float(wf_tolerance))
                    )
                if accepted:
                    reason = (
                        f"Accepted: Holdout {current_hold:.6f}→{v_hold:.6f} "
                        f"({d_hold:+.6f}); WF {current_wf:.6f}→{v_wf:.6f}"
                    )
                    last_improvement = d_hold
                    current_id = vid
                    current = load_experiment(data_dir, vid) or var
                    current_hold = v_hold
                    current_wf = v_wf
                    total_imp = current_hold - origin_hold
                    if best_improvement is None or total_imp > best_improvement:
                        best_improvement = total_imp
                    _set_scores(
                        current_id,
                        current_hold,
                        baseline_reps=baseline_reps,
                        fam_labels=fam_labels,
                        current_exp=current,
                    )
                    _sync_coverage()
                else:
                    bits = []
                    if strat == STRATEGY_SINGLE_SWAP:
                        bits.append(
                            f"not better than current best "
                            f"({current_id} Holdout {current_hold:.6f})"
                        )
                    else:
                        if d_hold < float(min_improvement):
                            bits.append(
                                f"Holdout Δ {d_hold:+.6f} < min {min_improvement}"
                            )
                        if v_wf < (current_wf - float(wf_tolerance)):
                            bits.append(
                                f"WF dropped {d_wf:.6f} (tol {wf_tolerance})"
                            )
                    reason = "Rejected: " + "; ".join(bits or ["no gain"])

            trial_log.append(
                {
                    "experiment_id": vid,
                    "parent_before": parent_before,
                    "accepted": accepted,
                    "holdout": v_hold,
                    "walk_forward": v_wf,
                    "delta_holdout": d_hold,
                    "delta_walk_forward": d_wf,
                    "reason": reason,
                    "round": round_i,
                    "families_changed": families_changed_count(
                        load_experiment(data_dir, vid), baseline_reps
                    ),
                    "family_id": fid,
                    "family_label": prop.get("family_label"),
                    "old_representative": prop.get("old_representative"),
                    "new_representative": prop.get("new_representative"),
                    "features_fingerprint": tres.get("features_fingerprint"),
                }
            )

        # End of iteration — hill climbing move (steepest ascent)
        round_best_delta = 0.0
        if strat == STRATEGY_HILL:
            if hill_best is not None:
                d_hold = float(hill_best["delta_holdout"])
                round_best_delta = d_hold
                last_improvement = d_hold
                current_id = str(hill_best["experiment_id"])
                current = hill_best["exp"]
                current_hold = float(hill_best["holdout"])
                current_wf = float(hill_best["walk_forward"])
                total_imp = current_hold - origin_hold
                if best_improvement is None or total_imp > best_improvement:
                    best_improvement = total_imp
                # Mark the chosen neighbour accepted in trial log
                for t in trial_log:
                    if str(t.get("experiment_id")) == current_id:
                        t["accepted"] = True
                        t["reason"] = (
                            f"Hill move: {round_start_id} → {current_id} "
                            f"({d_hold:+.6f})"
                        )
                _set_scores(
                    current_id,
                    current_hold,
                    baseline_reps=baseline_reps,
                    fam_labels=fam_labels,
                    current_exp=current,
                )
                _sync_coverage()
                research_history.append(
                    {
                        "round": round_i,
                        "champion_id": current_id,
                        "score": current_hold,
                        "last_improvement": last_improvement,
                        "improvement": total_imp,
                    }
                )
                dashboard["status"] = (
                    f"Iteration {round_i} · moved to {current_id}"
                )
                _prog(
                    min(0.92, 0.18 + 0.75 * (round_i / max(max_rounds, 1))),
                    f"Hill step → {current_id} ({current_hold:.5f}, "
                    f"{d_hold:+.5f})",
                )
            else:
                round_best_delta = 0.0
        elif strat == STRATEGY_GREEDY and current_id != round_start_id:
            round_best_delta = current_hold - round_start_hold
            last_improvement = round_best_delta
            total_imp = current_hold - origin_hold
            if best_improvement is None or total_imp > best_improvement:
                best_improvement = total_imp
            research_history.append(
                {
                    "round": round_i,
                    "champion_id": current_id,
                    "score": current_hold,
                    "last_improvement": last_improvement,
                    "improvement": total_imp,
                }
            )
            dashboard["status"] = (
                f"Iteration {round_i} complete · champion {current_id}"
            )
            _prog(
                min(0.92, 0.18 + 0.75 * (round_i / max(max_rounds, 1))),
                f"Round {round_i} best → {current_id} "
                f"(score {current_hold:.5f})",
            )
            _sync_coverage()
        elif strat == STRATEGY_SINGLE_SWAP and current_id != round_start_id:
            round_best_delta = current_hold - round_start_hold
            last_improvement = round_best_delta
            total_imp = current_hold - origin_hold
            best_improvement = total_imp
            research_history.append(
                {
                    "round": round_i,
                    "champion_id": current_id,
                    "score": current_hold,
                    "last_improvement": last_improvement,
                    "improvement": total_imp,
                }
            )
            _sync_coverage()
        else:
            # No move this iteration
            if strat != STRATEGY_SINGLE_SWAP:
                round_best_delta = 0.0

        if strat == STRATEGY_SINGLE_SWAP:
            stop_reason = (
                f"Single Swap complete (1 iteration). "
                f"Champion {current_id} "
                f"({neighbours_evaluated} neighbours evaluated)."
            )
            research_complete = True
            dashboard["converged"] = True
            break

        if round_best_delta < float(min_improvement):
            stop_reason = (
                f"Iteration {round_i}: no meaningful improvement "
                f"(best Δ {round_best_delta:.6f} < {min_improvement}). "
                f"Champion remains {current_id}."
            )
            research_complete = True
            dashboard["converged"] = True
            # Only append Stop if this iteration did not already record a move
            if not (
                research_history
                and research_history[-1].get("round") == round_i
                and research_history[-1].get("champion_id")
            ):
                research_history.append(
                    {
                        "round": round_i,
                        "stop": True,
                        "note": "Below threshold",
                    }
                )
            break
    else:
        stop_reason = f"Completed {rounds_done} iteration(s)."
        research_complete = rounds_done >= int(max_rounds)
        dashboard["converged"] = research_complete

    dashboard["stop_reason"] = stop_reason
    dashboard["status"] = "complete" if research_complete else "paused"
    dashboard["converged"] = bool(research_complete)
    _sync_coverage()
    _set_scores(
        current_id,
        current_hold,
        baseline_reps=baseline_reps,
        fam_labels=fam_labels,
        current_exp=current,
    )

    _prog(0.94, "Building recommendation…")
    experiments = list_experiments(data_dir, rid)
    auto_ids = set(all_train_ids)
    auto_exps = [
        e for e in experiments if str(e.get("experiment_id")) in auto_ids
    ]
    # Annotate families_changed for ranking
    for e in auto_exps:
        e["families_changed"] = families_changed_count(e, baseline_reps)

    champ = recommend_champion(auto_exps, baseline_reps=baseline_reps)
    champ_id = str(champ["experiment_id"]) if champ else current_id
    can_continue = (
        strat in CONTINUABLE_STRATEGIES
        and not research_complete
        and (
            int(dashboard.get("remaining_families") or 0) > 0
            or strat == STRATEGY_HILL
        )
    )

    explanation = explain_recommendation(
        auto_exps,
        baseline_id=baseline_id,
        recommended_id=champ_id,
        baseline_reps=baseline_reps,
        research_complete=research_complete,
        can_continue=can_continue and not research_complete,
    )
    if research_complete:
        explanation = explain_recommendation(
            auto_exps,
            baseline_id=baseline_id,
            recommended_id=champ_id,
            baseline_reps=baseline_reps,
            research_complete=True,
            can_continue=False,
        )

    promote_out: dict[str, Any] | None = None
    if promote and champ_id:
        try:
            promote_out = promote_champion(data_dir, champ_id)
        except Exception as exc:
            promote_out = {"error": str(exc), "experiment_id": champ_id}

    champ_exp = load_experiment(data_dir, champ_id) if champ_id else None
    champion_card = format_champion_card(
        experiment=champ_exp,
        baseline_reps=baseline_reps,
        fam_labels=fam_labels,
        stop_reason=stop_reason,
        overall_score=explanation.get("overall_score"),
        research_complete=research_complete,
    )
    if promote_out and promote_out.get("card_text"):
        champion_card = (
            str(promote_out["card_text"])
            + "\n\n"
            + champion_card
        )

    baseline_exp = load_experiment(data_dir, baseline_id)
    baseline_hold = None
    if baseline_exp and baseline_exp.get("holdout_score") is not None:
        baseline_hold = float(baseline_exp["holdout_score"])
    dataset_label = str(
        (payload.get("card") or {}).get("dataset") or rid
    )
    n_features_ds = (payload.get("card") or {}).get("n_features")
    if n_features_ds is None:
        n_features_ds = len(payload.get("discovery_ratings") or []) or None
    n_families_ds = (payload.get("card") or {}).get("n_families")
    if n_families_ds is None:
        n_families_ds = len(payload.get("families") or []) or None

    research_stats = build_research_statistics(
        run_id=rid,
        dataset=dataset_label,
        n_features=int(n_features_ds) if n_features_ds is not None else None,
        n_families=int(n_families_ds) if n_families_ds is not None else None,
        n_experiments=len(all_train_ids),
        n_models=len(all_train_ids) - len(train_errors),
        rounds=rounds_done,
        strategy=strat,
        strategy_label=STRATEGY_LABELS[strat],
        champion_id=champ_id,
        champion_holdout=(
            float(champ_exp["holdout_score"])
            if champ_exp and champ_exp.get("holdout_score") is not None
            else None
        ),
        champion_walk_forward=(
            float(champ_exp["walk_forward_score"])
            if champ_exp and champ_exp.get("walk_forward_score") is not None
            else None
        ),
        baseline_id=baseline_id,
        baseline_holdout=baseline_hold,
        research_complete=research_complete,
        stop_reason=stop_reason,
        families_tested=int(dashboard.get("families_tested") or 0),
        neighbours_evaluated=neighbours_evaluated,
        best_improvement=best_improvement,
        last_improvement=last_improvement,
        converged=bool(dashboard.get("converged")),
        search_space=search_space,
        research_history=research_history,
    )
    research_stats_text = format_research_statistics(research_stats)
    dashboard_text = research_stats_text

    _prog(1.0, f"Done · best {champ_id}")
    message = (
        f"Auto Research ({STRATEGY_LABELS[strat]}): {rounds_done} iteration(s), "
        f"baseline {baseline_id}, {neighbours_evaluated} neighbours. "
        f"Champion {champ_id}. {stop_reason}"
    )
    return {
        "run_id": rid,
        "discovery_bundle_id": bid,
        "baseline_experiment_id": baseline_id,
        "variant_experiment_ids": variant_ids,
        "proposals": proposals_all,
        "trial_log": trial_log,
        "research_history": research_history,
        "research_history_text": format_research_history(research_history),
        "rounds_done": rounds_done,
        "max_rounds": int(max_rounds),
        "stop_reason": stop_reason,
        "research_complete": research_complete,
        "min_improvement": min_improvement,
        "strategy": strat,
        "strategy_label": STRATEGY_LABELS[strat],
        "current_winner_id": current_id,
        "dashboard": dashboard,
        "dashboard_text": dashboard_text,
        "research_statistics": research_stats,
        "research_statistics_text": research_stats_text,
        "champion_card": champion_card,
        "created_count": 1 + len(variant_ids),
        "trained_count": len(all_train_ids) - len(train_errors),
        "neighbours_evaluated": neighbours_evaluated,
        "create_errors": create_errors,
        "train_errors": train_errors,
        "features_fingerprints": fingerprints,
        "unique_feature_fingerprints": len(set(fingerprints)),
        "recommended_champion_id": champ_id,
        "recommended_champion": champ,
        "recommendation": explanation,
        "promote": promote_out,
        "message": message,
        "can_continue": can_continue and strat in CONTINUABLE_STRATEGIES,
        "experiments": [
            load_experiment(data_dir, eid) for eid in all_train_ids
        ],
    }


__all__ = [
    "CONTINUABLE_STRATEGIES",
    "DEFAULT_MAX_ROUNDS",
    "DEFAULT_MIN_IMPROVEMENT",
    "DEFAULT_PER_ROUND",
    "DEFAULT_RESEARCH_LEVEL",
    "DEFAULT_WF_TOLERANCE",
    "IMPLEMENTED_STRATEGIES",
    "LEVEL_BALANCED",
    "LEVEL_DEEP",
    "LEVEL_HINTS",
    "LEVEL_LABELS",
    "LEVEL_QUICK",
    "LEVEL_TO_STRATEGY",
    "NEAR_TIE_EPS",
    "PER_ROUND_CHOICES",
    "STRATEGY_BEAM",
    "STRATEGY_GENETIC",
    "STRATEGY_GREEDY",
    "STRATEGY_HILL",
    "STRATEGY_LABELS",
    "STRATEGY_SINGLE_SWAP",
    "STRATEGY_TO_LEVEL",
    "auto_create_and_train",
    "build_research_statistics",
    "estimate_search_space",
    "explain_recommendation",
    "families_changed_count",
    "families_changed_labels",
    "format_champion_card",
    "format_dashboard",
    "format_research_history",
    "format_research_statistics",
    "propose_intelligent_variants",
    "propose_neighbour_swaps",
    "recommend_champion",
    "resolve_research_strategy",
]
