"""HTML training report for a saved model package."""

from __future__ import annotations

import html
import json
from typing import Any


def _esc(val: Any) -> str:
    return html.escape("" if val is None else str(val))


def _fmt_num(val: Any, digits: int = 4) -> str:
    if val is None:
        return "—"
    try:
        n = float(val)
        if not (n == n):  # NaN
            return "—"
        return f"{n:.{digits}f}"
    except (TypeError, ValueError):
        return _esc(val)


def build_training_report_html(
    *,
    summary: dict[str, Any],
    config: dict[str, Any],
    metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
    pipeline_fingerprint: dict[str, Any] | None,
    training_log: str,
) -> str:
    test = summary.get("test_metrics") or metrics.get("test") or {}
    val = metrics.get("validation") or {}
    params = config.get("parameters") or {}
    split = config.get("split") or {}
    features = config.get("features") or []
    training_meta = metrics.get("training_meta") or {}

    validation_rows = [
        ("Best Validation RMSE", training_meta.get("best_validation_rmse") or summary.get("validation_rmse") or val.get("rmse")),
        ("Best Validation MAE", summary.get("validation_mae") or val.get("mae")),
        ("Validation R²", summary.get("validation_r2") or val.get("r2")),
        ("Validation Directional Accuracy", summary.get("validation_directional_accuracy_pct") or val.get("directional_accuracy_pct")),
        ("Validation MAPE", summary.get("validation_mape") or val.get("mape")),
        ("Best Iteration", summary.get("best_iteration") or training_meta.get("best_iteration")),
        ("Early Stopping Round", summary.get("early_stopping_rounds") or training_meta.get("early_stopping_rounds")),
        ("Training RMSE", summary.get("train_rmse") or training_meta.get("train_rmse")),
    ]
    validation_html = "".join(
        f"<tr><td>{_esc(k)}</td><td class='num'>{_fmt_num(v)}</td></tr>" for k, v in validation_rows
    )

    metric_rows = [
        ("RMSE", test.get("rmse")),
        ("MAE", test.get("mae")),
        ("MAPE", test.get("mape")),
        ("R²", test.get("r2")),
        ("Median Error", test.get("median_error")),
        ("Max Error", test.get("max_error")),
        ("Directional Accuracy", test.get("directional_accuracy") or test.get("directional_accuracy_pct")),
        ("Validation RMSE", summary.get("validation_rmse") or val.get("rmse")),
    ]

    metrics_html = "".join(
        f"<tr><td>{_esc(k)}</td><td class='num'>{_fmt_num(v)}</td></tr>" for k, v in metric_rows
    )

    param_rows = [
        ("Learning rate", params.get("learning_rate")),
        ("Max depth", params.get("max_depth")),
        ("N estimators", params.get("n_estimators")),
        ("Early stopping rounds", params.get("early_stopping_rounds")),
        ("Subsample", params.get("subsample")),
        ("Colsample by tree", params.get("colsample_bytree")),
        ("Min child weight", params.get("min_child_weight")),
        ("Reg alpha", params.get("reg_alpha")),
        ("Reg lambda", params.get("reg_lambda")),
        ("Random seed", params.get("random_seed")),
        ("Train / Val / Test", f"{split.get('train', '—')}% / {split.get('validation', '—')}% / {split.get('test', '—')}%"),
        ("Split strategy", split.get("strategy")),
    ]
    params_html = "".join(
        f"<tr><td>{_esc(k)}</td><td class='num'>{_esc(v)}</td></tr>" for k, v in param_rows
    )

    fi_rows = "".join(
        f"<tr><td>{_esc(r.get('feature'))}</td><td class='num'>{_fmt_num(r.get('importance_pct'), 2)}</td></tr>"
        for r in feature_importance[:50]
    )

    ds_meta = config.get("dataset_metadata") or {}
    fp_json = json.dumps(pipeline_fingerprint or {}, indent=2)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Training Report — {_esc(summary.get('model_name'))}</title>
<style>
body {{ font-family: Consolas, Menlo, monospace; background: #0e1218; color: #e5e9f0; margin: 0; padding: 24px; font-size: 12px; }}
h1 {{ color: #58a6ff; font-size: 18px; margin: 0 0 4px; }}
.meta {{ color: #7a8392; margin-bottom: 20px; }}
section {{ background: #1a2030; border: 1px solid #2a3142; border-radius: 6px; padding: 14px 16px; margin-bottom: 14px; }}
h2 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #7a8392; margin: 0 0 10px; }}
table {{ width: 100%; border-collapse: collapse; }}
td, th {{ padding: 6px 8px; border-bottom: 1px solid #2a3142; text-align: left; }}
.num {{ text-align: right; }}
.kpi {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
.kpi div {{ background: rgba(0,0,0,0.25); padding: 10px; border-radius: 4px; }}
.kpi label {{ display: block; font-size: 9px; color: #7a8392; text-transform: uppercase; }}
.kpi span {{ font-size: 16px; font-weight: 700; color: #58a6ff; }}
pre {{ background: rgba(0,0,0,0.3); padding: 10px; border-radius: 4px; overflow: auto; font-size: 10px; white-space: pre-wrap; }}
.log {{ font-size: 11px; line-height: 1.6; color: #c9d1d9; }}
</style>
</head>
<body>
<h1>Training Report</h1>
<div class="meta">{_esc(summary.get('model_name'))} · {_esc(summary.get('algorithm'))} · {_esc(summary.get('target'))}</div>

<section>
<h2>Overview</h2>
<div class="kpi">
<div><label>Dataset</label><span style="font-size:12px">{_esc(summary.get('dataset'))}</span></div>
<div><label>Rows</label><span>{_esc(summary.get('rows'))}</span></div>
<div><label>Features</label><span>{_esc(summary.get('features'))}</span></div>
<div><label>Training Time</label><span style="font-size:12px">{_fmt_num(summary.get('training_time_sec'), 1)}s</span></div>
<div><label>Trees</label><span>{_esc(summary.get('trees_trained'))}</span></div>
<div><label>Early Stop</label><span style="font-size:12px">{'Yes' if summary.get('early_stopped') else 'No'}</span></div>
<div><label>Version</label><span style="font-size:12px">{_esc(config.get('model_version'))}</span></div>
<div><label>Val RMSE</label><span>{_fmt_num(summary.get('validation_rmse'))}</span></div>
</div>
</section>

<section><h2>Validation Metrics</h2><table>{validation_html}</table></section>
<section><h2>Test Metrics</h2><table>{metrics_html}</table></section>
<section><h2>Training Configuration</h2><table>{params_html}</table>
<p style="color:#7a8392;margin-top:10px">Features selected: {len(features)}</p></section>
<section><h2>Feature Importance (top 50)</h2><table><thead><tr><th>Feature</th><th>Importance %</th></tr></thead><tbody>{fi_rows}</tbody></table></section>
<section><h2>Pipeline Fingerprint</h2><pre>{_esc(fp_json)}</pre></section>
<section><h2>Training Log</h2><pre class="log">{_esc(training_log)}</pre></section>
</body>
</html>"""
