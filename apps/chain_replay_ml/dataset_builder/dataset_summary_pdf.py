"""Colorful PDF export for the dataset Summary tab."""

from __future__ import annotations

import io
from typing import Any

from .audit_investigation_engine import training_recommendation_display


def _fmt_num(v: Any) -> str:
    if v is None:
        return "—"
    try:
        n = float(v)
        if abs(n - int(n)) < 1e-9:
            return f"{int(n):,}"
        return f"{n:,.1f}"
    except (TypeError, ValueError):
        return str(v)


def _pct(v: Any) -> str:
    if v is None:
        return "—"
    return f"{v}%"


def _status_key(rec: str | None) -> str:
    key = str(rec or "NOT_READY").upper().replace(" ", "_")
    if "WARNING" in key:
        return "READY_WITH_WARNINGS"
    if key == "READY":
        return "READY"
    return "NOT_READY"


def build_summary_pdf(summary: dict[str, Any]) -> bytes:
    """Render dataset summary dict as a colorful PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError(
            "reportlab is not installed. Run: python -m pip install reportlab"
        ) from exc

    status_colors = {
        "READY": colors.HexColor("#1b5e20"),
        "READY_WITH_WARNINGS": colors.HexColor("#e65100"),
        "NOT_READY": colors.HexColor("#b71c1c"),
    }
    status_bg = {
        "READY": colors.HexColor("#e8f5e9"),
        "READY_WITH_WARNINGS": colors.HexColor("#fff8e1"),
        "NOT_READY": colors.HexColor("#ffebee"),
    }
    section_bg = colors.HexColor("#f5f7fb")
    header_bg = colors.HexColor("#1565c0")
    muted = colors.HexColor("#5f6368")
    grid_line = colors.HexColor("#dde3ea")

    def section_table(rows: list[tuple[str, str]]) -> Table:
        data = [[k, v] for k, v in rows]
        table = Table(data, colWidths=[62 * mm, 108 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR", (0, 0), (0, -1), muted),
                    ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#1a1a1a")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.5, grid_line),
                ]
            )
        )
        return table

    def section_header(title: str, styles: Any) -> Table:
        para = Paragraph(f"<b>{title}</b>", styles["DsSectionTitle"])
        table = Table([[para]], colWidths=[170 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), section_bg),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return table

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Dataset Summary — {summary.get('dataset_name', '')}",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BannerTitle",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.white,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DsSectionTitle",
            parent=styles["Heading3"],
            fontSize=10,
            textColor=colors.HexColor("#1565c0"),
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DsStatusLine",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DsBullet",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#2e7d32"),
            leftIndent=8,
        )
    )

    story: list[Any] = []
    rec_key = _status_key((summary.get("status") or {}).get("training_recommendation"))
    status_display = (summary.get("status") or {}).get("display") or training_recommendation_display(rec_key)

    banner = Table(
        [[Paragraph("DATASET SUMMARY", styles["BannerTitle"])]],
        colWidths=[174 * mm],
    )
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), header_bg),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(banner)
    story.append(Spacer(1, 8))

    story.append(section_table([("Name", str(summary.get("dataset_name") or "—"))]))
    story.append(Spacer(1, 6))

    conf = summary.get("dataset_confidence") or {}
    if conf.get("pct") is not None:
        story.append(section_header("Dataset Confidence", styles))
        story.append(Spacer(1, 4))
        story.append(section_table([
            ("Score", f"{conf.get('pct')}%"),
            ("Bar", str(conf.get("bar") or "—")),
        ]))
        formula = conf.get("formula") or {}
        formula_lines = list(formula.get("lines") or [])
        final_line = formula.get("final_score_label")
        if final_line:
            formula_lines.append(str(final_line))
        if formula_lines:
            story.append(section_table([
                (formula.get("title") or "Confidence Formula", "<br/>".join(formula_lines)),
            ]))
        story.append(Spacer(1, 8))

    status_box = Table(
        [[Paragraph(f"<b>Status</b><br/>{status_display}", styles["DsStatusLine"])]],
        colWidths=[174 * mm],
    )
    status_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), status_bg[rec_key]),
                ("TEXTCOLOR", (0, 0), (-1, -1), status_colors[rec_key]),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("BOX", (0, 0), (-1, -1), 1, status_colors[rec_key]),
            ]
        )
    )
    story.append(status_box)
    story.append(Spacer(1, 10))

    readiness = summary.get("model_training_readiness") or {}
    readiness_checks = readiness.get("checks") or []
    if readiness_checks:
        story.append(section_header("Model Training Readiness", styles))
        story.append(Spacer(1, 4))
        story.append(section_table([
            (str(row.get("label") or "—"), str(row.get("value") or "—"))
            for row in readiness_checks
        ]))
        overall = readiness.get("overall") or {}
        story.append(section_table([(
            "Overall",
            str(overall.get("display") or overall.get("short_label") or "—"),
        )]))
        story.append(Spacer(1, 8))

    d = summary.get("dataset") or {}
    story.append(section_header("Dataset", styles))
    story.append(Spacer(1, 4))
    story.append(
        section_table(
            [
                ("Market", str(d.get("market") or "—")),
                ("Trading Days", _fmt_num(d.get("trading_days"))),
                ("Rows", _fmt_num(d.get("rows"))),
                ("Features", _fmt_num(d.get("features"))),
                ("Targets", _fmt_num(d.get("targets"))),
                ("Metadata", _fmt_num(d.get("metadata_columns"))),
                ("Total Columns", _fmt_num(d.get("total_columns"))),
            ]
        )
    )
    story.append(Spacer(1, 8))

    samp = summary.get("sampling") or {}
    story.append(section_header("Sampling", styles))
    story.append(Spacer(1, 4))
    story.append(
        section_table(
            [
                ("Sampling Interval", str(samp.get("interval_label") or "—")),
                ("Method", str(samp.get("method") or "—")),
                ("ATM Band", str(samp.get("strikes_label") or "—")),
            ]
        )
    )
    story.append(Spacer(1, 8))

    lineage = summary.get("dataset_lineage") or {}
    story.append(section_header("Dataset Built From", styles))
    story.append(Spacer(1, 4))
    story.append(
        section_table(
            [
                ("Trading Days", _fmt_num(lineage.get("trading_days"))),
                ("Expiry", str(lineage.get("expiry") or "—")),
                ("Sampling", str(lineage.get("sampling") or "—")),
                ("ATM Band", str(lineage.get("atm_band") or "—")),
                ("Source DB", str(lineage.get("source_db") or "—")),
                ("Build Time", str(lineage.get("build_time") or "—")),
            ]
        )
    )
    story.append(Spacer(1, 8))

    pred_rows = summary.get("prediction_target_rows") or []
    if not pred_rows:
        pred_rows = [{"target": t, "predicts": "—"} for t in (summary.get("prediction_targets") or [])]
    story.append(section_header("Prediction Targets", styles))
    story.append(Spacer(1, 4))
    if pred_rows:
        for row in pred_rows:
            story.append(section_table([
                ("Target", str(row.get("target") or "—")),
                ("Predicts", str(row.get("predicts") or "—")),
            ]))
            story.append(Spacer(1, 2))
    else:
        story.append(Paragraph("—", styles["DsBullet"]))
    story.append(Spacer(1, 8))

    target_details = summary.get("target_details") or []
    if target_details:
        story.append(section_header("Target Details", styles))
        story.append(Spacer(1, 4))
        for t in target_details:
            story.append(section_table([
                ("Target", str(t.get("target") or "—")),
                ("Predicts", str(t.get("predicts") or "—")),
                ("Rows", _fmt_num(t.get("rows"))),
                ("Missing", _fmt_num(t.get("missing"))),
                ("Mean", str(t.get("mean") if t.get("mean") is not None else "—")),
                ("Std Dev", str(t.get("std_dev") if t.get("std_dev") is not None else "—")),
                ("Target Type", str(t.get("target_type") or "Regression")),
            ]))
            story.append(Spacer(1, 2))
        story.append(Spacer(1, 6))

    feat = summary.get("feature_summary") or {}
    story.append(section_header("Feature Summary", styles))
    story.append(Spacer(1, 4))
    story.append(
        section_table(
            [
                ("Expected Features", _fmt_num(feat.get("expected"))),
                ("Implemented", _fmt_num(feat.get("implemented"))),
                ("Feature Coverage", _pct(feat.get("coverage_pct"))),
                ("Formula Validation", str(feat.get("formula_display") or "—")),
                ("Replay Validation", str(feat.get("replay_display") or "—")),
            ]
        )
    )
    story.append(Spacer(1, 8))

    feature_groups = summary.get("feature_group_coverage") or []
    if feature_groups:
        story.append(section_header("Feature Groups", styles))
        story.append(Spacer(1, 4))
        for g in feature_groups:
            icon = "✓" if g.get("complete") else "○"
            story.append(Paragraph(f"{icon} {g.get('display') or g.get('label') or '—'}", styles["DsBullet"]))
        story.append(Spacer(1, 8))

    q = summary.get("quality") or {}
    story.append(section_header("Dataset Quality", styles))
    story.append(Spacer(1, 4))
    story.append(
        section_table(
            [
                ("Duplicate Rows", _fmt_num(q.get("duplicate_rows"))),
                ("Missing Targets", _fmt_num(q.get("missing_targets"))),
                ("Invalid Strike Rows", _fmt_num(q.get("invalid_strike_rows"))),
                ("Invalid Timestamps", _fmt_num(q.get("invalid_timestamps"))),
                ("Expected Nulls", _fmt_num(q.get("expected_nulls"))),
                ("Unexpected Nulls", _fmt_num(q.get("unexpected_nulls"))),
            ]
        )
    )
    story.append(Spacer(1, 8))

    aud = summary.get("audit") or {}
    audit_status = (
        aud.get("training_recommendation_display")
        or training_recommendation_display(aud.get("training_recommendation"))
        or aud.get("status")
        or "—"
    )
    story.append(section_header("Audit", styles))
    story.append(Spacer(1, 4))
    story.append(
        section_table(
            [
                ("Audit Status", str(audit_status)),
                ("Critical Issues", _fmt_num(aud.get("critical_issues"))),
                ("Warnings", _fmt_num(aud.get("warnings"))),
                ("Information", _fmt_num(aud.get("information"))),
                ("Builder Confidence", _pct(aud.get("builder_confidence_pct"))),
                ("Dataset Health", _pct(aud.get("dataset_health_pct"))),
                ("Training Recommendation", str(aud.get("training_recommendation_display") or status_display)),
            ]
        )
    )
    story.append(Spacer(1, 8))

    files = summary.get("files") or []
    story.append(section_header("Files", styles))
    story.append(Spacer(1, 4))
    for f in files:
        icon = "✓" if f.get("exists") else "○"
        color = "#2e7d32" if f.get("exists") else "#9e9e9e"
        label = f.get("label") or f.get("key") or "File"
        story.append(Paragraph(f'<font color="{color}">{icon} {label}</font>', styles["DsBullet"]))
    story.append(Spacer(1, 8))

    pipe = summary.get("pipeline") or {}
    fp_match = pipe.get("fingerprint_match")
    fp_label = "Match ✓" if fp_match is True else ("Mismatch ✗" if fp_match is False else "—")
    story.append(section_header("Pipeline Identity", styles))
    story.append(Spacer(1, 4))
    story.append(
        section_table(
            [
                ("Dataset Version", f"v{str(pipe.get('dataset_version') or '—').lstrip('v')}"),
                ("Builder Version", f"v{str(pipe.get('builder_version') or '—').lstrip('v')}"),
                ("Validator Version", f"v{str(pipe.get('validator_version') or '—').lstrip('v')}"),
                ("Spec Hash", str(pipe.get("spec_hash") or "—")),
                ("Fingerprint", fp_label),
            ]
        )
    )
    story.append(Spacer(1, 8))

    cert = summary.get("dataset_certification") or {}
    cert_checks = cert.get("checks") or []
    if cert_checks:
        story.append(section_header("Dataset Certification", styles))
        story.append(Spacer(1, 4))
        for row in cert_checks:
            story.append(section_table([
                (str(row.get("label") or "—"), str(row.get("display") or "—")),
            ]))
        overall = cert.get("overall") or {}
        story.append(section_table([("Overall", str(overall.get("display") or "—"))]))
        story.append(section_table([
            ("Certification Date", str(cert.get("certification_date") or "—")),
            ("Certified By", str(cert.get("certified_by") or "—")),
        ]))

    doc.build(story)
    return buffer.getvalue()
