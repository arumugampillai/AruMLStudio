"""Shared UI helpers for experiment duplicate warnings."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import Any, Callable


def confirm_experiment_after_similarity_check(
    parent: tk.Misc,
    check: dict[str, Any],
    *,
    action_label: str = "Create Experiment",
) -> bool:
    """Return True if the user chooses to proceed."""
    if not check.get("ok"):
        return True

    verdict = check.get("verdict") or "novel"
    matches = check.get("matches") or []
    if not matches:
        return True

    lines = [check.get("recommendation") or "", ""]
    for m in matches[:3]:
        outcome = m.get("outcome") or "unknown"
        outcome_label = {
            "improved": "Improved",
            "no_improvement": "No Improvement",
            "mixed": "Mixed",
            "unknown": "Unknown",
        }.get(outcome, outcome)
        pf = m.get("pf_change")
        pf_txt = f" · PF Δ {pf:+.2f}" if pf is not None else ""
        lines.append(
            f"Experiment #{m.get('experiment_number')} — {m.get('similarity_pct')}% similar — {outcome_label}{pf_txt}",
        )
        shared = m.get("shared_changes") or []
        if shared:
            lines.append(f"  Shared: {', '.join(shared[:4])}")

    title = {
        "very_similar": "Very Similar Experiment",
        "similar": "Similar Past Experiment",
        "novel": "Novel Experiment",
    }.get(verdict, "Experiment Check")

    if check.get("should_warn"):
        return messagebox.askyesno(
            title,
            "\n".join(lines) + f"\n\nProceed with {action_label} anyway?",
            parent=parent,
        )

    if verdict == "novel":
        return True

    messagebox.showinfo(title, "\n".join(lines), parent=parent)
    return True


def run_similarity_preflight(
    data_dir: str,
    check_fn: Callable[[], dict[str, Any]],
    parent: tk.Misc,
    *,
    action_label: str = "Create Experiment",
) -> bool:
    try:
        check = check_fn()
    except Exception as exc:
        messagebox.showerror("Similarity Check", str(exc), parent=parent)
        return False
    return confirm_experiment_after_similarity_check(parent, check, action_label=action_label)
