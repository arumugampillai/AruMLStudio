"""Cryptographic lineage graph resolver and tracker (Phase 4F.2).

Reconstructs parent-child lineage trees and enables full ancestral tracing from leaf candidates back to root.
"""

from __future__ import annotations

from typing import Any, Sequence
from .types import CandidateLineageRecord, CandidateSpec


def reconstruct_lineage_graph(candidates: Sequence[CandidateSpec]) -> dict[str, list[str]]:
    """Build an adjacency map of parent_candidate_id -> list of child candidate_ids."""
    graph: dict[str, list[str]] = {}
    for c in candidates:
        if c.lineage and c.lineage.parent_candidate_id:
            p_id = c.lineage.parent_candidate_id
            if p_id not in graph:
                graph[p_id] = []
            graph[p_id].append(c.candidate_id)
    return graph


def trace_ancestors(
    candidate_id: str,
    candidates_by_id: dict[str, CandidateSpec],
) -> list[CandidateLineageRecord]:
    """Trace complete ancestral lineage from a candidate back to the root ancestor."""
    trail: list[CandidateLineageRecord] = []
    curr_id: str | None = candidate_id

    visited = set()
    while curr_id and curr_id in candidates_by_id and curr_id not in visited:
        visited.add(curr_id)
        cand = candidates_by_id[curr_id]
        if cand.lineage:
            trail.append(cand.lineage)
            curr_id = cand.lineage.parent_candidate_id
        else:
            break

    return trail
