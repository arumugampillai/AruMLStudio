"""Research Timeline — walk catalog URI DAG + time order."""

from __future__ import annotations

from .store import ArtifactCatalogStore
from .types import TimelineEvent
from .uri import is_artifact_uri


def timeline_chronological(store: ArtifactCatalogStore) -> list[TimelineEvent]:
    """All artifacts ordered by created_at (research history)."""
    return [
        TimelineEvent(
            artifact_uri=r.artifact_uri,
            artifact_type=r.artifact_type,
            created_at=r.created_at,
            parent_artifact_uris=list(r.parent_artifact_uris),
            depth=0,
        )
        for r in store.list_all()
    ]


def lineage_ancestors(
    store: ArtifactCatalogStore,
    artifact_uri: str,
    *,
    max_depth: int = 32,
) -> list[TimelineEvent]:
    """Ancestors of ``artifact_uri`` (parents first by BFS depth).

    Suitable for 'show lineage for aruneo://model/…' (§9.4).
    """
    if not is_artifact_uri(artifact_uri):
        return []
    root = store.get(artifact_uri)
    if root is None:
        return []

    out: list[TimelineEvent] = []
    seen: set[str] = {artifact_uri}
    queue: list[tuple[str, int]] = [(p, 1) for p in root.parent_artifact_uris]
    while queue:
        uri, depth = queue.pop(0)
        if uri in seen or depth > max_depth:
            continue
        seen.add(uri)
        rec = store.get(uri)
        if rec is None:
            out.append(
                TimelineEvent(
                    artifact_uri=uri,
                    artifact_type="other",
                    created_at="",
                    parent_artifact_uris=[],
                    depth=depth,
                )
            )
            continue
        out.append(
            TimelineEvent(
                artifact_uri=rec.artifact_uri,
                artifact_type=rec.artifact_type,
                created_at=rec.created_at,
                parent_artifact_uris=list(rec.parent_artifact_uris),
                depth=depth,
            )
        )
        for p in rec.parent_artifact_uris:
            if p not in seen:
                queue.append((p, depth + 1))
    # Stable: deeper first then URI (roots of lineage toward leaves).
    out.sort(key=lambda e: (-e.depth, e.created_at, e.artifact_uri))
    # Append the subject last for display chains.
    out.append(
        TimelineEvent(
            artifact_uri=root.artifact_uri,
            artifact_type=root.artifact_type,
            created_at=root.created_at,
            parent_artifact_uris=list(root.parent_artifact_uris),
            depth=0,
        )
    )
    return out


def lineage_chain_uris(
    store: ArtifactCatalogStore,
    artifact_uri: str,
) -> list[str]:
    """Simplified URI chain: oldest ancestor → … → subject."""
    events = lineage_ancestors(store, artifact_uri)
    # ancestors sorted deep→shallow then subject; reverse for Master→…→Model
    body = [e for e in events if e.artifact_uri != artifact_uri]
    body.sort(key=lambda e: (e.depth, e.created_at, e.artifact_uri), reverse=True)
    return [e.artifact_uri for e in body] + [artifact_uri]
