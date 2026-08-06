"""In-process DAG navigation (Sprint 7) — load edges then traverse."""

from __future__ import annotations

from collections import defaultdict, deque


def _ascii_sort(ids: set[str] | list[str]) -> list[str]:
    return sorted(set(ids), key=lambda s: s.encode("ascii", "replace"))


def build_adjacency(
    edges: list[tuple[str, str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build parent→children and child→parents maps from (parent, child) pairs."""
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, list[str]] = defaultdict(list)
    seen_fwd: set[tuple[str, str]] = set()
    seen_rev: set[tuple[str, str]] = set()
    for parent, child in edges:
        if (parent, child) not in seen_fwd:
            children[parent].append(child)
            seen_fwd.add((parent, child))
        if (child, parent) not in seen_rev:
            parents[child].append(parent)
            seen_rev.add((child, parent))
    return children, parents


def parents_of(
    object_id: str, edges: list[tuple[str, str]]
) -> list[str]:
    """All parents where child_object = id (ASCII ascending)."""
    found = {p for p, c in edges if c == object_id}
    return _ascii_sort(found)


def children_of(
    object_id: str, edges: list[tuple[str, str]]
) -> list[str]:
    """All children where parent_object = id (ASCII ascending)."""
    found = {c for p, c in edges if p == object_id}
    return _ascii_sort(found)


def ancestors_of(
    object_id: str, edges: list[tuple[str, str]]
) -> list[str]:
    """Transitive closure of parents; exclude self; ASCII ascending result set."""
    _, parents_map = build_adjacency(edges)
    visited: set[str] = set()
    queue: deque[str] = deque(parents_map.get(object_id, []))
    while queue:
        node = queue.popleft()
        if node == object_id or node in visited:
            continue
        visited.add(node)
        for p in parents_map.get(node, []):
            if p not in visited and p != object_id:
                queue.append(p)
    return _ascii_sort(visited)


def descendants_of(
    object_id: str, edges: list[tuple[str, str]]
) -> list[str]:
    """Transitive closure of children; exclude self; ASCII ascending result set."""
    children_map, _ = build_adjacency(edges)
    visited: set[str] = set()
    queue: deque[str] = deque(children_map.get(object_id, []))
    while queue:
        node = queue.popleft()
        if node == object_id or node in visited:
            continue
        visited.add(node)
        for c in children_map.get(node, []):
            if c not in visited and c != object_id:
                queue.append(c)
    return _ascii_sort(visited)


def has_cycle(edges: list[tuple[str, str]]) -> bool:
    """Detect directed cycle via DFS color marking."""
    children_map, _ = build_adjacency(edges)
    nodes = set()
    for p, c in edges:
        nodes.add(p)
        nodes.add(c)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}

    def dfs(node: str) -> bool:
        color[node] = GRAY
        for nxt in children_map.get(node, []):
            if color.get(nxt, WHITE) == GRAY:
                return True
            if color.get(nxt, WHITE) == WHITE and dfs(nxt):
                return True
        color[node] = BLACK
        return False

    for n in nodes:
        if color[n] == WHITE and dfs(n):
            return True
    return False


def would_introduce_cycle(
    edges: list[tuple[str, str]], parent: str, child: str
) -> bool:
    """True if adding parent→child would create a directed cycle."""
    if parent == child:
        return True
    # cycle if child can already reach parent
    return parent in set(descendants_of(child, edges)) or parent == child


def max_dag_depth(edges: list[tuple[str, str]]) -> int:
    """Longest directed path length (edge count); 0 if empty."""
    if not edges:
        return 0
    children_map, parents_map = build_adjacency(edges)
    nodes = set()
    for p, c in edges:
        nodes.add(p)
        nodes.add(c)
    # memoized longest path from node
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def longest(node: str) -> int:
        if node in memo:
            return memo[node]
        if node in visiting:
            return 0  # cycle guard — depth undefined; treat as 0 spur
        visiting.add(node)
        best = 0
        for nxt in children_map.get(node, []):
            best = max(best, 1 + longest(nxt))
        visiting.discard(node)
        memo[node] = best
        return best

    return max((longest(n) for n in nodes), default=0)


def weakly_connected_components(edges: list[tuple[str, str]]) -> int:
    """Count weakly connected components over undirected projection."""
    if not edges:
        return 0
    adj: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for p, c in edges:
        adj[p].add(c)
        adj[c].add(p)
        nodes.add(p)
        nodes.add(c)
    seen: set[str] = set()
    components = 0
    for start in nodes:
        if start in seen:
            continue
        components += 1
        stack = [start]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj[n] - seen)
    return components
