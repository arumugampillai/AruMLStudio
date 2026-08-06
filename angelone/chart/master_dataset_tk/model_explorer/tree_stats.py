"""Parse XGBoost tree dumps into node details, feature usage, model summary, and per-tree stats."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModelSummary:
    n_trees: int
    max_depth: int
    objective: str
    learning_rate: float | None
    features_used: list[str]
    total_nodes: int
    n_leaves: int
    feature_names: list[str] = field(default_factory=list)
    raw_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureUsageRow:
    feature: str
    occurrences: int
    avg_gain: float
    avg_cover: float
    first_tree: int
    max_depth: int


@dataclass(frozen=True)
class NodeDetails:
    tree_id: int
    node_id: int
    id_label: str
    feature: str | None
    threshold: float | None
    gain: float | None
    cover: float | None
    depth: int
    leaf_value: float | None
    is_leaf: bool
    missing_direction: str | None
    parent_id: int | None
    left_id: int | None
    right_id: int | None
    yes_id: str | None = None
    no_id: str | None = None
    missing_id: str | None = None


@dataclass(frozen=True)
class TreeStatistics:
    """Per-tree structural stats for the Model Explorer Tree Statistics panel."""

    tree_index: int
    n_nodes: int
    n_leaves: int
    max_depth: int
    root_feature: str | None
    total_gain: float
    features_used: list[str] = field(default_factory=list)


def tree_count(booster: Any) -> int:
    """Return number of trees in the booster."""
    try:
        n = int(booster.num_boosted_rounds())
        if n > 0:
            return n
    except Exception:
        pass
    dump = booster.get_dump(dump_format="text")
    return len(dump)


def _trees_df(booster: Any) -> pd.DataFrame:
    df = booster.trees_to_dataframe()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(
            columns=[
                "Tree", "Node", "ID", "Feature", "Split", "Yes", "No",
                "Missing", "Gain", "Cover", "Category",
            ]
        )
    return df


def _parse_config(booster: Any) -> dict[str, Any]:
    try:
        raw = booster.save_config()
        return json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except Exception:
        return {}


def _config_get(cfg: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _feature_names(booster: Any) -> list[str]:
    try:
        names = getattr(booster, "feature_names", None)
        if names:
            return [str(x) for x in names]
    except Exception:
        pass
    return []


def _node_depths(tree_df: pd.DataFrame) -> dict[int, int]:
    """Compute depth of each node within one tree via Yes/No edges."""
    if tree_df.empty:
        return {}
    id_to_node = {str(r["ID"]): int(r["Node"]) for _, r in tree_df.iterrows()}
    children: dict[int, list[int]] = {int(r["Node"]): [] for _, r in tree_df.iterrows()}
    for _, r in tree_df.iterrows():
        parent = int(r["Node"])
        for col in ("Yes", "No"):
            child_id = r.get(col)
            if pd.isna(child_id):
                continue
            child_node = id_to_node.get(str(child_id))
            if child_node is not None:
                children[parent].append(child_node)
    depths: dict[int, int] = {}

    def walk(nid: int, depth: int) -> None:
        depths[nid] = depth
        for c in children.get(nid, []):
            walk(c, depth + 1)

    if 0 in children or 0 in {int(r["Node"]) for _, r in tree_df.iterrows()}:
        walk(0, 0)
    else:
        roots = sorted(int(r["Node"]) for _, r in tree_df.iterrows())
        if roots:
            walk(roots[0], 0)
    return depths


def _parent_map(tree_df: pd.DataFrame) -> dict[int, int]:
    if tree_df.empty:
        return {}
    id_to_node = {str(r["ID"]): int(r["Node"]) for _, r in tree_df.iterrows()}
    parents: dict[int, int] = {}
    for _, r in tree_df.iterrows():
        parent = int(r["Node"])
        for col in ("Yes", "No"):
            child_id = r.get(col)
            if pd.isna(child_id):
                continue
            child_node = id_to_node.get(str(child_id))
            if child_node is not None:
                parents[child_node] = parent
    return parents


def _missing_direction(row: pd.Series) -> str | None:
    if str(row.get("Feature", "")) == "Leaf" or pd.isna(row.get("Missing")):
        return None
    missing = str(row["Missing"])
    yes = str(row["Yes"]) if not pd.isna(row.get("Yes")) else None
    no = str(row["No"]) if not pd.isna(row.get("No")) else None
    if yes is not None and missing == yes:
        return "yes/left"
    if no is not None and missing == no:
        return "no/right"
    return missing


def _row_to_details(
    row: pd.Series,
    *,
    depths: dict[int, int],
    parents: dict[int, int],
    id_to_node: dict[str, int],
) -> NodeDetails:
    node_id = int(row["Node"])
    feature_raw = str(row.get("Feature", ""))
    is_leaf = feature_raw == "Leaf"
    gain = None if pd.isna(row.get("Gain")) else float(row["Gain"])
    cover = None if pd.isna(row.get("Cover")) else float(row["Cover"])
    split = None if pd.isna(row.get("Split")) else float(row["Split"])
    yes_s = None if pd.isna(row.get("Yes")) else str(row["Yes"])
    no_s = None if pd.isna(row.get("No")) else str(row["No"])
    miss_s = None if pd.isna(row.get("Missing")) else str(row["Missing"])
    left_id = id_to_node.get(yes_s) if yes_s else None
    right_id = id_to_node.get(no_s) if no_s else None
    return NodeDetails(
        tree_id=int(row["Tree"]),
        node_id=node_id,
        id_label=str(row["ID"]),
        feature=None if is_leaf else feature_raw,
        threshold=None if is_leaf else split,
        gain=None if is_leaf else gain,
        cover=cover,
        depth=int(depths.get(node_id, 0)),
        leaf_value=gain if is_leaf else None,
        is_leaf=is_leaf,
        missing_direction=_missing_direction(row),
        parent_id=parents.get(node_id),
        left_id=left_id,
        right_id=right_id,
        yes_id=yes_s,
        no_id=no_s,
        missing_id=miss_s,
    )


def list_nodes_for_tree(booster: Any, tree_id: int) -> list[NodeDetails]:
    """Return all nodes for ``tree_id`` sorted by node id."""
    df = _trees_df(booster)
    tree_df = df[df["Tree"] == int(tree_id)].copy()
    if tree_df.empty:
        return []
    depths = _node_depths(tree_df)
    parents = _parent_map(tree_df)
    id_to_node = {str(r["ID"]): int(r["Node"]) for _, r in tree_df.iterrows()}
    nodes = [
        _row_to_details(row, depths=depths, parents=parents, id_to_node=id_to_node)
        for _, row in tree_df.iterrows()
    ]
    nodes.sort(key=lambda n: n.node_id)
    return nodes


def node_details(booster: Any, tree_id: int, node_id: int) -> NodeDetails | None:
    for node in list_nodes_for_tree(booster, tree_id):
        if node.node_id == int(node_id):
            return node
    return None


def build_tree_statistics(booster: Any, tree_id: int) -> TreeStatistics:
    """Compute read-only structural statistics for a single tree.

    Fields: tree index, node/leaf counts, max depth, root split feature,
    total split gain, and unique features used in the tree.
    """
    tid = int(tree_id)
    nodes = list_nodes_for_tree(booster, tid)
    if not nodes:
        return TreeStatistics(
            tree_index=tid,
            n_nodes=0,
            n_leaves=0,
            max_depth=0,
            root_feature=None,
            total_gain=0.0,
            features_used=[],
        )

    n_leaves = sum(1 for n in nodes if n.is_leaf)
    max_depth = max(n.depth for n in nodes)
    root = next((n for n in nodes if n.node_id == 0), nodes[0])
    root_feature = None if root.is_leaf else root.feature
    total_gain = float(
        sum(n.gain for n in nodes if not n.is_leaf and n.gain is not None)
    )
    seen: set[str] = set()
    features_used: list[str] = []
    for n in nodes:
        if n.is_leaf or not n.feature:
            continue
        if n.feature not in seen:
            seen.add(n.feature)
            features_used.append(n.feature)
    features_used.sort()

    return TreeStatistics(
        tree_index=tid,
        n_nodes=len(nodes),
        n_leaves=n_leaves,
        max_depth=int(max_depth),
        root_feature=root_feature,
        total_gain=total_gain,
        features_used=features_used,
    )


def build_feature_usage(booster: Any) -> list[FeatureUsageRow]:
    """Aggregate split-feature usage across all trees."""
    df = _trees_df(booster)
    if df.empty:
        return []
    splits = df[df["Feature"].astype(str) != "Leaf"].copy()
    if splits.empty:
        return []

    # Depth per (tree, node)
    depth_lookup: dict[tuple[int, int], int] = {}
    for tid, tree_df in df.groupby("Tree"):
        depths = _node_depths(tree_df)
        for nid, d in depths.items():
            depth_lookup[(int(tid), int(nid))] = int(d)

    rows: list[FeatureUsageRow] = []
    for feat, g in splits.groupby(splits["Feature"].astype(str)):
        gains = g["Gain"].astype(float)
        covers = g["Cover"].astype(float)
        trees = g["Tree"].astype(int)
        max_d = 0
        for _, r in g.iterrows():
            max_d = max(max_d, depth_lookup.get((int(r["Tree"]), int(r["Node"])), 0))
        rows.append(
            FeatureUsageRow(
                feature=str(feat),
                occurrences=int(len(g)),
                avg_gain=float(gains.mean()) if len(gains) else 0.0,
                avg_cover=float(covers.mean()) if len(covers) else 0.0,
                first_tree=int(trees.min()),
                max_depth=int(max_d),
            )
        )
    rows.sort(key=lambda r: (-r.occurrences, -r.avg_gain, r.feature))
    return rows


def build_model_summary(booster: Any) -> ModelSummary:
    """Summarize booster structure and learner config."""
    df = _trees_df(booster)
    n_trees = tree_count(booster)
    if not df.empty:
        n_trees = max(n_trees, int(df["Tree"].max()) + 1)

    max_depth = 0
    total_nodes = int(len(df))
    n_leaves = int((df["Feature"].astype(str) == "Leaf").sum()) if not df.empty else 0
    for _, tree_df in df.groupby("Tree") if not df.empty else []:
        depths = _node_depths(tree_df)
        if depths:
            max_depth = max(max_depth, max(depths.values()))

    usage = build_feature_usage(booster)
    features_used = [u.feature for u in usage]

    cfg = _parse_config(booster)
    objective = (
        _config_get(cfg, "learner", "objective", "name", default=None)
        or _config_get(cfg, "learner", "learner_model_param", "objective", default=None)
        or ""
    )
    lr_raw = _config_get(cfg, "learner", "gradient_booster", "tree_train_param", "eta", default=None)
    if lr_raw is None:
        lr_raw = _config_get(
            cfg, "learner", "gradient_booster", "tree_train_param", "learning_rate", default=None
        )
    if lr_raw is None:
        lr_raw = _config_get(cfg, "learner", "gradient_booster", "gbtree_train_param", "eta", default=None)
    learning_rate: float | None
    try:
        learning_rate = float(lr_raw) if lr_raw is not None and str(lr_raw) != "" else None
    except (TypeError, ValueError):
        learning_rate = None

    # Prefer configured max_depth when dump is empty
    if max_depth == 0:
        md = _config_get(cfg, "learner", "gradient_booster", "tree_train_param", "max_depth", default=None)
        try:
            max_depth = int(md) if md is not None else 0
        except (TypeError, ValueError):
            max_depth = 0

    return ModelSummary(
        n_trees=int(n_trees),
        max_depth=int(max_depth),
        objective=str(objective or ""),
        learning_rate=learning_rate,
        features_used=features_used,
        total_nodes=total_nodes,
        n_leaves=n_leaves,
        feature_names=_feature_names(booster),
        raw_config=cfg,
    )


_NODE_LINE_RE = re.compile(
    r"^(?P<indent>\t*)(?P<node>\d+):"
    r"(?:\[(?P<feat>[^\<\]]+)\<(?P<thr>[^\]]+)\]\s+"
    r"yes=(?P<yes>\d+),no=(?P<no>\d+)(?:,missing=(?P<miss>\d+))?"
    r"|leaf=(?P<leaf>[-+eE0-9\.]+))"
)


def parse_text_dump_tree(dump_text: str, tree_id: int = 0) -> list[NodeDetails]:
    """Fallback parser for a single-tree text dump (tests / no dataframe)."""
    nodes: list[NodeDetails] = []
    parents_by_indent: dict[int, int] = {}
    for line in dump_text.splitlines():
        m = _NODE_LINE_RE.match(line)
        if not m:
            continue
        indent = len(m.group("indent") or "")
        node_id = int(m.group("node"))
        parent = parents_by_indent.get(indent - 1) if indent > 0 else None
        parents_by_indent[indent] = node_id
        if m.group("leaf") is not None:
            nodes.append(
                NodeDetails(
                    tree_id=tree_id,
                    node_id=node_id,
                    id_label=f"{tree_id}-{node_id}",
                    feature=None,
                    threshold=None,
                    gain=None,
                    cover=None,
                    depth=indent,
                    leaf_value=float(m.group("leaf")),
                    is_leaf=True,
                    missing_direction=None,
                    parent_id=parent,
                    left_id=None,
                    right_id=None,
                )
            )
        else:
            yes_id = int(m.group("yes"))
            no_id = int(m.group("no"))
            miss = m.group("miss")
            miss_dir = None
            if miss is not None:
                miss_i = int(miss)
                if miss_i == yes_id:
                    miss_dir = "yes/left"
                elif miss_i == no_id:
                    miss_dir = "no/right"
            nodes.append(
                NodeDetails(
                    tree_id=tree_id,
                    node_id=node_id,
                    id_label=f"{tree_id}-{node_id}",
                    feature=str(m.group("feat")),
                    threshold=float(m.group("thr")),
                    gain=None,
                    cover=None,
                    depth=indent,
                    leaf_value=None,
                    is_leaf=False,
                    missing_direction=miss_dir,
                    parent_id=parent,
                    left_id=yes_id,
                    right_id=no_id,
                    yes_id=f"{tree_id}-{yes_id}",
                    no_id=f"{tree_id}-{no_id}",
                    missing_id=f"{tree_id}-{miss}" if miss is not None else None,
                )
            )
    nodes.sort(key=lambda n: n.node_id)
    return nodes
