from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence


def _majority_label(labels: Sequence[str]) -> str:
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _gini(labels: Sequence[str]) -> float:
    counts = Counter(labels)
    total = float(len(labels))
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def _best_split_classification(
    rows: Sequence[Dict[str, float]],
    labels: Sequence[str],
    feature_names: Sequence[str],
    min_leaf: int,
):
    base = _gini(labels)
    best = None
    best_gain = 0.0
    for feature in feature_names:
        values = sorted(set(row[feature] for row in rows))
        if len(values) <= 1:
            continue
        thresholds = [(left + right) / 2.0 for left, right in zip(values, values[1:])]
        for threshold in thresholds:
            left_idx = [idx for idx, row in enumerate(rows) if row[feature] <= threshold]
            right_idx = [idx for idx, row in enumerate(rows) if row[feature] > threshold]
            if len(left_idx) < min_leaf or len(right_idx) < min_leaf:
                continue
            left_labels = [labels[idx] for idx in left_idx]
            right_labels = [labels[idx] for idx in right_idx]
            gain = base
            gain -= (len(left_labels) / len(labels)) * _gini(left_labels)
            gain -= (len(right_labels) / len(labels)) * _gini(right_labels)
            if gain > best_gain + 1e-12:
                best_gain = gain
                best = (feature, threshold, left_idx, right_idx)
    return best


def train_classification_tree(
    rows: Sequence[Dict[str, float]],
    labels: Sequence[str],
    feature_names: Sequence[str],
    max_depth: int,
    min_leaf: int,
):
    if not rows:
        raise ValueError("cannot train a tree on zero rows")
    if len(set(labels)) == 1 or max_depth <= 0 or len(rows) <= min_leaf:
        return {"leaf": True, "label": _majority_label(labels), "counts": dict(Counter(labels))}

    split = _best_split_classification(rows, labels, feature_names, min_leaf)
    if split is None:
        return {"leaf": True, "label": _majority_label(labels), "counts": dict(Counter(labels))}

    feature, threshold, left_idx, right_idx = split
    left_rows = [rows[idx] for idx in left_idx]
    right_rows = [rows[idx] for idx in right_idx]
    left_labels = [labels[idx] for idx in left_idx]
    right_labels = [labels[idx] for idx in right_idx]
    return {
        "leaf": False,
        "feature": feature,
        "threshold": float(threshold),
        "label": _majority_label(labels),
        "counts": dict(Counter(labels)),
        "left": train_classification_tree(left_rows, left_labels, feature_names, max_depth - 1, min_leaf),
        "right": train_classification_tree(right_rows, right_labels, feature_names, max_depth - 1, min_leaf),
    }


def predict_classification(tree: Dict, row: Dict[str, float], path: List[str] | None = None) -> str:
    if path is None:
        path = []
    if tree.get("leaf"):
        path.append(f"leaf:{tree['label']}")
        return tree["label"]
    feature = tree["feature"]
    threshold = tree["threshold"]
    branch = "L" if row[feature] <= threshold else "R"
    path.append(f"{feature}{branch}{threshold:g}")
    if branch == "L":
        return predict_classification(tree["left"], row, path)
    return predict_classification(tree["right"], row, path)


def tree_depth(tree: Dict) -> int:
    if tree.get("leaf"):
        return 1
    return 1 + max(tree_depth(tree["left"]), tree_depth(tree["right"]))


def tree_leaf_count(tree: Dict) -> int:
    if tree.get("leaf"):
        return 1
    return tree_leaf_count(tree["left"]) + tree_leaf_count(tree["right"])
