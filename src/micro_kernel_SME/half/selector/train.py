from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from micro_kernel_SME.half.selector.features import (
    LONG_TILE_LABELS,
    LONG_TO_SHORT_TILE,
    PACK_LABELS,
    SHORT_TILE_LABELS,
    build_features,
    combo_label,
    combo_to_parts,
    improvement_column_name,
    metric_column_name,
)
from micro_kernel_SME.half.selector.tree import (
    predict_classification,
    train_classification_tree,
    tree_depth,
    tree_leaf_count,
)


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
REF_DATA_DIR = REPO_ROOT / "ref" / "data"
RULES_PATH = PACKAGE_DIR / "rules.py"
ALL_CSV_PATH = REF_DATA_DIR / "bf16_all.csv"

SOURCE_PACK_FILES = {
    "nopack": REF_DATA_DIR / "bf16_nopack.csv",
    "packa": REF_DATA_DIR / "bf16_packa.csv",
    "packb": REF_DATA_DIR / "bf16_packb.csv",
    "packab": REF_DATA_DIR / "bf16_packab.csv",
}

ALL_CSV_COLUMNS = [
    "M",
    "N",
    "K",
    "transA",
    "transB",
]
for pack in PACK_LABELS:
    for tile in LONG_TILE_LABELS:
        ALL_CSV_COLUMNS.append(metric_column_name(pack, tile))
ALL_CSV_COLUMNS.append("Mflops_KPL_BLAS")
for pack in PACK_LABELS:
    for tile in LONG_TILE_LABELS:
        ALL_CSV_COLUMNS.append(improvement_column_name(pack, tile))
ALL_CSV_COLUMNS.extend(["BestImplementation", "BestMflops"])


def _parse_case_key(command: str) -> tuple[int, int, int, str, str]:
    parts = command.split()

    def arg(flag: str) -> str:
        idx = parts.index(flag)
        return parts[idx + 1]

    return int(arg("-m")), int(arg("-n")), int(arg("-k")), arg("-transa"), arg("-transb")


def _format_improvement(score: float, baseline: float) -> str:
    if baseline <= 0.0:
        return "N/A"
    improve_value = ((score / baseline) - 1.0) * 100.0
    return f"{improve_value:+.2f}%"


def _canonical_blas_value(rows_by_pack: Dict[str, Dict[str, str]]) -> float:
    return float(rows_by_pack["nopack"]["Mflops_KPL_BLAS"])


def synthesize_all_rows() -> List[Dict[str, object]]:
    cases: Dict[tuple[int, int, int, str, str], Dict[str, Dict[str, str]]] = {}
    for pack, path in SOURCE_PACK_FILES.items():
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = _parse_case_key(row["Command"])
                cases.setdefault(key, {})[pack] = row

    for key, rows_by_pack in cases.items():
        missing = [pack for pack in PACK_LABELS if pack not in rows_by_pack]
        if missing:
            raise ValueError(f"missing pack rows for case {key}: {', '.join(missing)}")

    all_rows: List[Dict[str, object]] = []
    for m, n, k, trans_a, trans_b in sorted(cases):
        rows_by_pack = cases[(m, n, k, trans_a, trans_b)]
        baseline = _canonical_blas_value(rows_by_pack)
        row: Dict[str, object] = {
            "M": m,
            "N": n,
            "K": k,
            "transA": trans_a,
            "transB": trans_b,
        }
        best_combo = ""
        best_score = float("-inf")
        for pack in PACK_LABELS:
            source_row = rows_by_pack[pack]
            for tile_long in LONG_TILE_LABELS:
                tile_short = LONG_TO_SHORT_TILE[tile_long]
                metric_name = metric_column_name(pack, tile_long)
                score = float(source_row[metric_name])
                row[metric_name] = score
                row[improvement_column_name(pack, tile_long)] = _format_improvement(score, baseline)
                if score > best_score:
                    best_combo = combo_label(pack, tile_short)
                    best_score = score
        row["Mflops_KPL_BLAS"] = baseline
        row["BestImplementation"] = best_combo
        row["BestMflops"] = best_score
        all_rows.append(row)
    return all_rows


def write_all_csv(rows: Sequence[Dict[str, object]], output_path: Path = ALL_CSV_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALL_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[TRAIN] wrote synthesized all.csv to {output_path}")


def load_samples(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for row in rows:
        best_combo = str(row["BestImplementation"])
        best_pack, best_tile = combo_to_parts(best_combo)
        combo_scores = {}
        for pack in PACK_LABELS:
            for tile_long in LONG_TILE_LABELS:
                tile_short = LONG_TO_SHORT_TILE[tile_long]
                combo_scores[combo_label(pack, tile_short)] = float(row[metric_column_name(pack, tile_long)])
        samples.append(
            {
                "M": int(row["M"]),
                "N": int(row["N"]),
                "K": int(row["K"]),
                "transA": str(row["transA"]),
                "transB": str(row["transB"]),
                "best_pack": best_pack,
                "best_tile": best_tile,
                "best_combo": best_combo,
                "best_mflops": float(row["BestMflops"]),
                "blas_mflops": float(row["Mflops_KPL_BLAS"]),
                "combo_scores": combo_scores,
            }
        )
    return samples


def _feature_row(sample: Dict[str, object]) -> Dict[str, float]:
    return build_features(
        int(sample["M"]),
        int(sample["N"]),
        int(sample["K"]),
        str(sample["transA"]),
        str(sample["transB"]),
    )


def _split_train_eval(samples: Sequence[Dict[str, object]], seed: int, eval_ratio: float) -> tuple[List[int], List[int]]:
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    split = int(len(indices) * (1.0 - eval_ratio))
    return indices[:split], indices[split:]


def _accuracy(preds: Sequence[str], labels: Sequence[str]) -> float:
    return sum(pred == label for pred, label in zip(preds, labels)) / float(len(labels)) if labels else 0.0


def _combo_ratio(predicted_combos: Sequence[str], samples: Sequence[Dict[str, object]]) -> float:
    if not samples:
        return 0.0
    ratios = []
    for combo, sample in zip(predicted_combos, samples):
        oracle = float(sample["best_mflops"])
        ratios.append(float(sample["combo_scores"][combo]) / oracle if oracle > 0.0 else 0.0)
    return sum(ratios) / float(len(ratios))


def _print_tree_summary(name: str, tree: Dict[str, object]) -> None:
    print(f"[MODEL] {name}: depth={tree_depth(tree)}, leaves={tree_leaf_count(tree)}")


def train_model(
    *,
    depth_pack: int = 3,
    depth_tile: int = 6,
    min_leaf: int = 3,
    seed: int = 0,
    eval_ratio: float = 0.2,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    rows = synthesize_all_rows()
    samples = load_samples(rows)
    train_idx, eval_idx = _split_train_eval(samples, seed=seed, eval_ratio=eval_ratio)
    train_samples = [samples[idx] for idx in train_idx]
    eval_samples = [samples[idx] for idx in eval_idx]
    train_rows = [_feature_row(sample) for sample in train_samples]
    eval_rows = [_feature_row(sample) for sample in eval_samples]
    feature_names = list(train_rows[0].keys())

    pack_labels = [str(sample["best_pack"]) for sample in train_samples]
    pack_tree = train_classification_tree(train_rows, pack_labels, feature_names, depth_pack, min_leaf)

    tile_trees: Dict[str, Dict[str, object]] = {}
    default_tile_by_pack: Dict[str, str] = {}
    for pack in PACK_LABELS:
        pack_samples = [sample for sample in train_samples if sample["best_pack"] == pack]
        if not pack_samples:
            default_tile_by_pack[pack] = Counter(str(sample["best_tile"]) for sample in train_samples).most_common(1)[0][0]
            continue
        tile_labels = [str(sample["best_tile"]) for sample in pack_samples]
        tile_rows = [_feature_row(sample) for sample in pack_samples]
        default_tile_by_pack[pack] = Counter(tile_labels).most_common(1)[0][0]
        tile_trees[pack] = train_classification_tree(tile_rows, tile_labels, feature_names, depth_tile, min_leaf)

    eval_pack_labels = [str(sample["best_pack"]) for sample in eval_samples]
    pack_preds = [predict_classification(pack_tree, row, []) for row in eval_rows]
    pack_acc = _accuracy(pack_preds, eval_pack_labels)

    tile_acc_by_pack: Dict[str, float] = {}
    hierarchical_combos: List[str] = []
    combo_labels = [str(sample["best_combo"]) for sample in eval_samples]
    for pack in PACK_LABELS:
        pack_eval_samples = [sample for sample in eval_samples if sample["best_pack"] == pack]
        if not pack_eval_samples:
            continue
        tile_tree = tile_trees.get(pack)
        if tile_tree is None:
            continue
        tile_preds = [
            predict_classification(tile_tree, _feature_row(sample), [])
            for sample in pack_eval_samples
        ]
        tile_labels = [str(sample["best_tile"]) for sample in pack_eval_samples]
        tile_acc_by_pack[pack] = _accuracy(tile_preds, tile_labels)

    for row, predicted_pack in zip(eval_rows, pack_preds):
        tile_tree = tile_trees.get(predicted_pack)
        if tile_tree is None:
            predicted_tile = default_tile_by_pack[predicted_pack]
        else:
            predicted_tile = predict_classification(tile_tree, row, [])
        hierarchical_combos.append(combo_label(predicted_pack, predicted_tile))

    combo_acc = _accuracy(hierarchical_combos, combo_labels)
    oracle_ratio = _combo_ratio(hierarchical_combos, eval_samples)

    print("[TRAIN] bf16 selector summary")
    print(f"[TRAIN] samples={len(samples)}, train={len(train_samples)}, eval={len(eval_samples)}")
    print(f"[TRAIN] pack holdout acc={pack_acc:.3f}")
    for pack, value in sorted(tile_acc_by_pack.items()):
        print(f"[TRAIN] tile holdout acc for {pack}={value:.3f}")
    print(f"[TRAIN] hierarchical combo holdout acc={combo_acc:.3f}")
    print(f"[TRAIN] predicted/oracle kernel ratio={oracle_ratio:.3f}")
    _print_tree_summary("pack_tree", pack_tree)
    for pack, tree in sorted(tile_trees.items()):
        _print_tree_summary(f"tile_tree[{pack}]", tree)

    model = {
        "version": 1,
        "feature_schema": "M/N/K + transpose + shape tags",
        "pack_tree": pack_tree,
        "tile_trees": tile_trees,
        "default_tile_by_pack": default_tile_by_pack,
        "metrics": {
            "pack_acc": pack_acc,
            "tile_acc_by_pack": tile_acc_by_pack,
            "combo_acc": combo_acc,
            "predicted_oracle_ratio": oracle_ratio,
            "train_size": len(train_samples),
            "eval_size": len(eval_samples),
            "seed": seed,
        },
    }
    return model, rows


def write_model(model: Dict[str, object], output_path: Path = RULES_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "# Auto-generated by micro_kernel_SME.half.selector.train\n"
    content += f"MODEL = {repr(model)}\n"
    output_path.write_text(content, encoding="utf-8")
    print(f"[TRAIN] wrote model to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize bf16_all.csv and train the BF16 pack/tile selector.")
    parser.add_argument("--depth-pack", type=int, default=3)
    parser.add_argument("--depth-tile", type=int, default=6)
    parser.add_argument("--min-leaf", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--min-oracle-ratio", type=float, default=0.90)
    parser.add_argument("--all-csv-output", type=Path, default=ALL_CSV_PATH)
    parser.add_argument("--rules-output", type=Path, default=RULES_PATH)
    args = parser.parse_args()

    model, rows = train_model(
        depth_pack=args.depth_pack,
        depth_tile=args.depth_tile,
        min_leaf=args.min_leaf,
        seed=args.seed,
        eval_ratio=args.eval_ratio,
    )
    write_all_csv(rows, args.all_csv_output)
    write_model(model, args.rules_output)
    oracle_ratio = float(model["metrics"]["predicted_oracle_ratio"])
    if oracle_ratio < args.min_oracle_ratio:
        raise SystemExit(
            f"selector oracle ratio {oracle_ratio:.3f} is below required threshold {args.min_oracle_ratio:.3f}"
        )


if __name__ == "__main__":
    main()
