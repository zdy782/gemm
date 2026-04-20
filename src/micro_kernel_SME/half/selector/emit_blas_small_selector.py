"""Train and emit the BLAS sbgemm small-matrix selector header."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from micro_kernel_SME.half.selector.tree import train_classification_tree, tree_depth, tree_leaf_count


PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_DIR.parents[3]
DEFAULT_CSV_PATH = WORKSPACE_ROOT / "gemm" / "ref" / "data" / "bf16_all.csv"
DEFAULT_HEADER_PATH = WORKSPACE_ROOT / "blas" / "source" / "include" / "sbgemm_small_selector.h"
TRANSPOSE_PAIRS = ("NN", "NT", "TN", "TT")
FEATURE_NAMES = ("M", "N", "K")
CHOICE_ENUMS = {
    "BLAS": "SBGEMM_SMALL_KERNEL_CHOICE_BLAS",
    "1X4": "SBGEMM_SMALL_KERNEL_CHOICE_TILE_1X4",
    "2X2": "SBGEMM_SMALL_KERNEL_CHOICE_TILE_2X2",
    "4X1": "SBGEMM_SMALL_KERNEL_CHOICE_TILE_4X1",
}
PAIR_CANDIDATES = {
    "NN": (
        ("1X4", "Mflops_autogemm_nopacking_1VLx4VL"),
        ("2X2", "Mflops_autogemm_nopacking_2VLx2VL"),
        ("4X1", "Mflops_autogemm_nopacking_4VLx1VL"),
    ),
    "NT": (
        ("1X4", "Mflops_autogemm_nopacking_1VLx4VL"),
        ("2X2", "Mflops_autogemm_nopacking_2VLx2VL"),
        ("4X1", "Mflops_autogemm_nopacking_4VLx1VL"),
    ),
    "TN": (
        ("BLAS", "Mflops_KPL_BLAS"),
        ("1X4", "Mflops_autogemm_nopacking_1VLx4VL"),
        ("2X2", "Mflops_autogemm_nopacking_2VLx2VL"),
        ("4X1", "Mflops_autogemm_nopacking_4VLx1VL"),
    ),
    "TT": (
        ("1X4", "Mflops_autogemm_nopacking_1VLx4VL"),
        ("2X2", "Mflops_autogemm_nopacking_2VLx2VL"),
        ("4X1", "Mflops_autogemm_nopacking_4VLx1VL"),
    ),
}
FEATURE_EXPRESSIONS = {
    "M": "m",
    "N": "n",
    "K": "k",
}


def _load_rows(csv_path: Path) -> dict[str, list[dict[str, object]]]:
    rows_by_pair: dict[str, list[dict[str, object]]] = {pair: [] for pair in TRANSPOSE_PAIRS}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trans_pair = f"{row['transA'].strip().upper()}{row['transB'].strip().upper()}"
            if trans_pair not in rows_by_pair:
                continue
            m = int(row["M"])
            n = int(row["N"])
            k = int(row["K"])
            if m > 128 or n > 128 or k > 128:
                continue
            rows_by_pair[trans_pair].append(
                {
                    "M": float(m),
                    "N": float(n),
                    "K": float(k),
                    "label": _best_label(row, trans_pair),
                }
            )
    return rows_by_pair


def _best_label(row: dict[str, str], trans_pair: str) -> str:
    candidates = PAIR_CANDIDATES[trans_pair]
    return max(candidates, key=lambda item: float(row[item[1]]))[0]


def _train_model(
    rows_by_pair: dict[str, list[dict[str, object]]],
    max_depth: int,
    min_leaf: int,
) -> dict[str, dict[str, object]]:
    model: dict[str, dict[str, object]] = {}
    for trans_pair in TRANSPOSE_PAIRS:
        samples = rows_by_pair[trans_pair]
        if not samples:
            raise ValueError(f"no samples found for transpose pair {trans_pair}")
        feature_rows = [{name: sample[name] for name in FEATURE_NAMES} for sample in samples]
        labels = [str(sample["label"]) for sample in samples]
        model[trans_pair] = train_classification_tree(feature_rows, labels, FEATURE_NAMES, max_depth, min_leaf)
    return model


def _emit_tree(tree: Dict[str, object], indent: int) -> str:
    prefix = " " * indent
    if tree.get("leaf"):
        return f"{prefix}return {CHOICE_ENUMS[str(tree['label'])]};\n"
    feature = str(tree["feature"])
    threshold = float(tree["threshold"])
    expr = FEATURE_EXPRESSIONS[feature]
    code = f"{prefix}if ({expr} <= {threshold:.17g}) {{\n"
    code += _emit_tree(tree["left"], indent + 4)
    code += f"{prefix}}}\n"
    code += _emit_tree(tree["right"], indent)
    return code


def _collect_leaves(tree: Dict[str, object]) -> set[str]:
    if tree.get("leaf"):
        return {str(tree["label"])}
    return _collect_leaves(tree["left"]) | _collect_leaves(tree["right"])


def _validate_model(model: dict[str, dict[str, object]]) -> None:
    for trans_pair, tree in model.items():
        leaves = _collect_leaves(tree)
        allowed = {label for label, _ in PAIR_CANDIDATES[trans_pair]}
        if not leaves <= allowed:
            raise ValueError(f"unexpected leaf labels for {trans_pair}: {sorted(leaves - allowed)}")


def _emit_header(model: dict[str, dict[str, object]]) -> str:
    function_blocks = []
    for trans_pair in TRANSPOSE_PAIRS:
        func_name = f"select_sbgemm_small_kernel_{trans_pair.lower()}"
        function_blocks.append(
            (
                f"static inline SbgemmSmallKernelChoice {func_name}(\n"
                f"    const BLASLONG m, const BLASLONG n, const BLASLONG k)\n"
                "{\n"
                f"{_emit_tree(model[trans_pair], 4)}}}\n"
            )
        )

    return (
        "/* Auto-generated by gemm/src/micro_kernel_SME/half/selector/emit_blas_small_selector.py */\n"
        "#ifndef SBGEMM_SMALL_SELECTOR_H\n"
        "#define SBGEMM_SMALL_SELECTOR_H\n\n"
        "#include \"blas_types_def.h\"\n\n"
        "typedef enum {\n"
        "    SBGEMM_SMALL_KERNEL_CHOICE_BLAS = 0,\n"
        "    SBGEMM_SMALL_KERNEL_CHOICE_TILE_1X4,\n"
        "    SBGEMM_SMALL_KERNEL_CHOICE_TILE_2X2,\n"
        "    SBGEMM_SMALL_KERNEL_CHOICE_TILE_4X1,\n"
        "} SbgemmSmallKernelChoice;\n\n"
        + "\n".join(function_blocks)
        + "\n#endif\n"
    )


def emit_selector_header(
    csv_path: Path,
    header_path: Path,
    max_depth: int,
    min_leaf: int,
) -> dict[str, dict[str, object]]:
    rows_by_pair = _load_rows(csv_path)
    model = _train_model(rows_by_pair, max_depth=max_depth, min_leaf=min_leaf)
    _validate_model(model)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(_emit_header(model), encoding="utf-8")
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit the BLAS sbgemm small-matrix selector header.")
    parser.add_argument("--csv-input", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--header-output", type=Path, default=DEFAULT_HEADER_PATH)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-leaf", type=int, default=1)
    return parser.parse_args()


def _print_summary(model: dict[str, dict[str, object]]) -> None:
    for trans_pair in TRANSPOSE_PAIRS:
        tree = model[trans_pair]
        leaves = ",".join(sorted(_collect_leaves(tree)))
        print(
            f"{trans_pair}: depth={tree_depth(tree)} leaves={tree_leaf_count(tree)} labels={leaves}"
        )


def main() -> None:
    args = _parse_args()
    model = emit_selector_header(
        args.csv_input.resolve(),
        args.header_output.resolve(),
        max_depth=args.max_depth,
        min_leaf=args.min_leaf,
    )
    _print_summary(model)
    print(args.header_output.resolve())


if __name__ == "__main__":
    main()
