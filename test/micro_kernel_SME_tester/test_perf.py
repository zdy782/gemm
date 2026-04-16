"""Batch performance benchmark for SME half kernels.

This harness benchmarks the 12 fixed ``pack x tile`` bundle variants and
exports one aggregate CSV per precision.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


BUNDLE_BUILDER = REPO_ROOT / "src" / "micro_kernel_SME" / "half" / "build_blas_bundle.py"
BUNDLE_LAYOUT_VERSION = "direct-driver-benchmark-v5"

CONFIG = {
    "numactl": ["numactl", "-m", "7"],
    "taskset": ["taskset", "-c", "266"],
    "env": {"OMP_NUM_THREADS": "1"},
    "inner_loops": 64,
    "alpha": 2.0,
    "beta": 3.0,
    "bundle_output_root": REPO_ROOT / "build" / "perf_bundles",
}


PACK_MODES: tuple[tuple[str, bool, bool], ...] = (
    ("nopack", False, False),
    ("packa", True, False),
    ("packb", False, True),
    ("packab", True, True),
)
KERNEL_SHAPES: tuple[tuple[str, int, int], ...] = (
    ("1VLx4VL", 1, 4),
    ("2VLx2VL", 2, 2),
    ("4VLx1VL", 4, 1),
)
ALL_VARIANTS: tuple[dict[str, object], ...] = tuple(
    {
        "pack_name": pack_name,
        "pack_a": pack_a,
        "pack_b": pack_b,
        "tile_name": tile_name,
        "m_vl": m_vl,
        "n_vl": n_vl,
        "short_label": f"{pack_name}_{m_vl}x{n_vl}",
    }
    for pack_name, pack_a, pack_b in PACK_MODES
    for tile_name, m_vl, n_vl in KERNEL_SHAPES
)


def _round_numeric_fields(rows: List[Dict[str, object]]) -> None:
    """Round float values in-place for stable CSV output."""

    for row in rows:
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = round(value, 2)


def _write_csv(rows: Sequence[Dict[str, object]], csv_file: str) -> None:
    """Write CSV output without requiring pandas."""

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(csv_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_blas_output(output: str) -> float:
    """Extract MFlops from a benchmark binary stdout."""

    match = re.search(r"MFlops_Effi_Time_avg:\[\s*(\d+\.\d+)\s*MFlops", output)
    if match:
        return float(match.group(1))
    raise ValueError("Failed to parse benchmark MFlops output")


def run_command(cmd: List[str], extra_env: Dict[str, str] | None = None) -> str:
    """Run a child process and surface stdout on failure."""

    env = os.environ.copy()
    env.update(CONFIG["env"])
    if extra_env is not None:
        env.update(extra_env)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        return result.stdout
    except subprocess.CalledProcessError as exc:
        captured_output = exc.stdout.strip() if exc.stdout else ""
        if captured_output:
            raise RuntimeError(f"Command execution failed: {exc}\n{captured_output}") from exc
        raise RuntimeError(f"Command execution failed: {exc}") from exc


def variant_metric_label(variant: Mapping[str, object], func_name: str) -> str:
    """Return the CSV metric suffix for one fixed variant."""

    pack_name = str(variant["pack_name"])
    if pack_name == "nopack":
        pack_part = "autogemm_nopacking"
    else:
        pack_part = f"{func_name}_{pack_name}"
    return f"{pack_part}_{str(variant['tile_name'])}"


def bundle_variant_dir(variant: Mapping[str, object]) -> Path:
    """Return the output directory for one fixed pack/tile bundle."""

    return CONFIG["bundle_output_root"] / f"{variant['pack_name']}_{variant['m_vl']}x{variant['n_vl']}"


def benchmark_binary_for_variant(variant: Mapping[str, object], data_type: str) -> Path:
    """Return the benchmark executable path for one variant and precision."""

    binary_name = "sbgemm.goto" if data_type == "bf16" else "shgemm.goto"
    return bundle_variant_dir(variant) / "bin" / binary_name


def bundle_version_file(variant: Mapping[str, object]) -> Path:
    """Return the bundle layout version file path."""

    return bundle_variant_dir(variant) / "bundle_version.txt"


def selector_bundle_dir() -> Path:
    """Return the output directory for the BF16 selector-aware bundle."""

    return CONFIG["bundle_output_root"] / "bf16_selector_fp16_nopack_1x4"


def selector_bundle_version_file() -> Path:
    """Return the selector bundle layout version file path."""

    return selector_bundle_dir() / "bundle_version.txt"


def selector_benchmark_binary(data_type: str) -> Path:
    """Return the benchmark executable path for the selector bundle."""

    binary_name = "sbgemm.goto" if data_type == "bf16" else "shgemm.goto"
    return selector_bundle_dir() / "bin" / binary_name


def ensure_bundle(variant: Mapping[str, object]) -> None:
    """Build a fixed variant bundle when it is missing or stale."""

    sbgemm_binary = benchmark_binary_for_variant(variant, "bf16")
    shgemm_binary = benchmark_binary_for_variant(variant, "fp16")
    version_file = bundle_version_file(variant)
    version_matches = (
        version_file.exists()
        and version_file.read_text(encoding="utf-8").strip() == BUNDLE_LAYOUT_VERSION
    )
    if sbgemm_binary.exists() and shgemm_binary.exists() and version_matches:
        return

    build_cmd = [
        sys.executable,
        str(BUNDLE_BUILDER),
        "--pack",
        str(variant["pack_name"]),
        "--m-vl",
        str(variant["m_vl"]),
        "--n-vl",
        str(variant["n_vl"]),
        "--output-dir",
        str(CONFIG["bundle_output_root"]),
    ]
    print(
        f"[INFO] Building fixed bundle for pack={variant['pack_name']}, "
        f"tile={variant['m_vl']}x{variant['n_vl']}"
    )
    run_command(build_cmd)


def ensure_selector_bundle() -> None:
    """Build the selector-aware BF16 bundle when it is missing or stale."""

    sbgemm_binary = selector_benchmark_binary("bf16")
    shgemm_binary = selector_benchmark_binary("fp16")
    version_file = selector_bundle_version_file()
    version_matches = (
        version_file.exists()
        and version_file.read_text(encoding="utf-8").strip() == BUNDLE_LAYOUT_VERSION
    )
    if sbgemm_binary.exists() and shgemm_binary.exists() and version_matches:
        return

    build_cmd = [
        sys.executable,
        str(BUNDLE_BUILDER),
        "--bf16-selector",
        "--output-dir",
        str(CONFIG["bundle_output_root"]),
    ]
    print("[INFO] Building BF16 selector bundle")
    run_command(build_cmd)


def prepare_kernel_binaries(variants: Sequence[Mapping[str, object]]) -> Dict[tuple[str, str], Path]:
    """Ensure required bundles exist and return benchmark binary paths."""

    binaries: Dict[tuple[str, str], Path] = {}
    for variant in variants:
        ensure_bundle(variant)
        binaries[("bf16", str(variant["short_label"]))] = benchmark_binary_for_variant(variant, "bf16")
        binaries[("fp16", str(variant["short_label"]))] = benchmark_binary_for_variant(variant, "fp16")
    return binaries


def generate_test_case_groups_by_trans() -> Dict[str, List[Dict[str, object]]]:
    """Build the fixed perf suite grouped by transpose pair."""

    trans_pairs = [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")]
    cases_by_trans = {f"{transa}{transb}": [] for transa, transb in trans_pairs}
    shape_ranges = (
        (range(16, 129, 16), range(16, 129, 16), range(16, 129, 16)),
        (range(256, 1025, 128), range(256, 1025, 128), range(256, 1025, 128)),
    )

    for m_values, n_values, k_values in shape_ranges:
        for m in m_values:
            for n in n_values:
                for k in k_values:
                    for transa, transb in trans_pairs:
                        cases_by_trans[f"{transa}{transb}"].append(
                            {"m": m, "n": n, "k": k, "transa": transa, "transb": transb}
                        )

    return cases_by_trans


def generate_test_cases() -> List[Dict[str, object]]:
    """Flatten grouped transpose cases into one ordered perf suite."""

    cases_by_trans = generate_test_case_groups_by_trans()
    ordered_cases: List[Dict[str, object]] = []
    for trans_key in ("NN", "NT", "TN", "TT"):
        ordered_cases.extend(cases_by_trans[trans_key])
    return ordered_cases


def default_leading_dimensions(case: Mapping[str, object]) -> tuple[int, int, int]:
    """Return BLAS-style lda/ldb/ldc defaults for one case."""

    m = int(case["m"])
    n = int(case["n"])
    k = int(case["k"])
    transa = str(case["transa"])
    transb = str(case["transb"])
    lda = k if transa == "T" else m
    ldb = n if transb == "T" else k
    ldc = m
    return lda, ldb, ldc


def run_test_case(case: Mapping[str, object], kernel_binary: Path) -> float:
    """Benchmark one case against one fixed kernel bundle."""

    m = int(case["m"])
    n = int(case["n"])
    k = int(case["k"])
    transa = str(case["transa"])
    transb = str(case["transb"])
    lda, ldb, ldc = default_leading_dimensions(case)

    kernel_cmd_parts = [
        *CONFIG["numactl"],
        *CONFIG["taskset"],
        str(kernel_binary),
        "-m", str(m),
        "-n", str(n),
        "-k", str(k),
        "-lda", str(lda),
        "-ldb", str(ldb),
        "-ldc", str(ldc),
        "-transa", transa,
        "-transb", transb,
        "-api", "F",
        "-order", "C",
        "-alphaR", str(CONFIG["alpha"]),
        "-betaR", str(CONFIG["beta"]),
        "-innerLoops", str(CONFIG["inner_loops"]),
    ]

    try:
        kernel_output = run_command(kernel_cmd_parts)
        return parse_blas_output(kernel_output)
    except Exception as exc:
        print(f"\n[ERROR] kernel benchmark failed: {' '.join(kernel_cmd_parts)}")
        print(f"[ERROR DETAIL] {exc}")
        return 0.0


def run_blas_case(case: Mapping[str, object], blas_binary: str) -> float:
    """Benchmark one case against the reference BLAS executable."""

    m = int(case["m"])
    n = int(case["n"])
    k = int(case["k"])
    transa = str(case["transa"])
    transb = str(case["transb"])
    lda, ldb, ldc = default_leading_dimensions(case)

    blas_cmd = [
        *CONFIG["numactl"],
        *CONFIG["taskset"],
        blas_binary,
        "-m", str(m),
        "-n", str(n),
        "-k", str(k),
        "-lda", str(lda),
        "-ldb", str(ldb),
        "-ldc", str(ldc),
        "-transa", transa,
        "-transb", transb,
        "-alphaR", str(CONFIG["alpha"]),
        "-betaR", str(CONFIG["beta"]),
        "-innerLoops", str(CONFIG["inner_loops"]),
    ]

    try:
        blas_output = run_command(blas_cmd)
        return parse_blas_output(blas_output)
    except Exception as exc:
        print(f"\n[ERROR] BLAS benchmark failed: {' '.join(blas_cmd)}")
        print(f"[ERROR DETAIL] {exc}")
        return 0.0


def _benchmark_binary_lookup(
    binaries: Mapping[tuple[str, str], Path],
    data_type: str,
    variant: Mapping[str, object],
) -> Path:
    return binaries[(data_type, str(variant["short_label"]))]


def run_exhaustive_evaluation(
    data_type: str,
    blas_binary: str,
    func_name: str,
    all_cases: Sequence[Mapping[str, object]],
    predict_combo: bool = False,
) -> None:
    """Benchmark all 12 fixed bundle variants and export one aggregate CSV."""

    kernel_binaries = prepare_kernel_binaries(ALL_VARIANTS)
    total_cases = len(all_cases)

    print("\n" + "=" * 100)
    print(f" >>> Starting Exhaustive Evaluation for {data_type.upper()} ({blas_binary}) <<<")
    print("=" * 100)

    results: List[Dict[str, object]] = []
    for case in all_cases:
        m = int(case["m"])
        n = int(case["n"])
        k = int(case["k"])
        transa = str(case["transa"])
        transb = str(case["transb"])
        row: Dict[str, object] = {
            "M": m,
            "N": n,
            "K": k,
            "transA": transa,
            "transB": transb,
        }
        for variant in ALL_VARIANTS:
            row[f"Mflops_{variant_metric_label(variant, func_name)}"] = 0.0
        results.append(row)

    print(f"\n>>> Starting KPL_BLAS baseline test for {data_type.upper()} <<<")
    for index, case in enumerate(all_cases):
        print(
            f"[{index + 1}/{total_cases}] BLAS: "
            f"M={int(case['m'])} N={int(case['n'])} K={int(case['k'])} "
            f"ta={str(case['transa'])} tb={str(case['transb'])} ... ",
            end="",
            flush=True,
        )
        score = run_blas_case(case, blas_binary)
        results[index]["Mflops_KPL_BLAS"] = score
        print(f"{score:,.2f}")

    for variant in ALL_VARIANTS:
        variant_label = variant_metric_label(variant, func_name)
        print("\n" + "=" * 100)
        print(
            f">>> Starting variant test: {variant['short_label']} "
            f"(pack_a={variant['pack_a']}, pack_b={variant['pack_b']}) <<<"
        )
        kernel_binary = _benchmark_binary_lookup(kernel_binaries, data_type, variant)
        print(f"[INFO] Using kernel benchmark binary: {kernel_binary}")

        for index, case in enumerate(all_cases):
            print(
                f"[{index + 1}/{total_cases}] {str(variant['short_label']):12}: "
                f"M={int(case['m'])} N={int(case['n'])} K={int(case['k'])} "
                f"ta={str(case['transa'])} tb={str(case['transb'])} ... ",
                end="",
                flush=True,
            )
            score = run_test_case(case, kernel_binary)
            results[index][f"Mflops_{variant_label}"] = score
            print(f"{score:,.2f}")

    if data_type == "bf16" and predict_combo:
        ensure_selector_bundle()
        selector_binary = selector_benchmark_binary("bf16")
        print("\n" + "=" * 100)
        print(">>> Starting selector test: predicted BF16 combo <<<")
        print(f"[INFO] Using selector benchmark binary: {selector_binary}")
        for index, case in enumerate(all_cases):
            print(
                f"[{index + 1}/{total_cases}] {'selector':12}: "
                f"M={int(case['m'])} N={int(case['n'])} K={int(case['k'])} "
                f"ta={str(case['transa'])} tb={str(case['transb'])} ... ",
                end="",
                flush=True,
            )
            score = run_test_case(case, selector_binary)
            results[index]["Mflops_selector_predicted"] = score
            print(f"{score:,.2f}")

    for variant in ALL_VARIANTS:
        metric_key = f"Mflops_{variant_metric_label(variant, func_name)}"
        improve_key = f"Improve_blas/{variant_metric_label(variant, func_name)}"
        for row in results:
            baseline = float(row["Mflops_KPL_BLAS"])
            score = float(row[metric_key])
            if baseline > 0.0:
                improve_value = ((score / baseline) - 1.0) * 100.0
                row[improve_key] = f"{improve_value:+.2f}%"
            else:
                row[improve_key] = "N/A"

    if data_type == "bf16" and predict_combo:
        for row in results:
            baseline = float(row["Mflops_KPL_BLAS"])
            score = float(row["Mflops_selector_predicted"])
            if baseline > 0.0:
                improve_value = ((score / baseline) - 1.0) * 100.0
                row["Improve_blas/selector_predicted"] = f"{improve_value:+.2f}%"
            else:
                row["Improve_blas/selector_predicted"] = "N/A"

    for row in results:
        best_label = "BLAS"
        best_value = float(row["Mflops_KPL_BLAS"])
        for variant in ALL_VARIANTS:
            variant_label = str(variant["short_label"])
            metric_key = f"Mflops_{variant_metric_label(variant, func_name)}"
            metric_value = float(row[metric_key])
            if metric_value > best_value:
                best_label = variant_label
                best_value = metric_value
        if data_type == "bf16" and predict_combo:
            selector_value = float(row["Mflops_selector_predicted"])
            if selector_value > best_value:
                best_label = "selector_predicted"
                best_value = selector_value
        row["BestImplementation"] = best_label
        row["BestMflops"] = best_value

    _round_numeric_fields(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"sme_kernel_{data_type}_12variants_{timestamp}.csv"
    _write_csv(results, csv_file)
    print(f"[File] CSV saved to: {csv_file}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of cases per precision for smoke runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected evaluation plan without building bundles or running benchmarks.",
    )
    parser.add_argument(
        "--predict_combo",
        action="store_true",
        help="For BF16, also benchmark one selector-aware bundle that predicts pack/tile at runtime.",
    )
    return parser.parse_args()


def _truncate_cases(
    all_cases: Sequence[Mapping[str, object]],
    cases_by_trans: Mapping[str, Sequence[Mapping[str, object]]],
    limit: int,
) -> tuple[List[Mapping[str, object]], Dict[str, List[Mapping[str, object]]]]:
    if limit <= 0:
        return list(all_cases), {key: list(value) for key, value in cases_by_trans.items()}

    truncated_cases = list(all_cases[:limit])
    allowed = {
        (case["m"], case["n"], case["k"], case["transa"], case["transb"])
        for case in truncated_cases
    }
    truncated_groups = {
        trans_key: [
            case for case in trans_cases
            if (case["m"], case["n"], case["k"], case["transa"], case["transb"]) in allowed
        ]
        for trans_key, trans_cases in cases_by_trans.items()
    }
    return truncated_cases, truncated_groups


def _print_header(
    cases_by_trans: Mapping[str, Sequence[Mapping[str, object]]],
    total_cases: int,
    predict_combo: bool,
) -> None:
    print("=" * 100)
    print("  SME Kernel vs BLAS (Automated BF16 & FP16 Testing)")
    print("=" * 100)
    print("[INFO] test_perf runs perf-only kernels; correctness is skipped via --perf_only")
    print("Perf suite: fixed")
    print(f"Total test cases per precision type: {total_cases}")
    for trans_key, trans_cases in cases_by_trans.items():
        print(f"  - {trans_key}: {len(trans_cases)} cases")
    print(
        f"[INFO] fixed alpha={CONFIG['alpha']}, beta={CONFIG['beta']}; "
        "all 12 fixed pack/tile variants are exported into one aggregate table"
    )
    if predict_combo:
        print("[INFO] BF16 will also export a selector_predicted lane from the selector-aware runtime bundle")


def _print_dry_run_plan(predict_combo: bool) -> None:
    print("[DRY RUN] No bundles will be built and no benchmark binaries will run.")
    print("[DRY RUN] Bundles that would be prepared:")
    for variant in ALL_VARIANTS:
        print(f"  - {variant['short_label']}")
    if predict_combo:
        print("  - bf16_selector_fp16_nopack_1x4")


def main() -> None:
    """Program entrypoint."""

    args = parse_args()
    cases_by_trans = generate_test_case_groups_by_trans()
    all_cases = generate_test_cases()
    all_cases, cases_by_trans = _truncate_cases(all_cases, cases_by_trans, args.limit)
    _print_header(cases_by_trans, len(all_cases), args.predict_combo)

    if args.dry_run:
        _print_dry_run_plan(args.predict_combo)
        return

    test_configs = [
        {"data_type": "bf16", "blas_binary": str(SCRIPT_DIR / "sbgemm.goto"), "func_name": "sbgemm"},
        {"data_type": "fp16", "blas_binary": str(SCRIPT_DIR / "shgemm.goto"), "func_name": "shgemm"},
    ]

    try:
        for config in test_configs:
            run_exhaustive_evaluation(
                data_type=config["data_type"],
                blas_binary=config["blas_binary"],
                func_name=config["func_name"],
                all_cases=all_cases,
                predict_combo=args.predict_combo,
            )
    except KeyboardInterrupt:
        print("\n[Interrupted by user] Halting execution and preserving earlier results.")


if __name__ == "__main__":
    main()
