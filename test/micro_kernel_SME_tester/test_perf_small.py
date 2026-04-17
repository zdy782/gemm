"""Small-matrix performance benchmark for SME half kernels.

This harness focuses on the small-matrix region ``M/N/K <= 128`` and exports
one CSV that contains:

- the ``.goto`` BLAS baseline
- the three no-pack SME bundle variants: ``1VLx4VL``, ``2VLx2VL``, ``4VLx1VL``

The default sweep is exhaustive over ``M, N, K in [1, 128]`` for all four
transpose pairs, so the total runtime can be very large. Results are written
incrementally to CSV to support long-running campaigns and resume workflows.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, Iterator, List, Mapping, Sequence, TextIO

from test_perf import (
    CONFIG,
    benchmark_binary_for_variant,
    default_leading_dimensions,
    ensure_bundle,
    parse_blas_output,
    run_command,
    variant_metric_label,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FULL_PRESET = "full"
PROTOTYPE_PRESET = "prototype"
TRANSPOSE_KEYS: tuple[str, ...] = ("NN", "NT", "TN", "TT")
PROTOTYPE_DIM_VALUES: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    18,
    19,
    20,
    21,
    22,
    24,
    26,
    27,
    28,
    29,
    30,
    32,
    36,
    38,
    40,
    42,
    44,
    48,
    52,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    113,
    114,
    115,
    116,
    120,
    124,
    128,
)
NOPACK_VARIANTS: tuple[dict[str, object], ...] = (
    {
        "pack_name": "nopack",
        "pack_a": False,
        "pack_b": False,
        "tile_name": "1VLx4VL",
        "m_vl": 1,
        "n_vl": 4,
        "short_label": "nopack_1x4",
    },
    {
        "pack_name": "nopack",
        "pack_a": False,
        "pack_b": False,
        "tile_name": "2VLx2VL",
        "m_vl": 2,
        "n_vl": 2,
        "short_label": "nopack_2x2",
    },
    {
        "pack_name": "nopack",
        "pack_a": False,
        "pack_b": False,
        "tile_name": "4VLx1VL",
        "m_vl": 4,
        "n_vl": 1,
        "short_label": "nopack_4x1",
    },
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the small-matrix perf sweep."""

    parser = argparse.ArgumentParser(
        description="Benchmark all nopack SME small kernels against the .goto BLAS baseline."
    )
    parser.add_argument("--data-type", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--preset",
        choices=(FULL_PRESET, PROTOTYPE_PRESET),
        default=FULL_PRESET,
        help="Case generation preset. 'prototype' uses a curated 48x48x48 sample grid per transpose.",
    )
    parser.add_argument("--m-min", type=int, default=1)
    parser.add_argument("--m-max", type=int, default=128)
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=128)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=128)
    parser.add_argument("--m-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--k-step", type=int, default=1)
    parser.add_argument(
        "--transposes",
        nargs="+",
        choices=TRANSPOSE_KEYS,
        default=list(TRANSPOSE_KEYS),
        help="Subset of transpose pairs to benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many generated cases. Zero means no limit.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to a timestamped file in the current directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing CSV by skipping cases already present in the file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output CSV instead of failing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without building bundles or running benchmarks.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print one progress line every N completed cases.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="Flush the CSV file every N newly written rows.",
    )
    parser.add_argument(
        "--median-repeats",
        type=int,
        default=1,
        help="Run each implementation this many times per point and store the median MFlops.",
    )
    parser.add_argument("--inner-loops", type=int, default=int(CONFIG["inner_loops"]))
    parser.add_argument("--alpha", type=float, default=float(CONFIG["alpha"]))
    parser.add_argument("--beta", type=float, default=float(CONFIG["beta"]))
    parser.add_argument(
        "--bundle-output-root",
        type=Path,
        default=Path(CONFIG["bundle_output_root"]),
        help="Bundle cache root passed to the shared bundle builder.",
    )
    parser.add_argument(
        "--blas-binary",
        type=Path,
        default=None,
        help="Override the .goto baseline binary path.",
    )
    return parser.parse_args()


def _validate_positive_range(name: str, lower: int, upper: int, step: int) -> None:
    """Validate one inclusive integer sweep range."""

    if lower <= 0:
        raise ValueError(f"{name}-min must be positive, got {lower}")
    if upper < lower:
        raise ValueError(f"{name}-max must be >= {name}-min, got {upper} < {lower}")
    if step <= 0:
        raise ValueError(f"{name}-step must be positive, got {step}")


def _configure_runtime(args: argparse.Namespace) -> None:
    """Apply CLI overrides to the shared benchmark configuration."""

    _validate_positive_range("m", args.m_min, args.m_max, args.m_step)
    _validate_positive_range("n", args.n_min, args.n_max, args.n_step)
    _validate_positive_range("k", args.k_min, args.k_max, args.k_step)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.limit < 0:
        raise ValueError(f"limit must be >= 0, got {args.limit}")
    if args.progress_every <= 0:
        raise ValueError(f"progress-every must be positive, got {args.progress_every}")
    if args.flush_every <= 0:
        raise ValueError(f"flush-every must be positive, got {args.flush_every}")
    if args.median_repeats <= 0:
        raise ValueError(f"median-repeats must be positive, got {args.median_repeats}")

    CONFIG["inner_loops"] = args.inner_loops
    CONFIG["alpha"] = args.alpha
    CONFIG["beta"] = args.beta
    CONFIG["bundle_output_root"] = args.bundle_output_root.resolve()


def _default_output_csv(data_type: str, preset: str, median_repeats: int) -> Path:
    """Return the default CSV output path."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"sme_kernel_{data_type}_small_nopack_{preset}_r{median_repeats}_{timestamp}.csv"


def _dimension_values(args: argparse.Namespace) -> tuple[List[int], List[int], List[int]]:
    """Return the sampled M/N/K values for the selected preset."""

    if args.preset == PROTOTYPE_PRESET:
        values = list(PROTOTYPE_DIM_VALUES)
        return values, values.copy(), values.copy()
    return (
        list(range(args.m_min, args.m_max + 1, args.m_step)),
        list(range(args.n_min, args.n_max + 1, args.n_step)),
        list(range(args.k_min, args.k_max + 1, args.k_step)),
    )


def _blas_binary_path(args: argparse.Namespace) -> Path:
    """Return the baseline .goto binary path."""

    if args.blas_binary is not None:
        return args.blas_binary.resolve()
    return (SCRIPT_DIR / ("sbgemm.goto" if args.data_type == "bf16" else "shgemm.goto")).resolve()


def _metric_keys(func_name: str) -> List[str]:
    """Return the three nopack metric keys in stable order."""

    return [f"Mflops_{variant_metric_label(variant, func_name)}" for variant in NOPACK_VARIANTS]


def _csv_fieldnames(func_name: str) -> List[str]:
    """Return the CSV schema for one precision sweep."""

    metric_keys = _metric_keys(func_name)
    fieldnames = ["M", "N", "K", "transA", "transB", "Mflops_KPL_BLAS"]
    fieldnames.extend(metric_keys)
    fieldnames.extend(f"Improve_blas/{metric_key[len('Mflops_'):]}" for metric_key in metric_keys)
    fieldnames.extend(("BestImplementation", "BestMflops"))
    return fieldnames


def _case_key(case: Mapping[str, object]) -> tuple[int, int, int, str, str]:
    """Build the stable identity key for one case."""

    return (
        int(case["m"]),
        int(case["n"]),
        int(case["k"]),
        str(case["transa"]),
        str(case["transb"]),
    )


def _row_key(row: Mapping[str, str]) -> tuple[int, int, int, str, str]:
    """Build the stable identity key for one CSV row."""

    return (int(row["M"]), int(row["N"]), int(row["K"]), row["transA"], row["transB"])


def _iter_cases(
    args: argparse.Namespace,
    m_values: Sequence[int],
    n_values: Sequence[int],
    k_values: Sequence[int],
) -> Iterator[Dict[str, object]]:
    """Yield the small-matrix case sweep in stable order."""

    yielded = 0
    for trans_key in args.transposes:
        transa = trans_key[0]
        transb = trans_key[1]
        for m in m_values:
            for n in n_values:
                for k in k_values:
                    if args.limit > 0 and yielded >= args.limit:
                        return
                    yield {"m": m, "n": n, "k": k, "transa": transa, "transb": transb}
                    yielded += 1


def _count_cases(
    args: argparse.Namespace,
    m_values: Sequence[int],
    n_values: Sequence[int],
    k_values: Sequence[int],
) -> int:
    """Return the total number of generated cases."""

    total = len(args.transposes) * len(m_values) * len(n_values) * len(k_values)
    if args.limit > 0:
        return min(total, args.limit)
    return total


def _load_completed_case_keys(output_csv: Path) -> set[tuple[int, int, int, str, str]]:
    """Load completed case keys from an existing CSV file."""

    completed: set[tuple[int, int, int, str, str]] = set()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            completed.add(_row_key(row))
    return completed


def _open_output_csv(
    output_csv: Path,
    fieldnames: Sequence[str],
    resume: bool,
    overwrite: bool,
) -> tuple[TextIO, csv.DictWriter, set[tuple[int, int, int, str, str]]]:
    """Open the output CSV and prepare optional resume state."""

    completed: set[tuple[int, int, int, str, str]] = set()
    write_header = True

    if output_csv.exists():
        if resume:
            completed = _load_completed_case_keys(output_csv)
            write_header = False
            mode = "a"
        elif overwrite:
            mode = "w"
        else:
            raise FileExistsError(
                f"Output CSV already exists: {output_csv}. Use --resume or --overwrite."
            )
    else:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        mode = "w"

    handle = output_csv.open(mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        handle.flush()
    return handle, writer, completed


def _print_header(
    args: argparse.Namespace,
    m_values: Sequence[int],
    n_values: Sequence[int],
    k_values: Sequence[int],
    output_csv: Path,
    blas_binary: Path,
    total_cases: int,
    completed_cases: int,
) -> None:
    """Print the benchmark plan summary."""

    print("=" * 100)
    print("  SME Small-Kernel vs .goto BLAS (No-Pack Small-Matrix Sweep)")
    print("=" * 100)
    print(f"Data type: {args.data_type}")
    print(f"Preset: {args.preset}")
    print(f"Transpose pairs: {', '.join(args.transposes)}")
    if args.preset == PROTOTYPE_PRESET:
        print(
            "Prototype values per axis: "
            f"M={len(m_values)}, N={len(n_values)}, K={len(k_values)} "
            f"(dense small values, boundary neighborhoods, sparse tail anchors)"
        )
        print(f"Prototype M/N/K sample values: {list(m_values)}")
    else:
        print(
            "Ranges: "
            f"M={args.m_min}:{args.m_max}:{args.m_step}, "
            f"N={args.n_min}:{args.n_max}:{args.n_step}, "
            f"K={args.k_min}:{args.k_max}:{args.k_step}"
        )
    print(f"Total generated cases: {total_cases}")
    print(f"Median repeats per implementation: {args.median_repeats}")
    print(f"Total benchmark invocations this run: {total_cases * (1 + len(NOPACK_VARIANTS)) * args.median_repeats}")
    if completed_cases > 0:
        print(f"Cases already present in CSV: {completed_cases}")
        print(f"Cases remaining this run: {max(total_cases - completed_cases, 0)}")
    print("Variants: nopack_1x4, nopack_2x2, nopack_4x1, plus .goto baseline")
    print(
        f"Benchmark settings: alpha={CONFIG['alpha']}, beta={CONFIG['beta']}, "
        f"inner_loops={CONFIG['inner_loops']}"
    )
    print(f"Bundle cache root: {Path(CONFIG['bundle_output_root']).resolve()}")
    print(f"Baseline binary: {blas_binary}")
    print(f"Output CSV: {output_csv}")


def _run_binary_once(case: Mapping[str, object], binary: Path, include_api_order: bool) -> float:
    """Benchmark one case against one executable once."""

    m = int(case["m"])
    n = int(case["n"])
    k = int(case["k"])
    transa = str(case["transa"])
    transb = str(case["transb"])
    lda, ldb, ldc = default_leading_dimensions(case)

    cmd = [
        *CONFIG["numactl"],
        *CONFIG["taskset"],
        str(binary),
        "-m",
        str(m),
        "-n",
        str(n),
        "-k",
        str(k),
        "-lda",
        str(lda),
        "-ldb",
        str(ldb),
        "-ldc",
        str(ldc),
        "-transa",
        transa,
        "-transb",
        transb,
        "-alphaR",
        str(CONFIG["alpha"]),
        "-betaR",
        str(CONFIG["beta"]),
        "-innerLoops",
        str(CONFIG["inner_loops"]),
    ]
    if include_api_order:
        cmd.extend(["-api", "F", "-order", "C"])

    try:
        output = run_command(cmd)
        return parse_blas_output(output)
    except Exception as exc:
        print(f"\n[ERROR] benchmark failed for {' '.join(map(str, _case_key(case)))}")
        print(f"[ERROR DETAIL] binary={binary} error={exc}")
        return 0.0


def _run_binary_case(
    case: Mapping[str, object],
    binary: Path,
    include_api_order: bool,
    median_repeats: int,
) -> float:
    """Benchmark one case repeatedly and return the median score."""

    if median_repeats == 1:
        return _run_binary_once(case, binary, include_api_order)
    scores = [_run_binary_once(case, binary, include_api_order) for _ in range(median_repeats)]
    return float(median(scores))


def _prepare_variant_binaries(data_type: str) -> Dict[str, Path]:
    """Ensure the three nopack bundles exist and return their binary paths."""

    binaries: Dict[str, Path] = {}
    for variant in NOPACK_VARIANTS:
        ensure_bundle(variant)
        binaries[str(variant["short_label"])] = benchmark_binary_for_variant(variant, data_type)
    return binaries


def _round_row_values(row: Dict[str, object]) -> None:
    """Round float values in-place for stable CSV output."""

    for key, value in list(row.items()):
        if isinstance(value, float):
            row[key] = round(value, 2)


def _build_result_row(
    case: Mapping[str, object],
    func_name: str,
    blas_binary: Path,
    variant_binaries: Mapping[str, Path],
    median_repeats: int,
) -> Dict[str, object]:
    """Run one case and assemble the CSV row."""

    row: Dict[str, object] = {
        "M": int(case["m"]),
        "N": int(case["n"]),
        "K": int(case["k"]),
        "transA": str(case["transa"]),
        "transB": str(case["transb"]),
    }
    baseline = _run_binary_case(case, blas_binary, include_api_order=False, median_repeats=median_repeats)
    row["Mflops_KPL_BLAS"] = baseline

    best_label = "BLAS"
    best_value = baseline
    for variant in NOPACK_VARIANTS:
        metric_suffix = variant_metric_label(variant, func_name)
        metric_key = f"Mflops_{metric_suffix}"
        score = _run_binary_case(
            case,
            variant_binaries[str(variant["short_label"])],
            include_api_order=True,
            median_repeats=median_repeats,
        )
        row[metric_key] = score
        if baseline > 0.0:
            improve = ((score / baseline) - 1.0) * 100.0
            row[f"Improve_blas/{metric_suffix}"] = f"{improve:+.2f}%"
        else:
            row[f"Improve_blas/{metric_suffix}"] = "N/A"
        if score > best_value:
            best_label = str(variant["short_label"])
            best_value = score

    row["BestImplementation"] = best_label
    row["BestMflops"] = best_value
    _round_row_values(row)
    return row


def _print_progress(
    completed_index: int,
    total_cases: int,
    row: Mapping[str, object],
    func_name: str,
) -> None:
    """Print one compact progress line."""

    metric_values = []
    for variant in NOPACK_VARIANTS:
        metric_key = f"Mflops_{variant_metric_label(variant, func_name)}"
        metric_values.append(f"{variant['short_label']}={row[metric_key]}")

    print(
        f"[{completed_index}/{total_cases}] "
        f"M={row['M']} N={row['N']} K={row['K']} "
        f"ta={row['transA']} tb={row['transB']} "
        f"BLAS={row['Mflops_KPL_BLAS']} "
        + " ".join(metric_values)
        + f" best={row['BestImplementation']}"
    )


def main() -> None:
    """Program entrypoint."""

    args = _parse_args()
    try:
        _configure_runtime(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    func_name = "sbgemm" if args.data_type == "bf16" else "shgemm"
    m_values, n_values, k_values = _dimension_values(args)
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else _default_output_csv(args.data_type, args.preset, args.median_repeats)
    )
    blas_binary = _blas_binary_path(args)
    total_cases = _count_cases(args, m_values, n_values, k_values)
    fieldnames = _csv_fieldnames(func_name)

    completed_cases: set[tuple[int, int, int, str, str]] = set()
    if output_csv.exists() and args.resume:
        completed_cases = _load_completed_case_keys(output_csv)

    _print_header(args, m_values, n_values, k_values, output_csv, blas_binary, total_cases, len(completed_cases))
    if args.dry_run:
        return

    handle, writer, completed_cases = _open_output_csv(output_csv, fieldnames, args.resume, args.overwrite)
    if not blas_binary.exists():
        handle.close()
        raise SystemExit(f"Baseline binary not found: {blas_binary}")
    variant_binaries = _prepare_variant_binaries(args.data_type)

    written_since_flush = 0
    completed_count = len(completed_cases)
    if completed_count >= total_cases:
        print("[INFO] All generated cases are already present in the output CSV.")
        handle.close()
        return
    try:
        for case in _iter_cases(args, m_values, n_values, k_values):
            if _case_key(case) in completed_cases:
                continue
            row = _build_result_row(case, func_name, blas_binary, variant_binaries, args.median_repeats)
            writer.writerow(row)
            written_since_flush += 1
            completed_count += 1
            if written_since_flush >= args.flush_every:
                handle.flush()
                written_since_flush = 0
            if completed_count == 1 or completed_count % args.progress_every == 0 or completed_count == total_cases:
                _print_progress(completed_count, total_cases, row, func_name)
    except KeyboardInterrupt:
        print("\n[Interrupted by user] Progress has been flushed to CSV.")
    finally:
        handle.flush()
        handle.close()


if __name__ == "__main__":
    main()
