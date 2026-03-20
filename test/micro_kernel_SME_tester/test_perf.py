import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_TYPE_CONFIG = {
    "bf16": {
        "blas_binary": "./sbgemm.goto",
        "function_name": "sbgemm",
    },
    "fp16": {
        "blas_binary": "./shgemm.goto",
        "function_name": "shgemm",
    },
}

CONFIG = {
    "numactl": "numactl -m 15",
    "taskset": "taskset -c 575",
    "omp_threads": "OMP_NUM_THREADS=1",
    "inner_loops": 64,
    "kernel_script": "./test_single.py",
    "runs_per_test": 1,
    "kernel_shapes": [
        ("1VLx4VL", 1, 4),
        ("2VLx2VL", 2, 2),
        ("4VLx1VL", 4, 1),
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", choices=["bf16", "fp16"], required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--numactl", type=str, default=CONFIG["numactl"])
    parser.add_argument("--taskset", type=str, default=CONFIG["taskset"])
    parser.add_argument("--inner_loops", type=int, default=CONFIG["inner_loops"])
    parser.add_argument("--runs_per_test", type=int, default=CONFIG["runs_per_test"])
    return parser


def parse_blas_output(output: str) -> float:
    match = re.search(r"MFlops_Effi_Time_avg:\[\s*(\d+\.\d+)\s*MFlops", output)
    if match:
        return float(match.group(1))
    raise ValueError("Failed to parse BLAS MFlops")


def parse_kernel_output(output: str) -> float:
    match = re.search(r"GFLOPS:\s*(\d+\.\d+)", output)
    if match:
        return float(match.group(1)) * 1000.0
    raise ValueError("Failed to parse kernel GFLOPS")


def run_command(cmd: str) -> str:
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=SCRIPT_DIR,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command execution failed: {cmd}\n{result.stdout}")
    return result.stdout


def build_prefixed_command(*parts: str) -> str:
    return " ".join([part for part in parts if part])


def average_score(runner, runs_per_test: int) -> float:
    scores = []
    for _ in range(runs_per_test):
        score = runner()
        if score > 0:
            scores.append(score)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def generate_test_cases_by_trans() -> List[Dict]:
    trans_pairs = [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")]
    cases_by_trans = {f"{ta}{tb}": [] for ta, tb in trans_pairs}

    sizes_s = [16, 24, 32, 48]
    k_list = [16, 24, 32, 48]

    def label(x: int) -> str:
        return "S" if x in sizes_s else "L"

    selected_sizes = [
        (16, 64), (24, 96), (32, 100), (48, 128), (64, 192),
        (96, 256), (128, 300), (256, 512), (512, 768),
        (64, 16), (96, 24), (100, 32), (128, 48), (192, 64),
        (256, 96), (300, 128), (512, 256), (768, 512),
    ]

    for k in k_list:
        for m, n in selected_sizes:
            shape_type = f"{label(m)}{label(n)}S"
            for ta, tb in trans_pairs:
                cases_by_trans[f"{ta}{tb}"].append(
                    {"m": m, "n": n, "k": k, "transa": ta, "transb": tb, "type": shape_type}
                )

    for m in [28, 64]:
        for n in [500, 1000]:
            for k in [16, 32]:
                for ta, tb in trans_pairs:
                    cases_by_trans[f"{ta}{tb}"].append(
                        {"m": m, "n": n, "k": k, "transa": ta, "transb": tb, "type": "verybigN"}
                    )

    for m in [500, 1000]:
        for n in [28, 64]:
            for k in [16, 32]:
                for ta, tb in trans_pairs:
                    cases_by_trans[f"{ta}{tb}"].append(
                        {"m": m, "n": n, "k": k, "transa": ta, "transb": tb, "type": "verybigM"}
                    )

    for m in [64, 128]:
        for n in [64, 128]:
            for k in [256, 300]:
                for ta, tb in trans_pairs:
                    cases_by_trans[f"{ta}{tb}"].append(
                        {"m": m, "n": n, "k": k, "transa": ta, "transb": tb, "type": "verybigK"}
                    )

    final_cases = []
    for cases in cases_by_trans.values():
        unique_dict = {}
        for case in cases:
            key = (case["m"], case["n"], case["k"], case["transa"], case["transb"])
            unique_dict[key] = case
        final_cases.extend(unique_dict.values())
    return final_cases


def run_test_case(case: Dict, tile_name: str, m_vl: int, n_vl: int, data_type: str, prefix: Dict) -> float:
    m, n, k = case["m"], case["n"], case["k"]
    ta, tb = case["transa"], case["transb"]
    lda = k if ta == "T" else m
    ldb = n if tb == "T" else k

    kernel_cmd = build_prefixed_command(
        prefix["numactl"],
        prefix["taskset"],
        sys.executable,
        CONFIG["kernel_script"],
        f"--M {m}",
        f"--N {n}",
        f"--K {k}",
        f"--lda {lda}",
        f"--ldb {ldb}",
        f"--transA {ta}",
        f"--transB {tb}",
        f"--data_type {data_type}",
        f"--m_vl {m_vl}",
        f"--n_vl {n_vl}",
    )

    try:
        return average_score(lambda: parse_kernel_output(run_command(kernel_cmd)), prefix["runs_per_test"])
    except Exception:
        return 0.0


def run_blas_case(case: Dict, data_type: str, prefix: Dict) -> float:
    m, n, k = case["m"], case["n"], case["k"]
    ta, tb = case["transa"], case["transb"]

    blas_cmd = build_prefixed_command(
        CONFIG["omp_threads"],
        prefix["numactl"],
        prefix["taskset"],
        DATA_TYPE_CONFIG[data_type]["blas_binary"],
        f"-m {m}",
        f"-n {n}",
        f"-k {k}",
        f"-transa {ta}",
        f"-transb {tb}",
        f"-innerLoops {prefix['inner_loops']}",
    )

    try:
        return average_score(lambda: parse_blas_output(run_command(blas_cmd)), prefix["runs_per_test"])
    except Exception:
        return 0.0


def build_output_path(data_type: str, output: str | None) -> Path:
    if output:
        return Path(output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / f"sme_kernel_perf_{data_type}_{timestamp}.csv"


def format_improvement(baseline: float, candidate: float) -> str:
    if baseline <= 0:
        return "N/A"
    improvement = ((candidate / baseline) - 1.0) * 100.0
    return f"{improvement:+.2f}%"


def export_results(results: List[Dict], data_type: str, output_path: Path) -> None:
    ordered_cols = [
        "Command",
        "Thread",
        "Function",
        "data_type",
        "type",
        "Mflops_2VLx2VL",
        "Mflops_1VLx4VL",
        "Mflops_4VLx1VL",
        "Mflops_GOTO_BLAS",
        "Improve_blas/2VLx2VL",
        "Improve_blas/1VLx4VL",
        "Improve_blas/4VLx1VL",
    ]

    rows = []
    for row in results:
        baseline = row["Mflops_GOTO_BLAS"]
        export_row = dict(row)
        export_row["Improve_blas/2VLx2VL"] = format_improvement(baseline, row["Mflops_2VLx2VL"])
        export_row["Improve_blas/1VLx4VL"] = format_improvement(baseline, row["Mflops_1VLx4VL"])
        export_row["Improve_blas/4VLx1VL"] = format_improvement(baseline, row["Mflops_4VLx1VL"])
        rows.append(export_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[File] {data_type} result saved to: {output_path}")


def main() -> None:
    args = build_parser().parse_args()
    data_type = args.data_type
    prefix = {
        "numactl": args.numactl,
        "taskset": args.taskset,
        "inner_loops": args.inner_loops,
        "runs_per_test": args.runs_per_test,
    }

    print("=" * 100)
    print(f"  SME Kernel vs GotoBLAS ({data_type})")
    print("=" * 100)

    all_cases = generate_test_cases_by_trans()
    total_cases = len(all_cases)
    print(f"Total test cases: {total_cases}")

    results = []
    function_name = DATA_TYPE_CONFIG[data_type]["function_name"]
    blas_binary = DATA_TYPE_CONFIG[data_type]["blas_binary"]

    for case in all_cases:
        m, n, k = case["m"], case["n"], case["k"]
        ta, tb = case["transa"], case["transb"]
        cmd_str = f"{blas_binary} -transa {ta} -transb {tb} -m {m} -n {n} -k {k}"
        results.append(
            {
                "Command": cmd_str,
                "Thread": 1,
                "Function": function_name,
                "data_type": data_type,
                "type": case["type"],
                "Mflops_2VLx2VL": 0.0,
                "Mflops_1VLx4VL": 0.0,
                "Mflops_4VLx1VL": 0.0,
                "Mflops_GOTO_BLAS": 0.0,
            }
        )

    try:
        print("\n>>> Starting GotoBLAS baseline test <<<")
        for i, case in enumerate(all_cases):
            print(f"[{i + 1}/{total_cases}] BLAS: {results[i]['Command']:60} ... ", end="", flush=True)
            score = run_blas_case(case, data_type, prefix)
            results[i]["Mflops_GOTO_BLAS"] = score
            print(f"{score:,.2f}")

        for tile_name, m_vl, n_vl in CONFIG["kernel_shapes"]:
            print("\n" + "=" * 100)
            print(f">>> Starting tile test: {tile_name} (m_vl={m_vl}, n_vl={n_vl}) <<<")
            for i, case in enumerate(all_cases):
                print(f"[{i + 1}/{total_cases}] {tile_name:8}: {results[i]['Command']:60} ... ", end="", flush=True)
                score = run_test_case(case, tile_name, m_vl, n_vl, data_type, prefix)
                results[i][f"Mflops_{tile_name}"] = score
                print(f"{score:,.2f}")
    except KeyboardInterrupt:
        print("\n[Interrupted by user] Saving collected data...")

    if results:
        output_path = build_output_path(data_type, args.output)
        export_results(results, data_type, output_path)


if __name__ == "__main__":
    main()
