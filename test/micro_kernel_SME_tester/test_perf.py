import subprocess
import re
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent

CONFIG = {
    "numactl": "numactl -m 15",
    "taskset": "taskset -c 575",
    "omp_threads": "OMP_NUM_THREADS=1",
    "inner_loops": 64,
    "kernel_script": str(SCRIPT_DIR / "test_single.py"),
    "runs_per_test": 1,
    "kernel_shapes": [
        ("1VLx4VL", 1, 4),
        ("2VLx2VL", 2, 2),
        ("4VLx1VL", 4, 1),
    ],
}

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
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command execution failed: {e}") from e

def generate_test_cases_by_trans() -> List[Dict]:
    trans_pairs = [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")]
    cases_by_trans = {f"{ta}{tb}": [] for ta, tb in trans_pairs}

    SIZES_S = [16, 24, 32, 48]
    K_LIST = [16, 24, 32, 48]

    def label(x: int) -> str:
        return "S" if x in SIZES_S else "L"

    selected_sizes = [
        (16, 64), (24, 96), (32, 100), (48, 128), (64, 192),
        (96, 256), (128, 300), (256, 512), (512, 768),
        (64, 16), (96, 24), (100, 32), (128, 48), (192, 64),
        (256, 96), (300, 128), (512, 256), (768, 512)
    ]

    for k in K_LIST:
        for m, n in selected_sizes:
            shape_type = f"{label(m)}{label(n)}S"
            for ta, tb in trans_pairs:
                cases_by_trans[f"{ta}{tb}"].append(
                    {"m": m, "n": n, "k": k, "transa": ta, "transb": tb, "type": shape_type}
                )

    # verybigN
    for m in [28, 64]:
        for n in [500, 1000]:
            for k in [16, 32]:
                for ta, tb in trans_pairs:
                    cases_by_trans[f"{ta}{tb}"].append(
                        {"m": m, "n": n, "k": k, "transa": ta, "transb": tb, "type": "verybigN"}
                    )

    # verybigM
    for m in [500, 1000]:
        for n in [28, 64]:
            for k in [16, 32]:
                for ta, tb in trans_pairs:
                    cases_by_trans[f"{ta}{tb}"].append(
                        {"m": m, "n": n, "k": k, "transa": ta, "transb": tb, "type": "verybigM"}
                    )

    # verybigK
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
        for c in cases:
            key = (c["m"], c["n"], c["k"], c["transa"], c["transb"])
            unique_dict[key] = c
        final_cases.extend(unique_dict.values())

    return final_cases

# Added data_type parameter to dynamically set precision type
def run_test_case(case: Dict, m_vl: int, n_vl: int, data_type: str) -> float:
    m, n, k = case["m"], case["n"], case["k"]
    ta, tb = case["transa"], case["transb"]
    lda = k if ta == "T" else m
    ldb = n if tb == "T" else k
    ldc = m

    kernel_cmd = " ".join([
        CONFIG["numactl"], CONFIG["taskset"], sys.executable, CONFIG["kernel_script"],
        f"--M {m}", f"--N {n}", f"--K {k}",
        f"--lda {lda}", f"--ldb {ldb}", f"--ldc {ldc}",
        f"--transA {ta}", f"--transB {tb}",
        f"--data_type {data_type}", f"--m_vl {m_vl}", f"--n_vl {n_vl}"
    ])

    try:
        kernel_output = run_command(kernel_cmd)
        return parse_kernel_output(kernel_output)
    except Exception:
        return 0.0

# Added blas_binary parameter to dynamically set target executable
def run_blas_case(case: Dict, blas_binary: str) -> float:
    m, n, k = case["m"], case["n"], case["k"]
    ta, tb = case["transa"], case["transb"]

    blas_cmd = " ".join([
        CONFIG["omp_threads"], CONFIG["numactl"], CONFIG["taskset"], blas_binary,
        f"-m {m}", f"-n {n}", f"-k {k}", f"-transa {ta}", f"-transb {tb}", f"-innerLoops {CONFIG['inner_loops']}"
    ])

    try:
        blas_output = run_command(blas_cmd)
        return parse_blas_output(blas_output)
    except Exception:
        return 0.0

def run_evaluation(data_type: str, blas_binary: str, func_name: str, all_cases: List[Dict]):
    """
    Core evaluation loop. Runs all test cases for a specific data type (e.g., bf16, fp16).
    """
    total_cases = len(all_cases)
    print("\n" + "=" * 100)
    print(f" >>> Starting Evaluation for {data_type.upper()} ({blas_binary}) <<<")
    print("=" * 100)

    results = []

    # Initialize result rows
    for case in all_cases:
        m, n, k = case["m"], case["n"], case["k"]
        ta, tb = case["transa"], case["transb"]
        cmd_str = f"{blas_binary} -transa {ta} -transb {tb} -m {m} -n {n} -k {k}"
        row = {
            "Command": cmd_str,
            "Thread": 1,
            "Function": func_name,
            "type": case["type"],
            "Mflops_2VLx2VL": 0.0,
            "Mflops_1VLx4VL": 0.0,
            "Mflops_4VLx1VL": 0.0,
            "Mflops_KPL_BLAS": 0.0,
        }
        results.append(row)

    # 1. Run BLAS once as baseline
    print(f"\n>>> Starting KPL_BLAS baseline test for {data_type.upper()} <<<")
    for i, case in enumerate(all_cases):
        print(f"[{i+1}/{total_cases}] BLAS: {results[i]['Command']:60} ... ", end="", flush=True)
        score = run_blas_case(case, blas_binary)
        results[i]["Mflops_KPL_BLAS"] = score
        print(f"{score:,.2f}")

    # 2. Switch kernel shapes and fill results horizontally
    for tile_name, m_vl, n_vl in CONFIG["kernel_shapes"]:
        print("\n" + "=" * 100)
        print(f">>> Starting tile test: {tile_name} (m_vl={m_vl}, n_vl={n_vl}) <<<")

        for i, case in enumerate(all_cases):
            print(f"[{i+1}/{total_cases}] {tile_name:8}: {results[i]['Command']:60} ... ", end="", flush=True)
            score = run_test_case(case, m_vl, n_vl, data_type)
            results[i][f"Mflops_{tile_name}"] = score
            print(f"{score:,.2f}")

    # Data processing and export
    if results:
        df = pd.DataFrame(results)

        # Calculate improvement percentages
        for tile_name, _, _ in CONFIG["kernel_shapes"]:
            col_name = f"Improve_blas/{tile_name}"
            mask = df["Mflops_KPL_BLAS"] > 0
            df.loc[mask, col_name] = (
                (df.loc[mask, f"Mflops_{tile_name}"] / df.loc[mask, "Mflops_KPL_BLAS"]) - 1.0
            ) * 100.0
            df.loc[~mask, col_name] = 0.0
            df[col_name] = df[col_name].map(lambda x: f"{x:+.2f}%" if x != 0.0 else "N/A")

        header_tuples = [
            ("Command", "", ""),
            ("Thread", "", ""),
            ("Function", "", ""),
            ("type", "", ""),
            ("Mflops", "autogemm nopacking", "2VLx2VL"),
            ("Mflops", "autogemm nopacking", "1VLx4VL"),
            ("Mflops", "autogemm nopacking", "4VLx1VL"),
            ("Mflops", "KPL_BLAS", ""),
            ("Improve", "blas/2VLx2VL", ""),
            ("Improve", "blas/1VLx4VL", ""),
            ("Improve", "blas/4VLx1VL", "")
        ]

        ordered_cols = [
            "Command", "Thread", "Function", "type",
            "Mflops_2VLx2VL", "Mflops_1VLx4VL", "Mflops_4VLx1VL", "Mflops_KPL_BLAS",
            "Improve_blas/2VLx2VL", "Improve_blas/1VLx4VL", "Improve_blas/4VLx1VL"
        ]

        export_df = df[ordered_cols].copy()
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Include data type in output file names
        excel_file = f"sme_kernel_{data_type}_wide_format_{timestamp}.xlsx"
        csv_file = f"sme_kernel_{data_type}_wide_format_{timestamp}.csv"

        # Export 1: Excel with multi-level header
        try:
            multi_df = export_df.copy()
            multi_df.columns = pd.MultiIndex.from_tuples(header_tuples)
            multi_df.to_excel(excel_file, index=False)
            print(f"\n[File] Formatted Excel with multi-level header saved to: {excel_file}")
        except Exception as e:
            print(f"\n[Notice] Failed to save Excel, openpyxl may be missing. Skipping Excel export. Error: {e}")

        # Export 2: Flat CSV fallback
        export_df.columns = [
            "Command", "Thread", "Function", "type",
            "Mflops_autogemm_nopacking_2VLx2VL",
            "Mflops_autogemm_nopacking_1VLx4VL",
            "Mflops_autogemm_nopacking_4VLx1VL",
            "Mflops_KPL_BLAS",
            "Improve_blas/2VLx2VL", "Improve_blas/1VLx4VL", "Improve_blas/4VLx1VL"
        ]
        export_df.to_csv(csv_file, index=False)
        print(f"[File] Flat CSV saved to: {csv_file}")

def main():
    print("=" * 100)
    print("  SME Kernel vs BLAS (Automated BF16 & FP16 Testing)")
    print("=" * 100)

    all_cases = generate_test_cases_by_trans()
    print(f"Total test cases per precision type: {len(all_cases)}")

    # Define the testing configurations: precision, executable path, and output function name
    test_configs = [
        {"data_type": "bf16", "blas_binary": str(SCRIPT_DIR / "sbgemm.goto"), "func_name": "sbgemm"},
        {"data_type": "fp16", "blas_binary": str(SCRIPT_DIR / "shgemm.goto"), "func_name": "shgemm"},
    ]

    try:
        # Loop over configurations to automatically test bf16 and then fp16
        for config in test_configs:
            run_evaluation(
                data_type=config["data_type"],
                blas_binary=config["blas_binary"],
                func_name=config["func_name"],
                all_cases=all_cases
            )

    except KeyboardInterrupt:
        print("\n[Interrupted by user] Halting execution and keeping collected data from earlier evaluations...")

if __name__ == "__main__":
    main()
