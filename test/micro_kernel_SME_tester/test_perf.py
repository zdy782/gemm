import argparse
import os
import subprocess
import re
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
BUNDLE_BUILDER = REPO_ROOT / "src" / "micro_kernel_SME" / "half" / "build_blas_bundle.py"
BUNDLE_LAYOUT_VERSION = "direct-driver-benchmark-v1"

CONFIG = {
    "numactl": ["numactl", "-m", "15"],
    "taskset": ["taskset", "-c", "575"],
    "env": {"OMP_NUM_THREADS": "1"},
    "inner_loops": 64,
    "bundle_output_root": REPO_ROOT / "build" / "perf_bundles",
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

def run_command(cmd: List[str], extra_env: Dict[str, str] = None) -> str:
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
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command execution failed: {e}") from e


def pack_label(pack_a: bool, pack_b: bool, func_name: str) -> str:
    if pack_a and pack_b:
        return f"{func_name}_packab"
    if pack_a:
        return f"{func_name}_packa"
    if pack_b:
        return f"{func_name}_packb"
    return "autogemm_nopacking"


def build_pack_name(pack_a: bool, pack_b: bool) -> str:
    if pack_a and pack_b:
        return "packab"
    if pack_a:
        return "packa"
    if pack_b:
        return "packb"
    return "nopack"


def bundle_variant_dir(pack_a: bool, pack_b: bool, m_vl: int, n_vl: int) -> Path:
    pack_name = build_pack_name(pack_a, pack_b)
    return CONFIG["bundle_output_root"] / f"{pack_name}_{m_vl}x{n_vl}"


def benchmark_binary_for_variant(pack_a: bool, pack_b: bool, m_vl: int, n_vl: int, data_type: str) -> Path:
    bundle_dir = bundle_variant_dir(pack_a, pack_b, m_vl, n_vl)
    binary_name = "sbgemm.goto" if data_type == "bf16" else "shgemm.goto"
    return bundle_dir / "bin" / binary_name


def bundle_version_file(pack_a: bool, pack_b: bool, m_vl: int, n_vl: int) -> Path:
    return bundle_variant_dir(pack_a, pack_b, m_vl, n_vl) / "bundle_version.txt"


def ensure_bundle(pack_a: bool, pack_b: bool, m_vl: int, n_vl: int) -> None:
    sbgemm_binary = benchmark_binary_for_variant(pack_a, pack_b, m_vl, n_vl, "bf16")
    shgemm_binary = benchmark_binary_for_variant(pack_a, pack_b, m_vl, n_vl, "fp16")
    version_file = bundle_version_file(pack_a, pack_b, m_vl, n_vl)
    version_matches = version_file.exists() and version_file.read_text(encoding="utf-8").strip() == BUNDLE_LAYOUT_VERSION
    if sbgemm_binary.exists() and shgemm_binary.exists() and version_matches:
        return

    build_cmd = [
        sys.executable,
        str(BUNDLE_BUILDER),
        "--pack",
        build_pack_name(pack_a, pack_b),
        "--m-vl",
        str(m_vl),
        "--n-vl",
        str(n_vl),
        "--output-dir",
        str(CONFIG["bundle_output_root"]),
    ]
    print(
        f"[INFO] Building fixed bundle for pack={build_pack_name(pack_a, pack_b)}, "
        f"tile={m_vl}x{n_vl}"
    )
    run_command(build_cmd)


def prepare_kernel_binaries(pack_a: bool, pack_b: bool) -> Dict[tuple, Path]:
    binaries = {}
    for tile_name, m_vl, n_vl in CONFIG["kernel_shapes"]:
        ensure_bundle(pack_a, pack_b, m_vl, n_vl)
        binaries[("bf16", tile_name)] = benchmark_binary_for_variant(pack_a, pack_b, m_vl, n_vl, "bf16")
        binaries[("fp16", tile_name)] = benchmark_binary_for_variant(pack_a, pack_b, m_vl, n_vl, "fp16")
    return binaries

def generate_test_case_groups_by_trans() -> Dict[str, List[Dict]]:
    trans_pairs = [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")]
    cases_by_trans = {f"{ta}{tb}": [] for ta, tb in trans_pairs}
    SIZES_S = [16, 24, 32, 48]
    k_list = [16, 24, 32, 48, 64, 96]
    selected_sizes = [
        (16, 64), (24, 96), (32, 100), (48, 128), (64, 192),
        (96, 256), (128, 300), (192, 384), (256, 512), (512, 768),
        (64, 16), (96, 24), (100, 32), (128, 48), (192, 64),
        (256, 96), (300, 128), (384, 192), (512, 256), (768, 512),
    ]
    verybig_n_m_list = [28, 64, 128]
    verybig_n_n_list = [500, 1000, 1536]
    verybig_n_k_list = [16, 32]
    verybig_m_m_list = [500, 1000, 1536]
    verybig_m_n_list = [28, 64, 128]
    verybig_m_k_list = [16, 32]
    verybig_k_m_list = [64, 128, 256]
    verybig_k_n_list = [64, 128, 256]
    verybig_k_k_list = [256, 300, 512]

    def label(x: int) -> str:
        return "S" if x in SIZES_S else "L"

    for k in k_list:
        for m, n in selected_sizes:
            shape_type = f"{label(m)}{label(n)}S"
            for ta, tb in trans_pairs:
                cases_by_trans[f"{ta}{tb}"].append(
                    {"m": m, "n": n, "k": k, "transa": ta, "transb": tb, "type": shape_type}
                )

    # verybigN
    for m in verybig_n_m_list:
        for n in verybig_n_n_list:
            for k in verybig_n_k_list:
                for ta, tb in trans_pairs:
                    cases_by_trans[f"{ta}{tb}"].append(
                        {"m": m, "n": n, "k": k, "transa": ta, "transb": tb, "type": "verybigN"}
                    )

    # verybigM
    for m in verybig_m_m_list:
        for n in verybig_m_n_list:
            for k in verybig_m_k_list:
                for ta, tb in trans_pairs:
                    cases_by_trans[f"{ta}{tb}"].append(
                        {"m": m, "n": n, "k": k, "transa": ta, "transb": tb, "type": "verybigM"}
                    )

    # verybigK
    for m in verybig_k_m_list:
        for n in verybig_k_n_list:
            for k in verybig_k_k_list:
                for ta, tb in trans_pairs:
                    cases_by_trans[f"{ta}{tb}"].append(
                        {"m": m, "n": n, "k": k, "transa": ta, "transb": tb, "type": "verybigK"}
                    )

    grouped_cases = {}
    for trans_key, cases in cases_by_trans.items():
        unique_dict = {}
        for c in cases:
            key = (c["m"], c["n"], c["k"], c["transa"], c["transb"])
            unique_dict[key] = c
        grouped_cases[trans_key] = list(unique_dict.values())

    return grouped_cases


def generate_test_cases() -> List[Dict]:
    cases_by_trans = generate_test_case_groups_by_trans()
    ordered_cases = []
    for trans_key in ("NN", "NT", "TN", "TT"):
        ordered_cases.extend(cases_by_trans[trans_key])
    return ordered_cases

def run_test_case(case: Dict, kernel_binary: Path) -> float:
    m, n, k = case["m"], case["n"], case["k"]
    ta, tb = case["transa"], case["transb"]
    lda = k if ta == "T" else m
    ldb = n if tb == "T" else k
    ldc = m

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
        "-transa", ta,
        "-transb", tb,
        "-api", "F",
        "-order", "C",
        "-alphaR", "1.0",
        "-betaR", "1.0",
        "-innerLoops", str(CONFIG["inner_loops"]),
    ]

    try:
        kernel_output = run_command(kernel_cmd_parts)
        return parse_blas_output(kernel_output)
    except Exception:
        return 0.0

def run_blas_case(case: Dict, blas_binary: str) -> float:
    m, n, k = case["m"], case["n"], case["k"]
    ta, tb = case["transa"], case["transb"]

    blas_cmd = [
        *CONFIG["numactl"],
        *CONFIG["taskset"],
        blas_binary,
        "-m", str(m),
        "-n", str(n),
        "-k", str(k),
        "-transa", ta,
        "-transb", tb,
        "-innerLoops", str(CONFIG["inner_loops"]),
    ]

    try:
        blas_output = run_command(blas_cmd)
        return parse_blas_output(blas_output)
    except Exception:
        return 0.0

def run_evaluation(
    data_type: str,
    blas_binary: str,
    func_name: str,
    all_cases: List[Dict],
    pack_a: bool,
    pack_b: bool,
):
    """
    Core evaluation loop. Runs all test cases for a specific data type (e.g., bf16, fp16).
    """
    total_cases = len(all_cases)
    current_pack_label = pack_label(pack_a, pack_b, func_name)
    kernel_binaries = prepare_kernel_binaries(pack_a, pack_b)
    print("\n" + "=" * 100)
    print(f" >>> Starting Evaluation for {data_type.upper()} {current_pack_label} ({blas_binary}) <<<")
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
            "Mflops_KPL_BLAS": 0.0,
        }
        for tile_name, _, _ in CONFIG["kernel_shapes"]:
            row[f"Mflops_{current_pack_label}_{tile_name}"] = 0.0
        results.append(row)

    # 1. Run BLAS once as baseline
    print(f"\n>>> Starting KPL_BLAS baseline test for {data_type.upper()} <<<")
    for i, case in enumerate(all_cases):
        print(f"[{i+1}/{total_cases}] BLAS: {results[i]['Command']:60} ... ", end="", flush=True)
        score = run_blas_case(case, blas_binary)
        results[i]["Mflops_KPL_BLAS"] = score
        print(f"{score:,.2f}")

    print("\n" + "=" * 100)
    print(f">>> Starting kernel mode: {current_pack_label} <<<")
    for tile_name, m_vl, n_vl in CONFIG["kernel_shapes"]:
        print("\n" + "=" * 100)
        print(
            f">>> Starting tile test: {tile_name} (m_vl={m_vl}, n_vl={n_vl}, "
            f"pack_a={pack_a}, pack_b={pack_b}) <<<"
        )
        kernel_binary = kernel_binaries[(data_type, tile_name)]
        print(f"[INFO] Using kernel benchmark binary: {kernel_binary}")

        for i, case in enumerate(all_cases):
            print(f"[{i+1}/{total_cases}] {tile_name:8}/{current_pack_label:18}: {results[i]['Command']:60} ... ", end="", flush=True)
            score = run_test_case(case, kernel_binary)
            results[i][f"Mflops_{current_pack_label}_{tile_name}"] = score
            print(f"{score:,.2f}")

    # Data processing and export
    if results:
        df = pd.DataFrame(results)

        ordered_cols = ["Command", "Thread", "Function", "type"]

        for tile_name, _, _ in CONFIG["kernel_shapes"]:
            metric_col = f"Mflops_{current_pack_label}_{tile_name}"
            improve_col = f"Improve_blas/{current_pack_label}_{tile_name}"
            mask = df["Mflops_KPL_BLAS"] > 0
            df.loc[mask, improve_col] = (
                (df.loc[mask, metric_col] / df.loc[mask, "Mflops_KPL_BLAS"]) - 1.0
            ) * 100.0
            df.loc[~mask, improve_col] = 0.0
            df[improve_col] = df[improve_col].map(lambda x: f"{x:+.2f}%" if x != 0.0 else "N/A")
            ordered_cols.append(metric_col)

        ordered_cols.append("Mflops_KPL_BLAS")

        for tile_name, _, _ in CONFIG["kernel_shapes"]:
            ordered_cols.append(f"Improve_blas/{current_pack_label}_{tile_name}")

        export_df = df[ordered_cols].copy()
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        csv_file = f"sme_kernel_{data_type}_{current_pack_label}_{timestamp}.csv"
        export_df.to_csv(csv_file, index=False)
        print(f"[File] CSV saved to: {csv_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    print("=" * 100)
    print("  SME Kernel vs BLAS (Automated BF16 & FP16 Testing)")
    print("=" * 100)
    print("[INFO] test_perf runs perf-only kernels; correctness is skipped via --perf_only")

    cases_by_trans = generate_test_case_groups_by_trans()
    all_cases = generate_test_cases()
    total_cases = len(all_cases)
    print("Perf suite: expanded")
    print(f"Total test cases per precision type: {total_cases}")
    for trans_key, trans_cases in cases_by_trans.items():
        print(f"  - {trans_key}: {len(trans_cases)} cases")
    print("[INFO] test_perf runs all four pack modes: nopack, packa, packb, packab")

    # Define the testing configurations: precision, executable path, and output function name
    test_configs = [
        {"data_type": "bf16", "blas_binary": str(SCRIPT_DIR / "sbgemm.goto"), "func_name": "sbgemm"},
        {"data_type": "fp16", "blas_binary": str(SCRIPT_DIR / "shgemm.goto"), "func_name": "shgemm"},
    ]
    pack_modes = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]

    try:
        # Loop over configurations to automatically test bf16 and then fp16
        for config in test_configs:
            for pack_a, pack_b in pack_modes:
                run_evaluation(
                    data_type=config["data_type"],
                    blas_binary=config["blas_binary"],
                    func_name=config["func_name"],
                    all_cases=all_cases,
                    pack_a=pack_a,
                    pack_b=pack_b,
                )

    except KeyboardInterrupt:
        print("\n[Interrupted by user] Halting execution and keeping collected data from earlier evaluations...")

if __name__ == "__main__":
    main()
