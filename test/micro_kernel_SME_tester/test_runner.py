import os
import sys
import random
import string
import shutil
import subprocess


def setup_environment():
    """Set up Python import paths."""
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_path, "..", "..")
    src_path = os.path.join(project_root, "src")
    micro_kernel_sme_src_path = os.path.join(src_path, "micro_kernel_SME")
    sys.path.insert(0, src_path)
    sys.path.insert(0, micro_kernel_sme_src_path)
    return current_path


def run_single_test(
    M,
    N,
    K,
    lda,
    ldb,
    ldc,
    gemm_type,
    transA,
    transB,
    repeat,
    data_type="fp32",
    m_vl=1,
    n_vl=4,
    pack_mode="nopack",
    verbose=True,
    keep_tmp=False,
):
    """Run the full workflow for a single test case."""
    current_path = setup_environment()

    from generate_sme_test import generate_sme_asm, generate_sme_driver_cpp, generate_sme_test_cpp
    from generate_makefile import generate_makefile
    from global_config import assert_valid_tile_combo

    assert_valid_tile_combo(m_vl, n_vl)

    uniq_id = "".join(random.choices(string.ascii_uppercase, k=8))
    test_path = os.path.join(current_path, "tmp", uniq_id)

    if os.path.exists(test_path):
        shutil.rmtree(test_path, ignore_errors=True)
    os.makedirs(test_path, exist_ok=True)

    try:
        asm_code = generate_sme_asm(
            M, N, K, lda, ldb, ldc, gemm_type, transA, transB, uniq_id, data_type, m_vl, n_vl, pack_mode
        )
        if not asm_code:
            if verbose:
                print(f"[ERROR] Failed to generate asm code for M={M}, N={N}, K={K}")
            return False

        with open(os.path.join(test_path, "kernel_asm.S"), "w") as f:
            f.write(asm_code)

        cpp_code = generate_sme_test_cpp(
            M, N, K, lda, ldb, ldc, gemm_type, transA, transB, uniq_id, repeat, data_type, m_vl, n_vl, pack_mode
        )
        if not cpp_code:
            if verbose:
                print(f"[ERROR] Failed to generate cpp code for M={M}, N={N}, K={K}")
            return False

        with open(os.path.join(test_path, "test.cpp"), "w") as f:
            f.write(cpp_code)

        driver_code = generate_sme_driver_cpp(
            M, N, K, lda, ldb, ldc, gemm_type, transA, transB, uniq_id, data_type, m_vl, n_vl, pack_mode
        )
        if not driver_code:
            if verbose:
                print(f"[ERROR] Failed to generate driver code for M={M}, N={N}, K={K}")
            return False
        with open(os.path.join(test_path, "driver.cpp"), "w") as f:
            f.write(driver_code)

        makefile = generate_makefile(data_type)
        with open(os.path.join(test_path, "Makefile"), "w") as f:
            f.write(makefile)

        test_h_path = os.path.join(current_path, "test.h")
        timer_h_path = os.path.join(current_path, "timer.h")

        if not os.path.exists(test_h_path):
            if verbose:
                print(f"[ERROR] test.h not found: {test_h_path}")
            return False
        if not os.path.exists(timer_h_path):
            if verbose:
                print(f"[ERROR] timer.h not found: {timer_h_path}")
            return False

        shutil.copy(test_h_path, test_path)
        shutil.copy(timer_h_path, test_path)

        if verbose:
            print(f"[INFO] Compiling test case: M={M}, N={N}, K={K}")
        compile_cmd = f"cd {test_path} && make -s"
        compile_result = subprocess.run(
            compile_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        if verbose and compile_result.stdout:
            print(f"[COMPILE OUTPUT]\n{compile_result.stdout}")
        if compile_result.returncode != 0 or "ERROR" in compile_result.stdout.upper():
            if verbose:
                print(f"[ERROR] Compilation failed for M={M}, N={N}, K={K}")
            return False

        if verbose:
            print(f"[INFO] Running test case: M={M}, N={N}, K={K}")
        run_cmd = f"cd {test_path} && ./benchmark_kernel"
        run_result = subprocess.run(
            run_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        if verbose and run_result.stdout:
            print(f"[RUN OUTPUT]\n{run_result.stdout}")
        if run_result.returncode != 0 or "ERROR" in run_result.stdout.upper():
            if verbose:
                print(f"[ERROR] Execution failed for M={M}, N={N}, K={K}")
            return False

        return True

    except Exception as e:
        if verbose:
            print(f"[ERROR] Exception occurred in test case M={M}, N={N}, K={K}: {e}")
        return False

    finally:
        if not keep_tmp and os.path.exists(test_path):
            shutil.rmtree(test_path, ignore_errors=True)
