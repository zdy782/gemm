import os
import sys
import random
import string
import shutil
import subprocess


def pack_label(pack_a=False, pack_b=False):
    if pack_a and pack_b:
        return "packab"
    if pack_a:
        return "packa"
    if pack_b:
        return "packb"
    return "nopack"


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
    pack_a=False,
    pack_b=False,
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
            M, N, K, lda, ldb, ldc, gemm_type, transA, transB, uniq_id, data_type, m_vl, n_vl, pack_a, pack_b
        )
        if not asm_code:
            if verbose:
                print(f"[ERROR] Failed to generate asm code for M={M}, N={N}, K={K}")
            return False

        with open(os.path.join(test_path, "kernel_asm.S"), "w") as f:
            f.write(asm_code)

        cpp_code = generate_sme_test_cpp(
            M, N, K, lda, ldb, ldc, gemm_type, transA, transB, uniq_id, repeat, data_type, m_vl, n_vl, pack_a, pack_b
        )
        if not cpp_code:
            if verbose:
                print(f"[ERROR] Failed to generate cpp code for M={M}, N={N}, K={K}")
            return False

        with open(os.path.join(test_path, "test.cpp"), "w") as f:
            f.write(cpp_code)

        driver_code = generate_sme_driver_cpp(
            M, N, K, lda, ldb, ldc, gemm_type, transA, transB, uniq_id, data_type, m_vl, n_vl, pack_a, pack_b
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


def _parse_range_spec(value):
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) == 1:
        start = end = int(parts[0])
        step = 1
    elif len(parts) == 2:
        start = int(parts[0])
        end = int(parts[1])
        step = 1
    elif len(parts) == 3:
        start = int(parts[0])
        end = int(parts[1])
        step = int(parts[2])
    else:
        raise ValueError(f"invalid range spec: {value}")

    if start <= 0 or end <= 0 or step <= 0:
        raise ValueError(f"range values must be positive: {value}")
    if end < start:
        raise ValueError(f"range end must be >= start: {value}")
    return start, end, step


def _range_point_count(value):
    start, end, step = _parse_range_spec(value)
    return ((end - start) // step) + 1


def _resolve_range_stride(value, default_value):
    text = str(value).strip().lower()
    if text in {"", "auto", "none"}:
        return default_value
    return int(text)


def run_range_test(
    M_range,
    N_range,
    K_range,
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
    pack_a=False,
    pack_b=False,
    verbose=True,
    keep_tmp=False,
):
    """Run one ref-style range UT row in a single generated tester."""
    current_path = setup_environment()

    from generate_sme_test import generate_sme_asm, generate_sme_driver_cpp, generate_sme_range_test_cpp
    from generate_makefile import generate_makefile
    from global_config import assert_valid_tile_combo

    assert_valid_tile_combo(m_vl, n_vl)

    m_start, m_end, m_step = _parse_range_spec(M_range)
    n_start, n_end, n_step = _parse_range_spec(N_range)
    k_start, k_end, k_step = _parse_range_spec(K_range)
    total_inner_tests = _range_point_count(M_range) * _range_point_count(N_range) * _range_point_count(K_range)

    lda_gen = _resolve_range_stride(lda, m_end if transA == "N" else k_end)
    ldb_gen = _resolve_range_stride(ldb, k_end if transB == "N" else n_end)
    ldc_gen = _resolve_range_stride(ldc, m_end)

    uniq_id = "".join(random.choices(string.ascii_uppercase, k=8))
    test_path = os.path.join(current_path, "tmp", uniq_id)

    if os.path.exists(test_path):
        shutil.rmtree(test_path, ignore_errors=True)
    os.makedirs(test_path, exist_ok=True)

    try:
        asm_code = generate_sme_asm(
            m_end, n_end, k_end, lda_gen, ldb_gen, ldc_gen,
            gemm_type, transA, transB, uniq_id, data_type, m_vl, n_vl, pack_a, pack_b
        )
        if not asm_code:
            if verbose:
                print(f"[ERROR] Failed to generate asm code for M={M_range}, N={N_range}, K={K_range}")
            return False

        with open(os.path.join(test_path, "kernel_asm.S"), "w") as f:
            f.write(asm_code)

        cpp_code = generate_sme_range_test_cpp(
            m_start, m_end, m_step,
            n_start, n_end, n_step,
            k_start, k_end, k_step,
            lda,
            ldb,
            ldc,
            lda_gen, ldb_gen, ldc_gen,
            gemm_type, transA, transB, uniq_id, repeat,
            data_type, m_vl, n_vl, pack_a, pack_b
        )
        if not cpp_code:
            if verbose:
                print(f"[ERROR] Failed to generate range cpp code for M={M_range}, N={N_range}, K={K_range}")
            return False

        with open(os.path.join(test_path, "test.cpp"), "w") as f:
            f.write(cpp_code)

        driver_code = generate_sme_driver_cpp(
            m_end, n_end, k_end, lda_gen, ldb_gen, ldc_gen,
            gemm_type, transA, transB, uniq_id, data_type, m_vl, n_vl, pack_a, pack_b
        )
        if not driver_code:
            if verbose:
                print(f"[ERROR] Failed to generate driver code for M={M_range}, N={N_range}, K={K_range}")
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
            print(f"[INFO] Compiling range test: M={M_range}, N={N_range}, K={K_range}")
            print(f"[INFO] Expanded inner tests: {total_inner_tests}")
        compile_cmd = f"cd {test_path} && make -s"
        if verbose:
            compile_result = subprocess.run(
                compile_cmd,
                shell=True,
                text=True,
                encoding="utf-8",
            )
        else:
            compile_result = subprocess.run(
                compile_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )
        if compile_result.returncode != 0:
            if verbose:
                print(f"[ERROR] Compilation failed for M={M_range}, N={N_range}, K={K_range}")
            return False

        if verbose:
            print(f"[INFO] Running range test: M={M_range}, N={N_range}, K={K_range}")
        run_cmd = f"cd {test_path} && ./benchmark_kernel"
        if verbose:
            run_result = subprocess.run(
                run_cmd,
                shell=True,
                text=True,
                encoding="utf-8",
            )
        else:
            run_result = subprocess.run(
                run_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )
        if run_result.returncode != 0:
            if verbose:
                print(f"[ERROR] Execution failed for M={M_range}, N={N_range}, K={K_range}")
            return False

        return True

    except Exception as e:
        if verbose:
            print(f"[ERROR] Exception occurred in range test M={M_range}, N={N_range}, K={K_range}: {e}")
        return False

    finally:
        if not keep_tmp and os.path.exists(test_path):
            shutil.rmtree(test_path, ignore_errors=True)
