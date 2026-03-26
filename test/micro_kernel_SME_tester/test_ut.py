import os
import csv
import argparse

from test_runner import run_range_test, run_single_test, setup_environment


current_path = os.path.dirname(os.path.abspath(__file__))
setup_environment()


def _run_mode(testcases, pack_mode: str):
    total = len(testcases)
    passed = 0
    failed = 0
    failed_cases = []

    print(f"\n[INFO] Start running {total} test cases with pack_mode={pack_mode}...")
    print("=" * 80)

    for idx, tc in enumerate(testcases, 1):
        try:
            M_spec = tc.get("M", "0")
            N_spec = tc.get("N", "0")
            K_spec = tc.get("K", "0")
            is_range_case = any(":" in str(value) for value in (M_spec, N_spec, K_spec))
            gemm_type = tc["gemm_type"]
            transA = tc["transA"]
            transB = tc["transB"]
            repeat = int(tc["REPEAT"])
            data_type = tc["data_type"]
            m_vl = int(tc["m_vl"])
            n_vl = int(tc["n_vl"])

            print(f"\n[{idx}/{total}] Running test case:")
            print(f"  pack_mode={pack_mode}")
            print(
                f"  M={M_spec}, N={N_spec}, K={K_spec}, "
                f"lda={tc.get('lda', 'auto')}, ldb={tc.get('ldb', 'auto')}, ldc={tc.get('ldc', 'auto')}"
            )
            print(
                f"  gemm_type={gemm_type}, transA={transA}, transB={transB}, "
                f"data_type={data_type}, m_vl={m_vl}, n_vl={n_vl}"
            )

            if is_range_case:
                ok = run_range_test(
                    M_spec,
                    N_spec,
                    K_spec,
                    tc.get("lda", "auto"),
                    tc.get("ldb", "auto"),
                    tc.get("ldc", "auto"),
                    gemm_type,
                    transA,
                    transB,
                    repeat,
                    data_type=data_type,
                    m_vl=m_vl,
                    n_vl=n_vl,
                    pack_mode=pack_mode,
                    verbose=True,
                )
            else:
                M = int(M_spec)
                N = int(N_spec)
                K = int(K_spec)
                lda = int(tc.get("lda", M))
                ldb = int(tc.get("ldb", K))
                ldc = int(tc.get("ldc", M))
                ok = run_single_test(
                    M, N, K, lda, ldb, ldc,
                    gemm_type, transA, transB, repeat,
                    data_type=data_type,
                    m_vl=m_vl,
                    n_vl=n_vl,
                    pack_mode=pack_mode,
                    verbose=True,
                )

            if ok:
                passed += 1
                print("  [RESULT] PASS")
            else:
                failed += 1
                failed_cases.append({
                    "index": idx,
                    "pack_mode": pack_mode,
                    "M": M_spec, "N": N_spec, "K": K_spec,
                    "lda": tc.get("lda", "auto"),
                    "ldb": tc.get("ldb", "auto"),
                    "ldc": tc.get("ldc", "auto"),
                    "gemm_type": gemm_type,
                    "transA": transA,
                    "transB": transB,
                    "data_type": data_type,
                    "m_vl": m_vl,
                    "n_vl": n_vl,
                })
                print("  [RESULT] FAIL")
        except Exception as e:
            failed += 1
            failed_cases.append({
                "index": idx,
                "pack_mode": pack_mode,
                "error": f"Parse/Execute exception: {e}",
                "raw_data": tc,
            })
            print(f"  [RESULT] FAIL (Exception: {e})")
        print("-" * 80)

    print("\n" + "=" * 80)
    print(f"[TEST SUMMARY] pack_mode={pack_mode}, Total: {total}, Passed: {passed}, Failed: {failed}")
    print("=" * 80)

    if failed > 0:
        print(f"\n[FAILED CASES DETAIL] (pack_mode={pack_mode}, Total: {failed} cases)")
        print("-" * 80)
        for i, case in enumerate(failed_cases, 1):
            print(f"\n[{i}] Test case #{case['index']}:")
            print(f"  pack_mode={case['pack_mode']}")
            if "error" in case:
                print(f"  Error: {case['error']}")
                print(f"  Raw data: {case['raw_data']}")
            else:
                print(f"  M={case['M']}, N={case['N']}, K={case['K']}")
                print(f"  lda={case['lda']}, ldb={case['ldb']}, ldc={case['ldc']}")
                print(f"  gemm_type={case['gemm_type']}, transA={case['transA']}, transB={case['transB']}")
                print(
                    f"  data_type={case['data_type']}, m_vl={case['m_vl']}, "
                    f"n_vl={case['n_vl']}"
                )
        print("-" * 80)

    return {
        "pack_mode": pack_mode,
        "total": total,
        "passed": passed,
        "failed": failed,
        "failed_cases": failed_cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-mode", type=str, default="nopack", choices=["nopack", "packed", "both"])
    parser.add_argument("--testcases", type=str, default=None)
    args = parser.parse_args()

    default_range_path = os.path.join(current_path, "testcases_ut_range.csv")
    default_single_path = os.path.join(current_path, "testcases_ut.csv")
    testcases_path = args.testcases or (default_range_path if os.path.exists(default_range_path) else default_single_path)
    required_columns = {
        "M", "N", "K", "lda", "ldb", "ldc",
        "gemm_type", "transA", "transB", "REPEAT",
        "data_type", "m_vl", "n_vl",
    }

    if not os.path.exists(testcases_path):
        print(f"[ERROR] Testcases file not found: {testcases_path}")
        print("[INFO] Running default test case...")
        pack_modes = ["nopack", "packed"] if args.pack_mode == "both" else [args.pack_mode]
        ok = True
        for pack_mode in pack_modes:
            ok = run_single_test(16, 64, 16, 16, 64, 64, "small", "N", "N", 64, pack_mode=pack_mode) and ok
        if ok:
            print("\n[PASS] Default test case passed")
        else:
            print("\n[FAIL] Default test case failed")
        return

    with open(testcases_path, "r") as f:
        reader = csv.DictReader(f)
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(missing_columns)}"
            )
        testcases = list(reader)

    pack_modes = ["nopack", "packed"] if args.pack_mode == "both" else [args.pack_mode]
    results = [_run_mode(testcases, pack_mode) for pack_mode in pack_modes]

    if len(results) > 1:
        total = sum(result["total"] for result in results)
        passed = sum(result["passed"] for result in results)
        failed = sum(result["failed"] for result in results)
        print("\n" + "=" * 80)
        print(f"[TEST SUMMARY] pack_mode=both, Total: {total}, Passed: {passed}, Failed: {failed}")
        print("=" * 80)


if __name__ == "__main__":
    main()
