import argparse
import os
import csv

from test_runner import run_single_test, setup_environment


current_path = os.path.dirname(os.path.abspath(__file__))
setup_environment()


def parse_args():
    parser = argparse.ArgumentParser(description="Run SME UT cases from CSV")
    parser.add_argument(
        "--ext_load_strategy",
        type=str,
        default="legacy_half_vl",
        choices=["legacy_half_vl", "experimental"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    testcases_path = os.path.join(current_path, "testcases_ut.csv")
    required_columns = {
        "M", "N", "K", "lda", "ldb", "ldc",
        "gemm_type", "transA", "transB", "REPEAT",
        "data_type", "m_vl", "n_vl",
    }

    if not os.path.exists(testcases_path):
        print(f"[ERROR] Testcases file not found: {testcases_path}")
        print("[INFO] Running default test case...")
        if run_single_test(16, 64, 16, 16, 64, 64, "small", "N", "N", 64):
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

    total = len(testcases)
    passed = 0
    failed = 0
    failed_cases = []

    print(f"\n[INFO] Start running {total} test cases...")
    print("=" * 80)

    for idx, tc in enumerate(testcases, 1):
        try:
            M = int(tc.get("M", 0))
            N = int(tc.get("N", 0))
            K = int(tc.get("K", 0))
            lda = int(tc.get("lda", M))
            ldb = int(tc.get("ldb", K))
            ldc = int(tc.get("ldc", M))
            gemm_type = tc["gemm_type"]
            transA = tc["transA"]
            transB = tc["transB"]
            repeat = int(tc["REPEAT"])
            data_type = tc["data_type"]
            m_vl = int(tc["m_vl"])
            n_vl = int(tc["n_vl"])

            print(f"\n[{idx}/{total}] Running test case:")
            print(f"  M={M}, N={N}, K={K}, lda={lda}, ldb={ldb}, ldc={ldc}")
            print(
                f"  gemm_type={gemm_type}, transA={transA}, transB={transB}, "
                f"data_type={data_type}, m_vl={m_vl}, n_vl={n_vl}, "
                f"ext_load_strategy={args.ext_load_strategy}"
            )

            ok = run_single_test(
                M, N, K, lda, ldb, ldc,
                gemm_type, transA, transB, repeat,
                data_type=data_type,
                m_vl=m_vl,
                n_vl=n_vl,
                ext_load_strategy=args.ext_load_strategy,
                verbose=True,
            )

            if ok:
                passed += 1
                print("  [RESULT] PASS")
            else:
                failed += 1
                failed_cases.append({
                    "index": idx,
                    "M": M, "N": N, "K": K,
                    "lda": lda, "ldb": ldb, "ldc": ldc,
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
                "error": f"Parse/Execute exception: {e}",
                "raw_data": tc,
            })
            print(f"  [RESULT] FAIL (Exception: {e})")
        print("-" * 80)

    print("\n" + "=" * 80)
    print(f"[TEST SUMMARY] Total: {total}, Passed: {passed}, Failed: {failed}")
    print("=" * 80)

    if failed > 0:
        print(f"\n[FAILED CASES DETAIL] (Total: {failed} cases)")
        print("-" * 80)
        for i, case in enumerate(failed_cases, 1):
            print(f"\n[{i}] Test case #{case['index']}:")
            if "error" in case:
                print(f"  Error: {case['error']}")
                print(f"  Raw data: {case['raw_data']}")
            else:
                print(f"  M={case['M']}, N={case['N']}, K={case['K']}")
                print(f"  lda={case['lda']}, ldb={case['ldb']}, ldc={case['ldc']}")
                print(f"  gemm_type={case['gemm_type']}, transA={case['transA']}, transB={case['transB']}")
                print(
                    f"  data_type={case['data_type']}, m_vl={case['m_vl']}, "
                    f"n_vl={case['n_vl']}, ext_load_strategy={args.ext_load_strategy}"
                )
        print("-" * 80)


if __name__ == "__main__":
    main()
