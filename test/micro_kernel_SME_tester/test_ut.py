import os
import csv

from test_runner import run_single_test, setup_environment


current_path = os.path.dirname(os.path.abspath(__file__))
setup_environment()


def main():
    testcases_path = os.path.join(current_path, "testcases_ut.csv")

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
            gemm_type = tc.get("gemm_type", "small")
            transA = tc.get("transA", "N")
            transB = tc.get("transB", "N")
            repeat = int(tc.get("REPEAT", 64))
            data_type = tc.get("data_type", "fp32")
            m_vl = int(tc.get("m_vl", 1))
            n_vl = int(tc.get("n_vl", 4))

            print(f"\n[{idx}/{total}] Running test case:")
            print(f"  M={M}, N={N}, K={K}, lda={lda}, ldb={ldb}, ldc={ldc}")
            print(
                f"  gemm_type={gemm_type}, transA={transA}, transB={transB}, "
                f"data_type={data_type}, m_vl={m_vl}, n_vl={n_vl}"
            )

            ok = run_single_test(
                M, N, K, lda, ldb, ldc,
                gemm_type, transA, transB, repeat,
                data_type=data_type,
                m_vl=m_vl,
                n_vl=n_vl,
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
                print(f"  data_type={case['data_type']}, m_vl={case['m_vl']}, n_vl={case['n_vl']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
