import os
import csv
import argparse
import sys

from test_runner import (
    consume_last_failure_detail,
    pack_label,
    run_range_test,
    run_single_test,
    setup_environment,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional on target machines.
    tqdm = None


current_path = os.path.dirname(os.path.abspath(__file__))
setup_environment()


def _tqdm_enabled():
    return tqdm is not None and sys.stdout.isatty()


def _log(message: str):
    if _tqdm_enabled():
        tqdm.write(message)
    else:
        print(message)

def _range_count(spec: str):
    parts = str(spec).strip().split(":")
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
        raise ValueError(f"invalid range spec: {spec}")
    return ((end - start) // step) + 1


def _expanded_points(tc):
    m_spec = str(tc.get("M", "0"))
    n_spec = str(tc.get("N", "0"))
    k_spec = str(tc.get("K", "0"))
    if not any(":" in value for value in (m_spec, n_spec, k_spec)):
        return 1
    return _range_count(m_spec) * _range_count(n_spec) * _range_count(k_spec)


def _run_mode(testcases, pack_a: bool, pack_b: bool, predict_combo: bool):
    total = len(testcases)
    passed = 0
    failed = 0
    failed_cases = []
    current_pack = pack_label(pack_a, pack_b)
    expanded_total = sum(_expanded_points(tc) for tc in testcases)

    _log(f"\n[INFO] Start running {total} test cases with pack={current_pack}...")
    _log(f"[INFO] Expanded inner tests for pack={current_pack}: {expanded_total}")
    _log("=" * 80)

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
            case_predict_combo = predict_combo and data_type == "bf16"

            _log(f"\n[{idx}/{total}] Running test case:")
            _log(f"  pack={current_pack}, pack_a={pack_a}, pack_b={pack_b}")
            _log(
                f"  M={M_spec}, N={N_spec}, K={K_spec}, "
                f"lda={tc.get('lda', 'auto')}, ldb={tc.get('ldb', 'auto')}, ldc={tc.get('ldc', 'auto')}"
            )
            _log(
                f"  gemm_type={gemm_type}, transA={transA}, transB={transB}, "
                f"data_type={data_type}, m_vl={m_vl}, n_vl={n_vl}"
            )
            if case_predict_combo:
                _log("  selector=enabled (BF16 pack/m_vl/n_vl come from the trained decision tree)")
            if is_range_case:
                _log(f"  expanded_inner_tests={_expanded_points(tc)}")

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
                    pack_a=pack_a,
                    pack_b=pack_b,
                    verbose=True,
                    show_inner_progress=_tqdm_enabled(),
                    predict_combo=case_predict_combo,
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
                    pack_a=pack_a,
                    pack_b=pack_b,
                    verbose=True,
                    predict_combo=case_predict_combo,
                )

            if ok:
                passed += 1
                _log("  [RESULT] PASS")
            else:
                failure_detail = consume_last_failure_detail()
                failed += 1
                failed_cases.append({
                    "index": idx,
                    "pack": current_pack,
                    "pack_a": pack_a,
                    "pack_b": pack_b,
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
                    "failure_stage": (failure_detail or {}).get("stage"),
                    "failure_message": (failure_detail or {}).get("message"),
                })
                _log("  [RESULT] FAIL")
        except Exception as e:
            failed += 1
            failed_cases.append({
                "index": idx,
                "pack": current_pack,
                "pack_a": pack_a,
                "pack_b": pack_b,
                "error": f"Parse/Execute exception: {e}",
                "raw_data": tc,
            })
            _log(f"  [RESULT] FAIL (Exception: {e})")
        _log("-" * 80)

    _log("\n" + "=" * 80)
    _log(f"[TEST SUMMARY] pack={current_pack}, Total: {total}, Passed: {passed}, Failed: {failed}")
    _log("=" * 80)

    if failed > 0:
        _log(f"\n[FAILED CASES DETAIL] (pack={current_pack}, Total: {failed} cases)")
        _log("-" * 80)
        for i, case in enumerate(failed_cases, 1):
            _log(f"\n[{i}] Test case #{case['index']}:")
            _log(f"  pack={case['pack']}, pack_a={case['pack_a']}, pack_b={case['pack_b']}")
            if "error" in case:
                _log(f"  Error: {case['error']}")
                _log(f"  Raw data: {case['raw_data']}")
            else:
                _log(f"  M={case['M']}, N={case['N']}, K={case['K']}")
                _log(f"  lda={case['lda']}, ldb={case['ldb']}, ldc={case['ldc']}")
                _log(f"  gemm_type={case['gemm_type']}, transA={case['transA']}, transB={case['transB']}")
                _log(
                    f"  data_type={case['data_type']}, m_vl={case['m_vl']}, "
                    f"n_vl={case['n_vl']}"
                )
                if case.get("failure_stage"):
                    _log(f"  Failure stage: {case['failure_stage']}")
                if case.get("failure_message"):
                    _log(f"  Failure detail: {case['failure_message']}")
        _log("-" * 80)

    return {
        "pack": current_pack,
        "pack_a": pack_a,
        "pack_b": pack_b,
        "total": total,
        "passed": passed,
        "failed": failed,
        "failed_cases": failed_cases,
    }


def _pack_modes(all_packs: bool, pack_a: bool, pack_b: bool):
    if all_packs:
        return [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
    return [(pack_a, pack_b)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack_a", action="store_true")
    parser.add_argument("--pack_b", action="store_true")
    parser.add_argument("--all_packs", action="store_true")
    parser.add_argument(
        "--predict_combo",
        action="store_true",
        help="For BF16 cases, ignore the CSV pack/tile combination and use the trained selector.",
    )
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
        ok = run_single_test(
            16, 64, 16, 16, 64, 64, "small", "N", "N", 64,
            pack_a=args.pack_a, pack_b=args.pack_b,
            predict_combo=args.predict_combo,
        )
        if ok:
            print("\n[PASS] Default test case passed")
        else:
            print("\n[FAIL] Default test case failed")
        sys.exit(0 if ok else 1)

    with open(testcases_path, "r") as f:
        reader = csv.DictReader(f)
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(missing_columns)}"
            )
        testcases = list(reader)

    mode_results = []
    pack_modes = _pack_modes(args.all_packs, args.pack_a, args.pack_b)
    for pack_a, pack_b in pack_modes:
        mode_results.append(_run_mode(testcases, pack_a, pack_b, args.predict_combo))

    total = sum(mode["total"] for mode in mode_results)
    passed = sum(mode["passed"] for mode in mode_results)
    failed = sum(mode["failed"] for mode in mode_results)

    if len(mode_results) > 1:
        print("\n" + "=" * 80)
        print(f"[OVERALL SUMMARY] modes={len(mode_results)}, Total: {total}, Passed: {passed}, Failed: {failed}")
        print("=" * 80)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
