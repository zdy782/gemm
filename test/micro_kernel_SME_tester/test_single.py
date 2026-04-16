import sys
import argparse

from test_runner import run_single_test

parser = argparse.ArgumentParser()
parser.add_argument("--M", type=int, default=16)
parser.add_argument("--N", type=int, default=64)
parser.add_argument("--K", type=int, default=16)
parser.add_argument("--lda", type=int, default=None)
parser.add_argument("--ldb", type=int, default=None)
parser.add_argument("--ldc", type=int, default=None)
parser.add_argument("--gemm_type", type=str, default="small")
parser.add_argument("--transA", type=str, default="N")
parser.add_argument("--transB", type=str, default="N")
parser.add_argument("--REPEAT", type=int, default=64)
parser.add_argument("--data_type", type=str, default="bf16", choices=["bf16", "fp16"])
parser.add_argument("--m_vl", type=int, default=argparse.SUPPRESS, choices=[1, 2, 3, 4])
parser.add_argument("--n_vl", type=int, default=argparse.SUPPRESS, choices=[1, 2, 3, 4])
parser.add_argument("--pack_a", action="store_true", default=argparse.SUPPRESS)
parser.add_argument("--pack_b", action="store_true", default=argparse.SUPPRESS)
parser.add_argument("--profile_pack", action="store_true")
parser.add_argument("--perf_only", action="store_true")
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--beta", type=float, default=None)

args = parser.parse_args()
explicit_combo = any(
    hasattr(args, name)
    for name in ("m_vl", "n_vl", "pack_a", "pack_b")
)
m_vl = getattr(args, "m_vl", 1)
n_vl = getattr(args, "n_vl", 4)
pack_a = getattr(args, "pack_a", False)
pack_b = getattr(args, "pack_b", False)

if m_vl * n_vl > 4:
    parser.error("Only tile combos with m_vl * n_vl <= 4 are supported")

M = args.M
N = args.N
K = args.K

lda = args.lda if args.lda is not None else (M if args.transA == 'N' else K)
ldb = args.ldb if args.ldb is not None else (K if args.transB == 'N' else N)
ldc = args.ldc if args.ldc is not None else M

success = run_single_test(
    M=M, N=N, K=K,
    lda=lda, ldb=ldb, ldc=ldc,
    gemm_type=args.gemm_type,
    transA=args.transA,
    transB=args.transB,
    repeat=args.REPEAT,
    data_type=args.data_type,
    m_vl=m_vl,
    n_vl=n_vl,
    pack_a=pack_a,
    pack_b=pack_b,
    profile_pack=args.profile_pack,
    validate_results=not args.perf_only,
    alpha=args.alpha,
    beta=args.beta,
    verbose=True,
    predict_combo=args.data_type == "bf16" and not explicit_combo,
)

sys.exit(0 if success else 1)
