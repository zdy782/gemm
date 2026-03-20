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
parser.add_argument("--data_type", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])

args = parser.parse_args()

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
    verbose=True
)

sys.exit(0 if success else 1)
