from laf_asm_code import laf_asm_code
from global_config import *

def generate_sme_asm(
    M: int,
    N: int,
    K: int,
    lda: int,
    ldb: int,
    ldc: int,
    gemm_type: str,
    transA: str,
    transB: str,
    uniq_id: str,
    data_type: str = "fp32",
    m_vl: int = 1,
    n_vl: int = 4,
):
    """Generate .S assembly file for SME GEMM kernel

    Args:
        M (int): M
        N (int): N
        K (int): K
        lda (int): lda
        ldb (int): ldb
        ldc (int): ldc
        gemm_type (str): gemm type, "small" or "general"
        transA (str): transA, "N" or "T"
        transB (str): transB, "N" or "T"
        uniq_id (str): a random 8 chars id to identify the kernel
        data_type (str): data type, "fp32", "bf16", or "fp16"
        m_vl (int): M tile size in units of s-precision VL
        n_vl (int): N tile size in units of s-precision VL

    Returns:
        asm_code (str): generated assembly code
    """
    func_name = f"gemm_{M}x{N}x{K}_{lda}_{ldb}_{ldc}_{uniq_id}"
    
    assert_valid_tile_combo(m_vl, n_vl)
    kernel_code = laf_asm_code(
        gemm_type, transA, transB, func_name, M, N, K, lda, ldb, ldc, data_type, m_vl, n_vl
    )
    
    if not kernel_code:
        return ""
    
    return kernel_code
def generate_sme_test_cpp(
    M: int,
    N: int,
    K: int,
    lda: int,
    ldb: int,
    ldc: int,
    gemm_type: str,
    transA: str,
    transB: str,
    uniq_id: str,
    repeat: int,
    data_type: str = "fp32",
    m_vl: int = 1,
    n_vl: int = 4,
):
    """Generate C++ test file for SME GEMM kernel

    Args:
        M (int): M
        N (int): N
        K (int): K
        lda (int): lda
        ldb (int): ldb
        ldc (int): ldc
        gemm_type (str): gemm type, "small" or "general"
        transA (str): transA, "N" or "T"
        transB (str): transB, "N" or "T"
        uniq_id (str): a random 8 chars id to identify the kernel
        repeat (int): number of repetitions for performance test
        data_type (str): data type, "fp32", "bf16", or "fp16"
        m_vl (int): M tile size in units of s-precision VL
        n_vl (int): N tile size in units of s-precision VL

    Returns:
        cc_code (str): generated C++ test code
    """
    set_data_type(data_type)
    assert_valid_tile_combo(m_vl, n_vl)
    
    if is_bf16():
        input_type = "__bf16"
        output_type = "float"
        input_type_include = "#include <arm_bf16.h>"
        guard_name = "__BGEMM_KERNEL_H"
        tol_val = TOL_BF16
    elif is_fp16():
        input_type = "__fp16"
        output_type = "float"
        input_type_include = ""
        guard_name = "__HGEMM_KERNEL_H"
        tol_val = TOL_FP16
    else:
        input_type = "float"
        output_type = "float"
        input_type_include = ""
        guard_name = "__SGEMM_KERNEL_H"
        tol_val = TOL

    # Column-major storage matrix sizes:
    # - A (transA='N'): M×K, needs K columns × lda = K * lda elements
    # - A (transA='T'): K×M, needs M columns × lda = M * lda elements
    # - B (transB='N'): K×N, needs N columns × ldb = N * ldb elements
    # - B (transB='T'): N×K, needs K columns × ldb = K * ldb elements
    # - C: M×N, needs N columns × ldc = N * ldc elements
    a_cols = K if transA == 'N' else M
    b_cols = N if transB == 'N' else K
    
    a_size = a_cols * lda
    b_size = b_cols * ldb
    c_size = N * ldc

    func_name = f"gemm_{M}x{N}x{K}_{lda}_{ldb}_{ldc}_{uniq_id}"

    cc_code = ""

    # Header
    cc_code += f"""
#ifndef {guard_name}
#define {guard_name}
#endif
"""

    # includes
    cc_code += f"""
#include <cmath>
#include <cstring>
#include <cassert>
#include <arm_neon.h>
{input_type_include}
#include <cstdlib>
#include <cstdio>
#include "test.h"
#include "timer.h"
"""

    # extern declaration
    cc_code += f"""
extern "C" int {func_name}(const {input_type} *A, const {input_type} *B, {output_type} *C, const int lda, const int ldb, const int ldc);
"""

    # test part
    cc_code += f"""
void* _mm_malloc(size_t align, size_t sz)
{{
    void *ptr;
    int alloc_result = posix_memalign(&ptr, align, sz);
    if(alloc_result != 0)
    {{
        return NULL;
    }}
    return ptr;
}}

int main()
{{
    printf("M={M}, N={N}, K={K}, lda={lda}, ldb={ldb}, ldc={ldc}, transA={transA}, transB={transB}, REPEAT={repeat}, data_type={data_type}, m_vl={m_vl}, n_vl={n_vl} ");
    
    #define M {M}
    #define N {N}
    #define K {K}
    #define lda {lda}
    #define ldb {ldb}
    #define ldc {ldc}
    const char transA = '{transA}';
    const char transB = '{transB}';

    // Allocate matrices (column-major storage):
    // A: {a_cols} columns × lda = {a_size} elements
    // B: {b_cols} columns × ldb = {b_size} elements
    // C: N columns × ldc = {c_size} elements
    {input_type} *A = static_cast<{input_type}*>(_mm_malloc(64, {a_size} * sizeof({input_type})));
    {input_type} *B = static_cast<{input_type}*>(_mm_malloc(64, {b_size} * sizeof({input_type})));
    {output_type} *C = static_cast<{output_type}*>(_mm_malloc(64, {c_size} * sizeof({output_type})));
    {output_type} *refC = static_cast<{output_type}*>(_mm_malloc(64, {c_size} * sizeof({output_type})));
    {output_type} *ourC = static_cast<{output_type}*>(_mm_malloc(64, {c_size} * sizeof({output_type})));

    test_utils::init(A, {a_size});
    test_utils::init(B, {b_size});
    test_utils::init(C, {c_size});

    int n_warming = 20;
    int n_loops =64;

    for (int i = 0; i < n_warming; ++i) {{
        {func_name}(A, B, C, lda, ldb, ldc);
    }}

    Timer t;
    for (int i = 0; i < n_loops; ++i) {{
        {func_name}(A, B, C, lda, ldb, ldc);
    }}

    float latency = t.getTime();
    float gflops = M * N * K * 2 / latency * n_loops / 1000000;
    printf("GFLOPS: %.5f ", gflops);

    // Test ACC=false: C = A * B
    bool ACC = false;
    test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC, transA, transB);
    {func_name}(A, B, ourC, lda, ldb, ldc);
    
    if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, {tol_val}, {tol_val})) {{
        int idx = test_utils::diff_index(refC, ourC, M, N, ldc, {tol_val}, {tol_val});
        printf("ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, ACC=%d, ref[%d]=%.6f, our[%d]=%.6f\\n",
            M, N, K, lda, ldb, ldc, ACC, idx, refC[idx], idx, ourC[idx]);
        test_utils::print_diff(refC, ourC, M, N, ldc);
    }} else {{
        printf("0------passed\\n");
    }}
    
    // Test ACC=true: C = A * B + C_init
    for (int i = 0; i < M; ++i) {{
        for (int j = 0; j < N; ++j) {{
            {output_type} c = 10.0f * rand() / RAND_MAX;
            refC[i + j * ldc] = c;
            ourC[i + j * ldc] = c;
        }}
    }}
    
    ACC = true;
    test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC, transA, transB);
    {func_name}(A, B, ourC, lda, ldb, ldc);
    
    if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, {tol_val}, {tol_val})) {{
        int idx = test_utils::diff_index(refC, ourC, M, N, ldc, {tol_val}, {tol_val});
        printf("ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, ACC=%d, ref[%d]=%.6f, our[%d]=%.6f\\n",
            M, N, K, lda, ldb, ldc, ACC, idx, refC[idx], idx, ourC[idx]);
        test_utils::print_diff(refC, ourC, M, N, ldc);
    }} else {{
        printf("1------passed\\n");
    }}

    free(A);
    A=NULL;
    free(B);
    B=NULL;
    free(C);
    C=NULL;
    free(refC);
    refC=NULL;
    free(ourC);
    ourC=NULL;
    
    return 0;
}}
"""
    return cc_code
