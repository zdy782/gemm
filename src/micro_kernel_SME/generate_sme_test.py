from gemm_config import resolve_model
from generate_gemm_driver import generate_gemm_driver
from global_config import assert_valid_tile_combo, get_tolerance_value
from laf_asm_code import laf_asm_code
from model_spec import GenerationContext, KernelSpec
from register_plan import DEFAULT_REGISTER_PLAN


def build_symbol_names(spec: KernelSpec, uniq_id: str):
    trans_suffix = f"{spec.transA.lower()}{spec.transB.lower()}"
    tile_suffix = f"{spec.tile.m_vl}x{spec.tile.n_vl}"
    suffix = f"{trans_suffix}_{tile_suffix}_{uniq_id}"
    prefix = spec.gemm_prefix()
    return (
        f"{prefix}_kernel_{suffix}",
        f"{prefix}_{spec.pack_suffix()}_{suffix}",
    )


def build_profile_symbol_names(uniq_id: str):
    return (
        f"reset_profile_{uniq_id}",
        f"get_profile_{uniq_id}",
    )


def test_cpp_types(spec: KernelSpec):
    if spec.is_bf16():
        return "__bf16", "float", "#include <arm_bf16.h>", "__BGEMM_KERNEL_H", get_tolerance_value(spec)
    if spec.is_fp16():
        return "__fp16", "float", "", "__HGEMM_KERNEL_H", get_tolerance_value(spec)
    raise ValueError(f"Unsupported precision: {spec.data_type}")


def test_cpp_prelude(guard_name: str, input_type_include: str) -> str:
    return f"""
#ifndef {guard_name}
#define {guard_name}
#endif

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
    data_type: str = "bf16",
    m_vl: int = 1,
    n_vl: int = 4,
    pack_a: bool = False,
    pack_b: bool = False,
):
    """Generate the final `.S` text for one concrete SME kernel.

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
        data_type (str): data type, "bf16" or "fp16"
        m_vl (int): M tile size in units of s-precision VL
        n_vl (int): N tile size in units of s-precision VL

    Returns:
        asm_code (str): generated assembly code
    """
    # The generator pipeline is:
    # 1. normalize CLI-like inputs into an immutable spec
    # 2. resolve the transpose-specific load model
    # 3. hand the context to `laf_asm_code`, which generates the loop nest and
    #    per-tile kernels
    spec = KernelSpec.from_args(
        M,
        N,
        K,
        lda,
        ldb,
        ldc,
        gemm_type,
        transA,
        transB,
        data_type,
        m_vl,
        n_vl,
        pack_a,
        pack_b,
    )
    assert_valid_tile_combo(spec.tile.m_vl, spec.tile.n_vl)
    func_name, _ = build_symbol_names(spec, uniq_id)
    kernel_spec = spec.kernel_view_spec()
    ctx = GenerationContext(
        spec=kernel_spec,
        registers=DEFAULT_REGISTER_PLAN,
        model=resolve_model(kernel_spec),
    )
    kernel_code = laf_asm_code(ctx, func_name)
    
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
    data_type: str = "bf16",
    m_vl: int = 1,
    n_vl: int = 4,
    pack_a: bool = False,
    pack_b: bool = False,
    profile_pack: bool = False,
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
        data_type (str): data type, "bf16" or "fp16"
        m_vl (int): M tile size in units of s-precision VL
        n_vl (int): N tile size in units of s-precision VL

    Returns:
        cc_code (str): generated C++ test code
    """
    spec = KernelSpec.from_args(
        M,
        N,
        K,
        lda,
        ldb,
        ldc,
        gemm_type,
        transA,
        transB,
        data_type,
        m_vl,
        n_vl,
        pack_a,
        pack_b,
    )
    assert_valid_tile_combo(spec.tile.m_vl, spec.tile.n_vl)

    input_type, output_type, input_type_include, guard_name, tol_val = test_cpp_types(spec)

    # Column-major storage matrix sizes:
    # - A (transA='N'): M×K, needs K columns × lda = K * lda elements
    # - A (transA='T'): K×M, needs M columns × lda = M * lda elements
    # - B (transB='N'): K×N, needs N columns × ldb = N * ldb elements
    # - B (transB='T'): N×K, needs K columns × ldb = K * ldb elements
    # - C: M×N, needs N columns × ldc = N * ldc elements
    a_cols = K if spec.transA == 'N' else M
    b_cols = N if spec.transB == 'N' else K
    
    a_size = a_cols * lda
    b_size = b_cols * ldb
    c_size = N * ldc

    _, entry_func_name = build_symbol_names(spec, uniq_id)
    reset_profile_name, get_profile_name = build_profile_symbol_names(uniq_id)

    cc_code = test_cpp_prelude(guard_name, input_type_include)

    # extern declaration
    cc_code += f"""
extern "C" int {entry_func_name}(const long M, const long N, const long K, const float alpha, const {input_type} *A, const {input_type} *B, const float beta, {output_type} *C, const long lda, const long ldb, const long ldc);
"""
    if profile_pack:
        cc_code += f"""
struct AutoGemmProfile {{
    double a_pack_ms;
    double b_pack_ms;
    double kernel_ms;
    double total_ms;
    unsigned long long a_pack_calls;
    unsigned long long b_pack_calls;
    unsigned long long kernel_calls;
}};

extern "C" void {reset_profile_name}();
extern "C" const AutoGemmProfile* {get_profile_name}();
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
    printf("M={M}, N={N}, K={K}, lda={lda}, ldb={ldb}, ldc={ldc}, transA={transA}, transB={transB}, REPEAT={repeat}, data_type={data_type}, m_vl={m_vl}, n_vl={n_vl}, pack_a={str(pack_a).lower()}, pack_b={str(pack_b).lower()} ");
    
    #define M {M}
    #define N {N}
    #define K {K}
    #define lda {lda}
    #define ldb {ldb}
    #define ldc {ldc}
    const char transA = '{spec.transA}';
    const char transB = '{spec.transB}';

    // Allocate matrices (column-major storage):
    // A: {a_cols} columns × lda = {a_size} elements
    // B: {b_cols} columns × ldb = {b_size} elements
    // C: N columns × ldc = {c_size} elements
    {input_type} *A = static_cast<{input_type}*>(_mm_malloc(64, {a_size} * sizeof({input_type})));
    {input_type} *B = static_cast<{input_type}*>(_mm_malloc(64, {b_size} * sizeof({input_type})));
    {output_type} *C = static_cast<{output_type}*>(_mm_malloc(64, {c_size} * sizeof({output_type})));
    {output_type} *refC = static_cast<{output_type}*>(_mm_malloc(64, {c_size} * sizeof({output_type})));
    {output_type} *ourC = static_cast<{output_type}*>(_mm_malloc(64, {c_size} * sizeof({output_type})));
    bool all_passed = true;
    const float alpha_bench = 1.0f;
    const float beta_bench = 1.0f;

    struct ScaleCase {{
        float alpha;
        float beta;
        const char* label;
    }};
    const ScaleCase scale_cases[] = {{
        {{1.0f, 1.0f, "alpha=1,beta=1"}},
        {{1.0f, 0.0f, "alpha=1,beta=0"}},
        {{0.5f, 1.0f, "alpha=0.5,beta=1"}},
        {{0.5f, 0.25f, "alpha=0.5,beta=0.25"}},
    }};

    test_utils::init(A, {a_size});
    test_utils::init(B, {b_size});
    test_utils::init(C, {c_size});

    int n_warming = 20;
    int n_loops =64;

    for (int i = 0; i < n_warming; ++i) {{
        {entry_func_name}(M, N, K, alpha_bench, A, B, beta_bench, C, lda, ldb, ldc);
    }}

"""
    if profile_pack:
        cc_code += f"""
    {reset_profile_name}();
"""
    cc_code += f"""

    Timer t;
    for (int i = 0; i < n_loops; ++i) {{
        {entry_func_name}(M, N, K, alpha_bench, A, B, beta_bench, C, lda, ldb, ldc);
    }}

    float latency = t.getTime();
    float gflops = M * N * K * 2 / latency * n_loops / 1000000;
    printf("GFLOPS: %.5f ", gflops);
"""
    if profile_pack:
        cc_code += f"""
    const AutoGemmProfile* profile = {get_profile_name}();
    const double pack_ms = profile->a_pack_ms + profile->b_pack_ms;
    const double inv_total_pct = profile->total_ms > 0.0 ? (100.0 / profile->total_ms) : 0.0;
    printf("PROFILE: total_ms=%.5f, a_pack_ms=%.5f, b_pack_ms=%.5f, pack_ms=%.5f, kernel_ms=%.5f, a_pack_pct=%.2f, b_pack_pct=%.2f, pack_pct=%.2f, kernel_pct=%.2f ",
        profile->total_ms,
        profile->a_pack_ms,
        profile->b_pack_ms,
        pack_ms,
        profile->kernel_ms,
        profile->a_pack_ms * inv_total_pct,
        profile->b_pack_ms * inv_total_pct,
        pack_ms * inv_total_pct,
        profile->kernel_ms * inv_total_pct);
"""
    cc_code += f"""
    for (const ScaleCase& scale_case : scale_cases) {{
        test_utils::init(C, {c_size});
        memcpy(refC, C, {c_size} * sizeof({output_type}));
        memcpy(ourC, C, {c_size} * sizeof({output_type}));

        test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, scale_case.alpha, scale_case.beta, transA, transB);
        {entry_func_name}(M, N, K, scale_case.alpha, A, B, scale_case.beta, ourC, lda, ldb, ldc);

        if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, {tol_val}, {tol_val})) {{
            int idx = test_utils::diff_index(refC, ourC, M, N, ldc, {tol_val}, {tol_val});
            printf(
                "ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, alpha=%.6f, beta=%.6f, ref[%d]=%.6f, our[%d]=%.6f\\n",
                M, N, K, lda, ldb, ldc, scale_case.alpha, scale_case.beta, idx, refC[idx], idx, ourC[idx]
            );
            test_utils::print_diff(refC, ourC, M, N, ldc);
            all_passed = false;
        }} else {{
            printf("%s------passed\\n", scale_case.label);
        }}
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
    
    return all_passed ? 0 : 1;
}}
"""
    return cc_code


def generate_sme_range_test_cpp(
    M_start: int,
    M_end: int,
    M_step: int,
    N_start: int,
    N_end: int,
    N_step: int,
    K_start: int,
    K_end: int,
    K_step: int,
    lda_spec,
    ldb_spec,
    ldc_spec,
    lda_gen: int,
    ldb_gen: int,
    ldc_gen: int,
    gemm_type: str,
    transA: str,
    transB: str,
    uniq_id: str,
    repeat: int,
    data_type: str = "bf16",
    m_vl: int = 1,
    n_vl: int = 4,
    pack_a: bool = False,
    pack_b: bool = False,
    emit_progress_markers: bool = False,
):
    spec = KernelSpec.from_args(
        M_end,
        N_end,
        K_end,
        lda_gen,
        ldb_gen,
        ldc_gen,
        gemm_type,
        transA,
        transB,
        data_type,
        m_vl,
        n_vl,
        pack_a,
        pack_b,
    )
    assert_valid_tile_combo(spec.tile.m_vl, spec.tile.n_vl)

    input_type, output_type, input_type_include, guard_name, tol_val = test_cpp_types(spec)
    _, entry_func_name = build_symbol_names(spec, uniq_id)

    lda_spec_text = str(lda_spec).strip()
    ldb_spec_text = str(ldb_spec).strip()
    ldc_spec_text = str(ldc_spec).strip()
    lda_is_auto = lda_spec_text.lower() in {"", "auto", "none"}
    ldb_is_auto = ldb_spec_text.lower() in {"", "auto", "none"}
    ldc_is_auto = ldc_spec_text.lower() in {"", "auto", "none"}
    lda_fixed = 0 if lda_is_auto else int(lda_spec_text)
    ldb_fixed = 0 if ldb_is_auto else int(ldb_spec_text)
    ldc_fixed = 0 if ldc_is_auto else int(ldc_spec_text)

    max_lda = lda_gen
    max_ldb = ldb_gen
    max_ldc = ldc_gen
    max_a_size = (K_end if spec.transA == "N" else M_end) * max_lda
    max_b_size = (N_end if spec.transB == "N" else K_end) * max_ldb
    max_c_size = N_end * max_ldc

    cc_code = test_cpp_prelude(guard_name, input_type_include)
    cc_code += f"""
extern "C" int {entry_func_name}(const long M, const long N, const long K, const float alpha, const {input_type} *A, const {input_type} *B, const float beta, {output_type} *C, const long lda, const long ldb, const long ldc);

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
    const char transA = '{spec.transA}';
    const char transB = '{spec.transB}';
    const int m_start = {M_start};
    const int m_end = {M_end};
    const int m_step = {M_step};
    const int n_start = {N_start};
    const int n_end = {N_end};
    const int n_step = {N_step};
    const int k_start = {K_start};
    const int k_end = {K_end};
    const int k_step = {K_step};
    const bool lda_auto = {"true" if lda_is_auto else "false"};
    const bool ldb_auto = {"true" if ldb_is_auto else "false"};
    const bool ldc_auto = {"true" if ldc_is_auto else "false"};
    const bool emit_progress_markers = {"true" if emit_progress_markers else "false"};
    const int lda_fixed = {lda_fixed};
    const int ldb_fixed = {ldb_fixed};
    const int ldc_fixed = {ldc_fixed};
    const bool perf_flag = false;
    (void)perf_flag;
    (void){repeat};
    struct ScaleCase {{
        float alpha;
        float beta;
        const char* label;
    }};
    const ScaleCase scale_cases[] = {{
        {{1.0f, 1.0f, "alpha=1,beta=1"}},
        {{1.0f, 0.0f, "alpha=1,beta=0"}},
        {{0.5f, 0.25f, "alpha=0.5,beta=0.25"}},
    }};

    if (!emit_progress_markers) {{
        printf("M=%d:%d:%d, N=%d:%d:%d, K=%d:%d:%d, transA=%c, transB=%c, data_type={data_type}, m_vl={m_vl}, n_vl={n_vl}, pack_a={str(pack_a).lower()}, pack_b={str(pack_b).lower()}\\n",
            m_start, m_end, m_step, n_start, n_end, n_step, k_start, k_end, k_step, transA, transB);
    }}

    const int total_tests =
        ((m_end - m_start) / m_step + 1) *
        ((n_end - n_start) / n_step + 1) *
        ((k_end - k_start) / k_step + 1);
    const int progress_stride =
        emit_progress_markers ? ((total_tests <= 100) ? 1 : ((total_tests + 99) / 100)) : total_tests;

    if (!emit_progress_markers) {{
        printf("---------------------------------------------------------------------------------------\\n");
        printf("%12s %12s %12s %14s %14s %14s\\n", "M", "N", "K", "lda", "ldb", "status");
        printf("---------------------------------------------------------------------------------------\\n");
    }}

    {input_type} *A = static_cast<{input_type}*>(_mm_malloc(64, {max_a_size} * sizeof({input_type})));
    {input_type} *B = static_cast<{input_type}*>(_mm_malloc(64, {max_b_size} * sizeof({input_type})));
    {output_type} *C = static_cast<{output_type}*>(_mm_malloc(64, {max_c_size} * sizeof({output_type})));
    {output_type} *refC = static_cast<{output_type}*>(_mm_malloc(64, {max_c_size} * sizeof({output_type})));
    {output_type} *ourC = static_cast<{output_type}*>(_mm_malloc(64, {max_c_size} * sizeof({output_type})));

    if (!A || !B || !C || !refC || !ourC) {{
        printf("ERROR: memory allocation failed\\n");
        free(A);
        free(B);
        free(C);
        free(refC);
        free(ourC);
        return 1;
    }}

    int passed_tests = 0;
    int failed_tests = 0;
    int completed_tests = 0;

    for (int M = m_start; M <= m_end; M += m_step) {{
        for (int N = n_start; N <= n_end; N += n_step) {{
            for (int K = k_start; K <= k_end; K += k_step) {{
                const int lda_required = (transA == 'N') ? M : K;
                const int ldb_required = (transB == 'N') ? K : N;
                const int ldc_required = M;
                const int lda = lda_auto ? lda_required : lda_fixed;
                const int ldb = ldb_auto ? ldb_required : ldb_fixed;
                const int ldc = ldc_auto ? ldc_required : ldc_fixed;

                if (lda < lda_required || ldb < ldb_required || ldc < ldc_required) {{
                    if (!emit_progress_markers) {{
                        printf("%12d %12d %12d %14d %14d %14s\\n", M, N, K, lda, ldb, "FAILED");
                    }}
                    printf("ERROR: invalid stride for M=%d, N=%d, K=%d, lda=%d (< %d), ldb=%d (< %d), ldc=%d (< %d)\\n",
                        M, N, K, lda, lda_required, ldb, ldb_required, ldc, ldc_required);
                    free(A);
                    free(B);
                    free(C);
                    free(refC);
                    free(ourC);
                    return 1;
                }}

                const size_t a_cols = (transA == 'N') ? K : M;
                const size_t b_cols = (transB == 'N') ? N : K;
                const size_t a_size = a_cols * lda;
                const size_t b_size = b_cols * ldb;
                const size_t c_size = static_cast<size_t>(N) * ldc;

                test_utils::init(A, a_size);
                test_utils::init(B, b_size);
                for (size_t idx = 0; idx < c_size; ++idx) {{
                    refC[idx] = 0;
                    ourC[idx] = 0;
                }}

                bool case_passed = true;
                for (const ScaleCase& scale_case : scale_cases) {{
                    test_utils::init(C, c_size);
                    memcpy(refC, C, c_size * sizeof({output_type}));
                    memcpy(ourC, C, c_size * sizeof({output_type}));

                    test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, scale_case.alpha, scale_case.beta, transA, transB);
                    {entry_func_name}(M, N, K, scale_case.alpha, A, B, scale_case.beta, ourC, lda, ldb, ldc);

                    if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, {tol_val}, {tol_val})) {{
                        ++failed_tests;
                        case_passed = false;
                        const int idx = test_utils::diff_index(refC, ourC, M, N, ldc, {tol_val}, {tol_val});
                        if (!emit_progress_markers) {{
                            printf("%12d %12d %12d %14d %14d %14s\\n", M, N, K, lda, ldb, "FAILED");
                        }}
                        printf("ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, alpha=%.6f, beta=%.6f, ref[%d]=%.6f, our[%d]=%.6f\\n",
                            M, N, K, lda, ldb, ldc, scale_case.alpha, scale_case.beta, idx, refC[idx], idx, ourC[idx]);
                        test_utils::print_diff(refC, ourC, M, N, ldc);
                        free(A);
                        free(B);
                        free(C);
                        free(refC);
                        free(ourC);
                        return 1;
                    }}
                }}

                if (case_passed) {{
                    ++passed_tests;
                    if (!emit_progress_markers) {{
                        printf("%12d %12d %12d %14d %14d %14s\\n", M, N, K, lda, ldb, "PASSED");
                    }}
                }}

                ++completed_tests;
                if (emit_progress_markers && ((completed_tests % progress_stride) == 0 || completed_tests == total_tests)) {{
                    printf("__AUTOGEMM_PROGRESS__ %d %d\\n", completed_tests, total_tests);
                    fflush(stdout);
                }}
            }}
        }}
    }}

    if (!emit_progress_markers) {{
        printf("---------------------------------------------------------------------------------------\\n");
    }}
    printf("tests=%d passed=%d failed=%d\\n", total_tests, passed_tests, failed_tests);

    free(A);
    free(B);
    free(C);
    free(refC);
    free(ourC);
    return (failed_tests == 0) ? 0 : 1;
}}
"""
    return cc_code


def generate_sme_driver_cpp(
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
    data_type: str = "bf16",
    m_vl: int = 1,
    n_vl: int = 4,
    pack_a: bool = False,
    pack_b: bool = False,
):
    spec = KernelSpec.from_args(
        M,
        N,
        K,
        lda,
        ldb,
        ldc,
        gemm_type,
        transA,
        transB,
        data_type,
        m_vl,
        n_vl,
        pack_a,
        pack_b,
    )
    assert_valid_tile_combo(spec.tile.m_vl, spec.tile.n_vl)
    kernel_func_name, driver_func_name = build_symbol_names(spec, uniq_id)
    reset_profile_name, get_profile_name = build_profile_symbol_names(uniq_id)
    return generate_gemm_driver(
        spec,
        kernel_func_name,
        driver_func_name,
        reset_profile_name,
        get_profile_name,
    )
