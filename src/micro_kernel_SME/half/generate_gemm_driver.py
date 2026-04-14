from .generate_gemm_ncopy import generate_gemm_ncopy
from .generate_gemm_tcopy import generate_gemm_tcopy


GEMM_P = 256
GEMM_Q = 512
GEMM_R = 1024


def precision_types(spec):
    if spec.is_bf16():
        return "__bf16", "float", "#include <arm_bf16.h>\n"
    if spec.is_fp16():
        return "__fp16", "float", ""
    raise ValueError(f"Unsupported precision: {spec.data_type}")


def gen_driver_kernel_call(kernel_func_name, alpha_name, a_ptr, b_ptr, beta_name, c_ptr, lda_name, ldb_name, ldc_name):
    return (
        f"{kernel_func_name}(minI, minJ, minL, {alpha_name}, {a_ptr}, {b_ptr}, {beta_name}, {c_ptr}, "
        f"{lda_name}, {ldb_name}, {ldc_name});"
    )


def generate_gemm_driver(
    spec,
    kernel_func_name: str,
    driver_func_name: str,
    reset_profile_name: str,
    get_profile_name: str,
) -> str:
    input_type, output_type, input_include = precision_types(spec)
    prefix = spec.gemm_prefix()
    incopy_name = f"{prefix}_incopy"
    itcopy_name = f"{prefix}_itcopy"
    oncopy_name = f"{prefix}_oncopy"
    otcopy_name = f"{prefix}_otcopy"

    b_workspace = GEMM_Q * GEMM_R
    a_needs_pack = spec.pack_a
    b_needs_pack = spec.pack_b

    if spec.transA == "N":
        a_pack_call = f"{itcopy_name}(minL, minI, A + is + ls * lda, lda, A_pack);"
        a_pack_after = (
            f"sa = A_pack;\n"
            f"                lda_kernel = minI;\n"
        )
        a_direct_stmt = (
            f"sa = A + is + ls * lda;\n"
            f"                lda_kernel = lda;\n"
        )
    else:
        a_pack_call = f"{incopy_name}(minL, minI, A + ls + is * lda, lda, A_pack);"
        a_pack_after = (
            f"sa = A_pack;\n"
            f"                lda_kernel = minI;\n"
        )
        a_direct_stmt = (
            f"sa = A + ls + is * lda;\n"
            f"                lda_kernel = lda;\n"
        )

    if spec.transB == "N":
        b_pack_call = f"{oncopy_name}(minL, minJ, B + ls + js * ldb, ldb, B_pack);"
        b_pack_after = (
            f"sb = B_pack;\n"
            f"            ldb_kernel = minJ;\n"
        )
        b_direct_stmt = (
            f"sb = B + ls + js * ldb;\n"
            f"            ldb_kernel = ldb;\n"
        )
    else:
        b_pack_call = f"{otcopy_name}(minL, minJ, B + js + ls * ldb, ldb, B_pack);"
        b_pack_after = (
            f"sb = B_pack;\n"
            f"            ldb_kernel = minJ;\n"
        )
        b_direct_stmt = (
            f"sb = B + js + ls * ldb;\n"
            f"            ldb_kernel = ldb;\n"
        )

    if b_needs_pack:
        b_copy_code = (
            "const uint64_t b_pack_start_ns = autogemm_now_ns();\n"
            f"            {b_pack_call}\n"
            "            autogemm_profile.b_pack_ms += autogemm_elapsed_ms(b_pack_start_ns, autogemm_now_ns());\n"
            "            ++autogemm_profile.b_pack_calls;\n"
            f"            {b_pack_after}"
        )
    else:
        b_copy_code = b_direct_stmt

    kernel_call_code = (
        "const float effective_beta = (ls == 0) ? beta : 1.0f;\n"
        "const uint64_t kernel_start_ns = autogemm_now_ns();\n"
        f"                {gen_driver_kernel_call(kernel_func_name, 'alpha', 'sa', 'sb', 'effective_beta', 'C + is + js * ldc', 'lda_kernel', 'ldb_kernel', 'ldc')}\n"
        "                autogemm_profile.kernel_ms += autogemm_elapsed_ms(kernel_start_ns, autogemm_now_ns());\n"
        "                ++autogemm_profile.kernel_calls;\n"
    )

    if a_needs_pack:
        a_prepack_code = (
            "long a_panel_count = 0;\n"
            "        long a_panel_cursor = 0;\n"
            "        for (long is = 0; is < M; ) {\n"
            "            long minI = M - is;\n"
            f"            if (minI > {GEMM_P}) {{\n"
            f"                minI = {GEMM_P};\n"
            "            }\n"
            "            const uint64_t a_pack_start_ns = autogemm_now_ns();\n"
            f"            {a_pack_call.replace('A_pack', 'A_pack + a_panel_cursor')}\n"
            "            autogemm_profile.a_pack_ms += autogemm_elapsed_ms(a_pack_start_ns, autogemm_now_ns());\n"
            "            ++autogemm_profile.a_pack_calls;\n"
            "            a_panel_offsets[a_panel_count] = a_panel_cursor;\n"
            "            a_panel_sizes[a_panel_count] = minI;\n"
            "            ++a_panel_count;\n"
            "            a_panel_cursor += minI * minL;\n"
            "            is += minI;\n"
            "        }\n"
        )
        a_reuse_code = (
            "long a_panel_index = 0;\n"
            "            for (long is = 0; is < M; ) {\n"
            "                long minI = M - is;\n"
            f"                if (minI > {GEMM_P}) {{\n"
            f"                    minI = {GEMM_P};\n"
            "                }\n"
            "                sa = A_pack + a_panel_offsets[a_panel_index];\n"
            "                lda_kernel = a_panel_sizes[a_panel_index];\n"
            "                ++a_panel_index;\n"
            f"                {kernel_call_code}"
            "                is += minI;\n"
            "            }\n"
        )
    else:
        a_copy_code = a_direct_stmt

    code = f"""
#include <stdint.h>
#include <arm_sve.h>
#if defined(__STDC_HOSTED__) && __STDC_HOSTED__
#include <time.h>
#endif
{input_include}

#if defined(__STDC_HOSTED__) && __STDC_HOSTED__
extern "C" void *malloc(unsigned long size);
extern "C" void free(void *ptr);

static inline void *autogemm_alloc_bytes(unsigned long size) {{
    return malloc(size);
}}

static inline void autogemm_free_bytes(void *ptr) {{
    free(ptr);
}}

static inline void autogemm_reset_alloc_state() {{
}}
#else
static unsigned char autogemm_freestanding_heap[64UL * 1024UL * 1024UL];
static unsigned long autogemm_freestanding_heap_cursor = 0;

static inline void *autogemm_alloc_bytes(unsigned long size) {{
    const unsigned long aligned_cursor = (autogemm_freestanding_heap_cursor + 63UL) & ~63UL;
    const unsigned long aligned_size = (size + 63UL) & ~63UL;
    if (aligned_cursor + aligned_size > sizeof(autogemm_freestanding_heap)) {{
        return nullptr;
    }}
    void *ptr = autogemm_freestanding_heap + aligned_cursor;
    autogemm_freestanding_heap_cursor = aligned_cursor + aligned_size;
    return ptr;
}}

static inline void autogemm_free_bytes(void *ptr) {{
    (void)ptr;
}}

static inline void autogemm_reset_alloc_state() {{
    autogemm_freestanding_heap_cursor = 0;
}}
#endif

extern "C" int {kernel_func_name}(const long M, const long N, const long K, const float alpha, const {input_type} *A, const {input_type} *B, const float beta, {output_type} *C, const long lda, const long ldb, const long ldc);

struct AutoGemmProfile {{
    double a_pack_ms;
    double b_pack_ms;
    double kernel_ms;
    double total_ms;
    unsigned long long a_pack_calls;
    unsigned long long b_pack_calls;
    unsigned long long kernel_calls;
}};

static AutoGemmProfile autogemm_profile = {{0.0, 0.0, 0.0, 0.0, 0ULL, 0ULL, 0ULL}};

static inline uint64_t autogemm_now_ns() {{
#if defined(__STDC_HOSTED__) && __STDC_HOSTED__
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
#else
    return 0ULL;
#endif
}}

static inline double autogemm_elapsed_ms(uint64_t start_ns, uint64_t end_ns) {{
    return static_cast<double>(end_ns - start_ns) / 1000000.0;
}}

extern "C" void {reset_profile_name}() {{
    autogemm_profile = {{0.0, 0.0, 0.0, 0.0, 0ULL, 0ULL, 0ULL}};
}}

extern "C" const AutoGemmProfile* {get_profile_name}() {{
    return &autogemm_profile;
}}
"""

    if b_needs_pack:
        code += f"""
alignas(64) static {input_type} B_pack[{b_workspace}];
"""
    if a_needs_pack or b_needs_pack:
        code += f"""
static constexpr int AUTOGEMM_COPY_UNROLL_M = {spec.tile.m_vl};
static constexpr int AUTOGEMM_COPY_UNROLL_N = {spec.tile.n_vl};
"""
    if a_needs_pack:
        if spec.transA == "N":
            code += generate_gemm_tcopy(itcopy_name, input_type, "IT")
        else:
            code += generate_gemm_ncopy(incopy_name, input_type, "IN")
    if b_needs_pack:
        if spec.transB == "N":
            code += generate_gemm_ncopy(oncopy_name, input_type, "ON")
        else:
            code += generate_gemm_tcopy(otcopy_name, input_type, "OT")

    code += f"""
extern "C" int {driver_func_name}(const long M, const long N, const long K, const float alpha, const {input_type} *A, const {input_type} *B, const float beta, {output_type} *C, const long lda, const long ldb, const long ldc)
{{
    const uint64_t total_start_ns = autogemm_now_ns();
    autogemm_reset_alloc_state();
    const {input_type} *sa = nullptr;
    const {input_type} *sb = nullptr;
    long lda_kernel = 0;
    long ldb_kernel = 0;
"""
    cleanup_code = ""
    if a_needs_pack:
        code += f"""
    {input_type} *A_pack = nullptr;
    long *a_panel_offsets = nullptr;
    long *a_panel_sizes = nullptr;
    if (M > 0) {{
        const unsigned long a_pack_elems = static_cast<unsigned long>(M) * static_cast<unsigned long>({GEMM_Q});
        const unsigned long max_a_panels = static_cast<unsigned long>((M + {GEMM_P} - 1) / {GEMM_P});
        A_pack = static_cast<{input_type}*>(autogemm_alloc_bytes(sizeof({input_type}) * a_pack_elems));
        a_panel_offsets = static_cast<long*>(autogemm_alloc_bytes(sizeof(long) * max_a_panels));
        a_panel_sizes = static_cast<long*>(autogemm_alloc_bytes(sizeof(long) * max_a_panels));
        if (A_pack == nullptr || a_panel_offsets == nullptr || a_panel_sizes == nullptr) {{
            autogemm_free_bytes(A_pack);
            autogemm_free_bytes(a_panel_offsets);
            autogemm_free_bytes(a_panel_sizes);
            return -1;
        }}
    }}
"""
        cleanup_code = """
    autogemm_free_bytes(A_pack);
    autogemm_free_bytes(a_panel_offsets);
    autogemm_free_bytes(a_panel_sizes);
"""
    if a_needs_pack:
        code += f"""
    for (long ls = 0; ls < K; ) {{
        long minL = K - ls;
        if (minL > {GEMM_Q}) {{
            minL = {GEMM_Q};
        }}
        {a_prepack_code}
        for (long js = 0; js < N; ) {{
            long minJ = N - js;
            if (minJ > {GEMM_R}) {{
                minJ = {GEMM_R};
            }}
            {b_copy_code}
            {a_reuse_code}
            js += minJ;
        }}
        ls += minL;
    }}
"""
    else:
        code += f"""
    for (long js = 0; js < N; ) {{
        long minJ = N - js;
        if (minJ > {GEMM_R}) {{
            minJ = {GEMM_R};
        }}
        for (long ls = 0; ls < K; ) {{
            long minL = K - ls;
            if (minL > {GEMM_Q}) {{
                minL = {GEMM_Q};
            }}
            {b_copy_code}
            for (long is = 0; is < M; ) {{
                long minI = M - is;
                if (minI > {GEMM_P}) {{
                    minI = {GEMM_P};
                }}
                {a_copy_code}
                {kernel_call_code}
                is += minI;
            }}
            ls += minL;
        }}
        js += minJ;
    }}
"""
    code += f"""
{cleanup_code}
    autogemm_profile.total_ms += autogemm_elapsed_ms(total_start_ns, autogemm_now_ns());
    return 0;
}}
"""
    return code
