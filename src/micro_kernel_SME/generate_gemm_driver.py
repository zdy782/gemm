from generate_gemm_ncopy import generate_gemm_ncopy
from generate_gemm_tcopy import generate_gemm_tcopy


GEMM_P = 256
GEMM_Q = 512
GEMM_R = 1024


def precision_types(spec):
    if spec.is_bf16():
        return "__bf16", "float", "#include <arm_bf16.h>\n"
    if spec.is_fp16():
        return "__fp16", "float", ""
    return "float", "float", ""


def gen_driver_kernel_call(kernel_func_name, a_ptr, b_ptr, c_ptr, lda_name, ldb_name, ldc_name):
    return (
        f"{kernel_func_name}(minI, minJ, minL, {a_ptr}, {b_ptr}, {c_ptr}, "
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

    a_workspace = GEMM_P * GEMM_Q
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

    if a_needs_pack:
        a_copy_code = (
            "const uint64_t a_pack_start_ns = autogemm_now_ns();\n"
            f"                {a_pack_call}\n"
            "                autogemm_profile.a_pack_ms += autogemm_elapsed_ms(a_pack_start_ns, autogemm_now_ns());\n"
            "                ++autogemm_profile.a_pack_calls;\n"
            f"                {a_pack_after}"
        )
    else:
        a_copy_code = a_direct_stmt

    kernel_call_code = (
        "const uint64_t kernel_start_ns = autogemm_now_ns();\n"
        f"                {gen_driver_kernel_call(kernel_func_name, 'sa', 'sb', 'C + is + js * ldc', 'lda_kernel', 'ldb_kernel', 'ldc')}\n"
        "                autogemm_profile.kernel_ms += autogemm_elapsed_ms(kernel_start_ns, autogemm_now_ns());\n"
        "                ++autogemm_profile.kernel_calls;\n"
    )

    code = f"""
#include <stdint.h>
#include <time.h>
#include <arm_sve.h>
{input_include}

extern "C" int {kernel_func_name}(const long M, const long N, const long K, const {input_type} *A, const {input_type} *B, {output_type} *C, const long lda, const long ldb, const long ldc);

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
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
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

    if a_needs_pack:
        code += f"""
alignas(64) static {input_type} A_pack[{a_workspace}];
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
        code += generate_gemm_ncopy(incopy_name, input_type, "IN")
        code += generate_gemm_tcopy(itcopy_name, input_type, "IT")
    if b_needs_pack:
        code += generate_gemm_ncopy(oncopy_name, input_type, "ON")
        code += generate_gemm_tcopy(otcopy_name, input_type, "OT")

    code += f"""
extern "C" int {driver_func_name}(const long M, const long N, const long K, const {input_type} *A, const {input_type} *B, {output_type} *C, const long lda, const long ldb, const long ldc)
{{
    const uint64_t total_start_ns = autogemm_now_ns();
    const {input_type} *sa = nullptr;
    const {input_type} *sb = nullptr;
    long lda_kernel = 0;
    long ldb_kernel = 0;

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
    autogemm_profile.total_ms += autogemm_elapsed_ms(total_start_ns, autogemm_now_ns());
    return 0;
}}
"""
    return code
