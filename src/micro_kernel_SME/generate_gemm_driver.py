from generate_gemm_ncopy import generate_gemm_ncopy
from generate_gemm_tcopy import generate_gemm_tcopy


GEMM_P = 256
GEMM_Q = 512
GEMM_R = 1024


def _precision_types(spec):
    if spec.is_bf16():
        return "__bf16", "float", "#include <arm_bf16.h>\n"
    if spec.is_fp16():
        return "__fp16", "float", ""
    return "float", "float", ""


def _driver_kernel_call(spec, kernel_func_name, a_ptr, b_ptr, c_ptr, lda_name, ldb_name, ldc_name):
    return (
        f"{kernel_func_name}(minI, minJ, minL, {a_ptr}, {b_ptr}, {c_ptr}, "
        f"{lda_name}, {ldb_name}, {ldc_name});"
    )


def generate_gemm_driver(spec, kernel_func_name: str, driver_func_name: str) -> str:
    input_type, output_type, input_include = _precision_types(spec)

    prefix = spec.gemm_prefix()
    incopy_name = f"{prefix}_incopy"
    itcopy_name = f"{prefix}_itcopy"
    oncopy_name = f"{prefix}_oncopy"
    otcopy_name = f"{prefix}_otcopy"

    a_workspace = GEMM_P * GEMM_Q
    b_workspace = GEMM_Q * GEMM_R

    if spec.transA == "N":
        a_pack_stmt = (
            f"{itcopy_name}(minL, minI, A + is + ls * lda, lda, A_pack);\n"
            f"                sa = A_pack;\n"
            f"                lda_kernel = minI;\n"
        )
        a_direct_stmt = (
            f"sa = A + is + ls * lda;\n"
            f"                lda_kernel = lda;\n"
        )
    else:
        a_pack_stmt = (
            f"{incopy_name}(minL, minI, A + ls + is * lda, lda, A_pack);\n"
            f"                sa = A_pack;\n"
            f"                lda_kernel = minL;\n"
        )
        a_direct_stmt = (
            f"sa = A + ls + is * lda;\n"
            f"                lda_kernel = lda;\n"
        )

    if spec.transB == "N":
        b_pack_stmt = (
            f"{oncopy_name}(minL, minJ, B + ls + js * ldb, ldb, B_pack);\n"
            f"            sb = B_pack;\n"
            f"            ldb_kernel = minL;\n"
        )
        b_direct_stmt = (
            f"sb = B + ls + js * ldb;\n"
            f"            ldb_kernel = ldb;\n"
        )
    else:
        b_pack_stmt = (
            f"{otcopy_name}(minL, minJ, B + js + ls * ldb, ldb, B_pack);\n"
            f"            sb = B_pack;\n"
            f"            ldb_kernel = minJ;\n"
        )
        b_direct_stmt = (
            f"sb = B + js + ls * ldb;\n"
            f"            ldb_kernel = ldb;\n"
        )

    b_copy_code = b_pack_stmt if spec.is_packed() else b_direct_stmt
    a_copy_code = a_pack_stmt if spec.is_packed() else a_direct_stmt

    code = f"""
#include <arm_sve.h>
{input_include}

extern "C" int {kernel_func_name}(const long M, const long N, const long K, const {input_type} *A, const {input_type} *B, {output_type} *C, const long lda, const long ldb, const long ldc);
"""

    if spec.is_packed():
        code += f"""
alignas(64) static {input_type} A_pack[{a_workspace}];
alignas(64) static {input_type} B_pack[{b_workspace}];
static constexpr int AUTOGEMM_COPY_UNROLL_M = {spec.tile.m_vl};
static constexpr int AUTOGEMM_COPY_UNROLL_N = {spec.tile.n_vl};
"""
        code += generate_gemm_ncopy(incopy_name, input_type, "IN")
        code += generate_gemm_tcopy(itcopy_name, input_type, "IT")
        code += generate_gemm_ncopy(oncopy_name, input_type, "ON")
        code += generate_gemm_tcopy(otcopy_name, input_type, "OT")

    code += f"""
extern "C" int {driver_func_name}(const long M, const long N, const long K, const {input_type} *A, const {input_type} *B, {output_type} *C, const long lda, const long ldb, const long ldc)
{{
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
                {_driver_kernel_call(spec, kernel_func_name, "sa", "sb", "C + is + js * ldc", "lda_kernel", "ldb_kernel", "ldc")}
                is += minI;
            }}
            ls += minL;
        }}
        js += minJ;
    }}
    return 0;
}}
"""
    return code
