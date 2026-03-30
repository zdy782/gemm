def tcopy_traits(input_type: str):
    if input_type == "float":
        return {
            "svcnt": "svcntw()",
            "pg_suffix": "b32",
            "load_ptr_type": "float32_t",
            "store_ptr_type": "float32_t",
            "vec_type": "svfloat32_t",
            "load_fn_short": "svld1_f32",
            "load_fn_large": "svldnt1_f32",
            "store_fn": "svst1_f32",
        }
    if input_type == "__fp16":
        return {
            "svcnt": "svcnth()",
            "pg_suffix": "b16",
            "load_ptr_type": "float16_t",
            "store_ptr_type": "float16_t",
            "vec_type": "svfloat16_t",
            "load_fn_short": "svld1_f16",
            "load_fn_large": "svldnt1_f16",
            "store_fn": "svst1_f16",
        }
    if input_type == "__bf16":
        return {
            "svcnt": "svcnth()",
            "pg_suffix": "b16",
            "load_ptr_type": "bfloat16_t",
            "store_ptr_type": "bfloat16_t",
            "vec_type": "svbfloat16_t",
            "load_fn_short": "svld1_bf16",
            "load_fn_large": "svldnt1_bf16",
            "store_fn": "svst1_bf16",
        }
    raise ValueError(f"Unsupported input_type for tcopy: {input_type}")


def generate_gemm_tcopy(cname: str, input_type: str, tcopy_type: str) -> str:
    if tcopy_type not in ("IT", "OT"):
        raise ValueError(f"Unsupported tcopy_type: {tcopy_type}")

    traits = tcopy_traits(input_type)
    unroll_name = "AUTOGEMM_COPY_UNROLL_M" if tcopy_type == "IT" else "AUTOGEMM_COPY_UNROLL_N"

    return f"""
static void {cname}(int k, int n, const {input_type} *src, int ldsrc, {input_type} *dst)
{{
    const int svcnt = {traits['svcnt']};
    const bool large_k = k > 64;
    const int eleSz = svcnt * {unroll_name} * (large_k ? 2 : 1);
    const {traits['load_ptr_type']} *a = reinterpret_cast<const {traits['load_ptr_type']} *>(src);
    {traits['store_ptr_type']} *b = reinterpret_cast<{traits['store_ptr_type']} *>(dst);
    int minI = 0;
    svbool_t pg;
    {traits['vec_type']} packed;

    for (int i = 0; i < n; i += minI) {{
        minI = ((eleSz > (n - i)) ? (n - i) : eleSz);
        const {traits['load_ptr_type']} *tmpa0 = a;
        for (int j = 0; j < k; ++j) {{
            int is = 0;
            const {traits['load_ptr_type']} *tmpa = tmpa0;
            {traits['store_ptr_type']} *tmpb = b + i + j * n;
            pg = svwhilelt_{traits['pg_suffix']}_s32(is, minI);
            do {{
                uint64_t cnt = svcntp_{traits['pg_suffix']}(svptrue_{traits['pg_suffix']}(), pg);
                packed = large_k ? {traits['load_fn_large']}(pg, tmpa) : {traits['load_fn_short']}(pg, tmpa);
                {traits['store_fn']}(pg, tmpb, packed);
                tmpa += cnt;
                tmpb += cnt;
                is += cnt;
                pg = svwhilelt_{traits['pg_suffix']}_s32(is, minI);
            }} while (svptest_any(svptrue_{traits['pg_suffix']}(), pg));
            tmpa0 += ldsrc;
        }}
        a += minI;
    }}
}}
"""
