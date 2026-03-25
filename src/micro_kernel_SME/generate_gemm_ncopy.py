def _ncopy_traits(input_type: str):
    if input_type == "float":
        return {
            "svcnt": "svcntw()",
            "pg_suffix": "b32",
            "load_ptr_type": "float32_t",
            "vec_type": "svfloat32_t",
            "load_fn": "svldnt1_f32",
            "store_fn": "svst1_f32",
        }
    if input_type == "__fp16":
        return {
            "svcnt": "svcnth()",
            "pg_suffix": "b16",
            "load_ptr_type": "float16_t",
            "vec_type": "svfloat16_t",
            "load_fn": "svldnt1_f16",
            "store_fn": "svst1_f16",
        }
    if input_type == "__bf16":
        return {
            "svcnt": "svcnth()",
            "pg_suffix": "b16",
            "load_ptr_type": "bfloat16_t",
            "vec_type": "svbfloat16_t",
            "load_fn": "svldnt1_bf16",
            "store_fn": "svst1_bf16",
        }
    raise ValueError(f"Unsupported input_type for ncopy: {input_type}")


def generate_gemm_ncopy(cname: str, input_type: str, ncopy_type: str) -> str:
    if ncopy_type not in ("IN", "ON"):
        raise ValueError(f"Unsupported ncopy_type: {ncopy_type}")

    traits = _ncopy_traits(input_type)
    unroll_name = "AUTOGEMM_COPY_UNROLL_M" if ncopy_type == "IN" else "AUTOGEMM_COPY_UNROLL_N"

    return f"""
static void {cname}(int k, int n, const {input_type} *src, int ldsrc, {input_type} *dst)
{{
    const int svcnt = {traits['svcnt']};
    const int eleSz = svcnt * {unroll_name};
    const {traits['load_ptr_type']} *a = reinterpret_cast<const {traits['load_ptr_type']} *>(src);
    int minJ = 0;
    svbool_t pg;
    {traits['vec_type']} packed;

    for (int i = 0; i < n; ++i) {{
        const {traits['load_ptr_type']} *tmpa_row = a + i * ldsrc;
        for (int j = 0; j < k; j += minJ) {{
            minJ = ((eleSz > (k - j)) ? (k - j) : eleSz);
            int js = 0;
            const {traits['load_ptr_type']} *tmpa = tmpa_row + j;
            pg = svwhilelt_{traits['pg_suffix']}_s32(js, minJ);
            do {{
                uint64_t cnt = svcntp_{traits['pg_suffix']}(svptrue_{traits['pg_suffix']}(), pg);
                packed = {traits['load_fn']}(pg, tmpa);
                {traits['store_fn']}(pg, dst, packed);
                dst += cnt;
                tmpa += cnt;
                js += cnt;
                pg = svwhilelt_{traits['pg_suffix']}_s32(js, minJ);
            }} while (svptest_any(svptrue_{traits['pg_suffix']}(), pg));
        }}
    }}
}}
"""
