def ncopy_traits(input_type: str):
    if input_type == "float":
        return {
            "svcnt": "svcntw()",
            "pg_suffix": "b32",
            "offset_type": "svuint32_t",
            "load_ptr_type": "float32_t",
            "vec_type": "svfloat32_t",
            "gather_fn": "svld1_gather_u32offset_f32",
            "load_fn": "svld1_f32",
            "store_fn": "svst1_f32",
            "element_size": "sizeof(float)",
            "scalar_fallback": False,
        }
    if input_type == "__fp16":
        return {
            "svcnt": "svcnth()",
            "pg_suffix": "b16",
            "offset_type": "svuint32_t",
            "load_ptr_type": "float16_t",
            "vec_type": "svfloat16_t",
            "gather_fn": None,
            "load_fn": "svld1_f16",
            "store_fn": "svst1_f16",
            "element_size": "sizeof(__fp16)",
            "scalar_fallback": True,
        }
    if input_type == "__bf16":
        return {
            "svcnt": "svcnth()",
            "pg_suffix": "b16",
            "offset_type": "svuint32_t",
            "load_ptr_type": "bfloat16_t",
            "vec_type": "svbfloat16_t",
            "gather_fn": None,
            "load_fn": "svld1_bf16",
            "store_fn": "svst1_bf16",
            "element_size": "sizeof(__bf16)",
            "scalar_fallback": True,
        }
    raise ValueError(f"Unsupported input_type for ncopy: {input_type}")


def generate_gemm_ncopy(cname: str, input_type: str, ncopy_type: str) -> str:
    if ncopy_type not in ("IN", "ON"):
        raise ValueError(f"Unsupported ncopy_type: {ncopy_type}")

    traits = ncopy_traits(input_type)
    unroll_name = "AUTOGEMM_COPY_UNROLL_M" if ncopy_type == "IN" else "AUTOGEMM_COPY_UNROLL_N"

    if traits["scalar_fallback"]:
        return f"""
static void {cname}(int k, int n, const {input_type} *src, int ldsrc, {input_type} *dst)
{{
    for (int i = 0; i < n; ++i) {{
        const {input_type} *tmpa0 = src + i * ldsrc;
        for (int j = 0; j < k; ++j) {{
            dst[i + j * n] = tmpa0[j];
        }}
    }}
}}
"""

    code = f"""
static void {cname}(int k, int n, const {input_type} *src, int ldsrc, {input_type} *dst)
{{
    const int svcnt = {traits['svcnt']};
    const int eleSz = svcnt * {unroll_name};
    const {traits['load_ptr_type']} *a = reinterpret_cast<const {traits['load_ptr_type']} *>(src);
    {traits['load_ptr_type']} *b = reinterpret_cast<{traits['load_ptr_type']} *>(dst);
    int minI = 0;
    svbool_t pg;
    {traits['vec_type']} packed;
"""
    code += f"    {traits['offset_type']} off = svindex_u32(0, ldsrc * {traits['element_size']});\n"

    code += f"""

    for (int i = 0; i < n; i += minI) {{
        minI = ((eleSz > (n - i)) ? (n - i) : eleSz);
        const {traits['load_ptr_type']} *tmpa0 = a + i * ldsrc;
        for (int j = 0; j < k; ++j) {{
            int is = 0;
            const {traits['load_ptr_type']} *tmpa = tmpa0 + j;
            {traits['load_ptr_type']} *tmpb = b + i + j * n;
            pg = svwhilelt_{traits['pg_suffix']}_s32(is, minI);
            do {{
                uint64_t cnt = svcntp_{traits['pg_suffix']}(svptrue_{traits['pg_suffix']}(), pg);
                packed = {traits['gather_fn']}(pg, tmpa, off);
                {traits['store_fn']}(pg, tmpb, packed);
                tmpa += cnt * ldsrc;
                tmpb += cnt;
                is += cnt;
                pg = svwhilelt_{traits['pg_suffix']}_s32(is, minI);
            }} while (svptest_any(svptrue_{traits['pg_suffix']}(), pg));
        }}
    }}
}}
"""
    return code
