def ncopy_traits(input_type: str):
    if input_type == "float":
        return {
            "svcnt": "svcntw()",
            "pg_suffix": "b32",
            "offset_type": "svuint32_t",
            "load_ptr_type": "float32_t",
            "vec_type": "svfloat32_t",
            "gather_fn": "svld1_gather_u32offset_f32",
            "store_fn": "svst1_f32",
            "element_size": "sizeof(float)",
            "ext_panel_copy": False,
        }
    if input_type == "__fp16":
        return {
            "svcnt": "svcnth()",
            "pg_suffix": "b16",
            "offset_type": "svuint32_t",
            "load_ptr_type": "float16_t",
            "vec_type": "svfloat16_t",
            "raw_vec_type": "svuint16_t",
            "load_fn": "svld1_f16",
            "scatter_store_fn": "svst1_scatter_u32offset_u32",
            "element_size": "sizeof(__fp16)",
            "ext_panel_copy": True,
        }
    if input_type == "__bf16":
        return {
            "svcnt": "svcnth()",
            "pg_suffix": "b16",
            "offset_type": "svuint32_t",
            "load_ptr_type": "bfloat16_t",
            "vec_type": "svbfloat16_t",
            "raw_vec_type": "svuint16_t",
            "load_fn": "svld1_bf16",
            "scatter_store_fn": "svst1_scatter_u32offset_u32",
            "element_size": "sizeof(__bf16)",
            "ext_panel_copy": True,
        }
    raise ValueError(f"Unsupported input_type for ncopy: {input_type}")


def generate_gemm_ncopy(cname: str, input_type: str, ncopy_type: str) -> str:
    if ncopy_type not in ("IN", "ON"):
        raise ValueError(f"Unsupported ncopy_type: {ncopy_type}")

    traits = ncopy_traits(input_type)
    unroll_name = "AUTOGEMM_COPY_UNROLL_M" if ncopy_type == "IN" else "AUTOGEMM_COPY_UNROLL_N"

    if traits["ext_panel_copy"]:
        return f"""
static inline void {cname}_store_pair(
    uint32_t *dst_words,
    svuint32_t offsets_lo,
    svuint32_t offsets_hi,
    svbool_t pg_lo,
    svbool_t pg_hi,
    {traits['vec_type']} row0,
    {traits['vec_type']} row1,
    uint32_t byte_bias)
{{
    {traits['raw_vec_type']} packed_lo = svzip1_u16(svreinterpret_u16(row0), svreinterpret_u16(row1));
    {traits['raw_vec_type']} packed_hi = svzip2_u16(svreinterpret_u16(row0), svreinterpret_u16(row1));
    if (byte_bias != 0) {{
        offsets_lo = svadd_n_u32_x(svptrue_b32(), offsets_lo, byte_bias);
        offsets_hi = svadd_n_u32_x(svptrue_b32(), offsets_hi, byte_bias);
    }}
    {traits['scatter_store_fn']}(pg_lo, dst_words, offsets_lo, svreinterpret_u32(packed_lo));
    {traits['scatter_store_fn']}(pg_hi, dst_words, offsets_hi, svreinterpret_u32(packed_hi));
}}

static void {cname}(int k, int n, const {input_type} *src, int ldsrc, {input_type} *dst)
{{
    const int vlh = svcnth();
    const int vlw = svcntw();
    const bool large_k = k > 64;
    const uint32_t col_bytes = static_cast<uint32_t>(n * sizeof({input_type}));
    const svuint32_t base_offsets_lo = svindex_u32(0, col_bytes);
    const svuint32_t base_offsets_hi = svindex_u32(vlw * col_bytes, col_bytes);
    const {traits['load_ptr_type']} *a = reinterpret_cast<const {traits['load_ptr_type']} *>(src);
    {traits['load_ptr_type']} *b = reinterpret_cast<{traits['load_ptr_type']} *>(dst);

    int i = 0;

    if (large_k) {{
        for (; i + 3 < n; i += 4) {{
            const {traits['load_ptr_type']} *row0 = a + i * ldsrc;
            const {traits['load_ptr_type']} *row1 = row0 + ldsrc;
            const {traits['load_ptr_type']} *row2 = row1 + ldsrc;
            const {traits['load_ptr_type']} *row3 = row2 + ldsrc;
            uint32_t *dst_words = reinterpret_cast<uint32_t *>(b + i);
            for (int j = 0; j < k; j += vlh) {{
                svbool_t pg16 = svwhilelt_b16_s32(j, k);
                svbool_t pg32_lo = svwhilelt_b32_s32(j, k);
                svbool_t pg32_hi = svwhilelt_b32_s32(j + vlw, k);
                uint32_t chunk_bias = static_cast<uint32_t>(j) * col_bytes;
                svuint32_t offsets_lo = svadd_n_u32_x(svptrue_b32(), base_offsets_lo, chunk_bias);
                svuint32_t offsets_hi = svadd_n_u32_x(svptrue_b32(), base_offsets_hi, chunk_bias);
                {traits['vec_type']} v0 = {traits['load_fn']}(pg16, row0 + j);
                {traits['vec_type']} v1 = {traits['load_fn']}(pg16, row1 + j);
                {traits['vec_type']} v2 = {traits['load_fn']}(pg16, row2 + j);
                {traits['vec_type']} v3 = {traits['load_fn']}(pg16, row3 + j);
                {cname}_store_pair(dst_words, offsets_lo, offsets_hi, pg32_lo, pg32_hi, v0, v1, 0);
                {cname}_store_pair(
                    dst_words,
                    offsets_lo,
                    offsets_hi,
                    pg32_lo,
                    pg32_hi,
                    v2,
                    v3,
                    static_cast<uint32_t>(2 * sizeof({input_type}))
                );
            }}
        }}
    }}

    for (; i + 1 < n; i += 2) {{
        const {traits['load_ptr_type']} *row0 = a + i * ldsrc;
        const {traits['load_ptr_type']} *row1 = row0 + ldsrc;
        uint32_t *dst_words = reinterpret_cast<uint32_t *>(b + i);
        for (int j = 0; j < k; j += vlh) {{
            svbool_t pg16 = svwhilelt_b16_s32(j, k);
            svbool_t pg32_lo = svwhilelt_b32_s32(j, k);
            svbool_t pg32_hi = svwhilelt_b32_s32(j + vlw, k);
            uint32_t chunk_bias = static_cast<uint32_t>(j) * col_bytes;
            svuint32_t offsets_lo = svadd_n_u32_x(svptrue_b32(), base_offsets_lo, chunk_bias);
            svuint32_t offsets_hi = svadd_n_u32_x(svptrue_b32(), base_offsets_hi, chunk_bias);
            {traits['vec_type']} v0 = {traits['load_fn']}(pg16, row0 + j);
            {traits['vec_type']} v1 = {traits['load_fn']}(pg16, row1 + j);
            {cname}_store_pair(dst_words, offsets_lo, offsets_hi, pg32_lo, pg32_hi, v0, v1, 0);
        }}
    }}

    for (; i < n; ++i) {{
        const {traits['load_ptr_type']} *row = a + i * ldsrc;
        for (int j = 0; j < k; ++j) {{
            b[i + j * n] = row[j];
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
