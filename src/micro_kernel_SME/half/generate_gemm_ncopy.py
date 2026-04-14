"""Generate ACLE-based half-precision ncopy helpers for packed SME drivers."""


def _validate_input_type(input_type: str) -> None:
    if input_type not in {"__bf16", "__fp16"}:
        raise ValueError(f"Unsupported input_type for ncopy: {input_type}")


def generate_gemm_ncopy(cname: str, input_type: str, ncopy_type: str) -> str:
    """Return a C++ helper that packs a half matrix into row-major [k][n]."""
    if ncopy_type not in {"IN", "ON"}:
        raise ValueError(f"Unsupported ncopy_type: {ncopy_type}")
    _validate_input_type(input_type)

    unroll_name = "AUTOGEMM_COPY_UNROLL_M" if ncopy_type == "IN" else "AUTOGEMM_COPY_UNROLL_N"
    return f"""
static void {cname}(int k, int n, const {input_type} *src, int ldsrc, {input_type} *dst)
{{
    const int svcnt = svcnth();
    const int ele_sz = svcnt * {unroll_name};
    const uint16_t *a = reinterpret_cast<const uint16_t *>(src);
    uint16_t *b = reinterpret_cast<uint16_t *>(dst);
    const svuint32_t off = svindex_u32(0, static_cast<uint32_t>(ldsrc * static_cast<int>(sizeof(uint16_t))));
    int min_j = 0;

    for (int j = 0; j < n; j += min_j) {{
        min_j = ((ele_sz > (n - j)) ? (n - j) : ele_sz);
        const uint16_t *tmpa0 = a + j * ldsrc;
        for (int i = 0; i < k; ++i) {{
            int js = 0;
            const uint16_t *tmpa = tmpa0;
            uint16_t *tmpb = b + i * n + j;
            svbool_t pg = svwhilelt_b32_s32(js, min_j);
            do {{
                const uint64_t cnt = svcntp_b32(svptrue_b32(), pg);
                const svuint32_t packed = svld1uh_gather_u32offset_u32(pg, tmpa, off);
                svst1h_u32(pg, tmpb, packed);
                tmpb += cnt;
                tmpa += cnt * ldsrc;
                js += cnt;
                pg = svwhilelt_b32_s32(js, min_j);
            }} while (svptest_any(svptrue_b32(), pg));
            tmpa0 += 1;
        }}
    }}
}}
"""
