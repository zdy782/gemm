"""Generate ACLE-based half-precision tcopy helpers for packed SME drivers."""


def _validate_input_type(input_type: str) -> None:
    if input_type not in {"__bf16", "__fp16"}:
        raise ValueError(f"Unsupported input_type for tcopy: {input_type}")


def generate_gemm_tcopy(cname: str, input_type: str, tcopy_type: str) -> str:
    """Return a C++ helper that packs a contiguous half matrix by n-sized rows."""
    if tcopy_type not in {"IT", "OT"}:
        raise ValueError(f"Unsupported tcopy_type: {tcopy_type}")
    _validate_input_type(input_type)

    unroll_name = "AUTOGEMM_COPY_UNROLL_M" if tcopy_type == "IT" else "AUTOGEMM_COPY_UNROLL_N"
    return f"""
static void {cname}(int k, int n, const {input_type} *src, int ldsrc, {input_type} *dst)
{{
    const int svcnt = svcnth();
    const int ele_sz = svcnt * {unroll_name};
    const uint16_t *a = reinterpret_cast<const uint16_t *>(src);
    uint16_t *b = reinterpret_cast<uint16_t *>(dst);
    int min_i = 0;

    for (int i = 0; i < n; i += min_i) {{
        min_i = ((ele_sz > (n - i)) ? (n - i) : ele_sz);
        const uint16_t *tmpa0 = a;
        for (int j = 0; j < k; ++j) {{
            int is = 0;
            const uint16_t *tmpa = tmpa0;
            uint16_t *tmpb = b + i + j * n;
            svbool_t pg = svwhilelt_b16_s32(is, min_i);
            do {{
                const uint64_t cnt = svcntp_b16(svptrue_b16(), pg);
                const svuint16_t packed = svldnt1_u16(pg, tmpa);
                svst1_u16(pg, tmpb, packed);
                tmpa += cnt;
                tmpb += cnt;
                is += cnt;
                pg = svwhilelt_b16_s32(is, min_i);
            }} while (svptest_any(svptrue_b16(), pg));
            tmpa0 += ldsrc;
        }}
        a += min_i;
    }}
}}
"""
