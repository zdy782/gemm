import gemm_config
from gemm_type_impl import *
from global_config import *
from kernel_mm_loop_L1 import kernel_mm_loop_n

def laf_asm_code(
    gemm_type,
    transA,
    transB,
    func_name,
    M,
    N,
    K,
    lda,
    ldb,
    ldc,
    data_type="fp32",
    m_vl=1,
    n_vl=4,
):
    set_data_type(data_type)
    gemm_config.set_type_value(gemm_type, transA, transB)
    assert_valid_tile_combo(m_vl, n_vl)

    if gemm_config.type == "small" and gemm_config.transa == "N" and gemm_config.transb == "N":
        logger.debug(f"{gemm_config.type}, {gemm_config.transa}, {gemm_config.transb}")
    elif gemm_config.type == "small" and gemm_config.transa == "N" and gemm_config.transb == "T":
        logger.debug(f"{gemm_config.type}, {gemm_config.transa}, {gemm_config.transb}")
    elif gemm_config.type == "small" and gemm_config.transa == "T" and gemm_config.transb == "N":
        logger.debug(f"{gemm_config.type}, {gemm_config.transa}, {gemm_config.transb}")
    elif gemm_config.type == "small" and gemm_config.transa == "T" and gemm_config.transb == "T":
        logger.debug(f"{gemm_config.type}, {gemm_config.transa}, {gemm_config.transb}")
    elif gemm_config.type == "general":
        logger.debug(f"{gemm_config.type}, {gemm_config.transa}, {gemm_config.transb}")

    gemm_config.get_gemm_type_model()
    logger.debug(f"currect_model:{gemm_config.currect_model}")

    code_str = f""
    code_str += PROLOGUE(func_name)
    code_str += SAVE_REGS()
    code_str += f"prfm    PLDL1KEEP, [{origPA}, #64]      // A矩阵预取\n"
    code_str += f"prfm    PLDL1KEEP, [{origPB}, #64]      // B矩阵预取\n"
    code_str += START_SME_FEATURE()
    code_str += f"mov     {pBt}, {origPB}\n"
    code_str += f"mov     {counterJ}, #0\n"
    code_str += f"mov     {LDA}, #{lda}\n"
    code_str += f"mov     {LDB}, #{ldb}\n"
    code_str += f"mov     {LDC}, #{ldc}\n"
    if is_ext_precision():
        code_str += f"ptrue   p0.h, all\n"
        code_str += f"pfalse  p7.b\n"
    else:
        code_str += f"ptrue   p0.s, all\n"
    code_str += gemm_config.currect_model.set_svindex()
    code_str += f"lsl     {LDC}, {LDC}, #2\n"
    
    if is_ext_precision():
        code_str += f"mov     z29.s, #0\n"
        code_str += f"index   z29.s, #0, #2\n"


    # Tile sizes are expressed in units of s-precision VL.
    m_size = tile_size_from_vl(m_vl)
    n_size = tile_size_from_vl(n_vl)

    code_str += kernel_mm_loop_n(M, N, K, n_size, m_size)

    code_str += STOP_SME_FEATURE()
    code_str += RESTORE_REGS()
    code_str += f"mov     w0, wzr\n"
    code_str += f"ret\n"


    return code_str
