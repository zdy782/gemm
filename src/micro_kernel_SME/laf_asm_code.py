import gemm_config
from gemm_type_impl import *
from global_config import *
from kernel_mm_loop_L1 import kernel_mm_loop_n

def laf_asm_code(gemm_type, transA, transB, func_name, M, N, K, lda, ldb, ldc, data_type="fp32"):
    set_data_type(data_type)
    gemm_config.set_type_value(gemm_type, transA, transB)

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


    # 控制M和N方向Kernel size的大小
    m_size = 16
    n_size = 64

    if (n_size <= 0 or m_size <= 0) or (n_size not in [16, 32, 48, 64]) or (m_size not in [16, 32, 48, 64]) or (n_size * m_size > 1024):
        logger.error(f"m_size:{m_size} * n_size:{n_size} error!")
        exit(1)

    code_str += kernel_mm_loop_n(M, N, K, n_size, m_size)

    code_str += STOP_SME_FEATURE()
    code_str += RESTORE_REGS()
    code_str += f"mov     w0, wzr\n"
    code_str += f"ret\n"


    return code_str