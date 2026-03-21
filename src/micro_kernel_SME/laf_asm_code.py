from global_config import (
    PROLOGUE,
    RESTORE_REGS,
    SAVE_REGS,
    START_SME_FEATURE,
    STOP_SME_FEATURE,
    get_element_size_shift,
    tile_size_from_vl,
)
from kernel_mm_loop_L1 import kernel_mm_loop_n


def laf_asm_code(ctx, func_name):
    spec = ctx.spec
    regs = ctx.registers
    code_str = f""
    code_str += PROLOGUE(func_name)
    code_str += SAVE_REGS(regs)
    code_str += f"prfm    PLDL1KEEP, [{regs.params.origPA}, #64]      // A矩阵预取\n"
    code_str += f"prfm    PLDL1KEEP, [{regs.params.origPB}, #64]      // B矩阵预取\n"
    code_str += START_SME_FEATURE(regs)
    code_str += f"mov     {regs.pointers.pBt}, {regs.params.origPB}\n"
    code_str += f"mov     {regs.counters.counterJ}, #0\n"
    code_str += f"mov     {regs.params.LDA}, #{spec.lda}\n"
    code_str += f"mov     {regs.params.LDB}, #{spec.ldb}\n"
    code_str += f"mov     {regs.params.LDC}, #{spec.ldc}\n"
    if ctx.is_ext_precision():
        code_str += f"ptrue   {regs.predicates.n_main}.h, all\n"
        code_str += f"pfalse  {regs.predicates.false_all}.b\n"
    else:
        code_str += f"ptrue   {regs.predicates.n_main}.s, all\n"
    code_str += ctx.model.set_svindex(ctx)
    code_str += f"lsl     {regs.params.LDC}, {regs.params.LDC}, #2\n"

    m_size = tile_size_from_vl(spec.tile.m_vl)
    n_size = tile_size_from_vl(spec.tile.n_vl)
    code_str += kernel_mm_loop_n(ctx, n_size=n_size, m_size=m_size)
    code_str += STOP_SME_FEATURE()
    code_str += RESTORE_REGS(regs)
    code_str += f"mov     w0, wzr\n"
    code_str += f"ret\n"
    return code_str
