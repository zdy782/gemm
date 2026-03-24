S_ELEMENTS_PER_VL = 16
H_ELEMENTS_PER_VL = 32
MAX_TILE_AREA_VL = 4

LD1 = "ld1w"
LDNT1 = "ldnt1w"
STNT1 = "stnt1w"
LD1_H = "ld1h"
LDNT1_H = "ldnt1h"

TOL = "1e-4"
TOL_BF16 = "5e-3"
TOL_FP16 = "1e-3"


def get_s_elements_per_vl():
    return S_ELEMENTS_PER_VL


def get_h_elements_per_vl():
    return H_ELEMENTS_PER_VL


def get_ext_logical_vl():
    return get_s_elements_per_vl()


def tile_size_from_vl(mult):
    return mult * get_s_elements_per_vl()


def get_save_subtile_count():
    return 4


def get_save_base_slice_stride():
    return get_s_elements_per_vl() // get_save_subtile_count()


def get_save_base_slice_indices():
    stride = get_save_base_slice_stride()
    return tuple(offset * stride for offset in range(get_save_subtile_count()))


def get_save_tail_mask():
    return get_s_elements_per_vl() - 1


def get_save_vl_offsets():
    return tuple(range(get_save_subtile_count()))


def is_valid_tile_combo(m_vl, n_vl):
    if m_vl not in (1, 2, 3, 4) or n_vl not in (1, 2, 3, 4):
        return False
    return (m_vl * n_vl) <= MAX_TILE_AREA_VL


def assert_valid_tile_combo(m_vl, n_vl):
    if not is_valid_tile_combo(m_vl, n_vl):
        raise ValueError(
            f"Unsupported tile combo m_vl={m_vl}, n_vl={n_vl}. "
            "Only combinations with m_vl * n_vl <= 4 are supported."
        )


def get_ld1(ctx):
    return LD1_H if ctx.is_ext_precision() else LD1


def get_non_temporal_ld1(ctx):
    return LDNT1_H if ctx.is_ext_precision() else LDNT1


def get_element_suffix(ctx):
    return ".h" if ctx.is_ext_precision() else ".s"


def get_ld_element_suffix(ctx):
    return ".h" if ctx.is_ext_precision() else ".s"


def get_mopa_inst(ctx):
    return "bfmopa" if ctx.is_bf16() else "fmopa"


def get_element_size_shift(ctx):
    return 1 if ctx.is_ext_precision() else 2


def get_predicate_suffix(ctx):
    return ".h" if ctx.is_ext_precision() else ".s"


def get_whilelt_increment(ctx):
    return get_h_elements_per_vl() if ctx.is_ext_precision() else get_s_elements_per_vl()


def get_k_step(ctx):
    return 2 if ctx.is_ext_precision() else 1


def get_k_remainder_mask(ctx):
    return 15 if ctx.is_ext_precision() else 7


def get_k_loop_shift(ctx):
    return 4 if ctx.is_ext_precision() else 3


def get_tolerance_value(spec):
    if spec.is_bf16():
        return TOL_BF16
    if spec.is_fp16():
        return TOL_FP16
    return TOL


def PROLOGUE(real_name):
    code_str = f""
    code_str += f".text\n"
    code_str += f".p2align 2\n"
    code_str += f".global {real_name}\n"
    code_str += f".type {real_name}, %function\n"
    code_str += f"{real_name}:\n"
    return code_str


def SAVE_REGS(registers):
    code_str = f""
    code_str += f".align 5\n"
    code_str += f"add     sp, sp, #-(11 * 16)\n"
    code_str += f"stp     d8, d9, [sp, #(0 * 16)]\n"
    code_str += f"stp     d10, d11, [sp, #(1 * 16)]\n"
    code_str += f"stp     d12, d13, [sp, #(2 * 16)]\n"
    code_str += f"stp     d14, d15, [sp, #(3 * 16)]\n"
    code_str += f"stp     d16, d17, [sp, #(4 * 16)]\n"
    code_str += f"stp     {registers.counters.counterI}, {registers.pointers.pB0}, [sp, #(5 * 16)]\n"
    code_str += f"stp     {registers.pointers.pC0}, {registers.pointers.pC1}, [sp, #(6 * 16)]\n"
    code_str += f"stp     {registers.pointers.pC2}, {registers.counters.wbk}, [sp, #(7 * 16)]\n"
    code_str += f"stp     {registers.pointers.pA0}, {registers.address.pA_OFFSET}, [sp, #(8 * 16)]\n"
    code_str += f"stp     {registers.pointers.pAn}, {registers.address.pB_OFFSET}, [sp, #(9 * 16)]\n"
    code_str += f"stp     {registers.address.TMP_PTR2}, {registers.pointers.pAt}, [sp, #(10 * 16)]\n"
    return code_str


def RESTORE_REGS(registers):
    code_str = f""
    code_str += f"ldp     d8, d9, [sp, #(0 * 16)]\n"
    code_str += f"ldp     d10, d11, [sp, #(1 * 16)]\n"
    code_str += f"ldp     d12, d13, [sp, #(2 * 16)]\n"
    code_str += f"ldp     d14, d15, [sp, #(3 * 16)]\n"
    code_str += f"ldp     d16, d17, [sp, #(4 * 16)]\n"
    code_str += f"ldp     {registers.counters.counterI}, {registers.pointers.pB0}, [sp, #(5 * 16)]\n"
    code_str += f"ldp     {registers.pointers.pC0}, {registers.pointers.pC1}, [sp, #(6 * 16)]\n"
    code_str += f"ldp     {registers.pointers.pC2}, {registers.counters.wbk}, [sp, #(7 * 16)]\n"
    code_str += f"ldp     {registers.pointers.pA0}, {registers.address.pA_OFFSET}, [sp, #(8 * 16)]\n"
    code_str += f"ldp     {registers.pointers.pAn}, {registers.address.pB_OFFSET}, [sp, #(9 * 16)]\n"
    code_str += f"ldp     {registers.address.TMP_PTR2}, {registers.pointers.pAt}, [sp, #(10 * 16)]\n"
    code_str += f"add     sp, sp, #(11*16)\n"
    return code_str


def START_SME_FEATURE(registers):
    code_str = f""
    code_str += f"fmov    {registers.counters.wbk}, d0\n"
    code_str += f"fmov    {registers.pointers.pA0}, d1\n"
    code_str += f"fmov    {registers.address.pA_OFFSET}, d2\n"
    code_str += f"fmov    {registers.pointers.pAn}, d3\n"
    code_str += f"msr     SVCRSMZA, #1\n"
    code_str += f"isb\n"
    code_str += f"fmov    d0, {registers.counters.wbk}\n"
    code_str += f"fmov    d1, {registers.pointers.pA0}\n"
    code_str += f"fmov    d2, {registers.address.pA_OFFSET}\n"
    code_str += f"fmov    d3, {registers.pointers.pAn}\n"
    return code_str


def STOP_SME_FEATURE():
    code_str = f""
    code_str += f"msr     SVCRSMZA, #0\n"
    code_str += f"isb\n"
    return code_str
