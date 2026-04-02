S_ELEMENTS_PER_VL = 16
H_ELEMENTS_PER_VL = 32
MAX_TILE_AREA_VL = 4

STNT1 = "stnt1w"
LDNT1 = "ldnt1w"
LD1_H = "ld1h"
LDNT1_H = "ldnt1h"

TOL_BF16 = "5e-3"
TOL_FP16 = "1e-3"
KERNEL_CALLEE_SAVE_SLOTS = 11
KERNEL_SCALAR_PARAM_SLOT = 11
KERNEL_FRAME_SLOTS = 12


def get_s_elements_per_vl():
    return S_ELEMENTS_PER_VL


def get_h_elements_per_vl():
    return H_ELEMENTS_PER_VL


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


def get_half_load_inst():
    return LD1_H


def get_half_non_temporal_load_inst():
    return LDNT1_H


def get_half_input_suffix():
    return ".h"


def get_mopa_inst(ctx):
    return "bfmopa" if ctx.is_bf16() else "fmopa"


def get_half_input_size_shift():
    return 1


def get_half_whilelt_increment():
    return get_h_elements_per_vl()


def get_half_k_step():
    return 2


def get_half_k_remainder_mask():
    return 15


def get_half_k_loop_shift():
    return 4


def get_tolerance_value(spec):
    if spec.is_bf16():
        return TOL_BF16
    if spec.is_fp16():
        return TOL_FP16
    raise ValueError(f"Unsupported precision: {spec.data_type}")


def get_kernel_frame_size():
    return KERNEL_FRAME_SLOTS * 16


def get_alpha_stack_offset():
    return KERNEL_SCALAR_PARAM_SLOT * 16


def get_beta_stack_offset():
    return KERNEL_SCALAR_PARAM_SLOT * 16 + 4


def PROLOGUE(real_name):
    code_str = f""
    code_str += f".text\n"
    code_str += f".p2align 2\n"
    code_str += f".global {real_name}\n"
    code_str += f".type {real_name}, %function\n"
    code_str += f"{real_name}:\n"
    return code_str


def NORMALIZE_RUNTIME_KERNEL_ABI(registers):
    code_str = f""
    # Runtime kernels follow the BLAS-style order:
    #   x0/x1/x2 = M/N/K
    #   s0/s1    = alpha/beta
    #   x3/x4/x5 = A/B/C
    #   x6/x7    = lda/ldb
    #   [sp]     = ldc (9th integer-class argument)
    # Internally we keep the historical register contract:
    #   x0/x1/x2 = A/B/C
    #   x3/x4/x5 = lda/ldb/ldc
    #   x6/x7/x8 = M/N/K
    code_str += f"ldr     x14, [sp]\n"
    code_str += f"mov     x15, x0\n"
    code_str += f"mov     x16, x1\n"
    code_str += f"mov     x17, x2\n"
    code_str += f"mov     {registers.params.origPA}, x3\n"
    code_str += f"mov     {registers.params.origPB}, x4\n"
    code_str += f"mov     {registers.params.pC}, x5\n"
    code_str += f"mov     {registers.params.LDA}, x6\n"
    code_str += f"mov     {registers.params.LDB}, x7\n"
    code_str += f"mov     {registers.params.LDC}, x14\n"
    code_str += f"mov     {registers.dims.origM}, x15\n"
    code_str += f"mov     {registers.dims.origN}, x16\n"
    code_str += f"mov     {registers.dims.origK}, x17\n"
    return code_str


def SAVE_REGS(registers):
    code_str = f""
    code_str += f".align 5\n"
    code_str += f"add     sp, sp, #-({get_kernel_frame_size()})\n"
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
    # Spill runtime alpha/beta once so the SME body may freely reuse z0-z31
    # while the save path later reloads the scalar coefficients.
    code_str += f"str     {registers.scalars.alpha_arg}, [sp, #{get_alpha_stack_offset()}]\n"
    code_str += f"str     {registers.scalars.beta_arg}, [sp, #{get_beta_stack_offset()}]\n"
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
    code_str += f"add     sp, sp, #({get_kernel_frame_size()})\n"
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
