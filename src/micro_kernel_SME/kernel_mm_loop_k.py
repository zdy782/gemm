from global_config import get_k_loop_shift, get_k_remainder_mask, get_k_step
from kernel_mvlxnvl import kernel_bc, kernel_ldntb_bc, kernel_m0, kernel_m0_last_k
from kernel_save import (
    kernel_save_c_1VL_1VL,
    kernel_save_c_1VL_2VL,
    kernel_save_c_1VL_3VL,
    kernel_save_c_1VL_4VL,
    kernel_save_c_2VL_1VL,
    kernel_save_c_2VL_2VL,
    kernel_save_c_3VL_1VL,
    kernel_save_c_4VL_1VL,
)

# The K loop has three responsibilities:
# - run the repeated full K blocks (`kernel_bc`)
# - run the ext-precision odd-K remainder (`kernel_m0_last_k`)
# - save ZA back to C once the whole K dimension is consumed


KERNEL_SAVE_FUN_MAP = {
    ("4VL", "1VL"): kernel_save_c_4VL_1VL,
    ("1VL", "4VL"): kernel_save_c_1VL_4VL,
    ("3VL", "1VL"): kernel_save_c_3VL_1VL,
    ("1VL", "3VL"): kernel_save_c_1VL_3VL,
    ("2VL", "2VL"): kernel_save_c_2VL_2VL,
    ("2VL", "1VL"): kernel_save_c_2VL_1VL,
    ("1VL", "2VL"): kernel_save_c_1VL_2VL,
    ("1VL", "1VL"): kernel_save_c_1VL_1VL,
}


def _get_kernel_save_fn(mvl, nvl):
    return KERNEL_SAVE_FUN_MAP[(mvl, nvl)]


def kernel_mm_loop_k_LU(ctx, label, kernel, kernel_last_k, cnt, mvl, nvl):
    regs = ctx.registers
    code_str = f""
    code_str += f".{label}_{mvl}_{nvl}_k:\n"
    if ctx.is_ext_precision():
        # Ext kernels accumulate two K elements per inner step. When the final
        # remaining chunk is a single element, jump to the dedicated last-k
        # loader instead of trying to reuse the normal paired body.
        code_str += f"cmp      {cnt}, #1\n"
        code_str += f"bne      .{label}_{mvl}_{nvl}_k_normal\n"
        code_str += f"tst      {regs.dims.origK}, #1\n"
        code_str += f"beq      .{label}_{mvl}_{nvl}_k_normal\n"
        code_str += kernel_last_k(ctx, mvl, nvl)
        code_str += f"b        .{label}_{mvl}_{nvl}_k_end\n"
        code_str += f".{label}_{mvl}_{nvl}_k_normal:\n"
    code_str += kernel(ctx, mvl, nvl)
    if ctx.is_ext_precision():
        code_str += f".{label}_{mvl}_{nvl}_k_end:\n"
    code_str += f"sub      {cnt}, {cnt}, #{get_k_step(ctx)}\n"
    code_str += f"cmp      {cnt}, #0\n"
    code_str += f"bgt      .{label}_{mvl}_{nvl}_k\n"
    return code_str


def kernel_mm_loop_kk(ctx, k_loop_label, mvl, nvl, kernel_bc_fn, kernel_mm_label):
    regs = ctx.registers
    code_str = f""
    code_str += f".{k_loop_label}_{mvl}_{nvl}_c:\n"
    code_str += f"cmp      {regs.counters.wbk}, #0\n"
    code_str += f"ble      {kernel_mm_label}\n"
    code_str += kernel_bc_fn(ctx, mvl, nvl)
    code_str += f"sub      {regs.counters.wbk}, {regs.counters.wbk}, #1\n"
    code_str += f"b.ne     .{k_loop_label}_{mvl}_{nvl}_c\n"
    return code_str


def kernel_mm_loop_k(ctx, label, mvl, nvl):
    regs = ctx.registers
    code_str = f""
    # `wbk` counts the full K blocks handled by the fixed kernel body. The
    # remainder path below only runs when `origK` leaves a partial block.
    code_str += f"lsr      {regs.counters.wbk}, {regs.dims.origK}, #{get_k_loop_shift(ctx)}\n"
    code_str += f"sub      {regs.counters.TMP_CNT}, {regs.dims.origM}, {regs.counters.counterI}\n"
    code_str += f"cmp      {regs.counters.TMP_CNT}, {regs.dims.MIN_M}\n"
    code_str += f"ble      .{label}_{mvl}_{nvl}_last_m_loopk\n"
    code_str += kernel_mm_loop_kk(ctx, f"k_loop_m_{label}", mvl, nvl, kernel_bc, f"KERNEL_MM_LU_{mvl}_{nvl}_K_{label}")
    code_str += f".{label}_{mvl}_{nvl}_last_m_loopk:\n"
    code_str += f"mul      {regs.counters.TMP_CNT}, {regs.dims.MIN_N}, {regs.params.LDC}\n"
    code_str += kernel_mm_loop_kk(ctx, f"k_loop_last_m_{label}", mvl, nvl, kernel_ldntb_bc, f"KERNEL_MM_LU_{mvl}_{nvl}_K_{label}")
    code_str += f"KERNEL_MM_LU_{mvl}_{nvl}_K_{label}:\n"
    code_str += f"tst      {regs.dims.origK}, #{get_k_remainder_mask(ctx)}\n"
    code_str += f"beq      .SAVE_C_{mvl}_{nvl}_{label}\n"
    code_str += f"ands     {regs.counters.TMP_CNT}, {regs.dims.origK}, #{get_k_remainder_mask(ctx)}\n"
    code_str += kernel_mm_loop_k_LU(ctx, f"k_mm_m_lu_k_{label}", kernel_m0, kernel_m0_last_k, regs.counters.TMP_CNT, mvl, nvl)
    code_str += f".SAVE_C_{mvl}_{nvl}_{label}:\n"
    code_str += _get_kernel_save_fn(mvl, nvl)(ctx, label)
    code_str += f"b        .end_of_loop_k_{nvl}_{label}\n"
    return code_str
