from global_config import get_k_loop_shift, get_k_remainder_mask, get_k_step
from kernel_mvlxnvl import kernel_bc, kernel_ldntb_bc, kernel_m0, kernel_m0_last_k, resolve_kernel_variant
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


def gen_ext_load_predicate_refresh(ctx):
    # Build the widened ext predicates once per active M tile so full paired load helpers can reuse them across the whole K loop.
    if not ctx.is_ext_precision():
        return ""
    regs = ctx.registers
    code_str = f"zip1      {regs.ext_predicate('m_main')}.h, {regs.predicates.m_main}.h, {regs.predicates.m_main}.h\n"
    code_str += f"zip1      {regs.ext_predicate('m_tail')}.h, {regs.predicates.m_tail}.h, {regs.predicates.m_tail}.h\n"
    code_str += f"zip1      {regs.ext_predicate('n_main')}.h, {regs.predicates.n_main}.h, {regs.predicates.n_main}.h\n"
    code_str += f"zip1      {regs.ext_predicate('n_tail')}.h, {regs.predicates.n_tail}.h, {regs.predicates.n_tail}.h\n"
    return code_str


def _gen_ext_hotpath_setup(ctx, mvl, nvl, kernel_variant):
    # The second contiguous `2VL` chunk always starts one full VL past the head pair, so precompute that offset once outside the K hot loop.
    if not ctx.is_ext_precision() or kernel_variant != "full":
        return ""
    if (mvl, nvl) not in {("1VL", "4VL"), ("4VL", "1VL")}:
        return ""
    regs = ctx.registers
    code_str = f"rdvl      {regs.address.TMP_PTR2}, #1\n"
    code_str += f"lsr       {regs.address.TMP_PTR2}, {regs.address.TMP_PTR2}, #1\n"
    return code_str


def _gen_ext_save_predicate_refresh(ctx):
    # Refresh the widened M-side save predicates right before save so exact-full paired paths do not depend on stale load-side p4/p5 state.
    if not ctx.is_ext_precision():
        return ""
    regs = ctx.registers
    code_str = f"zip1      {regs.ext_predicate('m_main')}.h, {regs.predicates.m_main}.h, {regs.predicates.m_main}.h\n"
    code_str += f"zip1      {regs.ext_predicate('m_tail')}.h, {regs.predicates.m_tail}.h, {regs.predicates.m_tail}.h\n"
    return code_str


def kernel_mm_loop_k_LU(ctx, label, kernel, kernel_last_k, cnt, mvl, nvl, kernel_variant):
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
        code_str += kernel_last_k(ctx, mvl, nvl, kernel_variant=kernel_variant)
        code_str += f"b        .{label}_{mvl}_{nvl}_k_end\n"
        code_str += f".{label}_{mvl}_{nvl}_k_normal:\n"
    code_str += kernel(ctx, mvl, nvl, kernel_variant=kernel_variant)
    if ctx.is_ext_precision():
        code_str += f".{label}_{mvl}_{nvl}_k_end:\n"
    code_str += f"sub      {cnt}, {cnt}, #{get_k_step(ctx)}\n"
    code_str += f"cmp      {cnt}, #0\n"
    code_str += f"bgt      .{label}_{mvl}_{nvl}_k\n"
    return code_str


def kernel_mm_loop_kk(ctx, k_loop_label, mvl, nvl, kernel_bc_fn, kernel_mm_label, kernel_variant):
    regs = ctx.registers
    code_str = f""
    code_str += f".{k_loop_label}_{mvl}_{nvl}_c:\n"
    code_str += f"cmp      {regs.counters.wbk}, #0\n"
    code_str += f"ble      {kernel_mm_label}\n"
    code_str += kernel_bc_fn(ctx, mvl, nvl, kernel_variant=kernel_variant)
    code_str += f"sub      {regs.counters.wbk}, {regs.counters.wbk}, #1\n"
    code_str += f"b.ne     .{k_loop_label}_{mvl}_{nvl}_c\n"
    return code_str


def kernel_mm_loop_k(ctx, label, mvl, nvl, m_fullness="single", n_fullness="single", exit_label=None):
    regs = ctx.registers
    if exit_label is None:
        exit_label = label
    kernel_variant = resolve_kernel_variant(ctx, mvl, nvl, m_fullness, n_fullness)
    code_str = f""
    code_str += _gen_ext_hotpath_setup(ctx, mvl, nvl, kernel_variant)
    # `wbk` counts the full K blocks handled by the fixed kernel body. The
    # remainder path below only runs when `origK` leaves a partial block.
    code_str += f"lsr      {regs.counters.wbk}, {regs.dims.origK}, #{get_k_loop_shift(ctx)}\n"
    code_str += f"sub      {regs.counters.TMP_CNT}, {regs.dims.origM}, {regs.counters.counterI}\n"
    code_str += f"cmp      {regs.counters.TMP_CNT}, {regs.dims.MIN_M}\n"
    code_str += f"ble      .{label}_{mvl}_{nvl}_last_m_loopk\n"
    code_str += kernel_mm_loop_kk(
        ctx,
        f"k_loop_m_{label}",
        mvl,
        nvl,
        kernel_bc,
        f"KERNEL_MM_LU_{mvl}_{nvl}_K_{label}",
        kernel_variant,
    )
    code_str += f".{label}_{mvl}_{nvl}_last_m_loopk:\n"
    code_str += f"mul      {regs.counters.TMP_CNT}, {regs.dims.MIN_N}, {regs.params.LDC}\n"
    code_str += kernel_mm_loop_kk(
        ctx,
        f"k_loop_last_m_{label}",
        mvl,
        nvl,
        kernel_ldntb_bc,
        f"KERNEL_MM_LU_{mvl}_{nvl}_K_{label}",
        kernel_variant,
    )
    code_str += f"KERNEL_MM_LU_{mvl}_{nvl}_K_{label}:\n"
    code_str += f"tst      {regs.dims.origK}, #{get_k_remainder_mask(ctx)}\n"
    code_str += f"beq      .SAVE_C_{mvl}_{nvl}_{label}\n"
    code_str += f"ands     {regs.counters.TMP_CNT}, {regs.dims.origK}, #{get_k_remainder_mask(ctx)}\n"
    code_str += kernel_mm_loop_k_LU(
        ctx,
        f"k_mm_m_lu_k_{label}",
        kernel_m0,
        kernel_m0_last_k,
        regs.counters.TMP_CNT,
        mvl,
        nvl,
        kernel_variant,
    )
    code_str += f".SAVE_C_{mvl}_{nvl}_{label}:\n"
    code_str += _gen_ext_save_predicate_refresh(ctx)
    code_str += _get_kernel_save_fn(mvl, nvl)(ctx, label)
    code_str += f"b        .end_of_loop_k_{nvl}_{exit_label}\n"
    return code_str
