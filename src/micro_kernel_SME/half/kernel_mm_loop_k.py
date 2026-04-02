from .global_config import get_half_k_loop_shift, get_half_k_remainder_mask, get_half_k_step
from .kernel_mvlxnvl import (
    UNPAIRED_SMALL_KERNEL_PLAN,
    kernel_bc,
    kernel_ldntb_bc,
    kernel_m0,
    kernel_m0_last_k,
    resolve_small_kernel_pair_plan,
)
from .kernel_save import (
    gen_save_alpha_setup,
    gen_save_beta_setup,
    gen_save_beta_zero_check,
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
# - run the half-input odd-K remainder (`kernel_m0_last_k`)
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


def gen_half_load_predicate_refresh(ctx, need_m_tail=True, need_n_tail=True):
    # Build only the widened half-input predicates that the selected tile can
    # actually consume in the upcoming K loop.
    regs = ctx.registers
    code_str = f"zip1      {regs.half_predicate('m_main')}.h, {regs.predicates.m_main}.h, {regs.predicates.m_main}.h\n"
    code_str += f"zip1      {regs.half_predicate('n_main')}.h, {regs.predicates.n_main}.h, {regs.predicates.n_main}.h\n"
    if need_m_tail:
        code_str += f"zip1      {regs.half_predicate('m_tail')}.h, {regs.predicates.m_tail}.h, {regs.predicates.m_tail}.h\n"
    if need_n_tail:
        code_str += f"zip1      {regs.half_predicate('n_tail')}.h, {regs.predicates.n_tail}.h, {regs.predicates.n_tail}.h\n"
    return code_str


def _gen_paired_half_hotpath_setup(ctx, pair_plan):
    # The second contiguous `2VL` chunk always starts one full VL past the head
    # pair, so precompute that offset once outside the paired half-load hot loop.
    if not ctx.use_paired_half_loads() or pair_plan is None or not pair_plan.precompute_second_chunk_offset:
        return ""
    regs = ctx.registers
    code_str = f"rdvl      {regs.address.TMP_PTR2}, #1\n"
    code_str += f"lsr       {regs.address.TMP_PTR2}, {regs.address.TMP_PTR2}, #1\n"
    return code_str

def kernel_mm_loop_k_LU(ctx, label, kernel, kernel_last_k, cnt, mvl, nvl, pair_plan):
    regs = ctx.registers
    code_str = f""
    code_str += f".{label}_{mvl}_{nvl}_k:\n"
    # Half kernels consume two logical K elements per inner step. When the
    # final remaining chunk is a single element, jump to the dedicated last-k
    # loader instead of trying to reuse the normal paired body.
    code_str += f"cmp      {cnt}, #1\n"
    code_str += f"bne      .{label}_{mvl}_{nvl}_k_normal\n"
    code_str += f"tst      {regs.dims.origK}, #1\n"
    code_str += f"beq      .{label}_{mvl}_{nvl}_k_normal\n"
    code_str += kernel_last_k(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN)
    code_str += f"b        .{label}_{mvl}_{nvl}_k_end\n"
    code_str += f".{label}_{mvl}_{nvl}_k_normal:\n"
    code_str += kernel(ctx, mvl, nvl, pair_plan=pair_plan)
    code_str += f".{label}_{mvl}_{nvl}_k_end:\n"
    code_str += f"sub      {cnt}, {cnt}, #{get_half_k_step()}\n"
    code_str += f"cmp      {cnt}, #0\n"
    code_str += f"bgt      .{label}_{mvl}_{nvl}_k\n"
    return code_str


def kernel_mm_loop_kk(ctx, k_loop_label, mvl, nvl, kernel_bc_fn, kernel_mm_label, pair_plan):
    regs = ctx.registers
    code_str = f""
    code_str += f".{k_loop_label}_{mvl}_{nvl}_c:\n"
    code_str += f"cmp      {regs.counters.wbk}, #0\n"
    code_str += f"ble      {kernel_mm_label}\n"
    code_str += kernel_bc_fn(ctx, mvl, nvl, pair_plan=pair_plan)
    code_str += f"sub      {regs.counters.wbk}, {regs.counters.wbk}, #1\n"
    code_str += f"b.ne     .{k_loop_label}_{mvl}_{nvl}_c\n"
    return code_str


def kernel_mm_loop_k(ctx, label, mvl, nvl, m_fullness="single", n_fullness="single", exit_label=None):
    regs = ctx.registers
    if exit_label is None:
        exit_label = label
    pair_plan = resolve_small_kernel_pair_plan(ctx, mvl, nvl, m_fullness, n_fullness)
    code_str = f""
    code_str += _gen_paired_half_hotpath_setup(ctx, pair_plan)
    # `wbk` counts the full K blocks handled by the fixed kernel body. The
    # remainder path below only runs when `origK` leaves a partial block.
    code_str += f"lsr      {regs.counters.wbk}, {regs.dims.origK}, #{get_half_k_loop_shift()}\n"
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
        pair_plan,
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
        pair_plan,
    )
    code_str += f"KERNEL_MM_LU_{mvl}_{nvl}_K_{label}:\n"
    code_str += f"tst      {regs.dims.origK}, #{get_half_k_remainder_mask()}\n"
    code_str += f"beq      .SAVE_C_{mvl}_{nvl}_{label}\n"
    code_str += f"ands     {regs.counters.TMP_CNT}, {regs.dims.origK}, #{get_half_k_remainder_mask()}\n"
    code_str += kernel_mm_loop_k_LU(
        ctx,
        f"k_mm_m_lu_k_{label}",
        kernel_m0,
        kernel_m0_last_k,
        regs.counters.TMP_CNT,
        mvl,
        nvl,
        pair_plan,
    )
    code_str += f".SAVE_C_{mvl}_{nvl}_{label}:\n"
    save_fn = _get_kernel_save_fn(mvl, nvl)
    code_str += gen_save_alpha_setup(ctx)
    code_str += gen_save_beta_zero_check(ctx)
    code_str += f"bne      .SAVE_C_{mvl}_{nvl}_{label}_beta_nonzero\n"
    code_str += save_fn(ctx, label, beta_zero=True, save_label_suffix="beta_zero")
    code_str += f"b        .SAVE_C_{mvl}_{nvl}_{label}_done\n"
    code_str += f".SAVE_C_{mvl}_{nvl}_{label}_beta_nonzero:\n"
    code_str += gen_save_beta_setup(ctx)
    code_str += save_fn(ctx, label, beta_zero=False, save_label_suffix="beta_nonzero")
    code_str += f".SAVE_C_{mvl}_{nvl}_{label}_done:\n"
    code_str += f"b        .end_of_loop_k_{nvl}_{exit_label}\n"
    return code_str
