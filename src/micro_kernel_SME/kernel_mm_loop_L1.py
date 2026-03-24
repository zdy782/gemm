from global_config import get_predicate_suffix, get_whilelt_increment, tile_size_from_vl
from kernel_mm_loop_L2 import kernel_mm_loop_L2

# L1 is the outer N loop that measures the remaining columns, chooses the widest legal `nvl`, and hands that chunk to L2.


def _gen_primary_n_predicate(ctx):
    # Build the first N predicate for the candidate chunk, using `ptrue` for ext paths and `whilelt` for scalar-sized paths.
    regs = ctx.registers
    if ctx.is_ext_precision():
        return f"ptrue   {regs.predicates.n_main}.h, vl16\n"
    return f"whilelt     {regs.predicates.n_main}.s, {regs.counters.TMP_CNT}, {regs.dims.MIN_N}\n"


def _gen_n_loop_block(ctx, multiplier, m_size):
    # Emit one N-width candidate block that falls through to the next narrower width when the remainder is too small.
    regs = ctx.registers
    next_label = f".loops_of_l1_{multiplier - 1}vl" if multiplier > 2 else ".loops_of_l1_1vl"
    threshold = tile_size_from_vl(multiplier - 1)
    pred_suffix = get_predicate_suffix(ctx)
    vl1 = tile_size_from_vl(1)
    code_str = f""
    code_str += f".loops_of_l1_{multiplier}vl:\n"
    code_str += f"cmp     {regs.dims.MIN_N}, #{threshold}\n"
    code_str += f"ble     {next_label}\n"
    code_str += f"sub     {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}\n"
    code_str += _gen_primary_n_predicate(ctx)
    for _ in range(multiplier - 1):
        code_str += f"add     {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}, #{vl1}\n"
    code_str += f"whilelt     {regs.predicates.n_tail}{pred_suffix}, {regs.counters.TMP_CNT}, {regs.dims.MIN_N}\n"
    code_str += f"add     {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}, #{vl1}\n"
    code_str += kernel_mm_loop_L2(ctx, m_size, "k_loop_m", f"{multiplier}VL")
    return code_str


def _gen_n_loop_condition(ctx, n_size):
    # Recompute the live N remainder after one tile so the next outer-loop iteration sees the exact columns still pending.
    regs = ctx.registers
    multiplier = n_size // tile_size_from_vl(1)
    code_str = f""
    if multiplier == 1:
        code_str += f"whilelt     {regs.predicates.n_main}.s, {regs.counters.counterJ}, {regs.dims.origN}\n"
        code_str += f"sub     {regs.dims.MIN_N}, {regs.dims.MIN_N}, {regs.dims.MIN_N}\n"
        code_str += f"cntp    {regs.dims.MIN_N}, {regs.predicates.n_main}, {regs.predicates.n_main}.s\n"
        code_str += f"b.first     .loops_of_n\n"
    elif multiplier == 2:
        code_str += f"whilelt     {regs.predicates.n_main}.h, {regs.counters.counterJ}, {regs.dims.origN}\n"
        code_str += f"sub     {regs.dims.MIN_N}, {regs.dims.MIN_N}, {regs.dims.MIN_N}\n"
        code_str += f"cntp    {regs.dims.MIN_N}, {regs.predicates.n_main}, {regs.predicates.n_main}.h\n"
        code_str += f"b.first     .loops_of_n\n"
    elif multiplier == 3:
        code_str += f"whilelt     {regs.predicates.n_main}.h, {regs.counters.counterJ}, {regs.dims.origN}\n"
        # A `3VL` tile is materialized as `2VL main + 1VL tail`, so the tail predicate must start at the logical `2VL` boundary.
        code_str += f"add         {regs.counters.TMP_CNT}, {regs.counters.counterJ}, #{tile_size_from_vl(2)}\n"
        code_str += f"whilelt     {regs.predicates.n_tail}.s, {regs.counters.TMP_CNT}, {regs.dims.origN}\n"
        code_str += f"sub     {regs.dims.MIN_N}, {regs.dims.MIN_N}, {regs.dims.MIN_N}\n"
        code_str += f"cntp    {regs.dims.MIN_N}, {regs.predicates.n_main}, {regs.predicates.n_main}.h\n"
        code_str += f"cntp    {regs.counters.TMP_CNT}, {regs.predicates.n_tail}, {regs.predicates.n_tail}.s\n"
        code_str += f"add     {regs.dims.MIN_N}, {regs.dims.MIN_N}, {regs.counters.TMP_CNT}\n"
        code_str += f"cmp     {regs.dims.MIN_N}, #0\n"
        code_str += f"bgt      .loops_of_n\n"
    elif multiplier == 4:
        code_str += f"whilelt     {regs.predicates.n_main}.b, {regs.counters.counterJ}, {regs.dims.origN}\n"
        code_str += f"sub     {regs.dims.MIN_N}, {regs.dims.MIN_N}, {regs.dims.MIN_N}\n"
        code_str += f"cntp    {regs.dims.MIN_N}, {regs.predicates.n_main}, {regs.predicates.n_main}.b\n"
        code_str += f"b.first     .loops_of_n\n"
    return code_str


def kernel_mm_loop_n(ctx, n_size=None, m_size=None):
    # Enter the outer N loop, seed the original dimensions, and dispatch from the widest legal N candidate down to the universal `1VL` fallback.
    regs = ctx.registers
    spec = ctx.spec
    if n_size is None:
        n_size = tile_size_from_vl(2)
    if m_size is None:
        m_size = tile_size_from_vl(2)
    code_str = f""
    code_str += f"mov  {regs.dims.origM}, #{spec.M}\n"
    code_str += f"mov  {regs.dims.origN}, #{spec.N}\n"
    code_str += f"mov  {regs.dims.origK}, #{spec.K}\n"
    code_str += f"b   .cond_of_loops_n      // A矩阵预取\n"
    code_str += f".loops_of_n:\n"
    code_str += f"mov     {regs.pointers.pC0}, {regs.params.pC}\n"
    code_str += f"sub     {regs.counters.counterI}, {regs.counters.counterI}, {regs.counters.counterI}\n"
    code_str += ctx.model.kernel_mm_loop_n_pre_func(ctx)

    # The legality rule `m_vl * n_vl <= 4` means we only need to test one 4VL, 3VL, or 2VL branch before falling back to `1VL`.
    vl1 = tile_size_from_vl(1)
    vl2 = tile_size_from_vl(2)
    vl3 = tile_size_from_vl(3)

    if n_size > vl3:
        code_str += _gen_n_loop_block(ctx, 4, m_size)
    if n_size > vl2:
        code_str += _gen_n_loop_block(ctx, 3, m_size)
    if n_size > vl1:
        code_str += _gen_n_loop_block(ctx, 2, m_size)

    code_str += f".loops_of_l1_1vl:\n"
    code_str += f"sub     {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}\n"
    code_str += f"whilelt     {regs.predicates.n_main}{get_predicate_suffix(ctx)}, {regs.counters.TMP_CNT}, {regs.dims.MIN_N}\n"
    code_str += f"add     {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}, #{get_whilelt_increment(ctx)}\n"
    code_str += kernel_mm_loop_L2(ctx, m_size, "k_loop_m", "1VL")
    code_str += f".end_of_loops_m:\n"
    code_str += f"add     {regs.counters.counterJ}, {regs.counters.counterJ}, {regs.dims.MIN_N}\n"
    code_str += ctx.model.kernel_mm_loop_n_post_func(ctx)
    code_str += f"mul     {regs.counters.TMP_CNT}, {regs.dims.MIN_N}, {regs.params.LDC}\n"
    code_str += f"add     {regs.params.pC}, {regs.params.pC}, {regs.counters.TMP_CNT}\n"
    code_str += f".cond_of_loops_n:\n"
    code_str += _gen_n_loop_condition(ctx, n_size)
    return code_str
