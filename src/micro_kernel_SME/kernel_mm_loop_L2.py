from global_config import get_predicate_suffix, tile_size_from_vl
from kernel_mm_loop_k import kernel_mm_loop_k


def _emit_m_count_init(ctx, m_size):
    regs = ctx.registers
    multiplier = m_size // tile_size_from_vl(1)
    if multiplier == 1:
        return f"cntp     {regs.dims.MIN_M}, {regs.predicates.m_main}, {regs.predicates.m_main}.s\n"
    if multiplier == 2:
        return f"cntp     {regs.dims.MIN_M}, {regs.predicates.m_main}, {regs.predicates.m_main}.h\n"
    return f"cntp     {regs.dims.MIN_M}, {regs.predicates.m_main}, {regs.predicates.m_main}.b\n"


def _emit_primary_m_predicate(ctx):
    regs = ctx.registers
    if ctx.is_ext_precision():
        return f"ptrue   {regs.predicates.m_main}.h, vl16\n"
    return f"whilelt     {regs.predicates.m_main}.s, {regs.counters.TMP_CNT}, {regs.dims.MIN_M}\n"


def _emit_m_loop_block(ctx, multiplier, nvl, label):
    regs = ctx.registers
    next_label = f".loop_m_{multiplier - 1}vl_{nvl}_{label}" if multiplier > 2 else f".loop_m_1vl_{nvl}_{label}"
    threshold = tile_size_from_vl(multiplier - 1)
    pred_suffix = get_predicate_suffix(ctx)
    vl1 = tile_size_from_vl(1)
    code_str = f""
    code_str += f".loop_m_{multiplier}vl_{nvl}_{label}:\n"
    code_str += f"cmp      {regs.dims.MIN_M}, #{threshold}\n"
    code_str += f"ble      {next_label}\n"
    code_str += f"sub      {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}\n"
    code_str += _emit_primary_m_predicate(ctx)
    for _ in range(multiplier - 1):
        code_str += f"add      {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}, #{vl1}\n"
    code_str += f"whilelt  {regs.predicates.m_tail}{pred_suffix}, {regs.counters.TMP_CNT}, {regs.dims.MIN_M}\n"
    code_str += f"add      {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}, #{vl1}\n"
    code_str += kernel_mm_loop_k(ctx, label, f"{multiplier}VL", nvl)
    return code_str


def _emit_m_loop_condition(ctx, m_size, nvl, label):
    regs = ctx.registers
    multiplier = m_size // tile_size_from_vl(1)
    code_str = f""
    if multiplier == 1 and nvl in ["1VL", "2VL", "3VL", "4VL"]:
        code_str += f"whilelt  {regs.predicates.m_main}.s, {regs.counters.counterI}, {regs.dims.origM}\n"
    elif multiplier == 2 and nvl in ["1VL", "2VL"]:
        code_str += f"whilelt  {regs.predicates.m_main}.h, {regs.counters.counterI}, {regs.dims.origM}\n"
    elif multiplier == 3 and nvl == "1VL":
        code_str += f"whilelt  {regs.predicates.m_main}.h, {regs.counters.counterI}, {regs.dims.origM}\n"
        code_str += f"add      {regs.counters.TMP_CNT}, {regs.counters.counterI}, #{tile_size_from_vl(2)}\n"
        code_str += f"whilelt  {regs.predicates.m_tail}.s, {regs.counters.TMP_CNT}, {regs.dims.origM}\n"
        code_str += f"cntp     {regs.dims.MIN_M}, {regs.predicates.m_main}, {regs.predicates.m_main}.h\n"
        code_str += f"cntp     {regs.counters.TMP_CNT}, {regs.predicates.m_tail}, {regs.predicates.m_tail}.s\n"
        code_str += f"add      {regs.counters.TMP_CNT}, {regs.dims.MIN_M}, {regs.counters.TMP_CNT}\n"
        code_str += f"whilelt  {regs.predicates.m_main}.b, xzr, {regs.counters.TMP_CNT}\n"
    elif multiplier == 4 and nvl == "1VL":
        code_str += f"whilelt  {regs.predicates.m_main}.b, {regs.counters.counterI}, {regs.dims.origM}\n"
    else:
        raise ValueError(f"Unsupported m_size={m_size}, nvl={nvl}")
    code_str += f"b.first  .loops_of_m_{nvl}_{label}\n"
    return code_str


def kernel_mm_loop_L2(ctx, m_size, label, nvl):
    regs = ctx.registers
    code_str = f""
    code_str += f"b    .cond_of_loops_m_{nvl}_{label}\n"
    code_str += f".loops_of_m_{nvl}_{label}:\n"
    code_str += f"zero     {{za0.b}}\n"
    code_str += f"sub      {regs.dims.MIN_M}, {regs.dims.MIN_M}, {regs.dims.MIN_M}\n"
    code_str += _emit_m_count_init(ctx, m_size)
    code_str += ctx.model.kernel_mm_loop_m_pre_func(ctx)

    vl1 = tile_size_from_vl(1)
    vl2 = tile_size_from_vl(2)
    vl3 = tile_size_from_vl(3)

    if m_size > vl3:
        code_str += _emit_m_loop_block(ctx, 4, nvl, label)
    if m_size > vl2:
        code_str += _emit_m_loop_block(ctx, 3, nvl, label)
    if m_size > vl1:
        code_str += _emit_m_loop_block(ctx, 2, nvl, label)

    code_str += f".loop_m_1vl_{nvl}_{label}:\n"
    code_str += f"sub      {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}\n"
    code_str += f"whilelt  {regs.predicates.m_main}{get_predicate_suffix(ctx)}, {regs.counters.TMP_CNT}, {regs.dims.MIN_M}\n"
    code_str += f"add      {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}, #{tile_size_from_vl(1)}\n"
    code_str += kernel_mm_loop_k(ctx, label, "1VL", nvl)
    code_str += f".end_of_loop_k_{nvl}_{label}:\n"
    code_str += f"add      {regs.counters.counterI}, {regs.counters.counterI}, {regs.dims.MIN_M}\n"
    code_str += ctx.model.kernel_mm_loop_m_post_func(ctx)
    code_str += f".cond_of_loops_m_{nvl}_{label}:\n"
    code_str += _emit_m_loop_condition(ctx, m_size, nvl, label)
    return code_str
