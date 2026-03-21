import gemm_config
from gemm_type_impl import *
from global_config import *
from kernel_mm_loop_k import kernel_mm_loop_k


def _emit_m_count_init(m_size):
    multiplier = m_size // tile_size_from_vl(1)
    if multiplier == 1:
        return f"cntp     {MIN_M}, p1, p1.s\n"
    if multiplier == 2:
        return f"cntp     {MIN_M}, p1, p1.h\n"
    return f"cntp     {MIN_M}, p1, p1.b\n"


def _emit_primary_m_predicate():
    if is_ext_precision():
        return f"ptrue   p1.h, vl{get_ext_logical_vl()}\n"
    return f"whilelt     p1.s, {TMP_CNT}, {MIN_M}\n"


def _emit_m_loop_block(multiplier, nvl, label):
    next_label = f".loop_m_{multiplier - 1}vl_{nvl}_{label}" if multiplier > 2 else f".loop_m_1vl_{nvl}_{label}"
    threshold = tile_size_from_vl(multiplier - 1)
    pred_suffix = get_predicate_suffix()
    vl1 = tile_size_from_vl(1)
    code_str = f""
    code_str += f".loop_m_{multiplier}vl_{nvl}_{label}:\n"
    code_str += f"cmp      {MIN_M}, #{threshold}\n"
    code_str += f"ble      {next_label}\n"
    code_str += f"sub      {TMP_CNT}, {TMP_CNT}, {TMP_CNT}\n"
    code_str += f"whilelt  p1{pred_suffix}, {TMP_CNT}, {MIN_M}\n"
    code_str += _emit_primary_m_predicate()
    for _ in range(multiplier - 1):
        code_str += f"add      {TMP_CNT}, {TMP_CNT}, #{vl1}\n"
    code_str += f"whilelt  p2{pred_suffix}, {TMP_CNT}, {MIN_M}\n"
    code_str += f"add      {TMP_CNT}, {TMP_CNT}, #{vl1}\n"
    code_str += kernel_mm_loop_k(label, f"{multiplier}VL", nvl)
    return code_str


def _emit_m_loop_condition(m_size, nvl, label):
    multiplier = m_size // tile_size_from_vl(1)
    code_str = f""
    if multiplier == 1 and nvl in ["1VL", "2VL", "3VL", "4VL"]:
        code_str += f"whilelt  p1.s, {counterI}, {origM}\n"
    elif multiplier == 2 and nvl in ["1VL", "2VL"]:
        code_str += f"whilelt  p1.h, {counterI}, {origM}\n"
    elif multiplier == 3 and nvl == "1VL":
        code_str += f"whilelt  p1.h, {counterI}, {origM}\n"
        code_str += f"add      {TMP_CNT}, {counterI}, #{tile_size_from_vl(2)}\n"
        code_str += f"whilelt  p2.s, {TMP_CNT}, {origM}\n"
        code_str += f"cntp     {MIN_M}, p1, p1.h\n"
        code_str += f"cntp     {TMP_CNT}, p2, p2.s\n"
        code_str += f"add      {TMP_CNT}, {MIN_M}, {TMP_CNT}\n"
        code_str += f"whilelt  p1.b, xzr, {TMP_CNT}\n"
    elif multiplier == 4 and nvl == "1VL":
        code_str += f"whilelt  p1.b, {counterI}, {origM}\n"
    else:
        logger.error(f"Unsupported m_size={m_size}, nvl={nvl}")
    code_str += f"b.first  .loops_of_m_{nvl}_{label}\n"
    return code_str


def kernel_mm_loop_L2(m_size, label, nvl):
    logger.debug(f"currect_model:{gemm_config.currect_model}")
    code_str = f""
    code_str += f"b    .cond_of_loops_m_{nvl}_{label}\n"
    code_str += f".loops_of_m_{nvl}_{label}:\n"
    code_str += f"zero     {{za0.b}}\n"
    code_str += f"sub      {MIN_M}, {MIN_M}, {MIN_M}\n"
    code_str += _emit_m_count_init(m_size)
    code_str += gemm_config.currect_model.kernel_mm_loop_m_pre_func()

    vl1 = tile_size_from_vl(1)
    vl2 = tile_size_from_vl(2)
    vl3 = tile_size_from_vl(3)

    if m_size > vl3:
        code_str += _emit_m_loop_block(4, nvl, label)
    if m_size > vl2:
        code_str += _emit_m_loop_block(3, nvl, label)
    if m_size > vl1:
        code_str += _emit_m_loop_block(2, nvl, label)

    code_str += f".loop_m_1vl_{nvl}_{label}:\n"
    code_str += f"sub      {TMP_CNT}, {TMP_CNT}, {TMP_CNT}\n"
    code_str += f"whilelt  p1{get_predicate_suffix()}, {TMP_CNT}, {MIN_M}\n"
    code_str += f"add      {TMP_CNT}, {TMP_CNT}, #{get_whilelt_increment()}\n"
    code_str += kernel_mm_loop_k(label, "1VL", nvl)
    code_str += f".end_of_loop_k_{nvl}_{label}:\n"
    code_str += f"add      {counterI}, {counterI}, {MIN_M}\n"
    code_str += gemm_config.currect_model.kernel_mm_loop_m_post_func()
    code_str += f".cond_of_loops_m_{nvl}_{label}:\n"
    code_str += _emit_m_loop_condition(m_size, nvl, label)
    return code_str
