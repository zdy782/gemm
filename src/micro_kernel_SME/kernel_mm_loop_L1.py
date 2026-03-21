import gemm_config
from gemm_type_impl import *
from global_config import *
from kernel_mm_loop_L2 import kernel_mm_loop_L2


def _emit_primary_n_predicate():
    if is_ext_precision():
        return f"ptrue   p0.h, vl{get_ext_logical_vl()}\n"
    return f"whilelt     p0.s, {TMP_CNT}, {MIN_N}\n"


def _emit_n_loop_block(multiplier, m_size):
    next_label = f".loops_of_l1_{multiplier - 1}vl" if multiplier > 2 else ".loops_of_l1_1vl"
    threshold = tile_size_from_vl(multiplier - 1)
    pred_suffix = get_predicate_suffix()
    vl1 = tile_size_from_vl(1)
    code_str = f""
    code_str += f".loops_of_l1_{multiplier}vl:\n"
    code_str += f"cmp     {MIN_N}, #{threshold}\n"
    code_str += f"ble     {next_label}\n"
    code_str += f"sub     {TMP_CNT}, {TMP_CNT}, {TMP_CNT}\n"
    code_str += _emit_primary_n_predicate()
    for _ in range(multiplier - 1):
        code_str += f"add     {TMP_CNT}, {TMP_CNT}, #{vl1}\n"
    code_str += f"whilelt     p3{pred_suffix}, {TMP_CNT}, {MIN_N}\n"
    code_str += f"add     {TMP_CNT}, {TMP_CNT}, #{vl1}\n"
    code_str += kernel_mm_loop_L2(m_size, "k_loop_m", f"{multiplier}VL")
    return code_str


def _emit_n_loop_condition(n_size):
    increment = get_whilelt_increment()
    multiplier = n_size // tile_size_from_vl(1)
    code_str = f""
    if multiplier == 1:
        code_str += f"whilelt     p0.s, {counterJ}, {origN}\n"
        code_str += f"sub     {MIN_N}, {MIN_N}, {MIN_N}\n"
        code_str += f"cntp    {MIN_N}, p0, p0.s\n"
        code_str += f"b.first     .loops_of_n\n"
    elif multiplier == 2:
        code_str += f"whilelt     p0.h, {counterJ}, {origN}\n"
        code_str += f"sub     {MIN_N}, {MIN_N}, {MIN_N}\n"
        code_str += f"cntp    {MIN_N}, p0, p0.h\n"
        code_str += f"b.first     .loops_of_n\n"
    elif multiplier == 3:
        code_str += f"whilelt     p0.h, {counterJ}, {origN}\n"
        code_str += f"add         {TMP_CNT}, {counterJ}, #{increment}\n"
        code_str += f"whilelt     p3.s, {TMP_CNT}, {origN}\n"
        code_str += f"sub     {MIN_N}, {MIN_N}, {MIN_N}\n"
        code_str += f"cntp    {MIN_N}, p0, p0.b\n"
        code_str += f"cntp    {TMP_CNT}, p3, p3.b\n"
        code_str += f"add     {MIN_N}, {MIN_N}, {TMP_CNT}\n"
        code_str += f"cmp     {MIN_N}, #0\n"
        code_str += f"bgt      .loops_of_n\n"
    elif multiplier == 4:
        code_str += f"whilelt     p0.b, {counterJ}, {origN}\n"
        code_str += f"sub     {MIN_N}, {MIN_N}, {MIN_N}\n"
        code_str += f"cntp    {MIN_N}, p0, p0.b\n"
        code_str += f"b.first     .loops_of_n\n"
    return code_str


def kernel_mm_loop_n(M, N, K, n_size=None, m_size=None):
    logger.debug(f"currect_model:{gemm_config.currect_model}")
    if n_size is None:
        n_size = tile_size_from_vl(2)
    if m_size is None:
        m_size = tile_size_from_vl(2)
    code_str = f""
    code_str += f"mov  {origM}, #{M}\n"
    code_str += f"mov  {origN}, #{N}\n"
    code_str += f"mov  {origK}, #{K}\n"
    code_str += f"b   .cond_of_loops_n      // A矩阵预取\n"
    code_str += f".loops_of_n:\n"
    code_str += f"mov     {pC0}, {pC}\n"
    code_str += f"sub     {counterI}, {counterI}, {counterI}\n"
    code_str += gemm_config.currect_model.kernel_mm_loop_n_pre_func()

    vl1 = tile_size_from_vl(1)
    vl2 = tile_size_from_vl(2)
    vl3 = tile_size_from_vl(3)

    if n_size > vl3:
        code_str += _emit_n_loop_block(4, m_size)
    if n_size > vl2:
        code_str += _emit_n_loop_block(3, m_size)
    if n_size > vl1:
        code_str += _emit_n_loop_block(2, m_size)

    code_str += f".loops_of_l1_1vl:\n"
    code_str += f"sub     {TMP_CNT}, {TMP_CNT}, {TMP_CNT}\n"
    code_str += f"whilelt     p0{get_predicate_suffix()}, {TMP_CNT}, {MIN_N}\n"
    code_str += f"add     {TMP_CNT}, {TMP_CNT}, #{get_whilelt_increment()}\n"
    code_str += kernel_mm_loop_L2(m_size, "k_loop_m", "1VL")
    code_str += f".end_of_loops_m:\n"
    code_str += f"add     {counterJ}, {counterJ}, {MIN_N}\n"
    code_str += gemm_config.currect_model.kernel_mm_loop_n_post_func()
    code_str += f"mul     {TMP_CNT}, {MIN_N}, {LDC}\n"
    code_str += f"add     {pC}, {pC}, {TMP_CNT}\n"
    code_str += f".cond_of_loops_n:\n"
    code_str += _emit_n_loop_condition(n_size)
    return code_str
