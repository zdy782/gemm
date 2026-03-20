import gemm_config
from gemm_type_impl import *
from global_config import *
from kernel_mm_loop_L2 import kernel_mm_loop_L2

def kernel_mm_loop_n(M, N, K, n_size = 32, m_size = 32):
    logger.debug(f"currect_model:{gemm_config.currect_model}")
    # KERNEL_MM_LOOP_L1
    code_str = f""
    code_str += f"mov  {origM}, #{M}\n"
    code_str += f"mov  {origN}, #{N}\n"
    code_str += f"mov  {origK}, #{K}\n"
    code_str += f"b   .cond_of_loops_n      // A矩阵预取\n"
    code_str += f".loops_of_n:\n"
    code_str += f"mov     {pC0}, {pC}\n"
    code_str += f"sub     {counterI}, {counterI}, {counterI}\n"
    code_str += gemm_config.currect_model.kernel_mm_loop_n_pre_func()

    pred_suffix = get_predicate_suffix()
    increment = get_whilelt_increment()

    if n_size > 48:
        code_str += f".loops_of_l1_4vl:\n"
        code_str += f"cmp     {MIN_N}, #48\n"
        code_str += f"ble     .loops_of_l1_3vl\n"
        code_str += f"sub     {TMP_CNT}, {TMP_CNT}, {TMP_CNT}\n"
        if is_ext_precision():
            code_str += f"ptrue   p0.h, vl16\n"
        else:
            code_str += f"whilelt     p0.s, {TMP_CNT}, {MIN_N}\n"
        code_str += f"add     {TMP_CNT}, {TMP_CNT}, #16\n"
        code_str += f"add     {TMP_CNT}, {TMP_CNT}, #16\n"
        code_str += f"add     {TMP_CNT}, {TMP_CNT}, #16\n"
        code_str += f"whilelt     p3{pred_suffix}, {TMP_CNT}, {MIN_N}\n"
        code_str += f"add     {TMP_CNT}, {TMP_CNT}, #16\n"

        code_str += kernel_mm_loop_L2(m_size, "k_loop_m", "4VL")
    
    if n_size > 32:
        code_str += f".loops_of_l1_3vl:\n"
        code_str += f"cmp     {MIN_N}, #32\n"
        code_str += f"ble     .loops_of_l1_2vl\n"
        code_str += f"sub     {TMP_CNT}, {TMP_CNT}, {TMP_CNT}\n"
        if is_ext_precision():
            code_str += f"ptrue   p0.h, vl16\n"
        else:
            code_str += f"whilelt     p0.s, {TMP_CNT}, {MIN_N}\n"
        code_str += f"add     {TMP_CNT}, {TMP_CNT}, #16\n"
        code_str += f"add     {TMP_CNT}, {TMP_CNT}, #16\n"
        code_str += f"whilelt     p3{pred_suffix}, {TMP_CNT}, {MIN_N}\n"
        code_str += f"add     {TMP_CNT}, {TMP_CNT}, #16\n"

        code_str += kernel_mm_loop_L2(m_size, "k_loop_m", "3VL")

    if n_size > 16:
        code_str += f".loops_of_l1_2vl:\n"
        code_str += f"cmp     {MIN_N}, #16\n"
        code_str += f"ble     .loops_of_l1_1vl\n"
        code_str += f"sub     {TMP_CNT}, {TMP_CNT}, {TMP_CNT}\n"
        if is_ext_precision():
            code_str += f"ptrue   p0.h, vl16\n"
        else:
            code_str += f"whilelt     p0.s, {TMP_CNT}, {MIN_N}\n"
        code_str += f"add     {TMP_CNT}, {TMP_CNT}, #16\n"
        code_str += f"whilelt     p3{pred_suffix}, {TMP_CNT}, {MIN_N}\n"
        code_str += f"add     {TMP_CNT}, {TMP_CNT}, #16\n"

        code_str += kernel_mm_loop_L2(m_size, "k_loop_m", "2VL")

    code_str += f".loops_of_l1_1vl:\n"
    code_str += f"sub     {TMP_CNT}, {TMP_CNT}, {TMP_CNT}\n"
    code_str += f"whilelt     p0{pred_suffix}, {TMP_CNT}, {MIN_N}\n"
    code_str += f"add     {TMP_CNT}, {TMP_CNT}, #{increment}\n"

    code_str += kernel_mm_loop_L2(m_size, "k_loop_m", "1VL")

    code_str += f".end_of_loops_m:\n"
    code_str += f"add     {counterJ}, {counterJ}, {MIN_N}\n"
    code_str += gemm_config.currect_model.kernel_mm_loop_n_post_func()
    code_str += f"mul     {TMP_CNT}, {MIN_N}, {LDC}\n"
    code_str += f"add     {pC}, {pC}, {TMP_CNT}\n"


    code_str += f".cond_of_loops_n:\n"
    if n_size == 16:
        code_str += f"whilelt     p0.s, {counterJ}, {origN}\n"
    elif n_size == 32:
        code_str += f"whilelt     p0.h, {counterJ}, {origN}\n"
    elif n_size == 48:
        code_str += f"whilelt     p0.h, {counterJ}, {origN}\n"
        code_str += f"add         {TMP_CNT}, {counterJ}, #{increment}\n"
        code_str += f"whilelt     p3.s, {TMP_CNT}, {origN}\n"
    elif n_size == 64:
        code_str += f"whilelt     p0.b, {counterJ}, {origN}\n"
    code_str += f"sub     {MIN_N}, {MIN_N}, {MIN_N}\n"
    if n_size == 16:
        code_str += f"cntp    {MIN_N}, p0, p0.s\n"
        code_str += f"b.first     .loops_of_n\n"
    elif n_size == 32:
        code_str += f"cntp    {MIN_N}, p0, p0.h\n"
        code_str += f"b.first     .loops_of_n\n"
    elif n_size == 48:
        code_str += f"cntp    {MIN_N}, p0, p0.b\n"
        code_str += f"cntp    {TMP_CNT}, p3, p3.b\n"
        code_str += f"add     {MIN_N}, {MIN_N}, {TMP_CNT}\n"
        code_str += f"cmp     {MIN_N}, #0\n"
        code_str += f"bgt      .loops_of_n\n"
    elif n_size == 64:
        code_str += f"cntp    {MIN_N}, p0, p0.b\n"
        code_str += f"b.first     .loops_of_n\n"

    return code_str