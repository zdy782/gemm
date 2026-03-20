import gemm_config
from gemm_type_impl import *
from global_config import *
from kernel_mm_loop_k import kernel_mm_loop_k

def kernel_mm_loop_L2(m_size, label, nvl):
    logger.debug(f"currect_model:{gemm_config.currect_model}")
    code_str = f""
    code_str += f"b    .cond_of_loops_m_{nvl}_{label}\n"
    code_str += f".loops_of_m_{nvl}_{label}:\n"
    code_str += f"zero     {{za0.b}}\n"
    code_str += f"sub      {MIN_M}, {MIN_M}, {MIN_M}\n"

    pred_suffix = get_predicate_suffix()
    increment = get_whilelt_increment()

    if m_size == 16:
        code_str += f"cntp     {MIN_M}, p1, p1.s\n"
    elif m_size == 32:
        code_str += f"cntp     {MIN_M}, p1, p1.h\n"
    elif m_size == 48:
        code_str += f"cntp     {MIN_M}, p1, p1.b\n"
    elif m_size == 64:
        code_str += f"cntp     {MIN_M}, p1, p1.b\n"

    code_str += gemm_config.currect_model.kernel_mm_loop_m_pre_func()

    if m_size > 48:
        code_str += f".loop_m_4vl_{nvl}_{label}:\n"
        code_str += f"cmp      {MIN_M}, #48\n"
        code_str += f"ble      .loop_m_3vl_{nvl}_{label}\n"
        code_str += f"sub      {TMP_CNT}, {TMP_CNT}, {TMP_CNT}\n"
        code_str += f"whilelt  p1{pred_suffix}, {TMP_CNT}, {MIN_M}\n"
        if is_ext_precision():
            code_str += f"ptrue   p1.h, vl16\n"
        else:
            code_str += f"whilelt     p1.s, {TMP_CNT}, {MIN_M}\n"
        code_str += f"add      {TMP_CNT}, {TMP_CNT}, #16\n"
        code_str += f"add      {TMP_CNT}, {TMP_CNT}, #16\n"
        code_str += f"add      {TMP_CNT}, {TMP_CNT}, #16\n"
        code_str += f"whilelt  p2{pred_suffix}, {TMP_CNT}, {MIN_M}\n"
        code_str += f"add      {TMP_CNT}, {TMP_CNT}, #16\n"

        code_str += kernel_mm_loop_k(label, "4VL", nvl)

    if m_size > 32:
        code_str += f".loop_m_3vl_{nvl}_{label}:\n"
        code_str += f"cmp      {MIN_M}, #32\n"
        code_str += f"ble      .loop_m_2vl_{nvl}_{label}\n"
        code_str += f"sub      {TMP_CNT}, {TMP_CNT}, {TMP_CNT}\n"
        if is_ext_precision():
            code_str += f"ptrue   p1.h, vl16\n"
        else:
            code_str += f"whilelt     p1.s, {TMP_CNT}, {MIN_M}\n"
        code_str += f"add      {TMP_CNT}, {TMP_CNT}, #16\n"
        code_str += f"add      {TMP_CNT}, {TMP_CNT}, #16\n"
        code_str += f"whilelt  p2{pred_suffix}, {TMP_CNT}, {MIN_M}\n"
        code_str += f"add      {TMP_CNT}, {TMP_CNT}, #16\n"

        code_str += kernel_mm_loop_k(label, "3VL", nvl)

    if m_size > 16:
        code_str += f".loop_m_2vl_{nvl}_{label}:\n"
        code_str += f"cmp      {MIN_M}, #16\n"
        code_str += f"ble      .loop_m_1vl_{nvl}_{label}\n"
        code_str += f"sub      {TMP_CNT}, {TMP_CNT}, {TMP_CNT}\n"
        if is_ext_precision():
            code_str += f"ptrue   p1.h, vl16\n"
        else:
            code_str += f"whilelt     p1.s, {TMP_CNT}, {MIN_M}\n"
        code_str += f"add      {TMP_CNT}, {TMP_CNT}, #16\n"
        code_str += f"whilelt  p2{pred_suffix}, {TMP_CNT}, {MIN_M}\n"
        code_str += f"add      {TMP_CNT}, {TMP_CNT}, #16\n"

        code_str += kernel_mm_loop_k(label, "2VL", nvl)

    code_str += f".loop_m_1vl_{nvl}_{label}:\n"
    code_str += f"sub      {TMP_CNT}, {TMP_CNT}, {TMP_CNT}\n"
    code_str += f"whilelt  p1{pred_suffix}, {TMP_CNT}, {MIN_M}\n"
    code_str += f"add      {TMP_CNT}, {TMP_CNT}, #{increment}\n"

    code_str += kernel_mm_loop_k(label, "1VL", nvl)

    code_str += f".end_of_loop_k_{nvl}_{label}:\n"
    code_str += f"add      {counterI}, {counterI}, {MIN_M}\n"

    code_str += gemm_config.currect_model.kernel_mm_loop_m_post_func()
    code_str += f".cond_of_loops_m_{nvl}_{label}:\n"
    if (m_size == 16) and (nvl in ["1VL", "2VL", "3VL", "4VL"]):
        code_str += f"whilelt  p1.s, {counterI}, {origM}\n"
    elif (m_size == 32 and (nvl in ["1VL", "2VL"])):
        code_str += f"whilelt  p1.h, {counterI}, {origM}\n"
    elif (m_size == 48) and (nvl == "1VL"):
        code_str += f"whilelt  p1.h, {counterI}, {origM}\n"
        code_str += f"add      {TMP_CNT}, {counterI}, #32\n"
        code_str += f"whilelt  p2.s, {TMP_CNT}, {origM}\n"
        code_str += f"cntp     {MIN_M}, p1, p1.h\n"
        code_str += f"cntp     {TMP_CNT}, p2, p2.s\n"
        code_str += f"add      {TMP_CNT}, {MIN_M}, {TMP_CNT}\n"
        code_str += f"whilelt  p1.b, xzr, {TMP_CNT}\n"
    elif (m_size == 64) and (nvl == "1VL"):
        code_str += f"whilelt  p1.b, {counterI}, {origM}\n"
    else:
        logger.error(f"m_size:{m_size} * n_size:{n_size} error!")

    code_str += f"b.first  .loops_of_m_{nvl}_{label}\n"

    return code_str