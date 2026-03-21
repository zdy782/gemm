from global_config import *
from kernel_save import *
from kernel_mvlxnvl import *


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


def kernel_mm_loop_k_LU(label, kernel, kernel_last_k, cnt, mvl, nvl):
    code_str = f""
    code_str += f".{label}_{mvl}_{nvl}_k:\n"
    if is_ext_precision():
        code_str += f"cmp      {cnt}, #1\n"
        code_str += f"bne      .{label}_{mvl}_{nvl}_k_normal\n"
        code_str += f"tst      {origK}, #1\n"
        code_str += f"beq      .{label}_{mvl}_{nvl}_k_normal\n"
        code_str += kernel_last_k(mvl, nvl)
        code_str += f"b        .{label}_{mvl}_{nvl}_k_end\n"
        code_str += f".{label}_{mvl}_{nvl}_k_normal:\n"
    code_str += kernel(mvl, nvl)
    if is_ext_precision():
        code_str += f".{label}_{mvl}_{nvl}_k_end:\n"
    code_str += f"sub      {cnt}, {cnt}, #{get_k_step()}\n"
    code_str += f"cmp      {cnt}, #0\n"
    code_str += f"bgt      .{label}_{mvl}_{nvl}_k\n"
    return code_str


def kernel_mm_loop_kk(k_loop_lable, mvl, nvl, kernel_bc, kernel_mm_lu_lable):
    code_str = f""
    code_str += f".{k_loop_lable}_{mvl}_{nvl}_c:\n"
    code_str += f"cmp      {wbk}, #0\n"
    code_str += f"ble      {kernel_mm_lu_lable}\n"
    code_str += kernel_bc(mvl, nvl)
    code_str += f"sub      {wbk}, {wbk}, #1\n"
    code_str += f"b.ne     .{k_loop_lable}_{mvl}_{nvl}_c\n"
    return code_str


def kernel_mm_loop_k(label, mvl, nvl):
    code_str = f""
    code_str += f"lsr      {wbk}, {origK}, #{get_k_loop_shift()}\n"
    code_str += f"sub      {TMP_CNT}, {origM}, {counterI}\n"
    code_str += f"cmp      {TMP_CNT}, {MIN_M}\n"
    code_str += f"ble      .{label}_{mvl}_{nvl}_last_m_loopk\n"
    code_str += kernel_mm_loop_kk(f"k_loop_m_{label}", mvl, nvl, kernel_bc, f"KERNEL_MM_LU_{mvl}_{nvl}_K_{label}")
    code_str += f".{label}_{mvl}_{nvl}_last_m_loopk:\n"
    code_str += f"mul      {TMP_CNT}, {MIN_N}, {LDC}\n"
    code_str += kernel_mm_loop_kk(f"k_loop_last_m_{label}", mvl, nvl, kernel_ldntb_bc, f"KERNEL_MM_LU_{mvl}_{nvl}_K_{label}")
    code_str += f"KERNEL_MM_LU_{mvl}_{nvl}_K_{label}:\n"
    code_str += f"tst      {origK}, #{get_k_remainder_mask()}\n"
    code_str += f"ble      .SAVE_C_{mvl}_{nvl}_{label}\n"
    code_str += f"ands     {TMP_CNT}, {origK}, #{get_k_remainder_mask()}\n"
    code_str += kernel_mm_loop_k_LU(f"k_mm_m_lu_k_{label}", kernel_m0, kernel_m0_last_k, TMP_CNT, mvl, nvl)
    code_str += f".SAVE_C_{mvl}_{nvl}_{label}:\n"
    code_str += _get_kernel_save_fn(mvl, nvl)(label)
    code_str += f"b        .end_of_loop_k_{nvl}_{label}\n"
    return code_str
