from global_config import *
from kernel_asm import *

def get_pg0():
    if is_ext_precision():
        return "p4"
    else:
        return "p1"

def get_pg1():
    if is_ext_precision():
        return "p5"
    else:
        return "p2"

def save_zacol_1VL(pc, c0, c1, c2, c3, base_idx, idx, pg0, pg1, rab0, rc0, rab1, rc1, t0, t1, t2 = "z30", t3 = "z31"):
    code_str = f""
    code_str += save_zacol(pc, "#0", c0, base_idx, idx, pg0, rab0, rc0)
    code_str += f"add      {pc}, {pc}, {LDC}, lsl #2\n"

    return code_str

def save_zacol_2VL(pc, c0, c1, c2, c3, base_idx, idx, pg0, pg1, rab0, rc0, rab1, rc1, t0, t1, t2 = "z30", t3 = "z31"):
    code_str = f""
    code_str += save_zacol(pc, "#0", c0, base_idx, idx, pg0, rab0, rc0)
    code_str += save_zacol(pc, "#1", c1, base_idx, idx, pg1, rab1, rc1)
    code_str += f"add      {pc}, {pc}, {LDC}, lsl #2\n"

    return code_str

def save_zacol_3VL(pc, c0, c1, c2, c3, base_idx, idx, pg0, pg1, rab0, rc0, rab1, rc1, t0, t1, t2 = "z30", t3 = "z31"):
    code_str = f""
    code_str += save_zacol(pc, "#0", c0, base_idx, idx, pg0, rab0, rc0)
    code_str += save_zacol(pc, "#1", c1, base_idx, idx, pg0, rab1, rc1)
    code_str += save_zacol(pc, "#2", c2, base_idx, idx, pg1, t0, t1)
    code_str += f"add      {pc}, {pc}, {LDC}, lsl #2\n"

    return code_str

def save_zacol_4VL(pc, c0, c1, c2, c3, base_idx, idx, pg0, pg1, rab0, rc0, rab1, rc1, t0, t1, t2 = "z30", t3 = "z31"):
    code_str = f""
    code_str += save_zacol(pc, "#0", c0, base_idx, idx, pg0, rab0, rc0)
    code_str += save_zacol(pc, "#1", c1, base_idx, idx, pg0, rab1, rc1)
    code_str += save_zacol(pc, "#2", c2, base_idx, idx, pg0, t0, t1)
    code_str += save_zacol(pc, "#3", c3, base_idx, idx, pg1, t2, t3)
    code_str += f"add      {pc}, {pc}, {LDC}, lsl #2\n"

    return code_str

save_zacol_map = {
    "save_zacol_4VL" : save_zacol_4VL,
    "save_zacol_3VL" : save_zacol_3VL,
    "save_zacol_2VL" : save_zacol_2VL,
    "save_zacol_1VL" : save_zacol_1VL,
}

def kernel_save_c_base_val(lable, num, mvl, nvl, pc, c0 = "za0", c1 = "za1", c2 = "za2", c3 = "za3"):
    pg0 = get_pg0()
    pg1 = get_pg1()
    code_str = f""
    code_str += f"mov      {TMP_CNT}, {num}\n"
    code_str += f"mov      w12, #0\n"
    code_str += f"mov      {TMP_PTR1}, {pc}\n"
    code_str += f".loop_save_c_j_{mvl}_{nvl}_{lable}:\n"
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pc, c0, c1, c2, c3, "w12", "#0", pg0, pg1, "z0", "z1", "z2", "z3", "z25", "z26")
    code_str += f"add      w12, w12, #1\n"
    code_str += f"add      {TMP_PTR1}, {TMP_PTR1}, {LDC}\n"
    code_str += f"mov      {pc}, {TMP_PTR1}\n"
    code_str += f"subs     {TMP_CNT}, {TMP_CNT}, #1\n"
    code_str += f"bgt      .loop_save_c_j_{mvl}_{nvl}_{lable}\n"

    return code_str

def kernel_save_c_base_n_1VL_(mvl, c0 = "za0", c1 = "za1", c2 = "za2", c3 = "za3"):
    pg0 = get_pg0()
    pg1 = get_pg1()
    code_str = f""
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC0, c0, c1, c2, c3, "w12", "#0", pg0, pg1, "z0", "z1", "z2", "z3", "z25", "z26")
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC1, c0, c1, c2, c3, "w12", "#1", pg0, pg1, "z4", "z5", "z6", "z7", "z25", "z26")
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC2, c0, c1, c2, c3, "w12", "#2", pg0, pg1, "z8", "z9", "z10", "z11", "z25", "z26")
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC3, c0, c1, c2, c3, "w12", "#3", pg0, pg1, "z12", "z13", "z14", "z15", "z25", "z26")

    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC0, c0, c1, c2, c3, "w13", "#0", pg0, pg1, "z0", "z1", "z2", "z3", "z25", "z26")
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC1, c0, c1, c2, c3, "w13", "#1", pg0, pg1, "z4", "z5", "z6", "z7", "z25", "z26")
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC2, c0, c1, c2, c3, "w13", "#2", pg0, pg1, "z8", "z9", "z10", "z11", "z25", "z26")
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC3, c0, c1, c2, c3, "w13", "#3", pg0, pg1, "z12", "z13", "z14", "z15", "z25", "z26")

    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC0, c0, c1, c2, c3, "w14", "#0", pg0, pg1, "z0", "z1", "z2", "z3", "z25", "z26")
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC1, c0, c1, c2, c3, "w14", "#1", pg0, pg1, "z4", "z5", "z6", "z7", "z25", "z26")
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC2, c0, c1, c2, c3, "w14", "#2", pg0, pg1, "z8", "z9", "z10", "z11", "z25", "z26")
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC3, c0, c1, c2, c3, "w14", "#3", pg0, pg1, "z12", "z13", "z14", "z15", "z25", "z26")

    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC0, c0, c1, c2, c3, "w15", "#0", pg0, pg1, "z0", "z1", "z2", "z3", "z25", "z26")
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC1, c0, c1, c2, c3, "w15", "#1", pg0, pg1, "z4", "z5", "z6", "z7", "z25", "z26")
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC2, c0, c1, c2, c3, "w15", "#2", pg0, pg1, "z8", "z9", "z10", "z11", "z25", "z26")
    code_str += save_zacol_map[f"save_zacol_{mvl}"](pC3, c0, c1, c2, c3, "w15", "#3", pg0, pg1, "z12", "z13", "z14", "z15", "z25", "z26")

    return code_str

def kernel_save_c_base_n_1VL(lable, mvl, nvl, c0 = "za0", c1 = "za1", c2 = "za2", c3 = "za3"):
    code_str = f""
    code_str += f"mov      w12, #0\n"
    code_str += f"mov      w13, #4\n"
    code_str += f"mov      w14, #8\n"
    code_str += f"mov      w15, #12\n"
    code_str += f"ands     {TMP_CNT_POST}, {MIN_N}, #15\n"
    code_str += f"bne      .kernel_save_c_val_{mvl}_{nvl}_{lable}\n"
    code_str += f"add      {pC1}, {pC0}, {LDC}\n"
    code_str += f"add      {pC2}, {pC0}, {LDC}, lsl #1\n"
    code_str += f"add      {pC3}, {pC1}, {LDC}, lsl #1\n"
    code_str += kernel_save_c_base_n_1VL_(mvl, c0, c1, c2, c3)
    code_str += f"b        .kernel_save_c_val_{mvl}_{nvl}_{lable}_end\n"
    code_str += f".kernel_save_c_val_{mvl}_{nvl}_{lable}:\n"
    code_str += kernel_save_c_base_val(lable, TMP_CNT_POST, mvl, nvl, pC0, c0, c1, c2, c3)
    code_str += f".kernel_save_c_val_{mvl}_{nvl}_{lable}_end:\n"

    return code_str 

def kernel_save_c_base_n_2VL(lable, mvl):
    code_str = f""
    code_str += f"mov      w12, #0\n"
    code_str += f"mov      w13, #4\n"
    code_str += f"mov      w14, #8\n"
    code_str += f"mov      w15, #12\n"
    code_str += f"mov      {TMP_PTR}, {pC0}\n"
    code_str += f"add      {pC1}, {pC0}, {LDC}\n"
    code_str += f"add      {pC2}, {pC0}, {LDC}, lsl #1\n"
    code_str += f"add      {pC3}, {pC1}, {LDC}, lsl #1\n"
    code_str += kernel_save_c_base_n_1VL_(mvl, "za0", "za2")
    code_str += kernel_save_c_base_n_1VL(lable, mvl, "2VL", "za1", "za3")
    code_str += f"add      {pC0}, {TMP_PTR}, {MIN_M}, lsl #2\n"

    return code_str

def kernel_save_c_base_n_4VL(lable, mvl):
    code_str = f""
    code_str += f"mov      w12, #0\n"
    code_str += f"mov      w13, #4\n"
    code_str += f"mov      w14, #8\n"
    code_str += f"mov      w15, #12\n"
    code_str += f"mov      {TMP_PTR}, {pC0}\n"
    code_str += f"add      {pC1}, {pC0}, {LDC}\n"
    code_str += f"add      {pC2}, {pC0}, {LDC}, lsl #1\n"
    code_str += f"add      {pC3}, {pC1}, {LDC}, lsl #1\n"
    code_str += kernel_save_c_base_n_1VL_(mvl, "za0")
    code_str += kernel_save_c_base_n_1VL_(mvl, "za1")
    code_str += kernel_save_c_base_n_1VL_(mvl, "za2")
    code_str += kernel_save_c_base_n_1VL(lable, mvl, "4VL", "za3")
    code_str += f"add      {pC0}, {TMP_PTR}, {MIN_M}, lsl #2\n"

    return code_str

def kernel_save_c_base_n_3VL(lable, mvl):
    code_str = f""
    code_str += f"mov      w12, #0\n"
    code_str += f"mov      w13, #4\n"
    code_str += f"mov      w14, #8\n"
    code_str += f"mov      w15, #12\n"
    code_str += f"mov      {TMP_PTR}, {pC0}\n"
    code_str += f"add      {pC1}, {pC0}, {LDC}\n"
    code_str += f"add      {pC2}, {pC0}, {LDC}, lsl #1\n"
    code_str += f"add      {pC3}, {pC1}, {LDC}, lsl #1\n"
    code_str += kernel_save_c_base_n_1VL_(mvl, "za0")
    code_str += kernel_save_c_base_n_1VL_(mvl, "za1")
    code_str += kernel_save_c_base_n_1VL(lable, mvl, "3VL", "za2")
    code_str += f"add      {pC0}, {TMP_PTR}, {MIN_M}, lsl #2\n"

    return code_str

def kernel_save_c_4VL_1VL(lable):
    code_str = f""
    code_str += f"mov      {TMP_PTR}, {pC0}\n"
    code_str += kernel_save_c_base_n_1VL(lable, "4VL", "1VL", "za0", "za1", "za2", "za3")
    code_str += f"add      {pC0}, {TMP_PTR}, {MIN_M}, lsl #2\n"

    return code_str

def kernel_save_c_1VL_4VL(lable):
    code_str = f""
    code_str += kernel_save_c_base_n_4VL(lable, "1VL")

    return code_str

def kernel_save_c_3VL_1VL(lable):
    code_str = f""
    code_str += f"mov      {TMP_PTR}, {pC0}\n"
    code_str += kernel_save_c_base_n_1VL(lable, "3VL", "1VL", "za0", "za1", "za2")
    code_str += f"add      {pC0}, {TMP_PTR}, {MIN_M}, lsl #2\n"

    return code_str


def kernel_save_c_1VL_3VL(lable):
    code_str = f""
    code_str += kernel_save_c_base_n_3VL(lable, "1VL")

    return code_str

def kernel_save_c_2VL_2VL(lable):
    code_str = f""
    code_str += kernel_save_c_base_n_2VL(lable, "2VL")

    return code_str

def kernel_save_c_1VL_2VL(lable):
    code_str = f""
    code_str += kernel_save_c_base_n_2VL(lable, "1VL")

    return code_str

def kernel_save_c_2VL_1VL(lable):
    code_str = f""
    code_str += f"mov      {TMP_PTR}, {pC0}\n"
    code_str += kernel_save_c_base_n_1VL(lable, "2VL", "1VL", "za0", "za2")
    code_str += f"add      {pC0}, {TMP_PTR}, {MIN_M}, lsl #2\n"

    return code_str

def kernel_save_c_1VL_1VL(lable):
    code_str = f""
    code_str += f"mov      {TMP_PTR}, {pC0}\n"
    code_str += kernel_save_c_base_n_1VL(lable, "1VL", "1VL", "za0", "za2")
    code_str += f"add      {pC0}, {TMP_PTR}, {MIN_M}, lsl #2\n"

    return code_str
