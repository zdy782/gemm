import gemm_config
from global_config import *
from kernel_asm import *

kernel_fun_map = {
    "kernel_4VL_1VL" : kernel_4VL_1VL,
    "kernel_1VL_4VL" : kernel_1VL_4VL,
    "kernel_3VL_1VL" : kernel_3VL_1VL,
    "kernel_1VL_3VL" : kernel_1VL_3VL,
    "kernel_2VL_2VL" : kernel_2VL_2VL,
    "kernel_2VL_1VL" : kernel_2VL_1VL,
    "kernel_1VL_2VL" : kernel_1VL_2VL,
    "kernel_1VL_1VL" : kernel_1VL_1VL,
}

kernel_fun_map_last_k = {
    "kernel_4VL_1VL" : kernel_4VL_1VL_last_k,
    "kernel_1VL_4VL" : kernel_1VL_4VL_last_k,
    "kernel_3VL_1VL" : kernel_3VL_1VL_last_k,
    "kernel_1VL_3VL" : kernel_1VL_3VL_last_k,
    "kernel_2VL_2VL" : kernel_2VL_2VL_last_k,
    "kernel_2VL_1VL" : kernel_2VL_1VL_last_k,
    "kernel_1VL_2VL" : kernel_1VL_2VL_last_k,
    "kernel_1VL_1VL" : kernel_1VL_1VL_last_k,
}

def set_load_inst(type):
    if is_bf16():
        load_inst = LDNT1_H
        if type == "small":
            load_inst = LD1_H
    else:
        load_inst = LDNT1
        if type == "small":
            load_inst = LD1
    return load_inst

def kernel_m0(mvl, nvl):
    code_str = f""
    code_str += kernel_fun_map[f"kernel_{mvl}_{nvl}"]("z0", "z1", "z2", "z3", "z16", "z17", "z24", "z25")

    return code_str

def kernel_m1(mvl, nvl):
    code_str = f""
    code_str += kernel_fun_map[f"kernel_{mvl}_{nvl}"]("z4", "z5", "z6", "z7", "z18", "z19", "z30", "z31")

    return code_str

def kernel_m2(mvl, nvl):
    code_str = f""
    code_str += kernel_fun_map[f"kernel_{mvl}_{nvl}"]("z8", "z9", "z10", "z11", "z20", "z21", "z24", "z25")

    return code_str

def kernel_m3(mvl, nvl):
    code_str = f""
    code_str += kernel_fun_map[f"kernel_{mvl}_{nvl}"]("z12", "z13", "z14", "z15", "z22", "z23", "z30", "z31")

    return code_str

def kernel_m0_last_k(mvl, nvl):
    code_str = f""
    code_str += kernel_fun_map_last_k[f"kernel_{mvl}_{nvl}"]("z0", "z1", "z2", "z3", "z16", "z17", "z24", "z25")

    return code_str

def kernel_m1_last_k(mvl, nvl):
    code_str = f""
    code_str += kernel_fun_map_last_k[f"kernel_{mvl}_{nvl}"]("z4", "z5", "z6", "z7", "z18", "z19", "z30", "z31")

    return code_str

def kernel_m2_last_k(mvl, nvl):
    code_str = f""
    code_str += kernel_fun_map_last_k[f"kernel_{mvl}_{nvl}"]("z8", "z9", "z10", "z11", "z20", "z21", "z24", "z25")

    return code_str

def kernel_m3_last_k(mvl, nvl):
    code_str = f""
    code_str += kernel_fun_map_last_k[f"kernel_{mvl}_{nvl}"]("z12", "z13", "z14", "z15", "z22", "z23", "z30", "z31")

    return code_str

def kernel_ldnt1d_m0(mvl, nvl):
    code_str = f""
    code_str += kernel_fun_map[f"kernel_{mvl}_{nvl}"]("z0", "z1", "z2", "z3", "z16", "z17", "z24", "z25", set_load_inst(gemm_config.type))

    return code_str

def kernel_ldnt1d_m1(mvl, nvl):
    code_str = f""
    code_str += kernel_fun_map[f"kernel_{mvl}_{nvl}"]("z4", "z5", "z6", "z7", "z18", "z19", "z30", "z31", set_load_inst(gemm_config.type))

    return code_str

def kernel_ldnt1d_m2(mvl, nvl):
    code_str = f""
    code_str += kernel_fun_map[f"kernel_{mvl}_{nvl}"]("z8", "z9", "z10", "z11", "z20", "z21", "z24", "z25", set_load_inst(gemm_config.type))

    return code_str

def kernel_ldnt1d_m3(mvl, nvl):
    code_str = f""
    code_str += kernel_fun_map[f"kernel_{mvl}_{nvl}"]("z12", "z13", "z14", "z15", "z22", "z23", "z30", "z31", set_load_inst(gemm_config.type))

    return code_str

def kernel_bc(mvl, nvl):
    code_str = f""
    code_str += kernel_m0(mvl, nvl)
    code_str += kernel_m1(mvl, nvl)
    code_str += kernel_m2(mvl, nvl)
    code_str += kernel_m3(mvl, nvl)
    code_str += kernel_m0(mvl, nvl)
    code_str += kernel_m1(mvl, nvl)
    code_str += kernel_m2(mvl, nvl)
    code_str += kernel_m3(mvl, nvl)

    return code_str

def kernel_ldntb_bc(mvl, nvl):
    code_str = f""
    code_str += kernel_ldnt1d_m0(mvl, nvl)
    code_str += kernel_ldnt1d_m1(mvl, nvl)
    code_str += kernel_ldnt1d_m2(mvl, nvl)
    code_str += kernel_ldnt1d_m3(mvl, nvl)
    code_str += kernel_ldnt1d_m0(mvl, nvl)
    code_str += kernel_ldnt1d_m1(mvl, nvl)
    code_str += kernel_ldnt1d_m2(mvl, nvl)
    code_str += kernel_ldnt1d_m3(mvl, nvl)

    return code_str