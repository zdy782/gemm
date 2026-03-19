import gemm_config
from gemm_type_impl import *
from global_config import *

def kernel_4VL_1VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    if is_bf16():
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1", b0, "p0", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za0.s, p4/m, p6/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_a1(a1, "p1", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za1.s, p4/m, p6/m, {a1}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_a2(a2, "p1", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za2.s, p4/m, p6/m, {a2}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += f"add          {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
        code_str += gemm_config.currect_model.load_a3(a3, "p2", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za3.s, p5/m, p6/m, {a3}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += f"add          {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
    else:
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1/z", b0, "p0/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za0.s, p1/m, p0/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_a1(a1, "p1/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za1.s, p1/m, p0/m, {a1}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_a2(a2, "p1/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za2.s, p1/m, p0/m, {a2}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += f"add          {pB0}, {pB0}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
        code_str += gemm_config.currect_model.load_a3(a3, "p2/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za3.s, p2/m, p0/m, {a3}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += f"add          {pA0}, {pA0}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"

    return code_str

def kernel_1VL_4VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    if is_bf16():
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1", b0, "p0", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za0.s, p4/m, p6/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_b1(b1, "p0", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za1.s, p4/m, p6/m, {a0}{get_element_suffix()}, {b1}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_b2(b2, "p0", ldopt, ldaopt)
        code_str += f"add          {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
        code_str += f"{get_mopa_inst()}        za2.s, p4/m, p6/m, {a0}{get_element_suffix()}, {b2}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_b3(b3, "p3", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za3.s, p4/m, p3/m, {a0}{get_element_suffix()}, {b3}{get_element_suffix()}\n"
        code_str += f"add          {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
    else:
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1/z", b0, "p0/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za0.s, p1/m, p0/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_b1(b1, "p0/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za1.s, p1/m, p0/m, {a0}{get_element_suffix()}, {b1}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_b2(b2, "p0/z", ldopt, ldaopt)
        code_str += f"add          {pA0}, {pA0}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
        code_str += f"{get_mopa_inst()}        za2.s, p1/m, p0/m, {a0}{get_element_suffix()}, {b2}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_b3(b3, "p3/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za3.s, p1/m, p3/m, {a0}{get_element_suffix()}, {b3}{get_element_suffix()}\n"
        code_str += f"add          {pB0}, {pB0}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"

    return code_str

def kernel_3VL_1VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    if is_bf16():
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1", b0, "p0", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za0.s, p4/m, p6/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_a1(a1, "p1", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za1.s, p4/m, p6/m, {a1}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += f"add          {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
        code_str += gemm_config.currect_model.load_a2(a2, "p2", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za2.s, p5/m, p6/m, {a2}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += f"add          {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
    else:
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1/z", b0, "p0/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za0.s, p1/m, p0/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_a1(a1, "p1/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za1.s, p1/m, p0/m, {a1}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += f"add          {pB0}, {pB0}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
        code_str += gemm_config.currect_model.load_a2(a2, "p2/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za2.s, p2/m, p0/m, {a2}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += f"add          {pA0}, {pA0}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"

    return code_str

def kernel_1VL_3VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    if is_bf16():
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1", b0, "p0", ldopt, ldaopt)
        code_str += f"bfmopa      za0.s, p4/m, p6/m, {a0}.h, {b0}.h\n"
        code_str += gemm_config.currect_model.load_b1(b1, "p0", ldopt, ldaopt)
        code_str += f"bfmopa      za1.s, p4/m, p6/m, {a0}.h, {b1}.h\n"
        code_str += f"add         {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
        code_str += gemm_config.currect_model.load_b2(b2, "p3", ldopt, ldaopt)
        code_str += f"bfmopa      za2.s, p4/m, p3/m, {a0}.h, {b2}.h\n"
        code_str += f"add         {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
    else:
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1/z", b0, "p0/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za0.s, p1/m, p0/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_b1(b1, "p0/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za1.s, p1/m, p0/m, {a0}{get_element_suffix()}, {b1}{get_element_suffix()}\n"
        code_str += f"add          {pA0}, {pA0}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
        code_str += gemm_config.currect_model.load_b2(b2, "p3/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za2.s, p1/m, p3/m, {a0}{get_element_suffix()}, {b2}{get_element_suffix()}\n"
        code_str += f"add          {pB0}, {pB0}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
    return code_str

def kernel_2VL_2VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    if is_bf16():
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1", b0, "p0", ldopt, ldaopt)
        code_str += f"bfmopa      za0.s, p4/m, p6/m, {a0}.h, {b0}.h\n"
        code_str += gemm_config.currect_model.load_b1(b1, "p3", ldopt, ldaopt)
        code_str += f"bfmopa      za1.s, p4/m, p3/m, {a0}.h, {b1}.h\n"
        code_str += gemm_config.currect_model.load_a1(a1, "p2", ldopt, ldaopt)
        code_str += f"bfmopa      za2.s, p5/m, p6/m, {a1}.h, {b0}.h\n"
        code_str += f"add         {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
        code_str += f"bfmopa      za3.s, p5/m, p3/m, {a1}.h, {b1}.h\n"
        code_str += f"add         {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
    else:
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1/z", b0, "p0/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za0.s, p1/m, p0/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_b1(b1, "p3/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za1.s, p1/m, p3/m, {a0}{get_element_suffix()}, {b1}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_a1(a1, "p2/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za2.s, p2/m, p0/m, {a1}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += f"add          {pB0}, {pB0}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
        code_str += f"{get_mopa_inst()}        za3.s, p2/m, p3/m, {a1}{get_element_suffix()}, {b1}{get_element_suffix()}\n"
        code_str += f"add          {pA0}, {pA0}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"

    return code_str

def kernel_1VL_2VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    if is_bf16():
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1", b0, "p0", ldopt, ldaopt)
        code_str += f"bfmopa      za0.s, p4/m, p6/m, {a0}.h, {b0}.h\n"
        code_str += gemm_config.currect_model.load_b1(b1, "p3", ldopt, ldaopt)
        code_str += f"bfmopa      za1.s, p4/m, p3/m, {a0}.h, {b1}.h\n"
        code_str += f"add         {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
        code_str += f"add         {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
    else:
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1/z", b0, "p0/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za0.s, p1/m, p0/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_b1(b1, "p3/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za1.s, p1/m, p3/m, {a0}{get_element_suffix()}, {b1}{get_element_suffix()}\n"
        code_str += f"add          {pB0}, {pB0}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
        code_str += f"add          {pA0}, {pA0}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"

    return code_str

def kernel_2VL_1VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    if is_bf16():
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1", b0, "p0", ldopt, ldaopt)
        code_str += f"bfmopa      za0.s, p4/m, p6/m, {a0}.h, {b0}.h\n"
        code_str += gemm_config.currect_model.load_a1(a1, "p2", ldopt, ldaopt)
        code_str += f"bfmopa      za2.s, p5/m, p6/m, {a1}.h, {b0}.h\n"
        code_str += f"add         {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
        code_str += f"add         {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
    else:
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1/z", b0, "p0/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za0.s, p1/m, p0/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += gemm_config.currect_model.load_a1(a1, "p2/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za2.s, p2/m, p0/m, {a1}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += f"add          {pB0}, {pB0}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
        code_str += f"add          {pA0}, {pA0}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"

    return code_str

def kernel_1VL_1VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    if is_bf16():
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1", b0, "p0", ldopt, ldaopt)
        code_str += f"bfmopa      za0.s, p4/m, p6/m, {a0}.h, {b0}.h\n"
        code_str += f"add         {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
        code_str += f"add         {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
    else:
        code_str += gemm_config.currect_model.load_a0b0(a0, "p1/z", b0, "p0/z", ldopt, ldaopt)
        code_str += f"{get_mopa_inst()}        za0.s, p1/m, p0/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
        code_str += f"add          {pB0}, {pB0}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
        code_str += f"add          {pA0}, {pA0}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"

    return code_str

def kernel_4VL_1VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    code_str += gemm_config.currect_model.load_a0b0_last_k(a0, "p1", b0, "p0", ldopt, ldaopt)
    code_str += f"{get_mopa_inst()}        za0.s, p4/m, p6/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
    code_str += gemm_config.currect_model.load_a1_last_k(a1, "p1", ldopt, ldaopt)
    code_str += f"{get_mopa_inst()}        za1.s, p4/m, p6/m, {a1}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
    code_str += gemm_config.currect_model.load_a2_last_k(a2, "p1", ldopt, ldaopt)
    code_str += f"{get_mopa_inst()}        za2.s, p4/m, p6/m, {a2}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
    code_str += f"add          {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
    code_str += gemm_config.currect_model.load_a3_last_k(a3, "p2", ldopt, ldaopt)
    code_str += f"{get_mopa_inst()}        za3.s, p5/m, p6/m, {a3}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
    code_str += f"add          {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"

    return code_str

def kernel_1VL_4VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    code_str += gemm_config.currect_model.load_a0b0_last_k(a0, "p1", b0, "p0", ldopt, ldaopt)
    code_str += f"bfmopa        za0.s, p4/m, p6/m, {a0}.h, {b0}.h\n"
    code_str += gemm_config.currect_model.load_b1_last_k(b1, "p0", ldopt, ldaopt)
    code_str += f"bfmopa        za1.s, p4/m, p6/m, {a0}.h, {b1}.h\n"
    code_str += gemm_config.currect_model.load_b2_last_k(b2, "p0", ldopt, ldaopt)
    code_str += f"add          {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
    code_str += f"bfmopa        za2.s, p4/m, p6/m, {a0}.h, {b2}.h\n"
    code_str += gemm_config.currect_model.load_b3_last_k(b3, "p3", ldopt, ldaopt)
    code_str += f"bfmopa        za3.s, p4/m, p3/m, {a0}.h, {b3}.h\n"
    code_str += f"add          {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
    return code_str

def kernel_3VL_1VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    code_str += gemm_config.currect_model.load_a0b0_last_k(a0, "p1", b0, "p0", ldopt, ldaopt)
    code_str += f"{get_mopa_inst()}        za0.s, p4/m, p6/m, {a0}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
    code_str += gemm_config.currect_model.load_a1_last_k(a1, "p1", ldopt, ldaopt)
    code_str += f"{get_mopa_inst()}        za1.s, p4/m, p6/m, {a1}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
    code_str += f"add          {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"
    code_str += gemm_config.currect_model.load_a2_last_k(a2, "p2", ldopt, ldaopt)
    code_str += f"{get_mopa_inst()}        za2.s, p5/m, p6/m, {a2}{get_element_suffix()}, {b0}{get_element_suffix()}\n"
    code_str += f"add          {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"

    return code_str

def kernel_1VL_3VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    code_str += gemm_config.currect_model.load_a0b0_last_k(a0, "p1", b0, "p0", ldopt, ldaopt)
    code_str += f"bfmopa      za0.s, p4/m, p6/m, {a0}.h, {b0}.h\n"

    code_str += gemm_config.currect_model.load_b1_last_k(b1, "p0", ldopt, ldaopt)
    code_str += f"bfmopa      za1.s, p4/m, p6/m, {a0}.h, {b1}.h\n"

    code_str += f"add         {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
    code_str += gemm_config.currect_model.load_b2_last_k(b2, "p3", ldopt, ldaopt)
    code_str += f"bfmopa      za2.s, p4/m, p3/m, {a0}.h, {b2}.h\n"

    code_str += f"add         {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"

    return code_str

def kernel_2VL_2VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    code_str += gemm_config.currect_model.load_a0b0_last_k(a0, "p1", b0, "p0", ldopt, ldaopt)
    code_str += f"bfmopa      za0.s, p4/m, p6/m, {a0}.h, {b0}.h\n"

    code_str += gemm_config.currect_model.load_b1_last_k(b1, "p3", ldopt, ldaopt)
    code_str += f"bfmopa      za1.s, p4/m, p3/m, {a0}.h, {b1}.h\n"

    code_str += gemm_config.currect_model.load_a1_last_k(a1, "p2", ldopt, ldaopt)
    code_str += f"bfmopa      za2.s, p5/m, p6/m, {a1}.h, {b0}.h\n"

    code_str += f"add         {pA0}, {TMP_PTR}, {OFFSET_A}, LSL #{get_element_size_shift()}\n"
    code_str += f"bfmopa      za3.s, p5/m, p3/m, {a1}.h, {b1}.h\n"

    code_str += f"add         {pB0}, {TMP_PTR1}, {OFFSET_B}, LSL #{get_element_size_shift()}\n"

    return code_str

def kernel_1VL_2VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    code_str += gemm_config.currect_model.load_a0b0_last_k(a0, "p1", b0, "p0", ldopt, ldaopt)
    code_str += f"bfmopa      za0.s, p4/m, p6/m, {a0}.h, {b0}.h\n"
    
    code_str += gemm_config.currect_model.load_b1_last_k(b1, "p3", ldopt, ldaopt)
    code_str += f"bfmopa      za1.s, p4/m, p3/m, {a0}.h, {b1}.h\n"

    return code_str

def kernel_2VL_1VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    code_str += gemm_config.currect_model.load_a0b0_last_k(a0, "p1", b0, "p0", ldopt, ldaopt)
    code_str += f"bfmopa      za0.s, p4/m, p6/m, {a0}.h, {b0}.h\n"
    
    code_str += gemm_config.currect_model.load_a1_last_k(a1, "p2", ldopt, ldaopt)
    code_str += f"bfmopa      za2.s, p5/m, p6/m, {a1}.h, {b0}.h\n"

    return code_str

def kernel_1VL_1VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    if ldopt is None:
        ldopt = get_ld1()
    if ldaopt is None:
        ldaopt = get_ld1()
    code_str = f""
    code_str += gemm_config.currect_model.load_a0b0_last_k(a0, "p1", b0, "p0", ldopt, ldaopt)
    code_str += f"bfmopa      za0.s, p4/m, p6/m, {a0}.h, {b0}.h\n"

    return code_str

def save_zacol(pc, off, za, base_idx, idx, pg, rab0, rc0):
    code_str = f""
    code_str += f"mova         {rab0}.s, {pg}/m, {za}v.s[{base_idx}, {idx}]\n"
    code_str += f"{LDNT1}      {rc0}.s, {pg}/z, [{pc}, {off}, MUL VL]\n"
    code_str += f"fadd         {rc0}.s, {pg}/m, {rc0}.s, {rab0}.s\n"
    code_str += f"{STNT1}      {rc0}.s, {pg}, [{pc}, {off}, MUL VL]\n"

    return code_str