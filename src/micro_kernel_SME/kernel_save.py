from global_config import *
from kernel_asm import *


def get_pg0():
    return "p4" if is_ext_precision() else "p1"


def get_pg1():
    return "p5" if is_ext_precision() else "p2"


def _vl_multiplier(vl_label):
    return int(vl_label.replace("VL", ""))


def _save_subtile_predicates(count, pg0, pg1):
    predicates = [pg0] * get_save_subtile_count()
    if count > 1:
        predicates[count - 1] = pg1
    return predicates[:count]


def _save_temp_pairs(rab0, rc0, rab1, rc1, t0, t1, t2="z30", t3="z31"):
    return [(rab0, rc0), (rab1, rc1), (t0, t1), (t2, t3)]


def _save_base_index_regs():
    return ("w12", "w13", "w14", "w15")[: get_save_subtile_count()]


def _save_column_ptrs():
    return (pC0, pC1, pC2, pC3)[: get_save_subtile_count()]


def _save_column_temp_groups():
    return (
        ("z0", "z1", "z2", "z3"),
        ("z4", "z5", "z6", "z7"),
        ("z8", "z9", "z10", "z11"),
        ("z12", "z13", "z14", "z15"),
    )[: get_save_subtile_count()]


def _save_column_configs():
    return [
        (pc, f"#{offset}", temps)
        for pc, offset, temps in zip(
            _save_column_ptrs(),
            get_save_vl_offsets(),
            _save_column_temp_groups(),
        )
    ]


def _emit_save_base_index_init():
    code_parts = []
    for reg, idx in zip(_save_base_index_regs(), get_save_base_slice_indices()):
        code_parts.append(f"mov      {reg}, #{idx}\n")
    return "".join(code_parts)


def _emit_save_column_ptr_setup():
    ptrs = _save_column_ptrs()
    code_parts = []
    prev_ptr = ptrs[0]
    for ptr in ptrs[1:]:
        code_parts.append(f"add      {ptr}, {prev_ptr}, {LDC}\n")
        prev_ptr = ptr
    return "".join(code_parts)


def _save_zacol_group(
    count,
    pc,
    za_regs,
    base_idx,
    idx,
    pg0,
    pg1,
    rab0,
    rc0,
    rab1,
    rc1,
    t0,
    t1,
    t2="z30",
    t3="z31",
):
    predicates = _save_subtile_predicates(count, pg0, pg1)
    temp_pairs = _save_temp_pairs(rab0, rc0, rab1, rc1, t0, t1, t2, t3)
    code_parts = []

    for za_reg, predicate, temps, offset in zip(
        za_regs[:count],
        predicates,
        temp_pairs,
        get_save_vl_offsets(),
    ):
        rab, rc = temps
        code_parts.append(save_zacol(pc, f"#{offset}", za_reg, base_idx, idx, predicate, rab, rc))
    code_parts.append(f"add      {pc}, {pc}, {LDC}, lsl #2\n")
    return "".join(code_parts)


def save_zacol_1VL(pc, c0, c1, c2, c3, base_idx, idx, pg0, pg1, rab0, rc0, rab1, rc1, t0, t1, t2="z30", t3="z31"):
    return _save_zacol_group(1, pc, [c0], base_idx, idx, pg0, pg1, rab0, rc0, rab1, rc1, t0, t1, t2, t3)


def save_zacol_2VL(pc, c0, c1, c2, c3, base_idx, idx, pg0, pg1, rab0, rc0, rab1, rc1, t0, t1, t2="z30", t3="z31"):
    return _save_zacol_group(2, pc, [c0, c1], base_idx, idx, pg0, pg1, rab0, rc0, rab1, rc1, t0, t1, t2, t3)


def save_zacol_3VL(pc, c0, c1, c2, c3, base_idx, idx, pg0, pg1, rab0, rc0, rab1, rc1, t0, t1, t2="z30", t3="z31"):
    return _save_zacol_group(3, pc, [c0, c1, c2], base_idx, idx, pg0, pg1, rab0, rc0, rab1, rc1, t0, t1, t2, t3)


def save_zacol_4VL(pc, c0, c1, c2, c3, base_idx, idx, pg0, pg1, rab0, rc0, rab1, rc1, t0, t1, t2="z30", t3="z31"):
    return _save_zacol_group(4, pc, [c0, c1, c2, c3], base_idx, idx, pg0, pg1, rab0, rc0, rab1, rc1, t0, t1, t2, t3)


save_zacol_map = {
    "save_zacol_4VL": save_zacol_4VL,
    "save_zacol_3VL": save_zacol_3VL,
    "save_zacol_2VL": save_zacol_2VL,
    "save_zacol_1VL": save_zacol_1VL,
}


def kernel_save_c_base_val(lable, num, mvl, nvl, pc, c0="za0", c1="za1", c2="za2", c3="za3"):
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


def kernel_save_c_base_n_1VL_(mvl, c0="za0", c1="za1", c2="za2", c3="za3"):
    pg0 = get_pg0()
    pg1 = get_pg1()
    za_regs = [c0, c1, c2, c3][: _vl_multiplier(mvl)]
    code_parts = []

    for base_idx in _save_base_index_regs():
        for pc, idx, temps in _save_column_configs():
            code_parts.append(
                _save_zacol_group(
                    len(za_regs),
                    pc,
                    za_regs,
                    base_idx,
                    idx,
                    pg0,
                    pg1,
                    *temps,
                    "z25",
                    "z26",
                )
            )
    return "".join(code_parts)


def kernel_save_c_base_n_1VL(lable, mvl, nvl, c0="za0", c1="za1", c2="za2", c3="za3"):
    code_str = f""
    code_str += _emit_save_base_index_init()
    code_str += f"ands     {TMP_CNT_POST}, {MIN_N}, #{get_save_tail_mask()}\n"
    code_str += f"bne      .kernel_save_c_val_{mvl}_{nvl}_{lable}\n"
    code_str += _emit_save_column_ptr_setup()
    code_str += kernel_save_c_base_n_1VL_(mvl, c0, c1, c2, c3)
    code_str += f"b        .kernel_save_c_val_{mvl}_{nvl}_{lable}_end\n"
    code_str += f".kernel_save_c_val_{mvl}_{nvl}_{lable}:\n"
    code_str += kernel_save_c_base_val(lable, TMP_CNT_POST, mvl, nvl, pC0, c0, c1, c2, c3)
    code_str += f".kernel_save_c_val_{mvl}_{nvl}_{lable}_end:\n"
    return code_str


def _kernel_save_c_base_n_multi(lable, mvl, full_groups, tail_group, tail_nvl):
    code_str = f""
    code_str += _emit_save_base_index_init()
    code_str += f"mov      {TMP_PTR}, {pC0}\n"
    code_str += _emit_save_column_ptr_setup()
    for za_regs in full_groups:
        code_str += kernel_save_c_base_n_1VL_(mvl, *za_regs)
    code_str += kernel_save_c_base_n_1VL(lable, mvl, tail_nvl, *tail_group)
    code_str += f"add      {pC0}, {TMP_PTR}, {MIN_M}, lsl #2\n"
    return code_str


def kernel_save_c_base_n_2VL(lable, mvl):
    return _kernel_save_c_base_n_multi(lable, mvl, [("za0", "za2")], ("za1", "za3"), "2VL")


def kernel_save_c_base_n_4VL(lable, mvl):
    return _kernel_save_c_base_n_multi(lable, mvl, [("za0",), ("za1",), ("za2",)], ("za3",), "4VL")


def kernel_save_c_base_n_3VL(lable, mvl):
    return _kernel_save_c_base_n_multi(lable, mvl, [("za0",), ("za1",)], ("za2",), "3VL")


def _kernel_save_c_single_n(lable, mvl, *za_regs):
    code_str = f""
    code_str += f"mov      {TMP_PTR}, {pC0}\n"
    code_str += kernel_save_c_base_n_1VL(lable, mvl, "1VL", *za_regs)
    code_str += f"add      {pC0}, {TMP_PTR}, {MIN_M}, lsl #2\n"
    return code_str


def kernel_save_c_4VL_1VL(lable):
    return _kernel_save_c_single_n(lable, "4VL", "za0", "za1", "za2", "za3")


def kernel_save_c_1VL_4VL(lable):
    return kernel_save_c_base_n_4VL(lable, "1VL")


def kernel_save_c_3VL_1VL(lable):
    return _kernel_save_c_single_n(lable, "3VL", "za0", "za1", "za2")


def kernel_save_c_1VL_3VL(lable):
    return kernel_save_c_base_n_3VL(lable, "1VL")


def kernel_save_c_2VL_2VL(lable):
    return kernel_save_c_base_n_2VL(lable, "2VL")


def kernel_save_c_1VL_2VL(lable):
    return kernel_save_c_base_n_2VL(lable, "1VL")


def kernel_save_c_2VL_1VL(lable):
    return _kernel_save_c_single_n(lable, "2VL", "za0", "za2")


def kernel_save_c_1VL_1VL(lable):
    return _kernel_save_c_single_n(lable, "1VL", "za0")
