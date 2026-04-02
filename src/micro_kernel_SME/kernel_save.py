from global_config import get_save_base_slice_indices, get_save_subtile_count, get_save_tail_mask, get_save_vl_offsets
from kernel_asm import save_zacol


def get_pg0(ctx):
    # Save uses the half-input row predicates even though ZA/C traffic stays in
    # `.s` because accumulation/output is float.
    return ctx.registers.half_predicate("m_main")


def get_pg1(ctx):
    # Tail saves keep the same rule: half-input predicate, float ZA/C vectors.
    return ctx.registers.half_predicate("m_tail")


def _vl_multiplier(vl_label):
    # Convert labels such as `2VL` into the integer subtile count used by the save helpers.
    return int(vl_label.replace("VL", ""))


def _save_subtile_predicates(ctx, count):
    # Fill the save predicate list with main predicates and swap the last one to tail when the save group is partial.
    predicates = [get_pg0(ctx)] * get_save_subtile_count()
    if count > 1:
        predicates[count - 1] = get_pg1(ctx)
    return predicates[:count]


def _save_temp_pairs(rab0, rc0, rab1, rc1, t0, t1, t2="z30", t3="z31"):
    # Group the scratch Z registers exactly as `save_zacol` expects so every saved column reuses the same scratch layout.
    return [(rab0, rc0), (rab1, rc1), (t0, t1), (t2, t3)]


def _save_base_index_regs(ctx):
    # Slice out the logical ZA base-index registers that correspond to the configured save subtile count.
    return ctx.registers.save.base_indices[: get_save_subtile_count()]


def _save_column_ptrs(ctx):
    # Slice out the live C-column pointers that back the active save subtile count.
    regs = ctx.registers.pointers
    return (regs.pC0, regs.pC1, regs.pC2, regs.pC3)[: get_save_subtile_count()]


def _save_column_temp_groups():
    # Group scratch Z registers by output column so each save stream gets a fixed temporary bundle.
    return (
        ("z0", "z1", "z2", "z3"),
        ("z4", "z5", "z6", "z7"),
        ("z8", "z9", "z10", "z11"),
        ("z12", "z13", "z14", "z15"),
    )[: get_save_subtile_count()]


def _save_column_configs(ctx):
    # Zip together column pointers, VL offsets, and scratch groups so the save loops can walk one configuration list.
    return [
        (pc, f"#{offset}", temps)
        for pc, offset, temps in zip(
            _save_column_ptrs(ctx),
            get_save_vl_offsets(),
            _save_column_temp_groups(),
        )
    ]


def _gen_save_base_index_init(ctx):
    # Seed the ZA slice indices used to read each saved column group out of the accumulator tile.
    code_parts = []
    for reg, idx in zip(_save_base_index_regs(ctx), get_save_base_slice_indices()):
        code_parts.append(f"mov      {reg}, #{idx}\n")
    return "".join(code_parts)


def _gen_save_column_ptr_setup(ctx):
    # Build the pC1/pC2/pC3 column pointers from pC0 and LDC before the save loops start streaming C back.
    regs = ctx.registers
    ptrs = _save_column_ptrs(ctx)
    code_parts = []
    prev_ptr = ptrs[0]
    for ptr in ptrs[1:]:
        code_parts.append(f"add      {ptr}, {prev_ptr}, {regs.params.LDC}\n")
        prev_ptr = ptr
    return "".join(code_parts)


def _save_zacol_group(ctx, count, pc, za_regs, base_idx, idx, rab0, rc0, rab1, rc1, t0, t1, t2="z30", t3="z31"):
    # Save one N-group by walking the participating ZA registers, predicates, and VL offsets in lockstep.
    predicates = _save_subtile_predicates(ctx, count)
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
    code_parts.append(f"add      {pc}, {pc}, {ctx.registers.params.LDC}, lsl #2\n")
    return "".join(code_parts)


def save_zacol_1VL(ctx, pc, c0, c1, c2, c3, base_idx, idx, rab0, rc0, rab1, rc1, t0, t1, t2="z30", t3="z31"):
    return _save_zacol_group(ctx, 1, pc, [c0], base_idx, idx, rab0, rc0, rab1, rc1, t0, t1, t2, t3)


def save_zacol_2VL(ctx, pc, c0, c1, c2, c3, base_idx, idx, rab0, rc0, rab1, rc1, t0, t1, t2="z30", t3="z31"):
    return _save_zacol_group(ctx, 2, pc, [c0, c1], base_idx, idx, rab0, rc0, rab1, rc1, t0, t1, t2, t3)


def save_zacol_3VL(ctx, pc, c0, c1, c2, c3, base_idx, idx, rab0, rc0, rab1, rc1, t0, t1, t2="z30", t3="z31"):
    return _save_zacol_group(ctx, 3, pc, [c0, c1, c2], base_idx, idx, rab0, rc0, rab1, rc1, t0, t1, t2, t3)


def save_zacol_4VL(ctx, pc, c0, c1, c2, c3, base_idx, idx, rab0, rc0, rab1, rc1, t0, t1, t2="z30", t3="z31"):
    return _save_zacol_group(ctx, 4, pc, [c0, c1, c2, c3], base_idx, idx, rab0, rc0, rab1, rc1, t0, t1, t2, t3)


save_zacol_map = {
    "save_zacol_4VL": save_zacol_4VL,
    "save_zacol_3VL": save_zacol_3VL,
    "save_zacol_2VL": save_zacol_2VL,
    "save_zacol_1VL": save_zacol_1VL,
}


def kernel_save_c_base_val(ctx, label, num, mvl, nvl, pc, c0="za0", c1="za1", c2="za2", c3="za3"):
    # Handle the final short N group by iterating column-by-column instead of using the fully unrolled save path.
    regs = ctx.registers
    code_str = f""
    code_str += f"mov      {regs.counters.TMP_CNT}, {num}\n"
    code_str += f"mov      {_save_base_index_regs(ctx)[0]}, #0\n"
    code_str += f"mov      {regs.address.TMP_PTR1}, {pc}\n"
    code_str += f".loop_save_c_j_{mvl}_{nvl}_{label}:\n"
    code_str += save_zacol_map[f"save_zacol_{mvl}"](
        ctx,
        pc,
        c0,
        c1,
        c2,
        c3,
        _save_base_index_regs(ctx)[0],
        "#0",
        "z0",
        "z1",
        "z2",
        "z3",
        ctx.registers.vectors.save_tmp,
        ctx.registers.vectors.save_tmp1,
    )
    code_str += f"add      {_save_base_index_regs(ctx)[0]}, {_save_base_index_regs(ctx)[0]}, #1\n"
    code_str += f"add      {regs.address.TMP_PTR1}, {regs.address.TMP_PTR1}, {regs.params.LDC}\n"
    code_str += f"mov      {pc}, {regs.address.TMP_PTR1}\n"
    code_str += f"subs     {regs.counters.TMP_CNT}, {regs.counters.TMP_CNT}, #1\n"
    code_str += f"bgt      .loop_save_c_j_{mvl}_{nvl}_{label}\n"
    return code_str


def kernel_save_c_base_n_1VL_(ctx, mvl, c0="za0", c1="za1", c2="za2", c3="za3"):
    # Save one logical N group of width `1VL` using the ZA registers that belong to the current outer save chunk.
    za_regs = [c0, c1, c2, c3][: _vl_multiplier(mvl)]
    code_parts = []

    for base_idx in _save_base_index_regs(ctx):
        for pc, idx, temps in _save_column_configs(ctx):
            code_parts.append(
                _save_zacol_group(
                    ctx,
                    len(za_regs),
                    pc,
                    za_regs,
                    base_idx,
                    idx,
                    *temps,
                    ctx.registers.vectors.save_tmp,
                    ctx.registers.vectors.save_tmp1,
                )
            )
    return "".join(code_parts)


def kernel_save_c_base_n_1VL(ctx, label, mvl, nvl, c0="za0", c1="za1", c2="za2", c3="za3"):
    # Save the final `1VL`-based N group, branching between the fully aligned path and the short-tail helper.
    regs = ctx.registers
    code_str = f""
    code_str += _gen_save_base_index_init(ctx)
    code_str += f"ands     {regs.counters.TMP_CNT_POST}, {regs.dims.MIN_N}, #{get_save_tail_mask()}\n"
    code_str += f"bne      .kernel_save_c_val_{mvl}_{nvl}_{label}\n"
    code_str += _gen_save_column_ptr_setup(ctx)
    code_str += kernel_save_c_base_n_1VL_(ctx, mvl, c0, c1, c2, c3)
    code_str += f"b        .kernel_save_c_val_{mvl}_{nvl}_{label}_end\n"
    code_str += f".kernel_save_c_val_{mvl}_{nvl}_{label}:\n"
    code_str += kernel_save_c_base_val(ctx, label, regs.counters.TMP_CNT_POST, mvl, nvl, regs.pointers.pC0, c0, c1, c2, c3)
    code_str += f".kernel_save_c_val_{mvl}_{nvl}_{label}_end:\n"
    return code_str


def _kernel_save_c_base_n_multi(ctx, label, mvl, full_groups, tail_group, tail_nvl):
    # Save wider N groups by generating their full `1VL` slices first and then delegating the last partial group to the tail helper.
    regs = ctx.registers
    code_str = f""
    code_str += _gen_save_base_index_init(ctx)
    code_str += f"mov      {regs.address.TMP_PTR}, {regs.pointers.pC0}\n"
    code_str += _gen_save_column_ptr_setup(ctx)
    for za_regs in full_groups:
        code_str += kernel_save_c_base_n_1VL_(ctx, mvl, *za_regs)
    code_str += kernel_save_c_base_n_1VL(ctx, label, mvl, tail_nvl, *tail_group)
    code_str += f"add      {regs.pointers.pC0}, {regs.address.TMP_PTR}, {regs.dims.MIN_M}, lsl #2\n"
    return code_str


def kernel_save_c_base_n_2VL(ctx, label, mvl):
    return _kernel_save_c_base_n_multi(ctx, label, mvl, [("za0", "za2")], ("za1", "za3"), "2VL")


def kernel_save_c_base_n_4VL(ctx, label, mvl):
    return _kernel_save_c_base_n_multi(ctx, label, mvl, [("za0",), ("za1",), ("za2",)], ("za3",), "4VL")


def kernel_save_c_base_n_3VL(ctx, label, mvl):
    return _kernel_save_c_base_n_multi(ctx, label, mvl, [("za0",), ("za1",)], ("za2",), "3VL")


def _kernel_save_c_single_n(ctx, label, mvl, *za_regs):
    # Save kernels whose N shape is already `1VL`, then advance pC0 to the next M tile origin.
    regs = ctx.registers
    code_str = f""
    code_str += f"mov      {regs.address.TMP_PTR}, {regs.pointers.pC0}\n"
    code_str += kernel_save_c_base_n_1VL(ctx, label, mvl, "1VL", *za_regs)
    code_str += f"add      {regs.pointers.pC0}, {regs.address.TMP_PTR}, {regs.dims.MIN_M}, lsl #2\n"
    return code_str


def kernel_save_c_4VL_1VL(ctx, label):
    return _kernel_save_c_single_n(ctx, label, "4VL", "za0", "za1", "za2", "za3")


def kernel_save_c_1VL_4VL(ctx, label):
    return kernel_save_c_base_n_4VL(ctx, label, "1VL")


def kernel_save_c_3VL_1VL(ctx, label):
    return _kernel_save_c_single_n(ctx, label, "3VL", "za0", "za1", "za2")


def kernel_save_c_1VL_3VL(ctx, label):
    return kernel_save_c_base_n_3VL(ctx, label, "1VL")


def kernel_save_c_2VL_2VL(ctx, label):
    return kernel_save_c_base_n_2VL(ctx, label, "2VL")


def kernel_save_c_1VL_2VL(ctx, label):
    return kernel_save_c_base_n_2VL(ctx, label, "1VL")


def kernel_save_c_2VL_1VL(ctx, label):
    return _kernel_save_c_single_n(ctx, label, "2VL", "za0", "za2")


def kernel_save_c_1VL_1VL(ctx, label):
    return _kernel_save_c_single_n(ctx, label, "1VL", "za0")
