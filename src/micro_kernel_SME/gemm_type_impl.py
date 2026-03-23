from dataclasses import dataclass

from global_config import get_element_size_shift, get_ld_element_suffix, tile_size_from_vl


LOAD_CONTIGUOUS = "contiguous"
LOAD_GATHER = "gather"
_LABEL_COUNTER = 0


def _new_label(prefix):
    global _LABEL_COUNTER
    _LABEL_COUNTER += 1
    return f".L_{prefix}_{_LABEL_COUNTER}"


def _pred(ctx, role):
    return ctx.registers.logical_predicate(role)


def _ext_pred(ctx, role):
    return ctx.registers.ext_predicate(role)


def _load_predicate(ctx, role):
    pred = _pred(ctx, role)
    return pred if ctx.is_ext_precision() else f"{pred}/z"


def _zip_ext_predicate(ctx, role):
    pred = _pred(ctx, role)
    ext_pred = _ext_pred(ctx, role)
    return f"zip1      {ext_pred}.h, {pred}.h, {pred}.h\n"


def _paired_ext_predicate(ctx, first_role, second_role=None):
    second_role = first_role if second_role is None else second_role
    if second_role == first_role:
        return _zip_ext_predicate(ctx, first_role), _ext_pred(ctx, first_role), ""

    target = _ext_pred(ctx, second_role)
    pre_load = f"zip1      {target}.h, {_pred(ctx, first_role)}.h, {_pred(ctx, second_role)}.h\n"
    post_load = _zip_ext_predicate(ctx, first_role)
    post_load += _zip_ext_predicate(ctx, second_role)
    return pre_load, target, post_load


def _axis_min_register(ctx, role):
    return ctx.registers.dims.MIN_M if role.startswith("m_") else ctx.registers.dims.MIN_N


def _emit_runtime_pair_select(ctx, role, full_threshold, full_code, partial_code):
    fast_label = _new_label("ext_pair_fast")
    done_label = _new_label("ext_pair_done")
    dim_reg = _axis_min_register(ctx, role)
    code_str = f"cmp       {dim_reg}, #{full_threshold}\n"
    code_str += f"bge       {fast_label}\n"
    code_str += partial_code
    code_str += f"b         {done_label}\n"
    code_str += f"{fast_label}:\n"
    code_str += full_code
    code_str += f"{done_label}:\n"
    return code_str


def _rdvl_lane_offset(registers, lane):
    tmp = registers.address.TMP_PTR2
    code_str = f"rdvl      {tmp}, #1\n"
    if lane == 1:
        code_str += f"lsr       {tmp}, {tmp}, #2\n"
    elif lane == 2:
        code_str += f"lsr       {tmp}, {tmp}, #1\n"
    elif lane == 3:
        code_str += f"add       {tmp}, {tmp}, {tmp}, LSL #1\n"
        code_str += f"lsr       {tmp}, {tmp}, #2\n"
    return code_str


def _scaled_add(dst, base, offset, lane):
    if lane == 1:
        return f"add          {dst}, {base}, {offset}\n"
    if lane == 2:
        return f"add          {dst}, {base}, {offset}, lsl #1\n"
    code_str = f"add          {dst}, {base}, {offset}\n"
    code_str += f"add          {dst}, {dst}, {offset}, lsl #1\n"
    return code_str


def _emit_ext_contiguous_head(ctx, base, stride, tmp_base, role, dst, low_tmp, high_tmp):
    pred = _pred(ctx, role)
    code_str = f""
    code_str += _zip_ext_predicate(ctx, role)
    code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}]\n"
    code_str += f"add       {tmp_base}, {base}, {stride}, LSL #1\n"
    code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{tmp_base}]\n"
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
    return code_str


def _emit_ext_contiguous_head_pair_fast(ctx, base, stride, tmp_base, role0, role1, dst0, dst1, low_tmp, high_tmp):
    code_str = f""
    pred_pre, pred, pred_post = _paired_ext_predicate(ctx, role0, role1)
    code_str += pred_pre
    code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}]\n"
    code_str += f"add       {tmp_base}, {base}, {stride}, LSL #1\n"
    code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{tmp_base}]\n"
    code_str += pred_post
    code_str += f"zip1      {dst0}.h, {low_tmp}.h, {high_tmp}.h\n"
    code_str += f"zip2      {dst1}.h, {low_tmp}.h, {high_tmp}.h\n"
    return code_str


def _emit_ext_contiguous_head_pair_safe(ctx, base, stride, tmp_base, role0, role1, dst0, dst1, low_tmp, high_tmp):
    code_str = _emit_ext_contiguous_head(ctx, base, stride, tmp_base, role0, dst0, low_tmp, high_tmp)
    code_str += _emit_ext_contiguous_lane(ctx, base, tmp_base, role1, dst1, 1, low_tmp, high_tmp)
    return code_str


def _emit_ext_contiguous_head_pair(ctx, base, stride, tmp_base, role0, role1, dst0, dst1, low_tmp, high_tmp):
    if role0 == role1:
        return _emit_ext_contiguous_head_pair_fast(
            ctx, base, stride, tmp_base, role0, role1, dst0, dst1, low_tmp, high_tmp
        )
    # Mixed main/tail pairs only pay off when the paired 2VL chunk is full-width.
    full_code = _emit_ext_contiguous_head_pair_fast(
        ctx, base, stride, tmp_base, role0, role0, dst0, dst1, low_tmp, high_tmp
    )
    full_code += _zip_ext_predicate(ctx, role1)
    partial_code = _emit_ext_contiguous_head_pair_safe(
        ctx, base, stride, tmp_base, role0, role1, dst0, dst1, low_tmp, high_tmp
    )
    return _emit_runtime_pair_select(ctx, role0, tile_size_from_vl(2), full_code, partial_code)


def _emit_ext_contiguous_head_mixed_pair(ctx, base, stride, tmp_base, role0, role1, dst0, dst1, low_tmp, high_tmp):
    regs = ctx.registers
    pred0 = _pred(ctx, role0)
    pred1 = _pred(ctx, role1)
    code_str = f""
    code_str += f"add       {tmp_base}, {base}, {stride}, LSL #1\n"
    code_str += f"ld1h      {low_tmp}.h, {pred0}/z, [{base}]\n"
    code_str += f"ld1h      {high_tmp}.h, {pred0}/z, [{tmp_base}]\n"
    code_str += f"mov       {dst0}.h, #0\n"
    code_str += f"mov       {dst1}.h, #0\n"
    code_str += f"mov       {dst0}.h, {pred0}/m, {low_tmp}.h\n"
    code_str += f"mov       {dst1}.h, {pred0}/m, {high_tmp}.h\n"
    code_str += _rdvl_lane_offset(regs, 1)
    code_str += f"ld1h      {low_tmp}.h, {pred1}/z, [{base}, {regs.address.TMP_PTR2}, LSL #1]\n"
    code_str += f"mov       {high_tmp}.h, #0\n"
    code_str += f"ext       {high_tmp}.b, {high_tmp}.b, {low_tmp}.b, #32\n"
    code_str += f"orr       {dst0}.d, {dst0}.d, {high_tmp}.d\n"
    code_str += f"ld1h      {low_tmp}.h, {pred1}/z, [{tmp_base}, {regs.address.TMP_PTR2}, LSL #1]\n"
    code_str += f"mov       {high_tmp}.h, #0\n"
    code_str += f"ext       {high_tmp}.b, {high_tmp}.b, {low_tmp}.b, #32\n"
    code_str += f"orr       {dst1}.d, {dst1}.d, {high_tmp}.d\n"
    code_str += f"mov       {low_tmp}.d, {dst0}.d\n"
    code_str += f"zip1      {dst0}.h, {dst0}.h, {dst1}.h\n"
    code_str += f"zip2      {dst1}.h, {low_tmp}.h, {dst1}.h\n"
    code_str += _zip_ext_predicate(ctx, role0)
    code_str += _zip_ext_predicate(ctx, role1)
    return code_str


def _emit_ext_contiguous_lane_fast_pair(ctx, base, paired_base, role, dst, lane, low_tmp, high_tmp, next_dst=None, next_role=None):
    code_str = f""
    pred_pre, pred, pred_post = _paired_ext_predicate(ctx, role, next_role if next_dst is not None else None)
    code_str += pred_pre
    code_str += _rdvl_lane_offset(ctx.registers, lane)
    code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
    code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{paired_base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
    code_str += pred_post
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
    if next_dst is not None:
        code_str += f"zip2      {next_dst}.h, {low_tmp}.h, {high_tmp}.h\n"
    return code_str


def _emit_ext_contiguous_lane_pair_safe(ctx, base, paired_base, role0, role1, dst0, dst1, lane, low_tmp, high_tmp):
    code_str = _emit_ext_contiguous_lane(ctx, base, paired_base, role0, dst0, lane, low_tmp, high_tmp)
    code_str += _emit_ext_contiguous_lane(ctx, base, paired_base, role1, dst1, lane + 1, low_tmp, high_tmp)
    return code_str


def _emit_ext_contiguous_lane(ctx, base, paired_base, role, dst, lane, low_tmp, high_tmp, next_dst=None, next_role=None):
    if next_dst is None:
        pred = _pred(ctx, role)
        code_str = f""
        code_str += _zip_ext_predicate(ctx, role)
        code_str += _rdvl_lane_offset(ctx.registers, lane)
        code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
        code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{paired_base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
        code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
        return code_str
    if next_dst is not None and next_role != role:
        full_code = _emit_ext_contiguous_lane_fast_pair(
            ctx,
            base,
            paired_base,
            role,
            dst,
            lane,
            low_tmp,
            high_tmp,
            next_dst=next_dst,
            next_role=role,
        )
        full_code += _zip_ext_predicate(ctx, next_role)
        partial_code = _emit_ext_contiguous_lane_pair_safe(
            ctx, base, paired_base, role, next_role, dst, next_dst, lane, low_tmp, high_tmp
        )
        return _emit_runtime_pair_select(ctx, role, tile_size_from_vl(lane + 2), full_code, partial_code)
    return _emit_ext_contiguous_lane_fast_pair(
        ctx, base, paired_base, role, dst, lane, low_tmp, high_tmp, next_dst=next_dst, next_role=next_role
    )


def _emit_ext_contiguous_lane_mixed_pair(ctx, base, paired_base, role0, role1, dst0, dst1, lane, low_tmp, high_tmp):
    regs = ctx.registers
    pred0 = _pred(ctx, role0)
    pred1 = _pred(ctx, role1)
    code_str = f""
    code_str += _rdvl_lane_offset(regs, lane)
    code_str += f"ld1h      {low_tmp}.h, {pred0}/z, [{base}, {regs.address.TMP_PTR2}, LSL #1]\n"
    code_str += f"ld1h      {high_tmp}.h, {pred0}/z, [{paired_base}, {regs.address.TMP_PTR2}, LSL #1]\n"
    code_str += f"mov       {dst0}.h, #0\n"
    code_str += f"mov       {dst1}.h, #0\n"
    code_str += f"mov       {dst0}.h, {pred0}/m, {low_tmp}.h\n"
    code_str += f"mov       {dst1}.h, {pred0}/m, {high_tmp}.h\n"
    code_str += _rdvl_lane_offset(regs, lane + 1)
    code_str += f"ld1h      {low_tmp}.h, {pred1}/z, [{base}, {regs.address.TMP_PTR2}, LSL #1]\n"
    code_str += f"mov       {high_tmp}.h, #0\n"
    code_str += f"ext       {high_tmp}.b, {high_tmp}.b, {low_tmp}.b, #32\n"
    code_str += f"orr       {dst0}.d, {dst0}.d, {high_tmp}.d\n"
    code_str += f"ld1h      {low_tmp}.h, {pred1}/z, [{paired_base}, {regs.address.TMP_PTR2}, LSL #1]\n"
    code_str += f"mov       {high_tmp}.h, #0\n"
    code_str += f"ext       {high_tmp}.b, {high_tmp}.b, {low_tmp}.b, #32\n"
    code_str += f"orr       {dst1}.d, {dst1}.d, {high_tmp}.d\n"
    code_str += f"mov       {low_tmp}.d, {dst0}.d\n"
    code_str += f"zip1      {dst0}.h, {dst0}.h, {dst1}.h\n"
    code_str += f"zip2      {dst1}.h, {low_tmp}.h, {dst1}.h\n"
    code_str += _zip_ext_predicate(ctx, role0)
    code_str += _zip_ext_predicate(ctx, role1)
    return code_str


def _emit_ext_contiguous_last_k_fast_pair(ctx, base, role, dst, lane, low_tmp, zero_tmp, next_dst=None, next_role=None):
    code_str = f""
    pred_pre, pred, pred_post = _paired_ext_predicate(ctx, role, next_role if next_dst is not None else None)
    code_str += pred_pre
    if lane > 0:
        code_str += _rdvl_lane_offset(ctx.registers, lane)
        code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
    else:
        code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}]\n"
    code_str += pred_post
    code_str += f"mov       {zero_tmp}.h, #0\n"
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {zero_tmp}.h\n"
    if next_dst is not None:
        code_str += f"zip2      {next_dst}.h, {low_tmp}.h, {zero_tmp}.h\n"
    return code_str


def _emit_ext_contiguous_last_k_pair_safe(ctx, base, role0, role1, dst0, dst1, lane, low_tmp, zero_tmp):
    code_str = _emit_ext_contiguous_last_k(ctx, base, role0, dst0, lane, low_tmp, zero_tmp)
    code_str += _emit_ext_contiguous_last_k(ctx, base, role1, dst1, lane + 1, low_tmp, zero_tmp)
    return code_str


def _emit_ext_contiguous_last_k(ctx, base, role, dst, lane, low_tmp, zero_tmp, next_dst=None, next_role=None):
    if next_dst is None:
        pred = _pred(ctx, role)
        code_str = f""
        code_str += _zip_ext_predicate(ctx, role)
        if lane > 0:
            code_str += _rdvl_lane_offset(ctx.registers, lane)
            code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
        else:
            code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}]\n"
        code_str += f"mov       {zero_tmp}.h, #0\n"
        code_str += f"zip1      {dst}.h, {low_tmp}.h, {zero_tmp}.h\n"
        return code_str
    if next_dst is not None and next_role != role:
        return _emit_ext_contiguous_last_k_pair_safe(
            ctx, base, role, next_role, dst, next_dst, lane, low_tmp, zero_tmp
        )
    return _emit_ext_contiguous_last_k_fast_pair(
        ctx, base, role, dst, lane, low_tmp, zero_tmp, next_dst=next_dst, next_role=next_role
    )


def _emit_ext_contiguous_last_k_mixed_pair(ctx, base, role0, role1, dst0, dst1, lane, low_tmp, zero_tmp):
    regs = ctx.registers
    pred0 = _pred(ctx, role0)
    pred1 = _pred(ctx, role1)
    code_str = f""
    if lane > 0:
        code_str += _rdvl_lane_offset(regs, lane)
        code_str += f"ld1h      {low_tmp}.h, {pred0}/z, [{base}, {regs.address.TMP_PTR2}, LSL #1]\n"
    else:
        code_str += f"ld1h      {low_tmp}.h, {pred0}/z, [{base}]\n"
    code_str += f"mov       {dst0}.h, #0\n"
    code_str += f"mov       {dst1}.h, #0\n"
    code_str += f"mov       {dst0}.h, {pred0}/m, {low_tmp}.h\n"
    code_str += _rdvl_lane_offset(regs, lane + 1)
    code_str += f"ld1h      {low_tmp}.h, {pred1}/z, [{base}, {regs.address.TMP_PTR2}, LSL #1]\n"
    code_str += f"mov       {zero_tmp}.h, #0\n"
    code_str += f"ext       {zero_tmp}.b, {zero_tmp}.b, {low_tmp}.b, #32\n"
    code_str += f"orr       {dst0}.d, {dst0}.d, {zero_tmp}.d\n"
    code_str += f"mov       {low_tmp}.d, {dst0}.d\n"
    code_str += f"mov       {dst1}.h, #0\n"
    code_str += f"zip1      {dst0}.h, {dst0}.h, {dst1}.h\n"
    code_str += f"zip2      {dst1}.h, {low_tmp}.h, {dst1}.h\n"
    code_str += _zip_ext_predicate(ctx, role0)
    code_str += _zip_ext_predicate(ctx, role1)
    return code_str


def _emit_non_ext_contiguous(load_inst, ctx, base, role, dst, lane):
    suffix = get_ld_element_suffix(ctx)
    pred = _load_predicate(ctx, role)
    if lane == 0:
        return f"{load_inst}      {dst}{suffix}, {pred}, [{base}]\n"
    return f"{load_inst}      {dst}{suffix}, {pred}, [{base}, #{lane}, MUL VL]\n"


def _emit_ext_gather_pair(ctx, base, role, index_vec, dst, low_tmp, high_tmp, pair_base, next_dst=None, next_role=None):
    if next_dst is None:
        code_str = f""
        code_str += _zip_ext_predicate(ctx, role)
        code_str += f"ld1h      {low_tmp}.s, {_ext_pred(ctx, role)}/z, [{base}, {index_vec}.s, UXTW]\n"
        code_str += f"add       {pair_base}, {base}, #2\n"
        code_str += f"ld1h      {high_tmp}.s, {_ext_pred(ctx, role)}/z, [{pair_base}, {index_vec}.s, UXTW]\n"
        code_str += f"uzp1      {low_tmp}.h, {low_tmp}.h, {low_tmp}.h\n"
        code_str += f"uzp1      {high_tmp}.h, {high_tmp}.h, {high_tmp}.h\n"
        code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
        return code_str
    code_str = f""
    pred_pre, pred, pred_post = _paired_ext_predicate(ctx, role, next_role if next_dst is not None else None)
    code_str += pred_pre
    code_str += f"ld1h      {low_tmp}.s, {pred}/z, [{base}, {index_vec}.s, UXTW]\n"
    code_str += f"add       {pair_base}, {base}, #2\n"
    code_str += f"ld1h      {high_tmp}.s, {pred}/z, [{pair_base}, {index_vec}.s, UXTW]\n"
    code_str += pred_post
    if next_dst is not None:
        code_str += f"uzp1      {low_tmp}.h, {low_tmp}.h, {low_tmp}.h\n"
        code_str += f"uzp1      {high_tmp}.h, {high_tmp}.h, {high_tmp}.h\n"
        code_str += f"uzp1      {dst}.h, {low_tmp}.h, {low_tmp}.h\n"
        code_str += f"uzp2      {next_dst}.h, {low_tmp}.h, {low_tmp}.h\n"
        code_str += f"uzp1      {low_tmp}.h, {high_tmp}.h, {high_tmp}.h\n"
        code_str += f"uzp2      {high_tmp}.h, {high_tmp}.h, {high_tmp}.h\n"
        code_str += f"zip1      {dst}.h, {dst}.h, {low_tmp}.h\n"
        code_str += f"zip1      {next_dst}.h, {next_dst}.h, {high_tmp}.h\n"
        return code_str
    code_str += f"uzp1      {low_tmp}.h, {low_tmp}.h, {low_tmp}.h\n"
    code_str += f"uzp1      {high_tmp}.h, {high_tmp}.h, {high_tmp}.h\n"
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
    return code_str


def _emit_ext_gather_pair_two_bases(
    ctx,
    base0,
    high0_base,
    role0,
    base1,
    role1,
    index_vec,
    dst0,
    dst1,
    low_tmp,
    high_tmp,
):
    pred0 = _ext_pred(ctx, role0)
    pred1 = _ext_pred(ctx, role1)
    pred0_logical = _pred(ctx, role0)
    code_str = f""

    code_str += _zip_ext_predicate(ctx, role0)
    code_str += f"ld1h      {low_tmp}.s, {pred0}/z, [{base0}, {index_vec}.s, UXTW]\n"
    code_str += f"ld1h      {high_tmp}.s, {pred0}/z, [{high0_base}, {index_vec}.s, UXTW]\n"
    code_str += f"uzp1      {low_tmp}.h, {low_tmp}.h, {low_tmp}.h\n"
    code_str += f"uzp1      {high_tmp}.h, {high_tmp}.h, {high_tmp}.h\n"
    code_str += f"mov       {dst0}.h, #0\n"
    code_str += f"mov       {dst1}.h, #0\n"
    code_str += f"mov       {dst0}.h, {pred0_logical}/m, {low_tmp}.h\n"
    code_str += f"mov       {dst1}.h, {pred0_logical}/m, {high_tmp}.h\n"

    code_str += _zip_ext_predicate(ctx, role1)
    code_str += f"ld1h      {low_tmp}.s, {pred1}/z, [{base1}, {index_vec}.s, UXTW]\n"
    code_str += f"uzp1      {low_tmp}.h, {low_tmp}.h, {low_tmp}.h\n"
    code_str += f"mov       {high_tmp}.h, #0\n"
    code_str += f"ext       {high_tmp}.b, {high_tmp}.b, {low_tmp}.b, #32\n"
    code_str += f"orr       {dst0}.d, {dst0}.d, {high_tmp}.d\n"

    code_str += f"add       {base1}, {base1}, #2\n"
    code_str += f"ld1h      {low_tmp}.s, {pred1}/z, [{base1}, {index_vec}.s, UXTW]\n"
    code_str += f"uzp1      {low_tmp}.h, {low_tmp}.h, {low_tmp}.h\n"
    code_str += f"mov       {high_tmp}.h, #0\n"
    code_str += f"ext       {high_tmp}.b, {high_tmp}.b, {low_tmp}.b, #32\n"
    code_str += f"orr       {dst1}.d, {dst1}.d, {high_tmp}.d\n"

    code_str += f"mov       {low_tmp}.d, {dst0}.d\n"
    code_str += f"zip1      {dst0}.h, {dst0}.h, {dst1}.h\n"
    code_str += f"zip2      {dst1}.h, {low_tmp}.h, {dst1}.h\n"
    code_str += _zip_ext_predicate(ctx, role0)
    code_str += _zip_ext_predicate(ctx, role1)
    return code_str


def _emit_ext_gather_last_k(ctx, base, role, index_vec, dst, low_tmp, zero_tmp, next_dst=None, next_role=None):
    if next_dst is None:
        return (
            _zip_ext_predicate(ctx, role)
            + f"ld1h      {dst}.s, {_ext_pred(ctx, role)}/z, [{base}, {index_vec}.s, UXTW]\n"
        )
    code_str = f""
    pred_pre, pred, pred_post = _paired_ext_predicate(ctx, role, next_role if next_dst is not None else None)
    code_str += pred_pre
    code_str += f"ld1h      {low_tmp}.s, {pred}/z, [{base}, {index_vec}.s, UXTW]\n"
    code_str += pred_post
    code_str += f"mov       {zero_tmp}.h, #0\n"
    if next_dst is not None:
        code_str += f"uzp1      {low_tmp}.h, {low_tmp}.h, {low_tmp}.h\n"
        code_str += f"uzp1      {dst}.h, {low_tmp}.h, {low_tmp}.h\n"
        code_str += f"uzp2      {next_dst}.h, {low_tmp}.h, {low_tmp}.h\n"
        code_str += f"zip1      {dst}.h, {dst}.h, {zero_tmp}.h\n"
        code_str += f"zip1      {next_dst}.h, {next_dst}.h, {zero_tmp}.h\n"
        return code_str
    code_str += f"uzp1      {low_tmp}.h, {low_tmp}.h, {low_tmp}.h\n"
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {zero_tmp}.h\n"
    return code_str


def _emit_ext_gather_last_k_two_bases(
    ctx,
    base0,
    role0,
    base1,
    role1,
    index_vec,
    dst0,
    dst1,
    low_tmp,
    zero_tmp,
):
    pred0 = _ext_pred(ctx, role0)
    pred1 = _ext_pred(ctx, role1)
    pred0_logical = _pred(ctx, role0)
    code_str = f""

    code_str += _zip_ext_predicate(ctx, role0)
    code_str += f"ld1h      {low_tmp}.s, {pred0}/z, [{base0}, {index_vec}.s, UXTW]\n"
    code_str += f"uzp1      {low_tmp}.h, {low_tmp}.h, {low_tmp}.h\n"
    code_str += f"mov       {dst0}.h, #0\n"
    code_str += f"mov       {dst1}.h, #0\n"
    code_str += f"mov       {dst0}.h, {pred0_logical}/m, {low_tmp}.h\n"

    code_str += _zip_ext_predicate(ctx, role1)
    code_str += f"ld1h      {low_tmp}.s, {pred1}/z, [{base1}, {index_vec}.s, UXTW]\n"
    code_str += f"uzp1      {low_tmp}.h, {low_tmp}.h, {low_tmp}.h\n"
    code_str += f"mov       {zero_tmp}.h, #0\n"
    code_str += f"ext       {zero_tmp}.b, {zero_tmp}.b, {low_tmp}.b, #32\n"
    code_str += f"orr       {dst0}.d, {dst0}.d, {zero_tmp}.d\n"
    code_str += f"mov       {low_tmp}.d, {dst0}.d\n"
    code_str += f"mov       {dst1}.h, #0\n"
    code_str += f"zip1      {dst0}.h, {dst0}.h, {dst1}.h\n"
    code_str += f"zip2      {dst1}.h, {low_tmp}.h, {dst1}.h\n"
    code_str += _zip_ext_predicate(ctx, role0)
    code_str += _zip_ext_predicate(ctx, role1)
    return code_str


def _emit_non_ext_gather(load_inst, ctx, base, role, index_vec, dst):
    suffix = get_ld_element_suffix(ctx)
    pred = _load_predicate(ctx, role)
    return f"{load_inst}      {dst}{suffix}, {pred}, [{base}, {index_vec}.s, UXTW]\n"


def _emit_a_head(ctx, config, a0, role, ldaopt, next_dst=None, next_role=None):
    regs = ctx.registers
    if config.a_mode == LOAD_CONTIGUOUS:
        if ctx.is_ext_precision():
            if ctx.use_ext_paired_fast_path() and next_dst is not None:
                return _emit_ext_contiguous_head_pair(
                    ctx,
                    regs.pointers.pA0,
                    regs.params.LDA,
                    regs.address.TMP_PTR,
                    role,
                    next_role,
                    a0,
                    next_dst,
                    regs.vectors.a_low,
                    regs.vectors.pair_high,
                )
            return _emit_ext_contiguous_head(
                ctx,
                regs.pointers.pA0,
                regs.params.LDA,
                regs.address.TMP_PTR,
                role,
                a0,
                regs.vectors.a_low,
                regs.vectors.pair_high,
            )
        return _emit_non_ext_contiguous(ldaopt, ctx, regs.pointers.pA0, role, a0, 0)
    if ctx.is_ext_precision():
        if ctx.use_ext_paired_fast_path() and next_dst is not None:
            code_str = _emit_ext_gather_pair(
                ctx,
                regs.pointers.pA0,
                role,
                regs.vectors.a_index,
                a0,
                regs.vectors.a_low,
                regs.vectors.pair_high,
                regs.address.TMP_PTR,
            )
            code_str += _emit_gather_base(regs.pointers.pAn, regs, 1, True)
            code_str += _emit_ext_gather_pair(
                ctx,
                regs.pointers.pAn,
                next_role,
                regs.vectors.a_index,
                next_dst,
                regs.vectors.a_low,
                regs.vectors.pair_high,
                regs.address.TMP_PTR,
            )
            return code_str
        return _emit_ext_gather_pair(
            ctx,
            regs.pointers.pA0,
            role,
            regs.vectors.a_index,
            a0,
            regs.vectors.a_low,
            regs.vectors.pair_high,
            regs.address.TMP_PTR,
            next_dst=next_dst if ctx.use_ext_paired_fast_path() and next_dst is not None else None,
            next_role=next_role,
        )
    return _emit_non_ext_gather(ldaopt, ctx, regs.pointers.pA0, role, regs.vectors.a_index, a0)


def _emit_b_head(ctx, config, b0, role, ldopt, next_dst=None, next_role=None):
    regs = ctx.registers
    if config.b_mode == LOAD_CONTIGUOUS:
        if ctx.is_ext_precision():
            if ctx.use_ext_paired_fast_path() and next_dst is not None:
                return _emit_ext_contiguous_head_pair(
                    ctx,
                    regs.pointers.pB0,
                    regs.params.LDB,
                    regs.address.TMP_PTR1,
                    role,
                    next_role,
                    b0,
                    next_dst,
                    config.b_low_tmp,
                    config.b_high_tmp,
                )
            return _emit_ext_contiguous_head(
                ctx,
                regs.pointers.pB0,
                regs.params.LDB,
                regs.address.TMP_PTR1,
                role,
                b0,
                config.b_low_tmp,
                config.b_high_tmp,
            )
        return _emit_non_ext_contiguous(ldopt, ctx, regs.pointers.pB0, role, b0, 0)
    if ctx.is_ext_precision():
        if ctx.use_ext_paired_fast_path() and next_dst is not None:
            code_str = _emit_ext_gather_pair(
                ctx,
                regs.pointers.pB0,
                role,
                regs.vectors.b_index,
                b0,
                config.b_low_tmp,
                config.b_high_tmp,
                regs.address.TMP_PTR1,
            )
            code_str += _emit_gather_base(regs.pointers.pBn, regs, 1, False)
            code_str += _emit_ext_gather_pair(
                ctx,
                regs.pointers.pBn,
                next_role,
                regs.vectors.b_index,
                next_dst,
                config.b_low_tmp,
                config.b_high_tmp,
                regs.address.TMP_PTR1,
            )
            return code_str
        return _emit_ext_gather_pair(
            ctx,
            regs.pointers.pB0,
            role,
            regs.vectors.b_index,
            b0,
            config.b_low_tmp,
            config.b_high_tmp,
            regs.address.TMP_PTR1,
            next_dst=next_dst if ctx.use_ext_paired_fast_path() and next_dst is not None else None,
            next_role=next_role,
        )
    return _emit_non_ext_gather(ldopt, ctx, regs.pointers.pB0, role, regs.vectors.b_index, b0)


def _emit_gather_base_setup(target, registers, lane, is_a):
    if lane == 0:
        return ""
    base = registers.pointers.pA0 if is_a else registers.pointers.pB0
    offset = registers.address.pA_OFFSET if is_a else registers.address.pB_OFFSET
    return _scaled_add(target, base, offset, lane)


def _gather_base_register(registers, lane, is_a):
    if lane == 0:
        return registers.pointers.pA0 if is_a else registers.pointers.pB0
    return registers.pointers.pAn if is_a else registers.pointers.pBn


def _emit_a_last_k(ctx, config, a_reg, role, lane, next_reg=None, next_role=None):
    regs = ctx.registers
    if config.a_mode == LOAD_CONTIGUOUS:
        return _emit_ext_contiguous_last_k(
            ctx,
            regs.pointers.pA0,
            role,
            a_reg,
            lane,
            regs.vectors.a_low,
            regs.vectors.pair_high,
            next_dst=next_reg if ctx.use_ext_paired_fast_path() and next_reg is not None else None,
            next_role=next_role,
        )
    code_str = _emit_gather_base_setup(regs.pointers.pAn, regs, lane, True)
    if ctx.use_ext_paired_fast_path() and next_reg is not None:
        code_str += _emit_ext_gather_last_k(
            ctx,
            _gather_base_register(regs, lane, True),
            role,
            regs.vectors.a_index,
            a_reg,
            regs.vectors.a_low,
            regs.vectors.pair_high,
        )
        code_str += _emit_gather_base(regs.address.TMP_PTR2, regs, lane + 1, True)
        code_str += _emit_ext_gather_last_k(
            ctx,
            regs.address.TMP_PTR2,
            next_role,
            regs.vectors.a_index,
            next_reg,
            regs.vectors.a_low,
            regs.vectors.pair_high,
        )
        return code_str
    code_str += _emit_ext_gather_last_k(
        ctx,
        _gather_base_register(regs, lane, True),
        role,
        regs.vectors.a_index,
        a_reg,
        regs.vectors.a_low,
        regs.vectors.pair_high,
        next_dst=next_reg if ctx.use_ext_paired_fast_path() and next_reg is not None else None,
        next_role=next_role,
    )
    return code_str


def _emit_b_last_k(ctx, config, b_reg, role, lane, next_reg=None, next_role=None):
    regs = ctx.registers
    if config.b_mode == LOAD_CONTIGUOUS:
        return _emit_ext_contiguous_last_k(
            ctx,
            regs.pointers.pB0,
            role,
            b_reg,
            lane,
            config.b_low_tmp,
            config.b_high_tmp,
            next_dst=next_reg if ctx.use_ext_paired_fast_path() and next_reg is not None else None,
            next_role=next_role,
        )
    code_str = _emit_gather_base_setup(regs.pointers.pBn, regs, lane, False)
    if ctx.use_ext_paired_fast_path() and next_reg is not None:
        code_str += _emit_ext_gather_last_k(
            ctx,
            _gather_base_register(regs, lane, False),
            role,
            regs.vectors.b_index,
            b_reg,
            config.b_low_tmp,
            config.b_high_tmp,
        )
        code_str += _emit_gather_base(regs.address.TMP_PTR2, regs, lane + 1, False)
        code_str += _emit_ext_gather_last_k(
            ctx,
            regs.address.TMP_PTR2,
            next_role,
            regs.vectors.b_index,
            next_reg,
            config.b_low_tmp,
            config.b_high_tmp,
        )
        return code_str
    code_str += _emit_ext_gather_last_k(
        ctx,
        _gather_base_register(regs, lane, False),
        role,
        regs.vectors.b_index,
        b_reg,
        config.b_low_tmp,
        config.b_high_tmp,
        next_dst=next_reg if ctx.use_ext_paired_fast_path() and next_reg is not None else None,
        next_role=next_role,
    )
    return code_str


def _emit_gather_base(target, registers, lane, is_a):
    return _emit_gather_base_setup(target, registers, lane, is_a)


def _emit_a_lane(ctx, config, dst, role, lane, ldaopt, next_dst=None, next_role=None):
    regs = ctx.registers
    if config.a_mode == LOAD_CONTIGUOUS:
        if ctx.is_ext_precision():
            return _emit_ext_contiguous_lane(
                ctx,
                regs.pointers.pA0,
                regs.address.TMP_PTR,
                role,
                dst,
                lane,
                regs.vectors.a_low,
                regs.vectors.pair_high,
                next_dst=next_dst if ctx.use_ext_paired_fast_path() and next_dst is not None else None,
                next_role=next_role,
            )
        return _emit_non_ext_contiguous(ldaopt, ctx, regs.pointers.pA0, role, dst, lane)

    code_str = _emit_gather_base(regs.pointers.pAn, regs, lane, True)
    if ctx.is_ext_precision():
        if ctx.use_ext_paired_fast_path() and next_dst is not None:
            code_str += _emit_ext_gather_pair(
                ctx,
                regs.pointers.pAn,
                role,
                regs.vectors.a_index,
                dst,
                regs.vectors.a_low,
                regs.vectors.pair_high,
                regs.address.TMP_PTR2,
            )
            code_str += _emit_gather_base(regs.address.TMP_PTR1, regs, lane + 1, True)
            code_str += _emit_ext_gather_pair(
                ctx,
                regs.address.TMP_PTR1,
                next_role,
                regs.vectors.a_index,
                next_dst,
                regs.vectors.a_low,
                regs.vectors.pair_high,
                regs.address.TMP_PTR2,
            )
            return code_str
        code_str += _emit_ext_gather_pair(
            ctx,
            regs.pointers.pAn,
            role,
            regs.vectors.a_index,
            dst,
            regs.vectors.a_low,
            regs.vectors.pair_high,
            regs.address.TMP_PTR2,
            next_dst=next_dst if ctx.use_ext_paired_fast_path() and next_dst is not None else None,
            next_role=next_role,
        )
        return code_str
    code_str += _emit_non_ext_gather(ldaopt, ctx, regs.pointers.pAn, role, regs.vectors.a_index, dst)
    return code_str


def _emit_b_lane(ctx, config, dst, role, lane, ldopt, next_dst=None, next_role=None):
    regs = ctx.registers
    if config.b_mode == LOAD_CONTIGUOUS:
        if ctx.is_ext_precision():
            return _emit_ext_contiguous_lane(
                ctx,
                regs.pointers.pB0,
                regs.address.TMP_PTR1,
                role,
                dst,
                lane,
                config.b_low_tmp,
                config.b_high_tmp,
                next_dst=next_dst if ctx.use_ext_paired_fast_path() and next_dst is not None else None,
                next_role=next_role,
            )
        return _emit_non_ext_contiguous(ldopt, ctx, regs.pointers.pB0, role, dst, lane)

    code_str = _emit_gather_base(regs.pointers.pBn, regs, lane, False)
    if ctx.is_ext_precision():
        if ctx.use_ext_paired_fast_path() and next_dst is not None:
            code_str += _emit_ext_gather_pair(
                ctx,
                regs.pointers.pBn,
                role,
                regs.vectors.b_index,
                dst,
                config.b_low_tmp,
                config.b_high_tmp,
                regs.address.TMP_PTR2,
            )
            code_str += _emit_gather_base(regs.address.TMP_PTR, regs, lane + 1, False)
            code_str += _emit_ext_gather_pair(
                ctx,
                regs.address.TMP_PTR,
                next_role,
                regs.vectors.b_index,
                next_dst,
                config.b_low_tmp,
                config.b_high_tmp,
                regs.address.TMP_PTR2,
            )
            return code_str
        code_str += _emit_ext_gather_pair(
            ctx,
            regs.pointers.pBn,
            role,
            regs.vectors.b_index,
            dst,
            config.b_low_tmp,
            config.b_high_tmp,
            regs.address.TMP_PTR2,
            next_dst=next_dst if ctx.use_ext_paired_fast_path() and next_dst is not None else None,
            next_role=next_role,
        )
        return code_str
    code_str += _emit_non_ext_gather(ldopt, ctx, regs.pointers.pBn, role, regs.vectors.b_index, dst)
    return code_str


def _emit_small_svindex(ctx, config):
    regs = ctx.registers
    shift = get_element_size_shift(ctx)
    code_str = f""
    if config.a_mode == LOAD_GATHER:
        code_str += f"lsl     {regs.counters.TMP_CNT}, {regs.params.LDA}, #{shift}\n"
        code_str += f"mov     {regs.vectors.a_index}.s, #0\n"
        code_str += f"index   {regs.vectors.a_index}.s, #0, {regs.counters.TMP_CNT_SIN}\n"
    if config.b_mode == LOAD_GATHER:
        code_str += f"lsl     {regs.counters.TMP_CNT}, {regs.params.LDB}, #{shift}\n"
        code_str += f"mov     {regs.vectors.b_index}.s, #0\n"
        code_str += f"index   {regs.vectors.b_index}.s, #0, {regs.counters.TMP_CNT_SIN}\n"
    return code_str


def _emit_small_n_pre(ctx):
    regs = ctx.registers
    code_str = f""
    code_str += f"mov     {regs.pointers.pAt}, {regs.params.origPA}\n"
    code_str += f"mov     {regs.pointers.pB0}, {regs.pointers.pBt}\n"
    return code_str


def _emit_small_n_post(ctx, config):
    regs = ctx.registers
    shift = get_element_size_shift(ctx)
    if config.b_mode == LOAD_GATHER:
        code_str = f""
        code_str += f"mul     {regs.counters.TMP_CNT}, {regs.params.LDB}, {regs.dims.MIN_N}\n"
        code_str += f"add     {regs.pointers.pBt}, {regs.pointers.pBt}, {regs.counters.TMP_CNT}, lsl #{shift}\n"
        return code_str
    return f"add     {regs.pointers.pBt}, {regs.pointers.pBt}, {regs.dims.MIN_N}, lsl #{shift}\n"


def _emit_small_m_pre(ctx, config):
    regs = ctx.registers
    shift = get_element_size_shift(ctx)
    code_str = f""
    code_str += f"mov      {regs.pointers.pB0}, {regs.pointers.pBt}\n"
    code_str += f"mov      {regs.pointers.pA0}, {regs.pointers.pAt}\n"
    if config.a_mode == LOAD_GATHER:
        code_str += f"lsl      {regs.address.pA_OFFSET}, {regs.params.LDA}, #{6 + shift - 2}\n"
        code_str += f"mov      {regs.address.OFFSET_A}, #1\n"
    else:
        code_str += f"mov      {regs.address.OFFSET_A}, {regs.params.LDA}\n"
    if config.b_mode == LOAD_GATHER:
        code_str += f"lsl      {regs.address.pB_OFFSET}, {regs.params.LDB}, #{6 + shift - 2}\n"
        code_str += f"mov      {regs.address.OFFSET_B}, #1\n"
    else:
        code_str += f"mov      {regs.address.OFFSET_B}, {regs.params.LDB}\n"
    return code_str


def _emit_small_m_post(ctx, config):
    regs = ctx.registers
    shift = get_element_size_shift(ctx)
    if config.a_mode == LOAD_GATHER:
        code_str = f""
        code_str += f"mul     {regs.counters.TMP_CNT}, {regs.params.LDA}, {regs.dims.MIN_M}\n"
        code_str += f"add      {regs.pointers.pAt}, {regs.pointers.pAt}, {regs.counters.TMP_CNT}, lsl #{shift}\n"
        return code_str
    return f"add      {regs.pointers.pAt}, {regs.pointers.pAt}, {regs.dims.MIN_M}, lsl #{shift}\n"


@dataclass(frozen=True)
class SmallModelConfig:
    name: str
    a_mode: str
    b_mode: str
    b_low_tmp: str
    b_high_tmp: str


@dataclass(frozen=True)
class SmallGemmModel:
    config: SmallModelConfig

    def load_a0b0(self, ctx, a0, a_role, b0, b_role, ldopt, ldaopt, a1=None, b1=None, a1_role=None, b1_role=None):
        code_str = _emit_a_head(ctx, self.config, a0, a_role, ldaopt, next_dst=a1, next_role=a1_role)
        code_str += _emit_b_head(ctx, self.config, b0, b_role, ldopt, next_dst=b1, next_role=b1_role)
        return code_str

    def load_a0b0_last_k(self, ctx, a0, a_role, b0, b_role, ldopt, ldaopt, a1=None, b1=None, a1_role=None, b1_role=None):
        code_str = _emit_a_last_k(ctx, self.config, a0, a_role, 0, next_reg=a1, next_role=a1_role)
        code_str += _emit_b_last_k(ctx, self.config, b0, b_role, 0, next_reg=b1, next_role=b1_role)
        return code_str

    def load_a1(self, ctx, a1, role, ldopt, ldaopt, a2=None, a2_role=None):
        return _emit_a_lane(ctx, self.config, a1, role, 1, ldaopt, next_dst=a2, next_role=a2_role)

    def load_a1_last_k(self, ctx, a1, role, ldopt, ldaopt, a2=None, a2_role=None):
        return _emit_a_last_k(ctx, self.config, a1, role, 1, next_reg=a2, next_role=a2_role)

    def load_a2(self, ctx, a2, role, ldopt, ldaopt, a3=None, a3_role=None):
        return _emit_a_lane(ctx, self.config, a2, role, 2, ldaopt, next_dst=a3, next_role=a3_role)

    def load_a2_last_k(self, ctx, a2, role, ldopt, ldaopt, a3=None, a3_role=None):
        return _emit_a_last_k(ctx, self.config, a2, role, 2, next_reg=a3, next_role=a3_role)

    def load_a3(self, ctx, a3, role, ldopt, ldaopt, a4=None):
        return _emit_a_lane(ctx, self.config, a3, role, 3, ldaopt)

    def load_a3_last_k(self, ctx, a3, role, ldopt, ldaopt, a4=None):
        return _emit_a_last_k(ctx, self.config, a3, role, 3)

    def load_b1(self, ctx, b1, role, ldopt, ldaopt, b2=None, b2_role=None):
        return _emit_b_lane(ctx, self.config, b1, role, 1, ldopt, next_dst=b2, next_role=b2_role)

    def load_b1_last_k(self, ctx, b1, role, ldopt, ldaopt, b2=None, b2_role=None):
        return _emit_b_last_k(ctx, self.config, b1, role, 1, next_reg=b2, next_role=b2_role)

    def load_b2(self, ctx, b2, role, ldopt, ldaopt, b3=None, b3_role=None):
        return _emit_b_lane(ctx, self.config, b2, role, 2, ldopt, next_dst=b3, next_role=b3_role)

    def load_b2_last_k(self, ctx, b2, role, ldopt, ldaopt, b3=None, b3_role=None):
        return _emit_b_last_k(ctx, self.config, b2, role, 2, next_reg=b3, next_role=b3_role)

    def load_b3(self, ctx, b3, role, ldopt, ldaopt, b4=None):
        return _emit_b_lane(ctx, self.config, b3, role, 3, ldopt)

    def load_b3_last_k(self, ctx, b3, role, ldopt, ldaopt, b4=None):
        return _emit_b_last_k(ctx, self.config, b3, role, 3)

    def set_svindex(self, ctx):
        return _emit_small_svindex(ctx, self.config)

    def kernel_mm_loop_n_pre_func(self, ctx):
        return _emit_small_n_pre(ctx)

    def kernel_mm_loop_n_post_func(self, ctx):
        return _emit_small_n_post(ctx, self.config)

    def kernel_mm_loop_m_pre_func(self, ctx):
        return _emit_small_m_pre(ctx, self.config)

    def kernel_mm_loop_m_post_func(self, ctx):
        return _emit_small_m_post(ctx, self.config)


class GeneralGemmModel:
    def load_a0b0(self, ctx, a0, a_role, b0, b_role, ldopt, ldaopt, a1=None, b1=None, a1_role=None, b1_role=None):
        regs = ctx.registers
        if ctx.is_ext_precision():
            code_str = f""
            a_pred_pre, a_pred, a_pred_post = _paired_ext_predicate(
                ctx,
                a_role,
                a1_role if ctx.use_ext_paired_fast_path() and a1 is not None else None,
            )
            b_pred_pre, b_pred, b_pred_post = _paired_ext_predicate(
                ctx,
                b_role,
                b1_role if ctx.use_ext_paired_fast_path() and b1 is not None else None,
            )
            code_str += a_pred_pre
            code_str += f"ld1h      {regs.vectors.a_low}.h, {a_pred}/z, [{regs.pointers.pA0}]\n"
            code_str += f"ld1h      {regs.vectors.pair_high}.h, {a_pred}/z, [{regs.pointers.pA0}, #1, MUL VL]\n"
            code_str += a_pred_post
            code_str += f"zip1      {a0}.h, {regs.vectors.a_low}.h, {regs.vectors.pair_high}.h\n"
            if ctx.use_ext_paired_fast_path() and a1 is not None:
                code_str += f"zip2      {a1}.h, {regs.vectors.a_low}.h, {regs.vectors.pair_high}.h\n"
            code_str += b_pred_pre
            code_str += f"ld1h      {self._b_low(ctx)}.h, {b_pred}/z, [{regs.pointers.pB0}]\n"
            code_str += f"ld1h      {self._b_high(ctx)}.h, {b_pred}/z, [{regs.pointers.pB0}, #1, MUL VL]\n"
            code_str += b_pred_post
            code_str += f"zip1      {b0}.h, {self._b_low(ctx)}.h, {self._b_high(ctx)}.h\n"
            if ctx.use_ext_paired_fast_path() and b1 is not None:
                code_str += f"zip2      {b1}.h, {self._b_low(ctx)}.h, {self._b_high(ctx)}.h\n"
            return code_str
        return (
            f"{ldopt}      {b0}{get_ld_element_suffix(ctx)}, {_load_predicate(ctx, b_role)}, [{regs.pointers.pB0}]\n"
            f"{ldaopt}     {a0}{get_ld_element_suffix(ctx)}, {_load_predicate(ctx, a_role)}, [{regs.pointers.pA0}]\n"
        )

    def _b_low(self, ctx):
        return ctx.registers.vectors.b_low

    def _b_high(self, ctx):
        return ctx.registers.vectors.b_high

    def _emit_general_a(self, ctx, dst, role, lane, ldaopt, next_dst=None, next_role=None):
        regs = ctx.registers
        if ctx.is_ext_precision():
            low_offset = lane * 2
            high_offset = low_offset + 1
            code_str = f""
            pred_pre, pred, pred_post = _paired_ext_predicate(
                ctx,
                role,
                next_role if ctx.use_ext_paired_fast_path() and next_dst is not None else None,
            )
            code_str += pred_pre
            code_str += f"ld1h     {regs.vectors.a_low}.h, {pred}/z, [{regs.pointers.pA0}, #{low_offset}, MUL VL]\n"
            code_str += f"ld1h     {regs.vectors.pair_high}.h, {pred}/z, [{regs.pointers.pA0}, #{high_offset}, MUL VL]\n"
            code_str += pred_post
            code_str += f"zip1     {dst}.h, {regs.vectors.a_low}.h, {regs.vectors.pair_high}.h\n"
            if ctx.use_ext_paired_fast_path() and next_dst is not None:
                code_str += f"zip2     {next_dst}.h, {regs.vectors.a_low}.h, {regs.vectors.pair_high}.h\n"
            return code_str
        return f"{ldaopt}     {dst}{get_ld_element_suffix(ctx)}, {_load_predicate(ctx, role)}, [{regs.pointers.pA0}, #{lane}, MUL VL]\n"

    def _emit_general_b(self, ctx, dst, role, lane, ldopt, next_dst=None, next_role=None):
        regs = ctx.registers
        if ctx.is_ext_precision():
            low_offset = lane * 2
            high_offset = low_offset + 1
            code_str = f""
            pred_pre, pred, pred_post = _paired_ext_predicate(
                ctx,
                role,
                next_role if ctx.use_ext_paired_fast_path() and next_dst is not None else None,
            )
            code_str += pred_pre
            code_str += f"ld1h      {self._b_low(ctx)}.h, {pred}/z, [{regs.pointers.pB0}, #{low_offset}, MUL VL]\n"
            code_str += f"ld1h      {self._b_high(ctx)}.h, {pred}/z, [{regs.pointers.pB0}, #{high_offset}, MUL VL]\n"
            code_str += pred_post
            code_str += f"zip1      {dst}.h, {self._b_low(ctx)}.h, {self._b_high(ctx)}.h\n"
            if ctx.use_ext_paired_fast_path() and next_dst is not None:
                code_str += f"zip2      {next_dst}.h, {self._b_low(ctx)}.h, {self._b_high(ctx)}.h\n"
            return code_str
        return f"{ldopt}      {dst}{get_ld_element_suffix(ctx)}, {_load_predicate(ctx, role)}, [{regs.pointers.pB0}, #{lane}, MUL VL]\n"

    def load_a0b0_last_k(self, ctx, a0, a_role, b0, b_role, ldopt, ldaopt, a1=None, b1=None, a1_role=None, b1_role=None):
        return self.load_a0b0(ctx, a0, a_role, b0, b_role, ldopt, ldaopt, a1=a1, b1=b1, a1_role=a1_role, b1_role=b1_role)

    def load_a1(self, ctx, a1, role, ldopt, ldaopt, a2=None, a2_role=None):
        return self._emit_general_a(ctx, a1, role, 1, ldaopt, next_dst=a2, next_role=a2_role)

    def load_a1_last_k(self, ctx, a1, role, ldopt, ldaopt, a2=None, a2_role=None):
        return self.load_a1(ctx, a1, role, ldopt, ldaopt, a2=a2, a2_role=a2_role)

    def load_a2(self, ctx, a2, role, ldopt, ldaopt, a3=None, a3_role=None):
        return self._emit_general_a(ctx, a2, role, 2, ldaopt, next_dst=a3, next_role=a3_role)

    def load_a2_last_k(self, ctx, a2, role, ldopt, ldaopt, a3=None, a3_role=None):
        return self.load_a2(ctx, a2, role, ldopt, ldaopt, a3=a3, a3_role=a3_role)

    def load_a3(self, ctx, a3, role, ldopt, ldaopt, a4=None, a4_role=None):
        return self._emit_general_a(ctx, a3, role, 3, ldaopt)

    def load_a3_last_k(self, ctx, a3, role, ldopt, ldaopt, a4=None, a4_role=None):
        return self.load_a3(ctx, a3, role, ldopt, ldaopt)

    def load_b1(self, ctx, b1, role, ldopt, ldaopt, b2=None, b2_role=None):
        return self._emit_general_b(ctx, b1, role, 1, ldopt, next_dst=b2, next_role=b2_role)

    def load_b1_last_k(self, ctx, b1, role, ldopt, ldaopt, b2=None, b2_role=None):
        return self.load_b1(ctx, b1, role, ldopt, ldaopt, b2=b2, b2_role=b2_role)

    def load_b2(self, ctx, b2, role, ldopt, ldaopt, b3=None, b3_role=None):
        return self._emit_general_b(ctx, b2, role, 2, ldopt, next_dst=b3, next_role=b3_role)

    def load_b2_last_k(self, ctx, b2, role, ldopt, ldaopt, b3=None, b3_role=None):
        return self.load_b2(ctx, b2, role, ldopt, ldaopt, b3=b3, b3_role=b3_role)

    def load_b3(self, ctx, b3, role, ldopt, ldaopt, b4=None, b4_role=None):
        return self._emit_general_b(ctx, b3, role, 3, ldopt)

    def load_b3_last_k(self, ctx, b3, role, ldopt, ldaopt, b4=None, b4_role=None):
        return self.load_b3(ctx, b3, role, ldopt, ldaopt)


    def set_svindex(self, ctx):
        return ""

    def kernel_mm_loop_n_pre_func(self, ctx):
        regs = ctx.registers
        return (
            f"mov     {regs.pointers.pAt}, {regs.params.origPA}\n"
            f"mov     {regs.pointers.pB0}, {regs.pointers.pBt}\n"
        )

    def kernel_mm_loop_n_post_func(self, ctx):
        return f"mov     {ctx.registers.pointers.pBt}, {ctx.registers.pointers.pB0}\n"

    def kernel_mm_loop_m_pre_func(self, ctx):
        regs = ctx.registers
        return (
            f"mov      {regs.pointers.pB0}, {regs.pointers.pBt}\n"
            f"mov      {regs.pointers.pA0}, {regs.pointers.pAt}\n"
            f"mov      {regs.address.OFFSET_A}, {regs.params.LDA}\n"
            f"mov      {regs.address.OFFSET_B}, {regs.params.LDB}\n"
        )

    def kernel_mm_loop_m_post_func(self, ctx):
        regs = ctx.registers
        return f"mov      {regs.pointers.pAt}, {regs.pointers.pA0}\n"


small_nn_model = SmallGemmModel(SmallModelConfig("small_nn", LOAD_CONTIGUOUS, LOAD_GATHER, "z26", "z29"))
small_nt_model = SmallGemmModel(SmallModelConfig("small_nt", LOAD_CONTIGUOUS, LOAD_CONTIGUOUS, "z28", "z30"))
small_tn_model = SmallGemmModel(SmallModelConfig("small_tn", LOAD_GATHER, LOAD_GATHER, "z26", "z29"))
small_tt_model = SmallGemmModel(SmallModelConfig("small_tt", LOAD_GATHER, LOAD_CONTIGUOUS, "z26", "z27"))
general_model = GeneralGemmModel()
