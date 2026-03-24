from copy import deepcopy

from global_config import LDNT1, STNT1, get_element_size_shift, get_element_suffix, get_ld1, get_mopa_inst, tile_size_from_vl

# `kernel_asm.py` owns the tile-local instruction schedule. The loop layers
# above decide *which* `mvl x nvl` shape is active; this file decides *how* that
# shape loads A/B lanes, issues mopa instructions, and advances pointers.
#
# The key abstraction is a small "plan": a list of `load_ab` / `a` / `b` /
# `mopa` / `update` records. For ext-precision small kernels we derive paired
# variants from the base plan instead of duplicating a separate schedule per
# transpose and tile.


def _resolve_load_inst(ctx, load_inst):
    return get_ld1(ctx) if load_inst is None else load_inst


def _vector_operand(ctx, reg):
    return f"{reg}{get_element_suffix(ctx)}"


def _pointer_update(ctx, ptr_name):
    regs = ctx.registers
    if ptr_name == "A":
        update_base = regs.address.TMP_PTR if ctx.is_ext_precision() else regs.pointers.pA0
        live_ptr = regs.pointers.pA0
        offset = regs.address.OFFSET_A
    else:
        update_base = regs.address.TMP_PTR1 if ctx.is_ext_precision() else regs.pointers.pB0
        live_ptr = regs.pointers.pB0
        offset = regs.address.OFFSET_B
    return f"add          {live_ptr}, {update_base}, {offset}, LSL #{get_element_size_shift(ctx)}\n"


def _emit_load(ctx, op, regs_a, regs_b, load_inst, lda_inst, last_k):
    model = ctx.model
    if op["kind"] == "load_ab":
        load_fn = model.load_a0b0_last_k if last_k else model.load_a0b0
        a_next = regs_a[op["next_a"]] if op.get("next_a") is not None else None
        b_next = regs_b[op["next_b"]] if op.get("next_b") is not None else None
        if a_next is None and b_next is None:
            return load_fn(
                ctx,
                regs_a[op["a"]],
                op["a_role"],
                regs_b[op["b"]],
                op["b_role"],
                load_inst,
                lda_inst,
            )
        return load_fn(
            ctx,
            regs_a[op["a"]],
            op["a_role"],
            regs_b[op["b"]],
            op["b_role"],
            load_inst,
            lda_inst,
            a_next,
            b_next,
            op.get("next_a_role"),
            op.get("next_b_role"),
        )

    reg_idx = op["idx"]
    suffix = "_last_k" if last_k else ""
    load_fn = getattr(model, f"load_{op['kind']}{reg_idx}{suffix}")
    regs = regs_a if op["kind"] == "a" else regs_b
    next_reg = regs[op["next_idx"]] if op.get("next_idx") is not None else None
    if next_reg is None:
        return load_fn(ctx, regs[reg_idx], op["role"], load_inst, lda_inst)
    return load_fn(ctx, regs[reg_idx], op["role"], load_inst, lda_inst, next_reg, op.get("next_role"))


def _emit_mopa(ctx, op, regs_a, regs_b):
    lhs_pred = ctx.registers.ext_predicate(op["m_role"]) if ctx.is_ext_precision() else ctx.registers.logical_predicate(op["m_role"])
    rhs_pred = ctx.registers.ext_predicate(op["n_role"]) if ctx.is_ext_precision() else ctx.registers.logical_predicate(op["n_role"])
    return (
        f"{get_mopa_inst(ctx)}        za{op['za']}.s, "
        f"{lhs_pred}/m, {rhs_pred}/m, "
        f"{_vector_operand(ctx, regs_a[op['a']])}, {_vector_operand(ctx, regs_b[op['b']])}\n"
    )


def _emit_kernel(ctx, plan, last_k, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    load_inst = _resolve_load_inst(ctx, ldopt)
    lda_inst = _resolve_load_inst(ctx, ldaopt)
    regs_a = [a0, a1, a2, a3]
    regs_b = [b0, b1, b2, b3]
    code_parts = []

    for op in plan:
        if op["kind"] in ("load_ab", "a", "b"):
            code_parts.append(_emit_load(ctx, op, regs_a, regs_b, load_inst, lda_inst, last_k))
        elif op["kind"] == "mopa":
            code_parts.append(_emit_mopa(ctx, op, regs_a, regs_b))
        elif op["kind"] == "update":
            code_parts.append(_pointer_update(ctx, op["ptr"]))

    return "".join(code_parts)


KERNEL_PLANS = {
    "4VL_1VL": [
        {"kind": "load_ab", "a": 0, "a_role": "m_main", "b": 0, "b_role": "n_main"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "a", "idx": 1, "role": "m_main"},
        {"kind": "mopa", "za": 1, "a": 1, "b": 0, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "a", "idx": 2, "role": "m_main"},
        {"kind": "mopa", "za": 2, "a": 2, "b": 0, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "update", "ptr": "B"},
        {"kind": "a", "idx": 3, "role": "m_tail"},
        {"kind": "mopa", "za": 3, "a": 3, "b": 0, "m_role": "m_tail", "n_role": "n_main"},
        {"kind": "update", "ptr": "A"},
    ],
    "1VL_4VL": [
        {"kind": "load_ab", "a": 0, "a_role": "m_main", "b": 0, "b_role": "n_main"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "b", "idx": 1, "role": "n_main"},
        {"kind": "mopa", "za": 1, "a": 0, "b": 1, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "b", "idx": 2, "role": "n_main"},
        {"kind": "update", "ptr": "A"},
        {"kind": "mopa", "za": 2, "a": 0, "b": 2, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "b", "idx": 3, "role": "n_tail"},
        {"kind": "mopa", "za": 3, "a": 0, "b": 3, "m_role": "m_main", "n_role": "n_tail"},
        {"kind": "update", "ptr": "B"},
    ],
    "3VL_1VL": [
        {"kind": "load_ab", "a": 0, "a_role": "m_main", "b": 0, "b_role": "n_main"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "a", "idx": 1, "role": "m_main"},
        {"kind": "mopa", "za": 1, "a": 1, "b": 0, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "update", "ptr": "B"},
        {"kind": "a", "idx": 2, "role": "m_tail"},
        {"kind": "mopa", "za": 2, "a": 2, "b": 0, "m_role": "m_tail", "n_role": "n_main"},
        {"kind": "update", "ptr": "A"},
    ],
    "1VL_3VL": [
        {"kind": "load_ab", "a": 0, "a_role": "m_main", "b": 0, "b_role": "n_main"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "b", "idx": 1, "role": "n_main"},
        {"kind": "mopa", "za": 1, "a": 0, "b": 1, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "update", "ptr": "A"},
        {"kind": "b", "idx": 2, "role": "n_tail"},
        {"kind": "mopa", "za": 2, "a": 0, "b": 2, "m_role": "m_main", "n_role": "n_tail"},
        {"kind": "update", "ptr": "B"},
    ],
    "2VL_2VL": [
        {"kind": "load_ab", "a": 0, "a_role": "m_main", "b": 0, "b_role": "n_main"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "b", "idx": 1, "role": "n_tail"},
        {"kind": "mopa", "za": 1, "a": 0, "b": 1, "m_role": "m_main", "n_role": "n_tail"},
        {"kind": "a", "idx": 1, "role": "m_tail"},
        {"kind": "mopa", "za": 2, "a": 1, "b": 0, "m_role": "m_tail", "n_role": "n_main"},
        {"kind": "update", "ptr": "A"},
        {"kind": "mopa", "za": 3, "a": 1, "b": 1, "m_role": "m_tail", "n_role": "n_tail"},
        {"kind": "update", "ptr": "B"},
    ],
    "1VL_2VL": [
        {"kind": "load_ab", "a": 0, "a_role": "m_main", "b": 0, "b_role": "n_main"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "b", "idx": 1, "role": "n_tail"},
        {"kind": "mopa", "za": 1, "a": 0, "b": 1, "m_role": "m_main", "n_role": "n_tail"},
        {"kind": "update", "ptr": "A"},
        {"kind": "update", "ptr": "B"},
    ],
    "2VL_1VL": [
        {"kind": "load_ab", "a": 0, "a_role": "m_main", "b": 0, "b_role": "n_main"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "a", "idx": 1, "role": "m_tail"},
        {"kind": "mopa", "za": 2, "a": 1, "b": 0, "m_role": "m_tail", "n_role": "n_main"},
        {"kind": "update", "ptr": "A"},
        {"kind": "update", "ptr": "B"},
    ],
    "1VL_1VL": [
        {"kind": "load_ab", "a": 0, "a_role": "m_main", "b": 0, "b_role": "n_main"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "m_role": "m_main", "n_role": "n_main"},
        {"kind": "update", "ptr": "A"},
        {"kind": "update", "ptr": "B"},
    ],
}

KERNEL_LAST_K_PLANS = {
    "4VL_1VL": KERNEL_PLANS["4VL_1VL"],
    "1VL_4VL": KERNEL_PLANS["1VL_4VL"],
    "3VL_1VL": KERNEL_PLANS["3VL_1VL"],
    "1VL_3VL": KERNEL_PLANS["1VL_3VL"],
    "2VL_2VL": KERNEL_PLANS["2VL_2VL"],
    "1VL_2VL": KERNEL_PLANS["1VL_2VL"][:4],
    "2VL_1VL": KERNEL_PLANS["2VL_1VL"][:4],
    "1VL_1VL": KERNEL_PLANS["1VL_1VL"][:2],
}

_KEY_TILE_VL = {
    "4VL_1VL": (4, 1),
    "1VL_4VL": (1, 4),
    "3VL_1VL": (3, 1),
    "1VL_3VL": (1, 3),
    "2VL_2VL": (2, 2),
    "1VL_2VL": (1, 2),
    "2VL_1VL": (2, 1),
    "1VL_1VL": (1, 1),
}


_KERNEL_LABEL_COUNTER = 0


def _new_kernel_label(prefix):
    global _KERNEL_LABEL_COUNTER
    _KERNEL_LABEL_COUNTER += 1
    return f".L_kernel_asm_{prefix}_{_KERNEL_LABEL_COUNTER}"


def _default_lane_roles(side, vl_count):
    lane_kinds = {
        1: ("main",),
        2: ("main", "tail"),
        3: ("main", "main", "tail"),
        4: ("main", "main", "main", "tail"),
    }
    prefix = "m" if side == "a" else "n"
    return {lane: f"{prefix}_{kind}" for lane, kind in enumerate(lane_kinds[vl_count])}


def _same_role_pair_specs(side, vl_count, leading_only=False):
    lane_roles = _default_lane_roles(side, vl_count)
    pair_specs = []
    for lane in range(0, vl_count - 1, 2):
        next_lane = lane + 1
        if lane_roles[lane] != lane_roles[next_lane]:
            continue
        pair_specs.append((lane, next_lane, lane_roles[lane]))
        if leading_only:
            break
    return tuple(pair_specs)


def _full_pair_specs(side, vl_count):
    return _same_role_pair_specs(side, vl_count, leading_only=False)


def _hybrid_pair_specs(side, vl_count):
    return _same_role_pair_specs(side, vl_count, leading_only=True)


def _pair_spec_set(pair_specs):
    return {(lane0, lane1) for lane0, lane1, _ in pair_specs}


def _lane_roles_with_pairs(side, vl_count, pair_specs):
    lane_roles = _default_lane_roles(side, vl_count)
    for lane0, lane1, role in pair_specs:
        lane_roles[lane0] = role
        lane_roles[lane1] = role
    return lane_roles


def _load_lane_index(op, side):
    if side == "a":
        if op["kind"] == "load_ab":
            return op["a"]
        if op["kind"] == "a":
            return op["idx"]
        return None
    if op["kind"] == "load_ab":
        return op["b"]
    if op["kind"] == "b":
        return op["idx"]
    return None


def _load_lane_role(op, side):
    if side == "a":
        if op["kind"] == "load_ab":
            return op["a_role"]
        if op["kind"] == "a":
            return op["role"]
        return None
    if op["kind"] == "load_ab":
        return op["b_role"]
    if op["kind"] == "b":
        return op["role"]
    return None


def _set_paired_load(op, side, next_idx, next_role):
    if op["kind"] == "load_ab":
        op[f"next_{side}"] = next_idx
        op[f"next_{side}_role"] = next_role
        return
    op["next_idx"] = next_idx
    op["next_role"] = next_role


def _rewrite_plan_roles(plan, a_roles, b_roles):
    for op in plan:
        if op["kind"] == "load_ab":
            op["a_role"] = a_roles[op["a"]]
            op["b_role"] = b_roles[op["b"]]
        elif op["kind"] == "a":
            op["role"] = a_roles[op["idx"]]
        elif op["kind"] == "b":
            op["role"] = b_roles[op["idx"]]
        elif op["kind"] == "mopa":
            op["m_role"] = a_roles[op["a"]]
            op["n_role"] = b_roles[op["b"]]


def _collapse_side_pairs(plan, side, lane_pairs):
    ptr_name = "A" if side == "a" else "B"
    i = 0
    while i < len(plan):
        op = plan[i]
        lane = _load_lane_index(op, side)
        if lane is None:
            i += 1
            continue
        next_lane = lane + 1
        if (lane, next_lane) not in lane_pairs:
            i += 1
            continue
        if op["kind"] == "load_ab" and op.get(f"next_{side}") is not None:
            i += 1
            continue
        if op["kind"] == side and op.get("next_idx") is not None:
            i += 1
            continue

        found = None
        for j in range(i + 1, len(plan)):
            next_op = plan[j]
            if next_op["kind"] == "update" and next_op["ptr"] == ptr_name:
                break
            if _load_lane_index(next_op, side) == next_lane:
                found = j
                break

        if found is not None:
            next_op = plan.pop(found)
            _set_paired_load(op, side, next_lane, _load_lane_role(next_op, side))
        i += 1
    return plan


def _build_small_plan_variant(key, last_k, a_pairs=(), b_pairs=()):
    # Start from the canonical scalar plan for this tile, then rewrite lane
    # roles and pair only the chunks that are explicitly allowed. This keeps the
    # exact/tail decision localized here instead of scattering it across load
    # helpers.
    base_plan = deepcopy(KERNEL_LAST_K_PLANS[key] if last_k else KERNEL_PLANS[key])
    m_vl, n_vl = _KEY_TILE_VL[key]
    a_roles = _lane_roles_with_pairs("a", m_vl, a_pairs)
    b_roles = _lane_roles_with_pairs("b", n_vl, b_pairs)
    _rewrite_plan_roles(base_plan, a_roles, b_roles)
    if a_pairs:
        base_plan = _collapse_side_pairs(base_plan, "a", _pair_spec_set(a_pairs))
    if b_pairs:
        base_plan = _collapse_side_pairs(base_plan, "b", _pair_spec_set(b_pairs))
    return base_plan


def _axis_min_register(ctx, side):
    return ctx.registers.dims.MIN_M if side == "a" else ctx.registers.dims.MIN_N


def _side_is_contiguous(ctx, side):
    config = getattr(ctx.model, "config", None)
    if config is None:
        return False
    attr = "a_mode" if side == "a" else "b_mode"
    return getattr(config, attr, None) == "contiguous"


def _emit_axis_plan_select(ctx, side, threshold, partial_code, full_code):
    # A single contiguous side can be upgraded from safe -> paired when the
    # remaining logical axis is wide enough to cover the whole chunk.
    fast_label = _new_kernel_label(f"{side}_full")
    done_label = _new_kernel_label(f"{side}_done")
    dim_reg = _axis_min_register(ctx, side)
    code_str = f"cmp       {dim_reg}, #{threshold}\n"
    code_str += f"bge       {fast_label}\n"
    code_str += partial_code
    code_str += f"b         {done_label}\n"
    code_str += f"{fast_label}:\n"
    code_str += full_code
    code_str += f"{done_label}:\n"
    return code_str


def _emit_dual_axis_plan_select(ctx, m_threshold, n_threshold, safe_code, a_code, b_code, both_code):
    # `2VL x 2VL` is the only small shape where both axes may independently
    # become pairable. Check the M and N remainder counts separately so mixed
    # exact/tail tiles can still use a one-sided paired plan.
    a_full_label = _new_kernel_label("m_full")
    b_check_label = _new_kernel_label("n_check")
    b_full_under_safe_label = _new_kernel_label("n_full_safe")
    both_full_label = _new_kernel_label("both_full")
    done_label = _new_kernel_label("dual_done")

    code_str = f"cmp       {ctx.registers.dims.MIN_M}, #{m_threshold}\n"
    code_str += f"bge       {a_full_label}\n"
    code_str += f"{b_check_label}:\n"
    code_str += f"cmp       {ctx.registers.dims.MIN_N}, #{n_threshold}\n"
    code_str += f"bge       {b_full_under_safe_label}\n"
    code_str += safe_code
    code_str += f"b         {done_label}\n"
    code_str += f"{b_full_under_safe_label}:\n"
    code_str += b_code
    code_str += f"b         {done_label}\n"
    code_str += f"{a_full_label}:\n"
    code_str += f"cmp       {ctx.registers.dims.MIN_N}, #{n_threshold}\n"
    code_str += f"bge       {both_full_label}\n"
    code_str += a_code
    code_str += f"b         {done_label}\n"
    code_str += f"{both_full_label}:\n"
    code_str += both_code
    code_str += f"{done_label}:\n"
    return code_str


def _emit_plan_code(ctx, key, last_k, plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, plan, last_k, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def _emit_active_kernel(ctx, key, last_k, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    safe_plan = _build_small_plan_variant(key, last_k)
    safe_code = _emit_plan_code(ctx, key, last_k, safe_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)

    if last_k or not (ctx.is_ext_precision() and ctx.use_ext_paired_fast_path()):
        return safe_code

    m_vl, n_vl = _KEY_TILE_VL[key]
    a_contig = _side_is_contiguous(ctx, "a")
    b_contig = _side_is_contiguous(ctx, "b")

    if key == "1VL_1VL" or (not a_contig and not b_contig):
        return safe_code

    if key == "1VL_2VL":
        if not b_contig:
            return safe_code
        # Only pair `1x2` when the second N lane is the same logical role. A
        # `main + tail` exact tile is still emitted with the safe plan.
        b_pairs = _full_pair_specs("b", n_vl)
        if not b_pairs:
            return safe_code
        full_plan = _build_small_plan_variant(key, last_k, b_pairs=b_pairs)
        full_code = _emit_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
        return _emit_axis_plan_select(ctx, "b", tile_size_from_vl(2), safe_code, full_code)

    if key == "2VL_1VL":
        if not a_contig:
            return safe_code
        a_pairs = _full_pair_specs("a", m_vl)
        if not a_pairs:
            return safe_code
        full_plan = _build_small_plan_variant(key, last_k, a_pairs=a_pairs)
        full_code = _emit_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
        return _emit_axis_plan_select(ctx, "a", tile_size_from_vl(2), safe_code, full_code)

    if key == "1VL_3VL":
        if not b_contig:
            return safe_code
        b_pairs = _hybrid_pair_specs("b", n_vl)
        if not b_pairs:
            return safe_code
        hybrid_plan = _build_small_plan_variant(key, last_k, b_pairs=b_pairs)
        return _emit_plan_code(ctx, key, last_k, hybrid_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)

    if key == "3VL_1VL":
        if not a_contig:
            return safe_code
        a_pairs = _hybrid_pair_specs("a", m_vl)
        if not a_pairs:
            return safe_code
        hybrid_plan = _build_small_plan_variant(key, last_k, a_pairs=a_pairs)
        return _emit_plan_code(ctx, key, last_k, hybrid_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)

    if key == "1VL_4VL":
        if not b_contig:
            return safe_code
        hybrid_pairs = _hybrid_pair_specs("b", n_vl)
        if not hybrid_pairs:
            return safe_code
        hybrid_plan = _build_small_plan_variant(key, last_k, b_pairs=hybrid_pairs)
        hybrid_code = _emit_plan_code(ctx, key, last_k, hybrid_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
        full_pairs = _full_pair_specs("b", n_vl)
        if full_pairs == hybrid_pairs:
            return hybrid_code
        full_plan = _build_small_plan_variant(key, last_k, b_pairs=full_pairs)
        full_code = _emit_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
        return _emit_axis_plan_select(ctx, "b", tile_size_from_vl(4), hybrid_code, full_code)

    if key == "4VL_1VL":
        if not a_contig:
            return safe_code
        hybrid_pairs = _hybrid_pair_specs("a", m_vl)
        if not hybrid_pairs:
            return safe_code
        hybrid_plan = _build_small_plan_variant(key, last_k, a_pairs=hybrid_pairs)
        hybrid_code = _emit_plan_code(ctx, key, last_k, hybrid_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
        full_pairs = _full_pair_specs("a", m_vl)
        if full_pairs == hybrid_pairs:
            return hybrid_code
        full_plan = _build_small_plan_variant(key, last_k, a_pairs=full_pairs)
        full_code = _emit_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
        return _emit_axis_plan_select(ctx, "a", tile_size_from_vl(4), hybrid_code, full_code)

    if key == "2VL_2VL":
        if a_contig and b_contig:
            # `2x2` can use four different schedules: fully safe, A-paired,
            # B-paired, or both-sided paired. The threshold checks choose among
            # those at runtime from the same source plan.
            a_pairs = _full_pair_specs("a", m_vl)
            b_pairs = _full_pair_specs("b", n_vl)
            if not a_pairs and not b_pairs:
                return safe_code
            if a_pairs and not b_pairs:
                full_plan = _build_small_plan_variant(key, last_k, a_pairs=a_pairs)
                full_code = _emit_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
                return _emit_axis_plan_select(ctx, "a", tile_size_from_vl(2), safe_code, full_code)
            if b_pairs and not a_pairs:
                full_plan = _build_small_plan_variant(key, last_k, b_pairs=b_pairs)
                full_code = _emit_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
                return _emit_axis_plan_select(ctx, "b", tile_size_from_vl(2), safe_code, full_code)
            a_plan = _build_small_plan_variant(key, last_k, a_pairs=a_pairs)
            b_plan = _build_small_plan_variant(key, last_k, b_pairs=b_pairs)
            both_plan = _build_small_plan_variant(
                key,
                last_k,
                a_pairs=a_pairs,
                b_pairs=b_pairs,
            )
            a_code = _emit_plan_code(ctx, key, last_k, a_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
            b_code = _emit_plan_code(ctx, key, last_k, b_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
            both_code = _emit_plan_code(ctx, key, last_k, both_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
            return _emit_dual_axis_plan_select(
                ctx,
                tile_size_from_vl(2),
                tile_size_from_vl(2),
                safe_code,
                a_code,
                b_code,
                both_code,
            )
        if a_contig:
            a_pairs = _full_pair_specs("a", m_vl)
            if not a_pairs:
                return safe_code
            full_plan = _build_small_plan_variant(key, last_k, a_pairs=a_pairs)
            full_code = _emit_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
            return _emit_axis_plan_select(ctx, "a", tile_size_from_vl(2), safe_code, full_code)
        if b_contig:
            b_pairs = _full_pair_specs("b", n_vl)
            if not b_pairs:
                return safe_code
            full_plan = _build_small_plan_variant(key, last_k, b_pairs=b_pairs)
            full_code = _emit_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
            return _emit_axis_plan_select(ctx, "b", tile_size_from_vl(2), safe_code, full_code)

    return safe_code


def kernel_4VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "4VL_1VL", False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_4VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "1VL_4VL", False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_3VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "3VL_1VL", False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_3VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "1VL_3VL", False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_2VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "2VL_2VL", False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_2VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "1VL_2VL", False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "2VL_1VL", False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "1VL_1VL", False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_4VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "4VL_1VL", True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_4VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "1VL_4VL", True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_3VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "3VL_1VL", True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_3VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "1VL_3VL", True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_2VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "2VL_2VL", True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_2VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "1VL_2VL", True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "2VL_1VL", True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_active_kernel(ctx, "1VL_1VL", True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def save_zacol(pc, off, za, base_idx, idx, pg, rab0, rc0):
    code_str = f""
    code_str += f"mova         {rab0}.s, {pg}/m, {za}v.s[{base_idx}, {idx}]\n"
    code_str += f"{LDNT1}      {rc0}.s, {pg}/z, [{pc}, {off}, MUL VL]\n"
    code_str += f"fadd         {rc0}.s, {pg}/m, {rc0}.s, {rab0}.s\n"
    code_str += f"{STNT1}      {rc0}.s, {pg}, [{pc}, {off}, MUL VL]\n"
    return code_str
