from copy import deepcopy

from global_config import (
    LDNT1,
    STNT1,
    get_half_input_size_shift,
    get_half_input_suffix,
    get_half_load_inst,
    get_mopa_inst,
)

# This file converts a chosen `mvl x nvl` tile into the exact load, mopa, and pointer-update schedule that will be generated.


def _resolve_load_inst(ctx, load_inst):
    # Let callers override the load opcode while defaulting to the standard
    # half-input `ld1h` choice.
    return get_half_load_inst() if load_inst is None else load_inst


def _vector_operand(ctx, reg):
    # Half inputs always reach mopa as `.h` vectors even though ZA accumulates
    # into `.s`.
    return f"{reg}{get_half_input_suffix()}"


def _pointer_update(ctx, ptr_name):
    # Small kernels walk from paired half-load chunk bases; general kernels keep
    # advancing the live A/B pointers directly.
    regs = ctx.registers
    if ptr_name == "A":
        update_base = regs.address.TMP_PTR if ctx.use_paired_half_loads() else regs.pointers.pA0
        live_ptr = regs.pointers.pA0
        offset = regs.address.OFFSET_A
    else:
        update_base = regs.address.TMP_PTR1 if ctx.use_paired_half_loads() else regs.pointers.pB0
        live_ptr = regs.pointers.pB0
        offset = regs.address.OFFSET_B
    return f"add          {live_ptr}, {update_base}, {offset}, LSL #{get_half_input_size_shift()}\n"


def _gen_load(ctx, op, regs_a, regs_b, load_inst, lda_inst, last_k):
    # Turn one logical `load_ab` / `a` / `b` record into the matching model callback, including optional paired mates.
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
            op.get("next_a_load_role"),
            op.get("next_b_load_role"),
        )

    reg_idx = op["idx"]
    suffix = "_last_k" if last_k else ""
    load_fn = getattr(model, f"load_{op['kind']}{reg_idx}{suffix}")
    regs = regs_a if op["kind"] == "a" else regs_b
    next_reg = regs[op["next_idx"]] if op.get("next_idx") is not None else None
    if next_reg is None:
        return load_fn(ctx, regs[reg_idx], op["role"], load_inst, lda_inst)
    return load_fn(
        ctx,
        regs[reg_idx],
        op["role"],
        load_inst,
        lda_inst,
        next_reg,
        op.get("next_role"),
        op.get("next_load_role"),
    )


def _gen_mopa(ctx, op, regs_a, regs_b):
    # Emit one outer-product using half-input predicates. The `.s` suffix on ZA
    # and C-side vectors is accumulator/output semantics, not an fp32 input path.
    lhs_pred = ctx.registers.half_predicate(op["m_role"])
    rhs_pred = ctx.registers.half_predicate(op["n_role"])
    return (
        f"{get_mopa_inst(ctx)}        za{op['za']}.s, "
        f"{lhs_pred}/m, {rhs_pred}/m, "
        f"{_vector_operand(ctx, regs_a[op['a']])}, {_vector_operand(ctx, regs_b[op['b']])}\n"
    )


def _gen_kernel(ctx, plan, last_k, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    # Walk the plan in order so every load, mopa, and pointer update is generated exactly once into the tile body.
    load_inst = _resolve_load_inst(ctx, ldopt)
    lda_inst = _resolve_load_inst(ctx, ldaopt)
    regs_a = [a0, a1, a2, a3]
    regs_b = [b0, b1, b2, b3]
    code_parts = []

    for op in plan:
        if op["kind"] in ("load_ab", "a", "b"):
            code_parts.append(_gen_load(ctx, op, regs_a, regs_b, load_inst, lda_inst, last_k))
        elif op["kind"] == "mopa":
            code_parts.append(_gen_mopa(ctx, op, regs_a, regs_b))
        elif op["kind"] == "update":
            code_parts.append(_pointer_update(ctx, op["ptr"]))

    return "".join(code_parts)


# Plan construction -----------------------------------------------------------------

def _load_ab(a_idx, a_role, b_idx, b_role):
    return {"kind": "load_ab", "a": a_idx, "a_role": a_role, "b": b_idx, "b_role": b_role}


def _load_a(idx, role):
    return {"kind": "a", "idx": idx, "role": role}


def _load_b(idx, role):
    return {"kind": "b", "idx": idx, "role": role}


def _mopa(za, a_idx, b_idx, m_role, n_role):
    return {"kind": "mopa", "za": za, "a": a_idx, "b": b_idx, "m_role": m_role, "n_role": n_role}


def _update(ptr):
    return {"kind": "update", "ptr": ptr}


KERNEL_PLANS = {
    "4VL_1VL": [
        _load_ab(0, "m_main", 0, "n_main"),
        _mopa(0, 0, 0, "m_main", "n_main"),
        _load_a(1, "m_main"),
        _mopa(1, 1, 0, "m_main", "n_main"),
        _load_a(2, "m_main"),
        _mopa(2, 2, 0, "m_main", "n_main"),
        _update("B"),
        _load_a(3, "m_tail"),
        _mopa(3, 3, 0, "m_tail", "n_main"),
        _update("A"),
    ],
    "1VL_4VL": [
        _load_ab(0, "m_main", 0, "n_main"),
        _mopa(0, 0, 0, "m_main", "n_main"),
        _load_b(1, "n_main"),
        _mopa(1, 0, 1, "m_main", "n_main"),
        _load_b(2, "n_main"),
        _update("A"),
        _mopa(2, 0, 2, "m_main", "n_main"),
        _load_b(3, "n_tail"),
        _mopa(3, 0, 3, "m_main", "n_tail"),
        _update("B"),
    ],
    "3VL_1VL": [
        _load_ab(0, "m_main", 0, "n_main"),
        _mopa(0, 0, 0, "m_main", "n_main"),
        _load_a(1, "m_main"),
        _mopa(1, 1, 0, "m_main", "n_main"),
        _update("B"),
        _load_a(2, "m_tail"),
        _mopa(2, 2, 0, "m_tail", "n_main"),
        _update("A"),
    ],
    "1VL_3VL": [
        _load_ab(0, "m_main", 0, "n_main"),
        _mopa(0, 0, 0, "m_main", "n_main"),
        _load_b(1, "n_main"),
        _mopa(1, 0, 1, "m_main", "n_main"),
        _update("A"),
        _load_b(2, "n_tail"),
        _mopa(2, 0, 2, "m_main", "n_tail"),
        _update("B"),
    ],
    "2VL_2VL": [
        _load_ab(0, "m_main", 0, "n_main"),
        _mopa(0, 0, 0, "m_main", "n_main"),
        _load_b(1, "n_tail"),
        _mopa(1, 0, 1, "m_main", "n_tail"),
        _load_a(1, "m_tail"),
        _mopa(2, 1, 0, "m_tail", "n_main"),
        _update("A"),
        _mopa(3, 1, 1, "m_tail", "n_tail"),
        _update("B"),
    ],
    "1VL_2VL": [
        _load_ab(0, "m_main", 0, "n_main"),
        _mopa(0, 0, 0, "m_main", "n_main"),
        _load_b(1, "n_tail"),
        _mopa(1, 0, 1, "m_main", "n_tail"),
        _update("A"),
        _update("B"),
    ],
    "2VL_1VL": [
        _load_ab(0, "m_main", 0, "n_main"),
        _mopa(0, 0, 0, "m_main", "n_main"),
        _load_a(1, "m_tail"),
        _mopa(2, 1, 0, "m_tail", "n_main"),
        _update("A"),
        _update("B"),
    ],
    "1VL_1VL": [
        _load_ab(0, "m_main", 0, "n_main"),
        _mopa(0, 0, 0, "m_main", "n_main"),
        _update("A"),
        _update("B"),
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


# Small-plan transforms --------------------------------------------------------------


def _emit_kernel_plan(ctx, plan, last_k, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _gen_kernel(ctx, plan, last_k, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def _load_lane_index(op, side):
    # Read the logical lane index touched by one load record so we can find collapsible neighbors on one axis.
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
    # Read the logical main/tail role attached to one load record on the requested axis.
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


def _set_paired_load(op, side, next_idx, next_role, next_load_role=None):
    # Record the second lane that should be materialized together with this load when a pair collapse succeeds.
    if op["kind"] == "load_ab":
        op[f"next_{side}"] = next_idx
        op[f"next_{side}_role"] = next_role
        op[f"next_{side}_load_role"] = next_load_role
        return
    op["next_idx"] = next_idx
    op["next_role"] = next_role
    op["next_load_role"] = next_load_role


def _collapse_side_pairs(plan, side, lane_pairs, pair_load_role=None):
    # Merge adjacent same-side load records into one paired load record without crossing a pointer update boundary.
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
            _set_paired_load(
                op,
                side,
                next_lane,
                _load_lane_role(next_op, side),
                next_load_role=pair_load_role or _load_lane_role(next_op, side),
            )
        i += 1
    return plan


def _promote_lane_role(plan, side, lane_idx, promoted_role):
    # When a runtime-selected chunk is fully covered, its "tail" lane should behave like another full main lane.
    role_field = "m_role" if side == "a" else "n_role"
    load_role_field = "a_role" if side == "a" else "b_role"
    load_idx_field = "a" if side == "a" else "b"
    single_kind = side
    for op in plan:
        if op["kind"] == "load_ab" and op[load_idx_field] == lane_idx:
            op[load_role_field] = promoted_role
        elif op["kind"] == single_kind and op["idx"] == lane_idx:
            op["role"] = promoted_role
        elif op["kind"] == "mopa" and op[load_idx_field] == lane_idx:
            op[role_field] = promoted_role
    return plan


def _pair_load_role(side):
    return "m_main" if side == "a" else "n_main"


def _build_small_plan(key, last_k, pair_plan=None):
    # Build one small-kernel plan and optionally collapse the legal `2VL`
    # chunks selected by the resolved pair descriptor.
    base_plan = deepcopy(KERNEL_LAST_K_PLANS[key] if last_k else KERNEL_PLANS[key])

    if pair_plan is None or not getattr(pair_plan, "paired_enabled", False):
        return base_plan

    for lane in pair_plan.promote_a_lanes:
        base_plan = _promote_lane_role(base_plan, "a", lane, "m_main")
    for lane in pair_plan.promote_b_lanes:
        base_plan = _promote_lane_role(base_plan, "b", lane, "n_main")
    if pair_plan.a_pairs:
        base_plan = _collapse_side_pairs(
            base_plan, "a", set(pair_plan.a_pairs), pair_load_role=_pair_load_role("a")
        )
    if pair_plan.b_pairs:
        base_plan = _collapse_side_pairs(
            base_plan, "b", set(pair_plan.b_pairs), pair_load_role=_pair_load_role("b")
        )
    return base_plan


def _side_is_contiguous(ctx, side):
    # Only contiguous sides may upgrade to paired loads because gather sides already spend their budget on index-based shaping.
    config = getattr(ctx.model, "config", None)
    if config is None:
        return False
    attr = "a_mode" if side == "a" else "b_mode"
    return getattr(config, attr, None) == "contiguous"


def _gen_active_kernel(ctx, key, last_k, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    # Emit one explicit tile body from a base plan plus the resolved legal pair
    # collapses, keeping fullness selection in L1/L2 instead of re-branching
    # inside the kernel body.
    base_plan = _build_small_plan(key, last_k)
    base_code = _emit_kernel_plan(ctx, base_plan, last_k, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)

    if last_k or not ctx.use_paired_half_loads() or pair_plan is None or not pair_plan.paired_enabled:
        return base_code

    paired_plan = _build_small_plan(key, last_k, pair_plan=pair_plan)
    return _emit_kernel_plan(ctx, paired_plan, last_k, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_4VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "4VL_1VL", False, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_4VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "1VL_4VL", False, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_3VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "3VL_1VL", False, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_3VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "1VL_3VL", False, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_2VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "2VL_2VL", False, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_2VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "1VL_2VL", False, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "2VL_1VL", False, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "1VL_1VL", False, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_4VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "4VL_1VL", True, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_4VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "1VL_4VL", True, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_3VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "3VL_1VL", True, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_3VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "1VL_3VL", True, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_2VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "2VL_2VL", True, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_2VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "1VL_2VL", True, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "2VL_1VL", True, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, pair_plan=None):
    return _gen_active_kernel(ctx, "1VL_1VL", True, pair_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def save_zacol(pc, off, za, base_idx, idx, pg, rab0, rc0):
    # Extract one ZA column slice, accumulate it with C, and stream the updated vector back to memory.
    code_str = f""
    code_str += f"mova         {rab0}.s, {pg}/m, {za}v.s[{base_idx}, {idx}]\n"
    code_str += f"{LDNT1}      {rc0}.s, {pg}/z, [{pc}, {off}, MUL VL]\n"
    code_str += f"fadd         {rc0}.s, {pg}/m, {rc0}.s, {rab0}.s\n"
    code_str += f"{STNT1}      {rc0}.s, {pg}, [{pc}, {off}, MUL VL]\n"
    return code_str
