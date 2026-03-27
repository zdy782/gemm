from copy import deepcopy

from global_config import LDNT1, STNT1, get_element_size_shift, get_element_suffix, get_ld1, get_mopa_inst, tile_size_from_vl

# This file converts a chosen `mvl x nvl` tile into the exact load, mopa, and pointer-update schedule that will be generated.


def _resolve_load_inst(ctx, load_inst):
    # Let callers override the load opcode while defaulting to the model-specific `ld1*` choice.
    return get_ld1(ctx) if load_inst is None else load_inst


def _vector_operand(ctx, reg):
    # Format one Z register with the current element suffix so mopa emission stays precision-agnostic.
    return f"{reg}{get_element_suffix(ctx)}"


def _pointer_update(ctx, ptr_name):
    # Advance the live A/B pointer from the ext temporary base or the scalar base after each tile-local update record.
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
    # Emit one outer-product using the logical main/tail predicates that belong to this plan record.
    lhs_pred = ctx.registers.ext_predicate(op["m_role"]) if ctx.is_ext_precision() else ctx.registers.logical_predicate(op["m_role"])
    rhs_pred = ctx.registers.ext_predicate(op["n_role"]) if ctx.is_ext_precision() else ctx.registers.logical_predicate(op["n_role"])
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
    # Mint unique local labels so multiple runtime fast/safe dispatches can coexist in one generated function.
    global _KERNEL_LABEL_COUNTER
    _KERNEL_LABEL_COUNTER += 1
    return f".L_kernel_asm_{prefix}_{_KERNEL_LABEL_COUNTER}"


def _full_chunk_pairs(vl_count):
    # A selected `2VL` or `4VL` tile may pair every physical `2VL` chunk even when the second logical lane is only partially full.
    pair_map = {
        1: (),
        2: ((0, 1),),
        3: ((0, 1),),
        4: ((0, 1), (2, 3)),
    }
    return pair_map[vl_count]


def _leading_chunk_pairs(vl_count):
    # `3VL` tiles only pair their leading `2VL` chunk and keep the last `1VL` as a scalar lane.
    if vl_count <= 1:
        return ()
    return ((0, 1),)


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


def _build_small_plan_variant(
    key,
    last_k,
    a_pairs=(),
    b_pairs=(),
    a_pair_load_role=None,
    b_pair_load_role=None,
    promote_a_lanes=(),
    promote_b_lanes=(),
):
    # Build one small-kernel plan by collapsing the physical `2VL` chunks that the current tile shape is allowed to pair.
    base_plan = deepcopy(KERNEL_LAST_K_PLANS[key] if last_k else KERNEL_PLANS[key])
    for lane in promote_a_lanes:
        base_plan = _promote_lane_role(base_plan, "a", lane, "m_main")
    for lane in promote_b_lanes:
        base_plan = _promote_lane_role(base_plan, "b", lane, "n_main")
    if a_pairs:
        base_plan = _collapse_side_pairs(base_plan, "a", set(a_pairs), pair_load_role=a_pair_load_role)
    if b_pairs:
        base_plan = _collapse_side_pairs(base_plan, "b", set(b_pairs), pair_load_role=b_pair_load_role)
    return base_plan


def _side_is_contiguous(ctx, side):
    # Only contiguous sides may upgrade to paired loads because gather sides already spend their budget on index-based shaping.
    config = getattr(ctx.model, "config", None)
    if config is None:
        return False
    attr = "a_mode" if side == "a" else "b_mode"
    return getattr(config, attr, None) == "contiguous"


def _gen_plan_code(ctx, key, last_k, plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    # Keep one wrapper so every runtime branch feeds through the same kernel generator entrypoint.
    return _gen_kernel(ctx, plan, last_k, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def _gen_active_kernel(ctx, key, last_k, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    # Emit one explicit tile variant so fullness selection stays in L1/L2 instead of re-branching inside the kernel body.
    safe_plan = _build_small_plan_variant(key, last_k)
    safe_code = _gen_plan_code(ctx, key, last_k, safe_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)

    if variant == "safe" or last_k or not (ctx.is_ext_precision() and ctx.use_ext_paired_fast_path()):
        return safe_code

    m_vl, n_vl = _KEY_TILE_VL[key]
    a_contig = _side_is_contiguous(ctx, "a")
    b_contig = _side_is_contiguous(ctx, "b")

    if key == "1VL_1VL" or (not a_contig and not b_contig):
        return safe_code

    if key == "1VL_2VL":
        if variant != "full" or not b_contig:
            return safe_code
        b_pairs = _full_chunk_pairs(n_vl)
        full_plan = _build_small_plan_variant(key, last_k, b_pairs=b_pairs, b_pair_load_role="n_main")
        return _gen_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)

    if key == "2VL_1VL":
        if variant != "full" or not a_contig:
            return safe_code
        a_pairs = _full_chunk_pairs(m_vl)
        full_plan = _build_small_plan_variant(key, last_k, a_pairs=a_pairs, a_pair_load_role="m_main")
        return _gen_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)

    if key == "1VL_3VL":
        if variant != "hybrid" or not b_contig:
            return safe_code
        b_pairs = _leading_chunk_pairs(n_vl)
        hybrid_plan = _build_small_plan_variant(key, last_k, b_pairs=b_pairs, b_pair_load_role="n_main")
        return _gen_plan_code(ctx, key, last_k, hybrid_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)

    if key == "3VL_1VL":
        if variant != "hybrid" or not a_contig:
            return safe_code
        a_pairs = _leading_chunk_pairs(m_vl)
        hybrid_plan = _build_small_plan_variant(key, last_k, a_pairs=a_pairs, a_pair_load_role="m_main")
        return _gen_plan_code(ctx, key, last_k, hybrid_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)

    if key == "1VL_4VL":
        if not b_contig:
            return safe_code
        if variant == "lead":
            lead_plan = _build_small_plan_variant(key, last_k, b_pairs=((0, 1),), b_pair_load_role="n_main")
            return _gen_plan_code(ctx, key, last_k, lead_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
        if variant == "full":
            full_plan = _build_small_plan_variant(
                key,
                last_k,
                b_pairs=((0, 1), (2, 3)),
                b_pair_load_role="n_main",
                promote_b_lanes=(3,),
            )
            return _gen_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
        return safe_code

    if key == "4VL_1VL":
        if not a_contig:
            return safe_code
        if variant == "lead":
            lead_plan = _build_small_plan_variant(key, last_k, a_pairs=((0, 1),), a_pair_load_role="m_main")
            return _gen_plan_code(ctx, key, last_k, lead_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
        if variant == "full":
            full_plan = _build_small_plan_variant(
                key,
                last_k,
                a_pairs=((0, 1), (2, 3)),
                a_pair_load_role="m_main",
                promote_a_lanes=(3,),
            )
            return _gen_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
        return safe_code

    if key == "2VL_2VL":
        if variant == "full":
            a_pairs = ((0, 1),) if a_contig else ()
            b_pairs = ((0, 1),) if b_contig else ()
            if not a_pairs and not b_pairs:
                return safe_code
            full_plan = _build_small_plan_variant(
                key,
                last_k,
                a_pairs=a_pairs,
                b_pairs=b_pairs,
                a_pair_load_role="m_main" if a_pairs else None,
                b_pair_load_role="n_main" if b_pairs else None,
                promote_a_lanes=(1,) if a_pairs else (),
                promote_b_lanes=(1,) if b_pairs else (),
            )
            return _gen_plan_code(ctx, key, last_k, full_plan, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)
        return safe_code

    return safe_code


def kernel_4VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "4VL_1VL", False, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_4VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "1VL_4VL", False, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_3VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="hybrid"):
    return _gen_active_kernel(ctx, "3VL_1VL", False, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_3VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="hybrid"):
    return _gen_active_kernel(ctx, "1VL_3VL", False, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_2VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "2VL_2VL", False, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_2VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "1VL_2VL", False, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "2VL_1VL", False, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "1VL_1VL", False, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_4VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "4VL_1VL", True, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_4VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "1VL_4VL", True, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_3VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "3VL_1VL", True, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_3VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "1VL_3VL", True, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_2VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "2VL_2VL", True, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_2VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "1VL_2VL", True, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "2VL_1VL", True, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None, variant="safe"):
    return _gen_active_kernel(ctx, "1VL_1VL", True, variant, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def save_zacol(pc, off, za, base_idx, idx, pg, rab0, rc0):
    # Extract one ZA column slice, accumulate it with C, and stream the updated vector back to memory.
    code_str = f""
    code_str += f"mova         {rab0}.s, {pg}/m, {za}v.s[{base_idx}, {idx}]\n"
    code_str += f"{LDNT1}      {rc0}.s, {pg}/z, [{pc}, {off}, MUL VL]\n"
    code_str += f"fadd         {rc0}.s, {pg}/m, {rc0}.s, {rab0}.s\n"
    code_str += f"{STNT1}      {rc0}.s, {pg}, [{pc}, {off}, MUL VL]\n"
    return code_str
