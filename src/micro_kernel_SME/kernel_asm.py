from global_config import LDNT1, STNT1, get_element_size_shift, get_element_suffix, get_ld1, get_mopa_inst


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
        return load_fn(
            ctx,
            regs_a[op["a"]],
            op["a_role"],
            regs_b[op["b"]],
            op["b_role"],
            load_inst,
            lda_inst,
        )

    reg_idx = op["idx"]
    suffix = "_last_k" if last_k else ""
    load_fn = getattr(model, f"load_{op['kind']}{reg_idx}{suffix}")
    regs = regs_a if op["kind"] == "a" else regs_b
    return load_fn(ctx, regs[reg_idx], op["role"], load_inst, lda_inst)


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


def kernel_4VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_PLANS["4VL_1VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_4VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_PLANS["1VL_4VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_3VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_PLANS["3VL_1VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_3VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_PLANS["1VL_3VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_2VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_PLANS["2VL_2VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_2VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_PLANS["1VL_2VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_PLANS["2VL_1VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_1VL(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_PLANS["1VL_1VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_4VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_LAST_K_PLANS["4VL_1VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_4VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_LAST_K_PLANS["1VL_4VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_3VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_LAST_K_PLANS["3VL_1VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_3VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_LAST_K_PLANS["1VL_3VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_2VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_LAST_K_PLANS["2VL_2VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_2VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_LAST_K_PLANS["1VL_2VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_LAST_K_PLANS["2VL_1VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_1VL_last_k(ctx, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(ctx, KERNEL_LAST_K_PLANS["1VL_1VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def save_zacol(pc, off, za, base_idx, idx, pg, rab0, rc0):
    code_str = f""
    code_str += f"mova         {rab0}.s, {pg}/m, {za}v.s[{base_idx}, {idx}]\n"
    code_str += f"{LDNT1}      {rc0}.s, {pg}/z, [{pc}, {off}, MUL VL]\n"
    code_str += f"fadd         {rc0}.s, {pg}/m, {rc0}.s, {rab0}.s\n"
    code_str += f"{STNT1}      {rc0}.s, {pg}, [{pc}, {off}, MUL VL]\n"
    return code_str
