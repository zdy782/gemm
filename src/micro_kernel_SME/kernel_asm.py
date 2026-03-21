import gemm_config
from gemm_type_impl import *
from global_config import *


def _resolve_load_inst(load_inst):
    return get_ld1() if load_inst is None else load_inst


def _vector_operand(reg):
    return f"{reg}{get_element_suffix()}"


def _load_predicate(pg):
    return pg if is_ext_precision() else f"{pg}/z"


def _mopa_predicates(ext_pair, std_pair):
    return ext_pair if is_ext_precision() else std_pair


def _pointer_update(ptr_name):
    base_ptr = TMP_PTR if ptr_name == "A" else TMP_PTR1
    live_ptr = pA0 if ptr_name == "A" else pB0
    offset = OFFSET_A if ptr_name == "A" else OFFSET_B
    update_base = base_ptr if is_ext_precision() else live_ptr
    return f"add          {live_ptr}, {update_base}, {offset}, LSL #{get_element_size_shift()}\n"


def _emit_load(op, regs_a, regs_b, load_inst, lda_inst, last_k):
    model = gemm_config.currect_model
    if op["kind"] == "load_ab":
        load_fn = model.load_a0b0_last_k if last_k else model.load_a0b0
        return load_fn(
            regs_a[op["a"]],
            _load_predicate(op["a_pred"]),
            regs_b[op["b"]],
            _load_predicate(op["b_pred"]),
            load_inst,
            lda_inst,
        )

    reg_idx = op["idx"]
    predicate = _load_predicate(op["pred"])
    suffix = "_last_k" if last_k else ""
    load_fn = getattr(model, f"load_{op['kind']}{reg_idx}{suffix}")
    regs = regs_a if op["kind"] == "a" else regs_b
    return load_fn(regs[reg_idx], predicate, load_inst, lda_inst)


def _emit_mopa(op, regs_a, regs_b):
    lhs_pred, rhs_pred = _mopa_predicates(op["ext_pred"], op["std_pred"])
    return (
        f"{get_mopa_inst()}        za{op['za']}.s, "
        f"{lhs_pred}/m, {rhs_pred}/m, "
        f"{_vector_operand(regs_a[op['a']])}, {_vector_operand(regs_b[op['b']])}\n"
    )


def _emit_kernel(plan, last_k, a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    load_inst = _resolve_load_inst(ldopt)
    lda_inst = _resolve_load_inst(ldaopt)
    regs_a = [a0, a1, a2, a3]
    regs_b = [b0, b1, b2, b3]
    code_parts = []

    for op in plan:
        if op["kind"] in ("load_ab", "a", "b"):
            code_parts.append(_emit_load(op, regs_a, regs_b, load_inst, lda_inst, last_k))
        elif op["kind"] == "mopa":
            code_parts.append(_emit_mopa(op, regs_a, regs_b))
        elif op["kind"] == "update":
            code_parts.append(_pointer_update(op["ptr"]))

    return "".join(code_parts)


KERNEL_PLANS = {
    "4VL_1VL": [
        {"kind": "load_ab", "a": 0, "a_pred": "p1", "b": 0, "b_pred": "p0"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "a", "idx": 1, "pred": "p1"},
        {"kind": "mopa", "za": 1, "a": 1, "b": 0, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "a", "idx": 2, "pred": "p1"},
        {"kind": "mopa", "za": 2, "a": 2, "b": 0, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "update", "ptr": "B"},
        {"kind": "a", "idx": 3, "pred": "p2"},
        {"kind": "mopa", "za": 3, "a": 3, "b": 0, "ext_pred": ("p5", "p6"), "std_pred": ("p2", "p0")},
        {"kind": "update", "ptr": "A"},
    ],
    "1VL_4VL": [
        {"kind": "load_ab", "a": 0, "a_pred": "p1", "b": 0, "b_pred": "p0"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "b", "idx": 1, "pred": "p0"},
        {"kind": "mopa", "za": 1, "a": 0, "b": 1, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "b", "idx": 2, "pred": "p0"},
        {"kind": "update", "ptr": "A"},
        {"kind": "mopa", "za": 2, "a": 0, "b": 2, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "b", "idx": 3, "pred": "p3"},
        {"kind": "mopa", "za": 3, "a": 0, "b": 3, "ext_pred": ("p4", "p3"), "std_pred": ("p1", "p3")},
        {"kind": "update", "ptr": "B"},
    ],
    "3VL_1VL": [
        {"kind": "load_ab", "a": 0, "a_pred": "p1", "b": 0, "b_pred": "p0"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "a", "idx": 1, "pred": "p1"},
        {"kind": "mopa", "za": 1, "a": 1, "b": 0, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "update", "ptr": "B"},
        {"kind": "a", "idx": 2, "pred": "p2"},
        {"kind": "mopa", "za": 2, "a": 2, "b": 0, "ext_pred": ("p5", "p6"), "std_pred": ("p2", "p0")},
        {"kind": "update", "ptr": "A"},
    ],
    "1VL_3VL": [
        {"kind": "load_ab", "a": 0, "a_pred": "p1", "b": 0, "b_pred": "p0"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "b", "idx": 1, "pred": "p0"},
        {"kind": "mopa", "za": 1, "a": 0, "b": 1, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "update", "ptr": "A"},
        {"kind": "b", "idx": 2, "pred": "p3"},
        {"kind": "mopa", "za": 2, "a": 0, "b": 2, "ext_pred": ("p4", "p3"), "std_pred": ("p1", "p3")},
        {"kind": "update", "ptr": "B"},
    ],
    "2VL_2VL": [
        {"kind": "load_ab", "a": 0, "a_pred": "p1", "b": 0, "b_pred": "p0"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "b", "idx": 1, "pred": "p3"},
        {"kind": "mopa", "za": 1, "a": 0, "b": 1, "ext_pred": ("p4", "p3"), "std_pred": ("p1", "p3")},
        {"kind": "a", "idx": 1, "pred": "p2"},
        {"kind": "mopa", "za": 2, "a": 1, "b": 0, "ext_pred": ("p5", "p6"), "std_pred": ("p2", "p0")},
        {"kind": "update", "ptr": "A"},
        {"kind": "mopa", "za": 3, "a": 1, "b": 1, "ext_pred": ("p5", "p3"), "std_pred": ("p2", "p3")},
        {"kind": "update", "ptr": "B"},
    ],
    "1VL_2VL": [
        {"kind": "load_ab", "a": 0, "a_pred": "p1", "b": 0, "b_pred": "p0"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "b", "idx": 1, "pred": "p3"},
        {"kind": "mopa", "za": 1, "a": 0, "b": 1, "ext_pred": ("p4", "p3"), "std_pred": ("p1", "p3")},
        {"kind": "update", "ptr": "A"},
        {"kind": "update", "ptr": "B"},
    ],
    "2VL_1VL": [
        {"kind": "load_ab", "a": 0, "a_pred": "p1", "b": 0, "b_pred": "p0"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
        {"kind": "a", "idx": 1, "pred": "p2"},
        {"kind": "mopa", "za": 2, "a": 1, "b": 0, "ext_pred": ("p5", "p6"), "std_pred": ("p2", "p0")},
        {"kind": "update", "ptr": "A"},
        {"kind": "update", "ptr": "B"},
    ],
    "1VL_1VL": [
        {"kind": "load_ab", "a": 0, "a_pred": "p1", "b": 0, "b_pred": "p0"},
        {"kind": "mopa", "za": 0, "a": 0, "b": 0, "ext_pred": ("p4", "p6"), "std_pred": ("p1", "p0")},
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


def kernel_4VL_1VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_PLANS["4VL_1VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_4VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_PLANS["1VL_4VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_3VL_1VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_PLANS["3VL_1VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_3VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_PLANS["1VL_3VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_2VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_PLANS["2VL_2VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_2VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_PLANS["1VL_2VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_1VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_PLANS["2VL_1VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_1VL(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_PLANS["1VL_1VL"], False, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_4VL_1VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_LAST_K_PLANS["4VL_1VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_4VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_LAST_K_PLANS["1VL_4VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_3VL_1VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_LAST_K_PLANS["3VL_1VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_3VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_LAST_K_PLANS["1VL_3VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_2VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_LAST_K_PLANS["2VL_2VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_2VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_LAST_K_PLANS["1VL_2VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_2VL_1VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_LAST_K_PLANS["2VL_1VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def kernel_1VL_1VL_last_k(a0, a1, a2, a3, b0, b1, b2, b3, ldopt=None, ldaopt=None):
    return _emit_kernel(KERNEL_LAST_K_PLANS["1VL_1VL"], True, a0, a1, a2, a3, b0, b1, b2, b3, ldopt, ldaopt)


def save_zacol(pc, off, za, base_idx, idx, pg, rab0, rc0):
    code_str = f""
    code_str += f"mova         {rab0}.s, {pg}/m, {za}v.s[{base_idx}, {idx}]\n"
    code_str += f"{LDNT1}      {rc0}.s, {pg}/z, [{pc}, {off}, MUL VL]\n"
    code_str += f"fadd         {rc0}.s, {pg}/m, {rc0}.s, {rab0}.s\n"
    code_str += f"{STNT1}      {rc0}.s, {pg}, [{pc}, {off}, MUL VL]\n"

    return code_str
