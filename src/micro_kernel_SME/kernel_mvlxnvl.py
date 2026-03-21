import gemm_config
from global_config import *
from kernel_asm import *


KERNEL_FUN_MAP = {
    ("4VL", "1VL"): kernel_4VL_1VL,
    ("1VL", "4VL"): kernel_1VL_4VL,
    ("3VL", "1VL"): kernel_3VL_1VL,
    ("1VL", "3VL"): kernel_1VL_3VL,
    ("2VL", "2VL"): kernel_2VL_2VL,
    ("2VL", "1VL"): kernel_2VL_1VL,
    ("1VL", "2VL"): kernel_1VL_2VL,
    ("1VL", "1VL"): kernel_1VL_1VL,
}

KERNEL_FUN_MAP_LAST_K = {
    ("4VL", "1VL"): kernel_4VL_1VL_last_k,
    ("1VL", "4VL"): kernel_1VL_4VL_last_k,
    ("3VL", "1VL"): kernel_3VL_1VL_last_k,
    ("1VL", "3VL"): kernel_1VL_3VL_last_k,
    ("2VL", "2VL"): kernel_2VL_2VL_last_k,
    ("2VL", "1VL"): kernel_2VL_1VL_last_k,
    ("1VL", "2VL"): kernel_1VL_2VL_last_k,
    ("1VL", "1VL"): kernel_1VL_1VL_last_k,
}

M_KERNEL_REGS = [
    ("z0", "z1", "z2", "z3", "z16", "z17", "z24", "z25"),
    ("z4", "z5", "z6", "z7", "z18", "z19", "z30", "z31"),
    ("z8", "z9", "z10", "z11", "z20", "z21", "z24", "z25"),
    ("z12", "z13", "z14", "z15", "z22", "z23", "z30", "z31"),
]


def _get_kernel_fn(mvl, nvl, last_k=False):
    kernel_map = KERNEL_FUN_MAP_LAST_K if last_k else KERNEL_FUN_MAP
    return kernel_map[(mvl, nvl)]


def _emit_kernel_variant(variant_idx, mvl, nvl, last_k=False, load_inst=None):
    kernel_fn = _get_kernel_fn(mvl, nvl, last_k=last_k)
    regs = M_KERNEL_REGS[variant_idx]
    if load_inst is None:
        return kernel_fn(*regs)
    return kernel_fn(*regs, load_inst)


def _kernel_load_inst():
    if is_ext_precision():
        return LD1_H if gemm_config.type == "small" else LDNT1_H
    return LD1 if gemm_config.type == "small" else LDNT1


def _emit_kernel_bc(mvl, nvl, last_k=False, load_inst=None):
    code_parts = []
    for variant_idx in (0, 1, 2, 3, 0, 1, 2, 3):
        code_parts.append(_emit_kernel_variant(variant_idx, mvl, nvl, last_k=last_k, load_inst=load_inst))
    return "".join(code_parts)


def kernel_m0(mvl, nvl):
    return _emit_kernel_variant(0, mvl, nvl)


def kernel_m1(mvl, nvl):
    return _emit_kernel_variant(1, mvl, nvl)


def kernel_m2(mvl, nvl):
    return _emit_kernel_variant(2, mvl, nvl)


def kernel_m3(mvl, nvl):
    return _emit_kernel_variant(3, mvl, nvl)


def kernel_m0_last_k(mvl, nvl):
    return _emit_kernel_variant(0, mvl, nvl, last_k=True)


def kernel_m1_last_k(mvl, nvl):
    return _emit_kernel_variant(1, mvl, nvl, last_k=True)


def kernel_m2_last_k(mvl, nvl):
    return _emit_kernel_variant(2, mvl, nvl, last_k=True)


def kernel_m3_last_k(mvl, nvl):
    return _emit_kernel_variant(3, mvl, nvl, last_k=True)


def kernel_ldnt1d_m0(mvl, nvl):
    return _emit_kernel_variant(0, mvl, nvl, load_inst=_kernel_load_inst())


def kernel_ldnt1d_m1(mvl, nvl):
    return _emit_kernel_variant(1, mvl, nvl, load_inst=_kernel_load_inst())


def kernel_ldnt1d_m2(mvl, nvl):
    return _emit_kernel_variant(2, mvl, nvl, load_inst=_kernel_load_inst())


def kernel_ldnt1d_m3(mvl, nvl):
    return _emit_kernel_variant(3, mvl, nvl, load_inst=_kernel_load_inst())


def kernel_bc(mvl, nvl):
    return _emit_kernel_bc(mvl, nvl)


def kernel_ldntb_bc(mvl, nvl):
    return _emit_kernel_bc(mvl, nvl, load_inst=_kernel_load_inst())
