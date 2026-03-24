from model_spec import GemmType

from global_config import LD1, LD1_H, LDNT1, LDNT1_H
from kernel_asm import (
    kernel_1VL_1VL,
    kernel_1VL_1VL_last_k,
    kernel_1VL_2VL,
    kernel_1VL_2VL_last_k,
    kernel_1VL_3VL,
    kernel_1VL_3VL_last_k,
    kernel_1VL_4VL,
    kernel_1VL_4VL_last_k,
    kernel_2VL_1VL,
    kernel_2VL_1VL_last_k,
    kernel_2VL_2VL,
    kernel_2VL_2VL_last_k,
    kernel_3VL_1VL,
    kernel_3VL_1VL_last_k,
    kernel_4VL_1VL,
    kernel_4VL_1VL_last_k,
)

# This layer is the bridge between loop code and the concrete kernel emitter.
# It maps a logical tile shape such as `2VL x 2VL` to the right assembly helper
# and also defines the repeated eight-variant K body used by `kernel_bc`.


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


def _get_kernel_fn(mvl, nvl, last_k=False):
    kernel_map = KERNEL_FUN_MAP_LAST_K if last_k else KERNEL_FUN_MAP
    return kernel_map[(mvl, nvl)]


def _gen_kernel_variant(ctx, variant_idx, mvl, nvl, last_k=False, load_inst=None):
    kernel_fn = _get_kernel_fn(mvl, nvl, last_k=last_k)
    variant = ctx.registers.kernel_variant(variant_idx)
    if load_inst is None:
        return kernel_fn(ctx, *variant.a_regs, *variant.b_regs)
    return kernel_fn(ctx, *variant.a_regs, *variant.b_regs, load_inst)


def _kernel_load_inst(ctx):
    if ctx.is_ext_precision():
        return LD1_H if ctx.spec.gemm_type is GemmType.SMALL else LDNT1_H
    return LD1 if ctx.spec.gemm_type is GemmType.SMALL else LDNT1


def _gen_kernel_bc(ctx, mvl, nvl, last_k=False, load_inst=None):
    code_parts = []
    # Each K block reuses the same four register variants twice. This matches
    # the original handwritten schedule and keeps the generated body regular.
    for variant_idx in (0, 1, 2, 3, 0, 1, 2, 3):
        code_parts.append(_gen_kernel_variant(ctx, variant_idx, mvl, nvl, last_k=last_k, load_inst=load_inst))
    return "".join(code_parts)


def kernel_m0(ctx, mvl, nvl):
    return _gen_kernel_variant(ctx, 0, mvl, nvl)


def kernel_m1(ctx, mvl, nvl):
    return _gen_kernel_variant(ctx, 1, mvl, nvl)


def kernel_m2(ctx, mvl, nvl):
    return _gen_kernel_variant(ctx, 2, mvl, nvl)


def kernel_m3(ctx, mvl, nvl):
    return _gen_kernel_variant(ctx, 3, mvl, nvl)


def kernel_m0_last_k(ctx, mvl, nvl):
    return _gen_kernel_variant(ctx, 0, mvl, nvl, last_k=True)


def kernel_m1_last_k(ctx, mvl, nvl):
    return _gen_kernel_variant(ctx, 1, mvl, nvl, last_k=True)


def kernel_m2_last_k(ctx, mvl, nvl):
    return _gen_kernel_variant(ctx, 2, mvl, nvl, last_k=True)


def kernel_m3_last_k(ctx, mvl, nvl):
    return _gen_kernel_variant(ctx, 3, mvl, nvl, last_k=True)


def kernel_ldnt1d_m0(ctx, mvl, nvl):
    return _gen_kernel_variant(ctx, 0, mvl, nvl, load_inst=_kernel_load_inst(ctx))


def kernel_ldnt1d_m1(ctx, mvl, nvl):
    return _gen_kernel_variant(ctx, 1, mvl, nvl, load_inst=_kernel_load_inst(ctx))


def kernel_ldnt1d_m2(ctx, mvl, nvl):
    return _gen_kernel_variant(ctx, 2, mvl, nvl, load_inst=_kernel_load_inst(ctx))


def kernel_ldnt1d_m3(ctx, mvl, nvl):
    return _gen_kernel_variant(ctx, 3, mvl, nvl, load_inst=_kernel_load_inst(ctx))


def kernel_bc(ctx, mvl, nvl):
    return _gen_kernel_bc(ctx, mvl, nvl)


def kernel_ldntb_bc(ctx, mvl, nvl):
    return _gen_kernel_bc(ctx, mvl, nvl, load_inst=_kernel_load_inst(ctx))
