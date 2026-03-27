from model_spec import GemmType

from global_config import LD1, LD1_H, LDNT1, LDNT1_H
from kernel_asm import (
    _side_is_contiguous,
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

# This layer bridges the loop nest and the tile generator by mapping each logical `mvl x nvl` shape to its concrete kernel helper.


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
def resolve_kernel_variant(ctx, mvl, nvl, m_fullness, n_fullness, last_k=False):
    # Collapse loop-selected fullness into one explicit kernel body variant so the tile generator no longer re-dispatches on MIN_M/MIN_N.
    if last_k or not (ctx.is_ext_precision() and ctx.use_ext_paired_fast_path()):
        return "safe"

    a_contig = _side_is_contiguous(ctx, "a")
    b_contig = _side_is_contiguous(ctx, "b")
    key = (mvl, nvl)

    if key == ("1VL", "1VL") or (not a_contig and not b_contig):
        return "safe"
    if key == ("1VL", "2VL"):
        return "full" if b_contig and n_fullness == "exact_2vl" else "safe"
    if key == ("2VL", "1VL"):
        return "full" if a_contig and m_fullness == "exact_2vl" else "safe"
    if key == ("1VL", "3VL"):
        return "hybrid" if b_contig else "safe"
    if key == ("3VL", "1VL"):
        return "hybrid" if a_contig else "safe"
    if key == ("1VL", "4VL"):
        if not b_contig:
            return "safe"
        if n_fullness == "exact_4vl":
            return "full"
        if n_fullness == "lead_2vl_only":
            return "lead"
        return "safe"
    if key == ("4VL", "1VL"):
        if not a_contig:
            return "safe"
        if m_fullness == "exact_4vl":
            return "full"
        if m_fullness == "lead_2vl_only":
            return "lead"
        return "safe"
    if key == ("2VL", "2VL"):
        if a_contig and b_contig:
            return "full" if m_fullness == "exact_2vl" and n_fullness == "exact_2vl" else "safe"
        if a_contig:
            return "full" if m_fullness == "exact_2vl" else "safe"
        if b_contig:
            return "full" if n_fullness == "exact_2vl" else "safe"
        return "safe"
    return "safe"


def _get_kernel_fn(mvl, nvl, last_k=False):
    # Select the regular or last-k kernel table without making the callers repeat the map choice.
    kernel_map = KERNEL_FUN_MAP_LAST_K if last_k else KERNEL_FUN_MAP
    return kernel_map[(mvl, nvl)]


def _gen_kernel_variant(ctx, variant_idx, mvl, nvl, last_k=False, load_inst=None, kernel_variant="safe"):
    # Materialize one register-variant copy of the kernel body, optionally forcing a specific load opcode.
    kernel_fn = _get_kernel_fn(mvl, nvl, last_k=last_k)
    variant = ctx.registers.kernel_variant(variant_idx)
    if load_inst is None:
        return kernel_fn(ctx, *variant.a_regs, *variant.b_regs, variant=kernel_variant)
    return kernel_fn(ctx, *variant.a_regs, *variant.b_regs, load_inst, variant=kernel_variant)


def _kernel_load_inst(ctx):
    # Pick the streaming or non-streaming load form that matches the active precision and small/general model.
    if ctx.is_ext_precision():
        return LD1_H if ctx.spec.gemm_type is GemmType.SMALL else LDNT1_H
    return LD1 if ctx.spec.gemm_type is GemmType.SMALL else LDNT1


def _gen_kernel_bc(ctx, mvl, nvl, last_k=False, load_inst=None, kernel_variant="safe"):
    code_parts = []
    # Reusing the four register variants twice matches the handwritten K-body cadence and keeps scheduling regular.
    for variant_idx in (0, 1, 2, 3, 0, 1, 2, 3):
        code_parts.append(
            _gen_kernel_variant(
                ctx,
                variant_idx,
                mvl,
                nvl,
                last_k=last_k,
                load_inst=load_inst,
                kernel_variant=kernel_variant,
            )
        )
    return "".join(code_parts)


def kernel_m0(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_variant(ctx, 0, mvl, nvl, kernel_variant=kernel_variant)


def kernel_m1(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_variant(ctx, 1, mvl, nvl, kernel_variant=kernel_variant)


def kernel_m2(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_variant(ctx, 2, mvl, nvl, kernel_variant=kernel_variant)


def kernel_m3(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_variant(ctx, 3, mvl, nvl, kernel_variant=kernel_variant)


def kernel_m0_last_k(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_variant(ctx, 0, mvl, nvl, last_k=True, kernel_variant=kernel_variant)


def kernel_m1_last_k(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_variant(ctx, 1, mvl, nvl, last_k=True, kernel_variant=kernel_variant)


def kernel_m2_last_k(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_variant(ctx, 2, mvl, nvl, last_k=True, kernel_variant=kernel_variant)


def kernel_m3_last_k(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_variant(ctx, 3, mvl, nvl, last_k=True, kernel_variant=kernel_variant)


def kernel_ldnt1d_m0(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_variant(ctx, 0, mvl, nvl, load_inst=_kernel_load_inst(ctx), kernel_variant=kernel_variant)


def kernel_ldnt1d_m1(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_variant(ctx, 1, mvl, nvl, load_inst=_kernel_load_inst(ctx), kernel_variant=kernel_variant)


def kernel_ldnt1d_m2(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_variant(ctx, 2, mvl, nvl, load_inst=_kernel_load_inst(ctx), kernel_variant=kernel_variant)


def kernel_ldnt1d_m3(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_variant(ctx, 3, mvl, nvl, load_inst=_kernel_load_inst(ctx), kernel_variant=kernel_variant)


def kernel_bc(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_bc(ctx, mvl, nvl, kernel_variant=kernel_variant)


def kernel_ldntb_bc(ctx, mvl, nvl, kernel_variant="safe"):
    return _gen_kernel_bc(ctx, mvl, nvl, load_inst=_kernel_load_inst(ctx), kernel_variant=kernel_variant)
