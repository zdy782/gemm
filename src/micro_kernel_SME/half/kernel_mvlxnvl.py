from dataclasses import dataclass

from .model_spec import GemmType

from .global_config import LD1_H, LDNT1_H
from .kernel_asm import (
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


@dataclass(frozen=True)
class SmallKernelPairPlan:
    paired_enabled: bool = False
    a_pairs: tuple[tuple[int, int], ...] = ()
    b_pairs: tuple[tuple[int, int], ...] = ()
    promote_a_lanes: tuple[int, ...] = ()
    promote_b_lanes: tuple[int, ...] = ()
    precompute_second_chunk_offset: bool = False


UNPAIRED_SMALL_KERNEL_PLAN = SmallKernelPairPlan()


def resolve_small_kernel_pair_plan(ctx, mvl, nvl, m_fullness, n_fullness, last_k=False):
    # Collapse loop-selected fullness into one explicit set of legal pair
    # collapses so the tile generator no longer re-dispatches on MIN_M/MIN_N.
    if last_k or not ctx.use_paired_half_loads():
        return UNPAIRED_SMALL_KERNEL_PLAN

    a_contig = _side_is_contiguous(ctx, "a")
    b_contig = _side_is_contiguous(ctx, "b")
    key = (mvl, nvl)

    if key == ("1VL", "1VL") or (not a_contig and not b_contig):
        return UNPAIRED_SMALL_KERNEL_PLAN
    if key == ("1VL", "2VL"):
        if b_contig and n_fullness == "exact_2vl":
            return SmallKernelPairPlan(paired_enabled=True, b_pairs=((0, 1),))
        return UNPAIRED_SMALL_KERNEL_PLAN
    if key == ("2VL", "1VL"):
        if a_contig and m_fullness == "exact_2vl":
            return SmallKernelPairPlan(paired_enabled=True, a_pairs=((0, 1),))
        return UNPAIRED_SMALL_KERNEL_PLAN
    if key == ("1VL", "3VL"):
        if b_contig:
            return SmallKernelPairPlan(paired_enabled=True, b_pairs=((0, 1),))
        return UNPAIRED_SMALL_KERNEL_PLAN
    if key == ("3VL", "1VL"):
        if a_contig:
            return SmallKernelPairPlan(paired_enabled=True, a_pairs=((0, 1),))
        return UNPAIRED_SMALL_KERNEL_PLAN
    if key == ("1VL", "4VL"):
        if not b_contig:
            return UNPAIRED_SMALL_KERNEL_PLAN
        if n_fullness == "exact_4vl":
            return SmallKernelPairPlan(
                paired_enabled=True,
                b_pairs=((0, 1), (2, 3)),
                promote_b_lanes=(3,),
                precompute_second_chunk_offset=True,
            )
        if n_fullness == "lead_2vl_only":
            return SmallKernelPairPlan(paired_enabled=True, b_pairs=((0, 1),))
        return UNPAIRED_SMALL_KERNEL_PLAN
    if key == ("4VL", "1VL"):
        if not a_contig:
            return UNPAIRED_SMALL_KERNEL_PLAN
        if m_fullness == "exact_4vl":
            return SmallKernelPairPlan(
                paired_enabled=True,
                a_pairs=((0, 1), (2, 3)),
                promote_a_lanes=(3,),
                precompute_second_chunk_offset=True,
            )
        if m_fullness == "lead_2vl_only":
            return SmallKernelPairPlan(paired_enabled=True, a_pairs=((0, 1),))
        return UNPAIRED_SMALL_KERNEL_PLAN
    if key == ("2VL", "2VL"):
        a_pairs = ((0, 1),) if a_contig and m_fullness == "exact_2vl" else ()
        b_pairs = ((0, 1),) if b_contig and n_fullness == "exact_2vl" else ()
        if a_pairs or b_pairs:
            return SmallKernelPairPlan(
                paired_enabled=True,
                a_pairs=a_pairs,
                b_pairs=b_pairs,
                promote_a_lanes=(1,) if a_pairs else (),
                promote_b_lanes=(1,) if b_pairs else (),
            )
        return UNPAIRED_SMALL_KERNEL_PLAN
    return UNPAIRED_SMALL_KERNEL_PLAN


def _get_kernel_fn(mvl, nvl, last_k=False):
    # Select the regular or last-k kernel table without making the callers repeat the map choice.
    kernel_map = KERNEL_FUN_MAP_LAST_K if last_k else KERNEL_FUN_MAP
    return kernel_map[(mvl, nvl)]


def _gen_kernel_variant(ctx, variant_idx, mvl, nvl, last_k=False, load_inst=None, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    # Materialize one register-variant copy of the kernel body, optionally forcing a specific load opcode.
    kernel_fn = _get_kernel_fn(mvl, nvl, last_k=last_k)
    variant = ctx.registers.kernel_variant(variant_idx)
    return kernel_fn(ctx, *variant.a_regs, *variant.b_regs, ldopt=load_inst, pair_plan=pair_plan)


def _kernel_load_inst(ctx):
    # Pick the streaming or non-streaming half-input load form that matches
    # the active small/general model.
    return LD1_H if ctx.spec.gemm_type is GemmType.SMALL else LDNT1_H


def _gen_prefetch_hints(ctx):
    regs = ctx.registers
    code_str = f"prfm    PLDL1KEEP, [{regs.pointers.pA0}, #256]\n"
    code_str += f"prfm    PLDL1KEEP, [{regs.pointers.pB0}, #256]\n"
    return code_str


def _gen_kernel_bc(ctx, mvl, nvl, last_k=False, load_inst=None, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    code_parts = []
    code_parts.append(_gen_prefetch_hints(ctx))
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
                pair_plan=pair_plan,
            )
        )
    return "".join(code_parts)


def kernel_m0(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_variant(ctx, 0, mvl, nvl, pair_plan=pair_plan)


def kernel_m1(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_variant(ctx, 1, mvl, nvl, pair_plan=pair_plan)


def kernel_m2(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_variant(ctx, 2, mvl, nvl, pair_plan=pair_plan)


def kernel_m3(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_variant(ctx, 3, mvl, nvl, pair_plan=pair_plan)


def kernel_m0_last_k(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_variant(ctx, 0, mvl, nvl, last_k=True, pair_plan=pair_plan)


def kernel_m1_last_k(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_variant(ctx, 1, mvl, nvl, last_k=True, pair_plan=pair_plan)


def kernel_m2_last_k(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_variant(ctx, 2, mvl, nvl, last_k=True, pair_plan=pair_plan)


def kernel_m3_last_k(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_variant(ctx, 3, mvl, nvl, last_k=True, pair_plan=pair_plan)


def kernel_ldnt1d_m0(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_variant(ctx, 0, mvl, nvl, load_inst=_kernel_load_inst(ctx), pair_plan=pair_plan)


def kernel_ldnt1d_m1(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_variant(ctx, 1, mvl, nvl, load_inst=_kernel_load_inst(ctx), pair_plan=pair_plan)


def kernel_ldnt1d_m2(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_variant(ctx, 2, mvl, nvl, load_inst=_kernel_load_inst(ctx), pair_plan=pair_plan)


def kernel_ldnt1d_m3(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_variant(ctx, 3, mvl, nvl, load_inst=_kernel_load_inst(ctx), pair_plan=pair_plan)


def kernel_bc(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_bc(ctx, mvl, nvl, pair_plan=pair_plan)


def kernel_ldntb_bc(ctx, mvl, nvl, pair_plan=UNPAIRED_SMALL_KERNEL_PLAN):
    return _gen_kernel_bc(ctx, mvl, nvl, load_inst=_kernel_load_inst(ctx), pair_plan=pair_plan)
