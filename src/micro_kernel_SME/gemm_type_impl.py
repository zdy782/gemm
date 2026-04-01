from dataclasses import dataclass

from global_config import get_element_size_shift, get_element_suffix

# This file turns logical A/B lanes into concrete load-and-shape assembly for the chosen small/general model.


LOAD_CONTIGUOUS = "contiguous"
LOAD_GATHER = "gather"


def _pred(ctx, role):
    # Return the logical predicate that matches the requested main/tail role.
    return ctx.registers.logical_predicate(role)


def _ext_pred(ctx, role):
    # Return the widened ext predicate used by bf16/fp16 `.h` loads.
    return ctx.registers.ext_predicate(role)


def _load_predicate(ctx, role):
    # Non-ext loads keep the `/z` suffix inline while ext loads use a separate zipped predicate register.
    pred = _pred(ctx, role)
    return pred if ctx.is_ext_precision() else f"{pred}/z"


def _paired_ext_load_predicate(ctx, first_role, second_role=None, load_second_role=None):
    # Full paired paths reuse the prebuilt widened predicates. Only mixed pairs need a dedicated temporary load predicate.
    second_role = first_role if second_role is None else second_role
    load_second_role = second_role if load_second_role is None else load_second_role
    if first_role == load_second_role:
        return "", _ext_pred(ctx, first_role)
    target = ctx.registers.predicates.load_pair_tmp
    code_str = f"zip1      {target}.h, {_pred(ctx, first_role)}.h, {_pred(ctx, load_second_role)}.h\n"
    return code_str, target


def _rdvl_lane_offset(registers, lane):
    # Convert a logical lane index into the byte offset pattern used by ext contiguous loads.
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
    # Reuse one add helper for gather lanes so lane-1/2/3 addressing stays consistent.
    if lane == 1:
        return f"add          {dst}, {base}, {offset}\n"
    if lane == 2:
        return f"add          {dst}, {base}, {offset}, lsl #1\n"
    code_str = f"add          {dst}, {base}, {offset}\n"
    code_str += f"add          {dst}, {dst}, {offset}, lsl #1\n"
    return code_str


def _gen_zip_pair(dst0, dst1, low_tmp, high_tmp):
    # Emit `zip2` first when dst0 aliases an input so the later `zip1` still sees the original source.
    if dst1 is None:
        return f"zip1      {dst0}.h, {low_tmp}.h, {high_tmp}.h\n"
    if dst0 in {low_tmp, high_tmp}:
        code_str = f"zip2      {dst1}.h, {low_tmp}.h, {high_tmp}.h\n"
        code_str += f"zip1      {dst0}.h, {low_tmp}.h, {high_tmp}.h\n"
        return code_str
    code_str = f"zip1      {dst0}.h, {low_tmp}.h, {high_tmp}.h\n"
    code_str += f"zip2      {dst1}.h, {low_tmp}.h, {high_tmp}.h\n"
    return code_str


def _gen_ext_contiguous_head(ctx, base, stride, tmp_base, role, dst, low_tmp, high_tmp):
    # Materialize the first contiguous ext lane as one logical operand with a plain `zip1`, reusing the prebuilt widened predicates.
    pred = _pred(ctx, role)
    code_str = f""
    code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}]\n"
    code_str += f"add       {tmp_base}, {base}, {stride}, LSL #1\n"
    code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{tmp_base}]\n"
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
    return code_str


def _gen_ext_contiguous_head_pair_fast(
    ctx, base, stride, tmp_base, role0, role1, dst0, dst1, low_tmp, high_tmp, load_role1=None
):
    # Materialize the first contiguous `2VL` chunk as two operands with one load pair plus `zip1+zip2`.
    code_str = f""
    pred_pre, pred = _paired_ext_load_predicate(ctx, role0, role1, load_second_role=load_role1)
    code_str += pred_pre
    code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}]\n"
    code_str += f"add       {tmp_base}, {base}, {stride}, LSL #1\n"
    code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{tmp_base}]\n"
    code_str += _gen_zip_pair(dst0, dst1, low_tmp, high_tmp)
    return code_str


def _gen_ext_contiguous_lane_fast_pair(
    ctx, base, paired_base, role, dst, lane, low_tmp, high_tmp, next_dst=None, next_role=None, next_load_role=None
):
    # Load one later contiguous `2VL` chunk and shape it with `zip1+zip2`, reusing the hoisted lane-2 offset for the full hot path.
    code_str = f""
    pred_pre, pred = _paired_ext_load_predicate(
        ctx,
        role,
        next_role if next_dst is not None else None,
        load_second_role=next_load_role if next_dst is not None else None,
    )
    code_str += pred_pre
    if lane != 2:
        code_str += _rdvl_lane_offset(ctx.registers, lane)
    code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
    code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{paired_base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
    code_str += _gen_zip_pair(dst, next_dst, low_tmp, high_tmp)
    return code_str


def _gen_ext_contiguous_lane(
    ctx, base, paired_base, role, dst, lane, low_tmp, high_tmp, next_dst=None, next_role=None, next_load_role=None
):
    # This is the main lane materializer for ext contiguous paths after the first head load.
    if next_dst is None:
        pred = _pred(ctx, role)
        code_str = f""
        code_str += _rdvl_lane_offset(ctx.registers, lane)
        code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
        code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{paired_base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
        code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
        return code_str
    return _gen_ext_contiguous_lane_fast_pair(
        ctx,
        base,
        paired_base,
        role,
        dst,
        lane,
        low_tmp,
        high_tmp,
        next_dst=next_dst,
        next_role=next_role,
        next_load_role=next_load_role,
    )


def _gen_ext_contiguous_last_k_fast_pair(ctx, base, role, dst, lane, low_tmp, zero_tmp, next_dst=None, next_role=None):
    # The last-k fast path still pairs only one real load because the missing half is synthesized from zero.
    code_str = f""
    pred_pre, pred = _paired_ext_load_predicate(ctx, role, next_role if next_dst is not None else None)
    code_str += pred_pre
    if lane > 0:
        code_str += _rdvl_lane_offset(ctx.registers, lane)
        code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
    else:
        code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}]\n"
    code_str += f"mov       {zero_tmp}.h, #0\n"
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {zero_tmp}.h\n"
    if next_dst is not None:
        code_str += f"zip2      {next_dst}.h, {low_tmp}.h, {zero_tmp}.h\n"
    return code_str


def _gen_ext_contiguous_last_k_pair_safe(ctx, base, role0, role1, dst0, dst1, lane, low_tmp, zero_tmp):
    # Last-k mixed pairs are generated as two safe scalar loads to avoid partial-lane pairing mistakes.
    code_str = _gen_ext_contiguous_last_k(ctx, base, role0, dst0, lane, low_tmp, zero_tmp)
    code_str += _gen_ext_contiguous_last_k(ctx, base, role1, dst1, lane + 1, low_tmp, zero_tmp)
    return code_str


def _gen_ext_contiguous_last_k(ctx, base, role, dst, lane, low_tmp, zero_tmp, next_dst=None, next_role=None):
    # Last-k loads mirror the contiguous path but always zip against zero instead of a real high half.
    if next_dst is None:
        pred = _pred(ctx, role)
        code_str = f""
        if lane > 0:
            code_str += _rdvl_lane_offset(ctx.registers, lane)
            code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
        else:
            code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}]\n"
        code_str += f"mov       {zero_tmp}.h, #0\n"
        code_str += f"zip1      {dst}.h, {low_tmp}.h, {zero_tmp}.h\n"
        return code_str
    if next_dst is not None and next_role != role:
        return _gen_ext_contiguous_last_k_pair_safe(
            ctx, base, role, next_role, dst, next_dst, lane, low_tmp, zero_tmp
        )
    return _gen_ext_contiguous_last_k_fast_pair(
        ctx, base, role, dst, lane, low_tmp, zero_tmp, next_dst=next_dst, next_role=next_role
    )


def _gen_non_ext_contiguous(load_inst, ctx, base, role, dst, lane):
    # FP32 contiguous loads are already naturally aligned, so they never need the ext zip shaping machinery.
    suffix = get_element_suffix(ctx)
    pred = _load_predicate(ctx, role)
    if lane == 0:
        return f"{load_inst}      {dst}{suffix}, {pred}, [{base}]\n"
    return f"{load_inst}      {dst}{suffix}, {pred}, [{base}, #{lane}, MUL VL]\n"


def _gen_ext_gather_pair(ctx, base, role, index_vec, dst, low_tmp, high_tmp, pair_base):
    # Small gather paths only materialize one logical operand at a time, so they never need a temporary paired-load predicate.
    code_str = f""
    code_str += f"ld1h      {low_tmp}.s, {_ext_pred(ctx, role)}/z, [{base}, {index_vec}.s, UXTW]\n"
    code_str += f"add       {pair_base}, {base}, #2\n"
    code_str += f"ld1h      {high_tmp}.s, {_ext_pred(ctx, role)}/z, [{pair_base}, {index_vec}.s, UXTW]\n"
    code_str += f"uzp1      {low_tmp}.h, {low_tmp}.h, {low_tmp}.h\n"
    code_str += f"uzp1      {high_tmp}.h, {high_tmp}.h, {high_tmp}.h\n"
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
    return code_str


def _gen_ext_gather_last_k(ctx, base, role, index_vec, dst, low_tmp, zero_tmp):
    # Small gather last-k stays single-output as well; the missing high half is synthesized from zero without any paired predicate.
    return f"ld1h      {dst}.s, {_ext_pred(ctx, role)}/z, [{base}, {index_vec}.s, UXTW]\n"


def _gen_non_ext_gather(load_inst, ctx, base, role, index_vec, dst):
    # FP32 gather paths stay as direct vector gathers with no extra shaping.
    suffix = get_element_suffix(ctx)
    pred = _load_predicate(ctx, role)
    return f"{load_inst}      {dst}{suffix}, {pred}, [{base}, {index_vec}.s, UXTW]\n"


def _gen_a_head(ctx, config, a0, role, ldaopt, next_dst=None, next_role=None, next_load_role=None):
    # Choose the A-side head loader from contiguous/gather and ext/non-ext mode for the active transpose model.
    regs = ctx.registers
    if config.a_mode == LOAD_CONTIGUOUS:
        if ctx.is_ext_precision():
            if ctx.use_ext_paired_fast_path() and next_dst is not None:
                return _gen_ext_contiguous_head_pair_fast(
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
                    load_role1=next_load_role,
                )
            return _gen_ext_contiguous_head(
                ctx,
                regs.pointers.pA0,
                regs.params.LDA,
                regs.address.TMP_PTR,
                role,
                a0,
                regs.vectors.a_low,
                regs.vectors.pair_high,
            )
        return _gen_non_ext_contiguous(ldaopt, ctx, regs.pointers.pA0, role, a0, 0)
    if ctx.is_ext_precision():
        # Gather-based A paths still shape data with zip/uzp, but they never opt into paired fast execution.
        return _gen_ext_gather_pair(
            ctx,
            regs.pointers.pA0,
            role,
            regs.vectors.a_index,
            a0,
            regs.vectors.a_low,
            regs.vectors.pair_high,
            regs.address.TMP_PTR,
        )
    return _gen_non_ext_gather(ldaopt, ctx, regs.pointers.pA0, role, regs.vectors.a_index, a0)


def _gen_b_head(ctx, config, b0, role, ldopt, next_dst=None, next_role=None, next_load_role=None):
    # Choose the B-side head loader with the same policy, but using B-specific temporaries and strides.
    regs = ctx.registers
    if config.b_mode == LOAD_CONTIGUOUS:
        if ctx.is_ext_precision():
            if ctx.use_ext_paired_fast_path() and next_dst is not None:
                return _gen_ext_contiguous_head_pair_fast(
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
                    load_role1=next_load_role,
                )
            return _gen_ext_contiguous_head(
                ctx,
                regs.pointers.pB0,
                regs.params.LDB,
                regs.address.TMP_PTR1,
                role,
                b0,
                config.b_low_tmp,
                config.b_high_tmp,
            )
        return _gen_non_ext_contiguous(ldopt, ctx, regs.pointers.pB0, role, b0, 0)
    if ctx.is_ext_precision():
        return _gen_ext_gather_pair(
            ctx,
            regs.pointers.pB0,
            role,
            regs.vectors.b_index,
            b0,
            config.b_low_tmp,
            config.b_high_tmp,
            regs.address.TMP_PTR1,
        )
    return _gen_non_ext_gather(ldopt, ctx, regs.pointers.pB0, role, regs.vectors.b_index, b0)


def _gen_gather_base_setup(target, registers, lane, is_a):
    # Later gather lanes need an adjusted base pointer because the index vector itself always starts from zero.
    if lane == 0:
        return ""
    base = registers.pointers.pA0 if is_a else registers.pointers.pB0
    offset = registers.address.pA_OFFSET if is_a else registers.address.pB_OFFSET
    return _scaled_add(target, base, offset, lane)


def _gather_base_register(registers, lane, is_a):
    # Pick either the original base or the pre-shifted gather base depending on which lane is being loaded.
    if lane == 0:
        return registers.pointers.pA0 if is_a else registers.pointers.pB0
    return registers.pointers.pAn if is_a else registers.pointers.pBn


def _gen_a_last_k(ctx, config, a_reg, role, lane, next_reg=None, next_role=None):
    # Last-k A loads reuse the normal path selection but replace the missing high half with zero.
    regs = ctx.registers
    if config.a_mode == LOAD_CONTIGUOUS:
        return _gen_ext_contiguous_last_k(
            ctx,
            regs.pointers.pA0,
            role,
            a_reg,
            lane,
            regs.vectors.a_low,
            regs.vectors.pair_high,
        )
    code_str = _gen_gather_base_setup(regs.pointers.pAn, regs, lane, True)
    code_str += _gen_ext_gather_last_k(
        ctx,
        _gather_base_register(regs, lane, True),
        role,
        regs.vectors.a_index,
        a_reg,
        regs.vectors.a_low,
        regs.vectors.pair_high,
    )
    return code_str


def _gen_b_last_k(ctx, config, b_reg, role, lane, next_reg=None, next_role=None):
    # Last-k B loads mirror A but route through the B-side temporaries and gather index.
    regs = ctx.registers
    if config.b_mode == LOAD_CONTIGUOUS:
        return _gen_ext_contiguous_last_k(
            ctx,
            regs.pointers.pB0,
            role,
            b_reg,
            lane,
            config.b_low_tmp,
            config.b_high_tmp,
        )
    code_str = _gen_gather_base_setup(regs.pointers.pBn, regs, lane, False)
    code_str += _gen_ext_gather_last_k(
        ctx,
        _gather_base_register(regs, lane, False),
        role,
        regs.vectors.b_index,
        b_reg,
        config.b_low_tmp,
        config.b_high_tmp,
    )
    return code_str


def _gen_gather_base(target, registers, lane, is_a):
    # Keep one small wrapper so the lane loaders can treat gather base setup as a single step.
    return _gen_gather_base_setup(target, registers, lane, is_a)


def _gen_a_lane(ctx, config, dst, role, lane, ldaopt, next_dst=None, next_role=None, next_load_role=None):
    # Load later A lanes after the head, preserving the same contiguous/gather and fast/safe policy.
    regs = ctx.registers
    if config.a_mode == LOAD_CONTIGUOUS:
        if ctx.is_ext_precision():
            return _gen_ext_contiguous_lane(
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
                next_load_role=next_load_role,
            )
        return _gen_non_ext_contiguous(ldaopt, ctx, regs.pointers.pA0, role, dst, lane)

    code_str = _gen_gather_base(regs.pointers.pAn, regs, lane, True)
    if ctx.is_ext_precision():
        code_str += _gen_ext_gather_pair(
            ctx,
            regs.pointers.pAn,
            role,
            regs.vectors.a_index,
            dst,
            regs.vectors.a_low,
            regs.vectors.pair_high,
            regs.address.TMP_PTR2,
        )
        return code_str
    code_str += _gen_non_ext_gather(ldaopt, ctx, regs.pointers.pAn, role, regs.vectors.a_index, dst)
    return code_str


def _gen_b_lane(ctx, config, dst, role, lane, ldopt, next_dst=None, next_role=None, next_load_role=None):
    # Load later B lanes after the head, again reusing the same policy decisions as the A path.
    regs = ctx.registers
    if config.b_mode == LOAD_CONTIGUOUS:
        if ctx.is_ext_precision():
            return _gen_ext_contiguous_lane(
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
                next_load_role=next_load_role,
            )
        return _gen_non_ext_contiguous(ldopt, ctx, regs.pointers.pB0, role, dst, lane)

    code_str = _gen_gather_base(regs.pointers.pBn, regs, lane, False)
    if ctx.is_ext_precision():
        code_str += _gen_ext_gather_pair(
            ctx,
            regs.pointers.pBn,
            role,
            regs.vectors.b_index,
            dst,
            config.b_low_tmp,
            config.b_high_tmp,
            regs.address.TMP_PTR2,
        )
        return code_str
    code_str += _gen_non_ext_gather(ldopt, ctx, regs.pointers.pBn, role, regs.vectors.b_index, dst)
    return code_str


def _gen_small_svindex(ctx, config):
    # Small gather models precompute one stride-scaled index vector per side before entering the tile loops.
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


def _gen_small_n_pre(ctx):
    # Reset the A/B walking pointers at the start of each outer N tile.
    regs = ctx.registers
    code_str = f""
    code_str += f"mov     {regs.pointers.pAt}, {regs.params.origPA}\n"
    code_str += f"mov     {regs.pointers.pB0}, {regs.pointers.pBt}\n"
    return code_str


def _gen_small_n_post(ctx, config):
    # Advance the persistent B pointer by the actual N chunk that just finished.
    regs = ctx.registers
    shift = get_element_size_shift(ctx)
    if config.b_mode == LOAD_GATHER:
        code_str = f""
        code_str += f"mul     {regs.counters.TMP_CNT}, {regs.params.LDB}, {regs.dims.MIN_N}\n"
        code_str += f"add     {regs.pointers.pBt}, {regs.pointers.pBt}, {regs.counters.TMP_CNT}, lsl #{shift}\n"
        return code_str
    return f"add     {regs.pointers.pBt}, {regs.pointers.pBt}, {regs.dims.MIN_N}, lsl #{shift}\n"


def _gen_small_m_pre(ctx, config):
    # Rebuild the live A/B pointers and scaled offsets for each inner M chunk.
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


def _gen_small_m_post(ctx, config):
    # Advance the persistent A pointer by the actual M chunk that just finished.
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

    # `SmallGemmModel` gives the plan layer one uniform API while hiding transpose-specific load mechanics.

    def load_a0b0(
        self,
        ctx,
        a0,
        a_role,
        b0,
        b_role,
        ldopt,
        ldaopt,
        a1=None,
        b1=None,
        a1_role=None,
        b1_role=None,
        a1_load_role=None,
        b1_load_role=None,
    ):
        code_str = _gen_a_head(
            ctx, self.config, a0, a_role, ldaopt, next_dst=a1, next_role=a1_role, next_load_role=a1_load_role
        )
        code_str += _gen_b_head(
            ctx, self.config, b0, b_role, ldopt, next_dst=b1, next_role=b1_role, next_load_role=b1_load_role
        )
        return code_str

    def load_a0b0_last_k(
        self,
        ctx,
        a0,
        a_role,
        b0,
        b_role,
        ldopt,
        ldaopt,
        a1=None,
        b1=None,
        a1_role=None,
        b1_role=None,
        a1_load_role=None,
        b1_load_role=None,
    ):
        code_str = _gen_a_last_k(ctx, self.config, a0, a_role, 0, next_reg=a1, next_role=a1_role)
        code_str += _gen_b_last_k(ctx, self.config, b0, b_role, 0, next_reg=b1, next_role=b1_role)
        return code_str

    def load_a1(self, ctx, a1, role, ldopt, ldaopt, a2=None, a2_role=None, a2_load_role=None):
        return _gen_a_lane(
            ctx, self.config, a1, role, 1, ldaopt, next_dst=a2, next_role=a2_role, next_load_role=a2_load_role
        )

    def load_a1_last_k(self, ctx, a1, role, ldopt, ldaopt, a2=None, a2_role=None, a2_load_role=None):
        return _gen_a_last_k(ctx, self.config, a1, role, 1, next_reg=a2, next_role=a2_role)

    def load_a2(self, ctx, a2, role, ldopt, ldaopt, a3=None, a3_role=None, a3_load_role=None):
        return _gen_a_lane(
            ctx, self.config, a2, role, 2, ldaopt, next_dst=a3, next_role=a3_role, next_load_role=a3_load_role
        )

    def load_a2_last_k(self, ctx, a2, role, ldopt, ldaopt, a3=None, a3_role=None, a3_load_role=None):
        return _gen_a_last_k(ctx, self.config, a2, role, 2, next_reg=a3, next_role=a3_role)

    def load_a3(self, ctx, a3, role, ldopt, ldaopt, a4=None):
        return _gen_a_lane(ctx, self.config, a3, role, 3, ldaopt)

    def load_a3_last_k(self, ctx, a3, role, ldopt, ldaopt, a4=None):
        return _gen_a_last_k(ctx, self.config, a3, role, 3)

    def load_b1(self, ctx, b1, role, ldopt, ldaopt, b2=None, b2_role=None, b2_load_role=None):
        return _gen_b_lane(
            ctx, self.config, b1, role, 1, ldopt, next_dst=b2, next_role=b2_role, next_load_role=b2_load_role
        )

    def load_b1_last_k(self, ctx, b1, role, ldopt, ldaopt, b2=None, b2_role=None, b2_load_role=None):
        return _gen_b_last_k(ctx, self.config, b1, role, 1, next_reg=b2, next_role=b2_role)

    def load_b2(self, ctx, b2, role, ldopt, ldaopt, b3=None, b3_role=None, b3_load_role=None):
        return _gen_b_lane(
            ctx, self.config, b2, role, 2, ldopt, next_dst=b3, next_role=b3_role, next_load_role=b3_load_role
        )

    def load_b2_last_k(self, ctx, b2, role, ldopt, ldaopt, b3=None, b3_role=None, b3_load_role=None):
        return _gen_b_last_k(ctx, self.config, b2, role, 2, next_reg=b3, next_role=b3_role)

    def load_b3(self, ctx, b3, role, ldopt, ldaopt, b4=None):
        return _gen_b_lane(ctx, self.config, b3, role, 3, ldopt)

    def load_b3_last_k(self, ctx, b3, role, ldopt, ldaopt, b4=None):
        return _gen_b_last_k(ctx, self.config, b3, role, 3)

    def set_svindex(self, ctx):
        return _gen_small_svindex(ctx, self.config)

    def kernel_mm_loop_n_pre_func(self, ctx):
        return _gen_small_n_pre(ctx)

    def kernel_mm_loop_n_post_func(self, ctx):
        return _gen_small_n_post(ctx, self.config)

    def kernel_mm_loop_m_pre_func(self, ctx):
        return _gen_small_m_pre(ctx, self.config)

    def kernel_mm_loop_m_post_func(self, ctx):
        return _gen_small_m_post(ctx, self.config)


class GeneralGemmModel:
    # `GeneralGemmModel` keeps the older straightforward schedule used outside the specialized small path.
    def load_a0b0(
        self,
        ctx,
        a0,
        a_role,
        b0,
        b_role,
        ldopt,
        ldaopt,
        a1=None,
        b1=None,
        a1_role=None,
        b1_role=None,
        a1_load_role=None,
        b1_load_role=None,
    ):
        regs = ctx.registers
        if ctx.is_ext_precision():
            code_str = f""
            a_pred_pre, a_pred = _paired_ext_load_predicate(
                ctx,
                a_role,
                a1_role if ctx.use_ext_paired_fast_path() and a1 is not None else None,
                load_second_role=a1_load_role if ctx.use_ext_paired_fast_path() and a1 is not None else None,
            )
            b_pred_pre, b_pred = _paired_ext_load_predicate(
                ctx,
                b_role,
                b1_role if ctx.use_ext_paired_fast_path() and b1 is not None else None,
                load_second_role=b1_load_role if ctx.use_ext_paired_fast_path() and b1 is not None else None,
            )
            code_str += a_pred_pre
            code_str += f"ld1h      {regs.vectors.a_low}.h, {a_pred}/z, [{regs.pointers.pA0}]\n"
            code_str += f"ld1h      {regs.vectors.pair_high}.h, {a_pred}/z, [{regs.pointers.pA0}, #1, MUL VL]\n"
            code_str += f"zip1      {a0}.h, {regs.vectors.a_low}.h, {regs.vectors.pair_high}.h\n"
            if ctx.use_ext_paired_fast_path() and a1 is not None:
                code_str += f"zip2      {a1}.h, {regs.vectors.a_low}.h, {regs.vectors.pair_high}.h\n"
            code_str += b_pred_pre
            code_str += f"ld1h      {self._b_low(ctx)}.h, {b_pred}/z, [{regs.pointers.pB0}]\n"
            code_str += f"ld1h      {self._b_high(ctx)}.h, {b_pred}/z, [{regs.pointers.pB0}, #1, MUL VL]\n"
            code_str += f"zip1      {b0}.h, {self._b_low(ctx)}.h, {self._b_high(ctx)}.h\n"
            if ctx.use_ext_paired_fast_path() and b1 is not None:
                code_str += f"zip2      {b1}.h, {self._b_low(ctx)}.h, {self._b_high(ctx)}.h\n"
            return code_str
        return (
            f"{ldopt}      {b0}{get_element_suffix(ctx)}, {_load_predicate(ctx, b_role)}, [{regs.pointers.pB0}]\n"
            f"{ldaopt}     {a0}{get_element_suffix(ctx)}, {_load_predicate(ctx, a_role)}, [{regs.pointers.pA0}]\n"
        )

    def _b_low(self, ctx):
        return ctx.registers.vectors.b_low

    def _b_high(self, ctx):
        return ctx.registers.vectors.b_high

    def _gen_general_a(self, ctx, dst, role, lane, ldaopt, next_dst=None, next_role=None, next_load_role=None):
        # General A loads use fixed MUL VL offsets instead of the chunk-aware small-kernel loaders.
        regs = ctx.registers
        if ctx.is_ext_precision():
            low_offset = lane * 2
            high_offset = low_offset + 1
            code_str = f""
            pred_pre, pred = _paired_ext_load_predicate(
                ctx,
                role,
                next_role if ctx.use_ext_paired_fast_path() and next_dst is not None else None,
                load_second_role=next_load_role if ctx.use_ext_paired_fast_path() and next_dst is not None else None,
            )
            code_str += pred_pre
            code_str += f"ld1h     {regs.vectors.a_low}.h, {pred}/z, [{regs.pointers.pA0}, #{low_offset}, MUL VL]\n"
            code_str += f"ld1h     {regs.vectors.pair_high}.h, {pred}/z, [{regs.pointers.pA0}, #{high_offset}, MUL VL]\n"
            code_str += f"zip1     {dst}.h, {regs.vectors.a_low}.h, {regs.vectors.pair_high}.h\n"
            if ctx.use_ext_paired_fast_path() and next_dst is not None:
                code_str += f"zip2     {next_dst}.h, {regs.vectors.a_low}.h, {regs.vectors.pair_high}.h\n"
            return code_str
        return f"{ldaopt}     {dst}{get_element_suffix(ctx)}, {_load_predicate(ctx, role)}, [{regs.pointers.pA0}, #{lane}, MUL VL]\n"

    def _gen_general_b(self, ctx, dst, role, lane, ldopt, next_dst=None, next_role=None, next_load_role=None):
        # General B loads mirror general A and keep the old simple paired-via-offset schedule.
        regs = ctx.registers
        if ctx.is_ext_precision():
            low_offset = lane * 2
            high_offset = low_offset + 1
            code_str = f""
            pred_pre, pred = _paired_ext_load_predicate(
                ctx,
                role,
                next_role if ctx.use_ext_paired_fast_path() and next_dst is not None else None,
                load_second_role=next_load_role if ctx.use_ext_paired_fast_path() and next_dst is not None else None,
            )
            code_str += pred_pre
            code_str += f"ld1h      {self._b_low(ctx)}.h, {pred}/z, [{regs.pointers.pB0}, #{low_offset}, MUL VL]\n"
            code_str += f"ld1h      {self._b_high(ctx)}.h, {pred}/z, [{regs.pointers.pB0}, #{high_offset}, MUL VL]\n"
            code_str += f"zip1      {dst}.h, {self._b_low(ctx)}.h, {self._b_high(ctx)}.h\n"
            if ctx.use_ext_paired_fast_path() and next_dst is not None:
                code_str += f"zip2      {next_dst}.h, {self._b_low(ctx)}.h, {self._b_high(ctx)}.h\n"
            return code_str
        return f"{ldopt}      {dst}{get_element_suffix(ctx)}, {_load_predicate(ctx, role)}, [{regs.pointers.pB0}, #{lane}, MUL VL]\n"

    def load_a0b0_last_k(
        self,
        ctx,
        a0,
        a_role,
        b0,
        b_role,
        ldopt,
        ldaopt,
        a1=None,
        b1=None,
        a1_role=None,
        b1_role=None,
        a1_load_role=None,
        b1_load_role=None,
    ):
        return self.load_a0b0(
            ctx,
            a0,
            a_role,
            b0,
            b_role,
            ldopt,
            ldaopt,
            a1=a1,
            b1=b1,
            a1_role=a1_role,
            b1_role=b1_role,
            a1_load_role=a1_load_role,
            b1_load_role=b1_load_role,
        )

    def load_a1(self, ctx, a1, role, ldopt, ldaopt, a2=None, a2_role=None, a2_load_role=None):
        return self._gen_general_a(ctx, a1, role, 1, ldaopt, next_dst=a2, next_role=a2_role, next_load_role=a2_load_role)

    def load_a1_last_k(self, ctx, a1, role, ldopt, ldaopt, a2=None, a2_role=None, a2_load_role=None):
        return self.load_a1(ctx, a1, role, ldopt, ldaopt, a2=a2, a2_role=a2_role, a2_load_role=a2_load_role)

    def load_a2(self, ctx, a2, role, ldopt, ldaopt, a3=None, a3_role=None, a3_load_role=None):
        return self._gen_general_a(ctx, a2, role, 2, ldaopt, next_dst=a3, next_role=a3_role, next_load_role=a3_load_role)

    def load_a2_last_k(self, ctx, a2, role, ldopt, ldaopt, a3=None, a3_role=None, a3_load_role=None):
        return self.load_a2(ctx, a2, role, ldopt, ldaopt, a3=a3, a3_role=a3_role, a3_load_role=a3_load_role)

    def load_a3(self, ctx, a3, role, ldopt, ldaopt, a4=None, a4_role=None):
        return self._gen_general_a(ctx, a3, role, 3, ldaopt)

    def load_a3_last_k(self, ctx, a3, role, ldopt, ldaopt, a4=None, a4_role=None):
        return self.load_a3(ctx, a3, role, ldopt, ldaopt)

    def load_b1(self, ctx, b1, role, ldopt, ldaopt, b2=None, b2_role=None, b2_load_role=None):
        return self._gen_general_b(ctx, b1, role, 1, ldopt, next_dst=b2, next_role=b2_role, next_load_role=b2_load_role)

    def load_b1_last_k(self, ctx, b1, role, ldopt, ldaopt, b2=None, b2_role=None, b2_load_role=None):
        return self.load_b1(ctx, b1, role, ldopt, ldaopt, b2=b2, b2_role=b2_role, b2_load_role=b2_load_role)

    def load_b2(self, ctx, b2, role, ldopt, ldaopt, b3=None, b3_role=None, b3_load_role=None):
        return self._gen_general_b(ctx, b2, role, 2, ldopt, next_dst=b3, next_role=b3_role, next_load_role=b3_load_role)

    def load_b2_last_k(self, ctx, b2, role, ldopt, ldaopt, b3=None, b3_role=None, b3_load_role=None):
        return self.load_b2(ctx, b2, role, ldopt, ldaopt, b3=b3, b3_role=b3_role, b3_load_role=b3_load_role)

    def load_b3(self, ctx, b3, role, ldopt, ldaopt, b4=None, b4_role=None):
        return self._gen_general_b(ctx, b3, role, 3, ldopt)

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
