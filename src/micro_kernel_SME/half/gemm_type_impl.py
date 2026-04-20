from dataclasses import dataclass

from .global_config import get_half_input_size_shift, get_half_input_suffix

# This file turns logical A/B lanes into concrete load-and-shape assembly for the chosen small/general model.


LOAD_CONTIGUOUS = "contiguous"
LOAD_GATHER = "gather"


@dataclass(frozen=True)
class SmallSideLoadSpec:
    mode: str
    base_ptr: str
    stride_reg: str
    paired_base: str
    gather_ptr: str
    gather_index: str
    gather_offset: str
    low_tmp: str
    high_tmp: str
    gather_index_k1: str = ""


@dataclass(frozen=True)
class GeneralSideLoadSpec:
    base_ptr: str
    low_tmp: str
    high_tmp: str


def _pred(ctx, role):
    # Return the logical predicate that matches the requested main/tail role.
    return ctx.registers.logical_predicate(role)


def _half_pred(ctx, role):
    # Return the widened half-input predicate used by `.h` loads.
    return ctx.registers.half_predicate(role)


def _paired_half_load_predicate(ctx, first_role, second_role=None, load_second_role=None):
    # Full paired paths reuse the prebuilt half predicates. Only mixed pairs
    # need a dedicated temporary load predicate.
    second_role = first_role if second_role is None else second_role
    load_second_role = second_role if load_second_role is None else load_second_role
    if first_role == load_second_role:
        return "", _half_pred(ctx, first_role)
    target = ctx.registers.predicates.load_pair_tmp
    code_str = f"zip1      {target}.h, {_pred(ctx, first_role)}.h, {_pred(ctx, load_second_role)}.h\n"
    return code_str, target


def _rdvl_lane_offset(registers, lane):
    # Convert a logical lane index into the byte offset pattern used by paired
    # contiguous half-input loads.
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


def _gen_half_contiguous_head(ctx, base, stride, tmp_base, role, dst, low_tmp, high_tmp):
    # Materialize the first contiguous half-input lane as one logical operand
    # with a plain `zip1`, reusing the prebuilt widened predicates.
    pred = _pred(ctx, role)
    code_str = f""
    code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}]\n"
    code_str += f"add       {tmp_base}, {base}, {stride}, LSL #1\n"
    code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{tmp_base}]\n"
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
    return code_str


def _gen_half_contiguous_head_pair_fast(
    ctx, base, stride, tmp_base, role0, role1, dst0, dst1, low_tmp, high_tmp, load_role1=None
):
    # Materialize the first contiguous `2VL` chunk as two half-input operands
    # with one load pair plus `zip1+zip2`.
    code_str = f""
    pred_pre, pred = _paired_half_load_predicate(ctx, role0, role1, load_second_role=load_role1)
    code_str += pred_pre
    code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}]\n"
    code_str += f"add       {tmp_base}, {base}, {stride}, LSL #1\n"
    code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{tmp_base}]\n"
    code_str += _gen_zip_pair(dst0, dst1, low_tmp, high_tmp)
    return code_str


def _gen_half_contiguous_lane_fast_pair(
    ctx, base, paired_base, role, dst, lane, low_tmp, high_tmp, next_dst=None, next_role=None, next_load_role=None
):
    # Load one later contiguous `2VL` chunk and shape it with `zip1+zip2`,
    # reusing the hoisted lane-2 offset for the full hot path.
    code_str = f""
    pred_pre, pred = _paired_half_load_predicate(
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


def _gen_half_contiguous_lane(
    ctx, base, paired_base, role, dst, lane, low_tmp, high_tmp, next_dst=None, next_role=None, next_load_role=None
):
    # This is the main lane materializer for contiguous half-input paths after
    # the first head load.
    if next_dst is None:
        pred = _pred(ctx, role)
        code_str = f""
        code_str += _rdvl_lane_offset(ctx.registers, lane)
        code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
        code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{paired_base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
        code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
        return code_str
    return _gen_half_contiguous_lane_fast_pair(
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


def _gen_half_contiguous_last_k_fast_pair(
    ctx, base, role, dst, lane, low_tmp, zero_tmp, next_dst=None, next_role=None
):
    # The last-k fast path still pairs only one real load because the missing
    # half is synthesized from zero.
    code_str = f""
    pred_pre, pred = _paired_half_load_predicate(ctx, role, next_role if next_dst is not None else None)
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


def _gen_half_contiguous_last_k_pair_safe(ctx, base, role0, role1, dst0, dst1, lane, low_tmp, zero_tmp):
    # Last-k mixed pairs are generated as two safe scalar loads to avoid
    # partial-lane pairing mistakes.
    code_str = _gen_half_contiguous_last_k(ctx, base, role0, dst0, lane, low_tmp, zero_tmp)
    code_str += _gen_half_contiguous_last_k(ctx, base, role1, dst1, lane + 1, low_tmp, zero_tmp)
    return code_str


def _gen_half_contiguous_last_k(ctx, base, role, dst, lane, low_tmp, zero_tmp, next_dst=None, next_role=None):
    # Last-k loads mirror the contiguous path but always zip against zero
    # instead of a real high half.
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
        return _gen_half_contiguous_last_k_pair_safe(
            ctx, base, role, next_role, dst, next_dst, lane, low_tmp, zero_tmp
        )
    return _gen_half_contiguous_last_k_fast_pair(
        ctx, base, role, dst, lane, low_tmp, zero_tmp, next_dst=next_dst, next_role=next_role
    )


def _gen_half_gather(ctx, base, role, index_vec, dst, low_tmp, high_tmp, pair_base, index_k1=""):
    code_str = f""
    code_str += f"ld1h      {low_tmp}.s, {_half_pred(ctx, role)}/z, [{base}, {index_vec}.s, UXTW]\n"
    if index_k1:
        code_str += f"ld1h      {high_tmp}.s, {_half_pred(ctx, role)}/z, [{base}, {index_k1}.s, UXTW]\n"
    else:
        code_str += f"add       {pair_base}, {base}, #2\n"
        code_str += f"ld1h      {high_tmp}.s, {_half_pred(ctx, role)}/z, [{pair_base}, {index_vec}.s, UXTW]\n"
    code_str += f"lsl       {high_tmp}.s, {high_tmp}.s, #16\n"
    code_str += f"orr       {dst}.d, {low_tmp}.d, {high_tmp}.d\n"
    return code_str


def _gen_half_gather_last_k(ctx, base, role, index_vec, dst, low_tmp, zero_tmp):
    # Small gather last-k stays single-output as well; the missing high half is
    # synthesized from zero without any paired predicate.
    return f"ld1h      {dst}.s, {_half_pred(ctx, role)}/z, [{base}, {index_vec}.s, UXTW]\n"


# Small-kernel side adapters ---------------------------------------------------------

def _small_side_spec(ctx, config, side):
    regs = ctx.registers
    if side == "a":
        return SmallSideLoadSpec(
            mode=config.a_mode,
            base_ptr=regs.pointers.pA0,
            stride_reg=regs.params.LDA,
            paired_base=regs.address.TMP_PTR,
            gather_ptr=regs.pointers.pAn,
            gather_index=regs.vectors.a_index,
            gather_offset=regs.address.pA_OFFSET,
            low_tmp=regs.vectors.a_low,
            high_tmp=regs.vectors.pair_high,
            gather_index_k1=config.a_gather_index_k1,
        )
    return SmallSideLoadSpec(
        mode=config.b_mode,
        base_ptr=regs.pointers.pB0,
        stride_reg=regs.params.LDB,
        paired_base=regs.address.TMP_PTR1,
        gather_ptr=regs.pointers.pBn,
        gather_index=regs.vectors.b_index,
        gather_offset=regs.address.pB_OFFSET,
        low_tmp=config.b_low_tmp,
        high_tmp=config.b_high_tmp,
        gather_index_k1=config.b_gather_index_k1,
    )


def _gen_gather_base_setup(side, lane):
    # Later gather lanes need an adjusted base pointer because the index vector
    # itself always starts from zero.
    if lane == 0:
        return ""
    return _scaled_add(side.gather_ptr, side.base_ptr, side.gather_offset, lane)


def _gather_base_register(side, lane):
    # Pick either the original base or the pre-shifted gather base depending on
    # which lane is being loaded.
    return side.base_ptr if lane == 0 else side.gather_ptr


def _gen_small_side_head(ctx, side, dst, role, next_dst=None, next_role=None, next_load_role=None):
    if side.mode == LOAD_CONTIGUOUS:
        if next_dst is not None:
            return _gen_half_contiguous_head_pair_fast(
                ctx,
                side.base_ptr,
                side.stride_reg,
                side.paired_base,
                role,
                next_role,
                dst,
                next_dst,
                side.low_tmp,
                side.high_tmp,
                load_role1=next_load_role,
            )
        return _gen_half_contiguous_head(
            ctx,
            side.base_ptr,
            side.stride_reg,
            side.paired_base,
            role,
            dst,
            side.low_tmp,
            side.high_tmp,
        )
    return _gen_half_gather(
        ctx,
        side.base_ptr,
        role,
        side.gather_index,
        dst,
        side.low_tmp,
        side.high_tmp,
        side.paired_base,
    )


def _gen_small_side_last_k(ctx, side, dst, role, lane):
    if side.mode == LOAD_CONTIGUOUS:
        return _gen_half_contiguous_last_k(
            ctx,
            side.base_ptr,
            role,
            dst,
            lane,
            side.low_tmp,
            side.high_tmp,
        )
    code_str = _gen_gather_base_setup(side, lane)
    code_str += _gen_half_gather_last_k(
        ctx,
        _gather_base_register(side, lane),
        role,
        side.gather_index,
        dst,
        side.low_tmp,
        side.high_tmp,
    )
    return code_str


def _gen_small_side_lane(ctx, side, dst, role, lane, next_dst=None, next_role=None, next_load_role=None):
    if side.mode == LOAD_CONTIGUOUS:
        return _gen_half_contiguous_lane(
            ctx,
            side.base_ptr,
            side.paired_base,
            role,
            dst,
            lane,
            side.low_tmp,
            side.high_tmp,
            next_dst=next_dst,
            next_role=next_role,
            next_load_role=next_load_role,
        )

    code_str = _gen_gather_base_setup(side, lane)
    code_str += _gen_half_gather(
        ctx,
        side.gather_ptr,
        role,
        side.gather_index,
        dst,
        side.low_tmp,
        side.high_tmp,
        ctx.registers.address.TMP_PTR2,
        index_k1=side.gather_index_k1,
    )
    return code_str


def _gen_small_svindex(ctx, config):
    regs = ctx.registers
    shift = get_half_input_size_shift()
    code_str = f""
    if config.a_mode == LOAD_GATHER:
        code_str += f"lsl     {regs.counters.TMP_CNT}, {regs.params.LDA}, #{shift}\n"
        code_str += f"mov     {regs.vectors.a_index}.s, #0\n"
        code_str += f"index   {regs.vectors.a_index}.s, #0, {regs.counters.TMP_CNT_SIN}\n"
        if config.a_gather_index_k1:
            code_str += f"index   {config.a_gather_index_k1}.s, #2, {regs.counters.TMP_CNT_SIN}\n"
    if config.b_mode == LOAD_GATHER:
        code_str += f"lsl     {regs.counters.TMP_CNT}, {regs.params.LDB}, #{shift}\n"
        code_str += f"mov     {regs.vectors.b_index}.s, #0\n"
        code_str += f"index   {regs.vectors.b_index}.s, #0, {regs.counters.TMP_CNT_SIN}\n"
        if config.b_gather_index_k1:
            code_str += f"index   {config.b_gather_index_k1}.s, #2, {regs.counters.TMP_CNT_SIN}\n"
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
    shift = get_half_input_size_shift()
    if config.b_mode == LOAD_GATHER:
        code_str = f""
        code_str += f"mul     {regs.counters.TMP_CNT}, {regs.params.LDB}, {regs.dims.MIN_N}\n"
        code_str += f"add     {regs.pointers.pBt}, {regs.pointers.pBt}, {regs.counters.TMP_CNT}, lsl #{shift}\n"
        return code_str
    return f"add     {regs.pointers.pBt}, {regs.pointers.pBt}, {regs.dims.MIN_N}, lsl #{shift}\n"


def _gen_small_m_pre(ctx, config):
    # Rebuild the live A/B pointers and scaled offsets for each inner M chunk.
    regs = ctx.registers
    shift = get_half_input_size_shift()
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
    shift = get_half_input_size_shift()
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
    b_gather_index_k1: str = ""
    a_gather_index_k1: str = ""


@dataclass(frozen=True)
class SmallGemmModel:
    config: SmallModelConfig

    # `SmallGemmModel` gives the plan layer one uniform API while hiding transpose-specific load mechanics.

    def _side(self, ctx, side):
        return _small_side_spec(ctx, self.config, side)

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
        code_str = _gen_small_side_head(
            ctx, self._side(ctx, "a"), a0, a_role, next_dst=a1, next_role=a1_role, next_load_role=a1_load_role
        )
        code_str += _gen_small_side_head(
            ctx, self._side(ctx, "b"), b0, b_role, next_dst=b1, next_role=b1_role, next_load_role=b1_load_role
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
        code_str = _gen_small_side_last_k(ctx, self._side(ctx, "a"), a0, a_role, 0)
        code_str += _gen_small_side_last_k(ctx, self._side(ctx, "b"), b0, b_role, 0)
        return code_str

    def load_a1(self, ctx, a1, role, ldopt, ldaopt, a2=None, a2_role=None, a2_load_role=None):
        return _gen_small_side_lane(
            ctx, self._side(ctx, "a"), a1, role, 1, next_dst=a2, next_role=a2_role, next_load_role=a2_load_role
        )

    def load_a1_last_k(self, ctx, a1, role, ldopt, ldaopt, a2=None, a2_role=None, a2_load_role=None):
        return _gen_small_side_last_k(ctx, self._side(ctx, "a"), a1, role, 1)

    def load_a2(self, ctx, a2, role, ldopt, ldaopt, a3=None, a3_role=None, a3_load_role=None):
        return _gen_small_side_lane(
            ctx, self._side(ctx, "a"), a2, role, 2, next_dst=a3, next_role=a3_role, next_load_role=a3_load_role
        )

    def load_a2_last_k(self, ctx, a2, role, ldopt, ldaopt, a3=None, a3_role=None, a3_load_role=None):
        return _gen_small_side_last_k(ctx, self._side(ctx, "a"), a2, role, 2)

    def load_a3(self, ctx, a3, role, ldopt, ldaopt, a4=None):
        return _gen_small_side_lane(ctx, self._side(ctx, "a"), a3, role, 3)

    def load_a3_last_k(self, ctx, a3, role, ldopt, ldaopt, a4=None):
        return _gen_small_side_last_k(ctx, self._side(ctx, "a"), a3, role, 3)

    def load_b1(self, ctx, b1, role, ldopt, ldaopt, b2=None, b2_role=None, b2_load_role=None):
        return _gen_small_side_lane(
            ctx, self._side(ctx, "b"), b1, role, 1, next_dst=b2, next_role=b2_role, next_load_role=b2_load_role
        )

    def load_b1_last_k(self, ctx, b1, role, ldopt, ldaopt, b2=None, b2_role=None, b2_load_role=None):
        return _gen_small_side_last_k(ctx, self._side(ctx, "b"), b1, role, 1)

    def load_b2(self, ctx, b2, role, ldopt, ldaopt, b3=None, b3_role=None, b3_load_role=None):
        return _gen_small_side_lane(
            ctx, self._side(ctx, "b"), b2, role, 2, next_dst=b3, next_role=b3_role, next_load_role=b3_load_role
        )

    def load_b2_last_k(self, ctx, b2, role, ldopt, ldaopt, b3=None, b3_role=None, b3_load_role=None):
        return _gen_small_side_last_k(ctx, self._side(ctx, "b"), b2, role, 2)

    def load_b3(self, ctx, b3, role, ldopt, ldaopt, b4=None):
        return _gen_small_side_lane(ctx, self._side(ctx, "b"), b3, role, 3)

    def load_b3_last_k(self, ctx, b3, role, ldopt, ldaopt, b4=None):
        return _gen_small_side_last_k(ctx, self._side(ctx, "b"), b3, role, 3)

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


# General-kernel side adapters -------------------------------------------------------

class GeneralGemmModel:
    # `GeneralGemmModel` keeps the older straightforward schedule used outside
    # the specialized small path.

    def _side(self, ctx, side):
        regs = ctx.registers
        if side == "a":
            return GeneralSideLoadSpec(regs.pointers.pA0, regs.vectors.a_low, regs.vectors.pair_high)
        return GeneralSideLoadSpec(regs.pointers.pB0, regs.vectors.b_low, regs.vectors.b_high)

    def _gen_general_side_head(
        self,
        ctx,
        side,
        dst,
        role,
        next_dst=None,
        next_role=None,
        next_load_role=None,
    ):
        pair_next = ctx.use_paired_half_loads() and next_dst is not None
        pred_pre, pred = _paired_half_load_predicate(
            ctx,
            role,
            next_role if pair_next else None,
            load_second_role=next_load_role if pair_next else None,
        )
        code_str = pred_pre
        code_str += f"ld1h      {side.low_tmp}.h, {pred}/z, [{side.base_ptr}]\n"
        code_str += f"ld1h      {side.high_tmp}.h, {pred}/z, [{side.base_ptr}, #1, MUL VL]\n"
        code_str += f"zip1      {dst}.h, {side.low_tmp}.h, {side.high_tmp}.h\n"
        if pair_next:
            code_str += f"zip2      {next_dst}.h, {side.low_tmp}.h, {side.high_tmp}.h\n"
        return code_str

    def _gen_general_side_lane(self, ctx, side, dst, role, lane, next_dst=None, next_role=None, next_load_role=None):
        low_offset = lane * 2
        high_offset = low_offset + 1
        pair_next = ctx.use_paired_half_loads() and next_dst is not None
        pred_pre, pred = _paired_half_load_predicate(
            ctx,
            role,
            next_role if pair_next else None,
            load_second_role=next_load_role if pair_next else None,
        )
        code_str = pred_pre
        code_str += f"ld1h     {side.low_tmp}.h, {pred}/z, [{side.base_ptr}, #{low_offset}, MUL VL]\n"
        code_str += f"ld1h     {side.high_tmp}.h, {pred}/z, [{side.base_ptr}, #{high_offset}, MUL VL]\n"
        code_str += f"zip1     {dst}.h, {side.low_tmp}.h, {side.high_tmp}.h\n"
        if pair_next:
            code_str += f"zip2     {next_dst}.h, {side.low_tmp}.h, {side.high_tmp}.h\n"
        return code_str

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
        code_str = self._gen_general_side_head(
            ctx, self._side(ctx, "a"), a0, a_role, next_dst=a1, next_role=a1_role, next_load_role=a1_load_role
        )
        code_str += self._gen_general_side_head(
            ctx, self._side(ctx, "b"), b0, b_role, next_dst=b1, next_role=b1_role, next_load_role=b1_load_role
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
        return self._gen_general_side_lane(
            ctx, self._side(ctx, "a"), a1, role, 1, next_dst=a2, next_role=a2_role, next_load_role=a2_load_role
        )

    def load_a1_last_k(self, ctx, a1, role, ldopt, ldaopt, a2=None, a2_role=None, a2_load_role=None):
        return self.load_a1(ctx, a1, role, ldopt, ldaopt, a2=a2, a2_role=a2_role, a2_load_role=a2_load_role)

    def load_a2(self, ctx, a2, role, ldopt, ldaopt, a3=None, a3_role=None, a3_load_role=None):
        return self._gen_general_side_lane(
            ctx, self._side(ctx, "a"), a2, role, 2, next_dst=a3, next_role=a3_role, next_load_role=a3_load_role
        )

    def load_a2_last_k(self, ctx, a2, role, ldopt, ldaopt, a3=None, a3_role=None, a3_load_role=None):
        return self.load_a2(ctx, a2, role, ldopt, ldaopt, a3=a3, a3_role=a3_role, a3_load_role=a3_load_role)

    def load_a3(self, ctx, a3, role, ldopt, ldaopt, a4=None, a4_role=None):
        return self._gen_general_side_lane(ctx, self._side(ctx, "a"), a3, role, 3)

    def load_a3_last_k(self, ctx, a3, role, ldopt, ldaopt, a4=None, a4_role=None):
        return self.load_a3(ctx, a3, role, ldopt, ldaopt)

    def load_b1(self, ctx, b1, role, ldopt, ldaopt, b2=None, b2_role=None, b2_load_role=None):
        return self._gen_general_side_lane(
            ctx, self._side(ctx, "b"), b1, role, 1, next_dst=b2, next_role=b2_role, next_load_role=b2_load_role
        )

    def load_b1_last_k(self, ctx, b1, role, ldopt, ldaopt, b2=None, b2_role=None, b2_load_role=None):
        return self.load_b1(ctx, b1, role, ldopt, ldaopt, b2=b2, b2_role=b2_role, b2_load_role=b2_load_role)

    def load_b2(self, ctx, b2, role, ldopt, ldaopt, b3=None, b3_role=None, b3_load_role=None):
        return self._gen_general_side_lane(
            ctx, self._side(ctx, "b"), b2, role, 2, next_dst=b3, next_role=b3_role, next_load_role=b3_load_role
        )

    def load_b2_last_k(self, ctx, b2, role, ldopt, ldaopt, b3=None, b3_role=None, b3_load_role=None):
        return self.load_b2(ctx, b2, role, ldopt, ldaopt, b3=b3, b3_role=b3_role, b3_load_role=b3_load_role)

    def load_b3(self, ctx, b3, role, ldopt, ldaopt, b4=None, b4_role=None):
        return self._gen_general_side_lane(ctx, self._side(ctx, "b"), b3, role, 3)

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


small_nn_model = SmallGemmModel(SmallModelConfig("small_nn", LOAD_CONTIGUOUS, LOAD_GATHER, "z26", "z29", b_gather_index_k1="z28"))
small_nt_model = SmallGemmModel(SmallModelConfig("small_nt", LOAD_CONTIGUOUS, LOAD_CONTIGUOUS, "z28", "z30"))
small_tn_model = SmallGemmModel(SmallModelConfig("small_tn", LOAD_GATHER, LOAD_GATHER, "z26", "z29"))
small_tt_model = SmallGemmModel(SmallModelConfig("small_tt", LOAD_GATHER, LOAD_CONTIGUOUS, "z26", "z27"))
general_model = GeneralGemmModel()
