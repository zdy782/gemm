from dataclasses import dataclass

from global_config import get_element_size_shift, get_ld_element_suffix


LOAD_CONTIGUOUS = "contiguous"
LOAD_GATHER = "gather"


def _pred(ctx, role):
    return ctx.registers.logical_predicate(role)


def _ext_pred(ctx, role):
    return ctx.registers.ext_predicate(role)


def _load_predicate(ctx, role):
    pred = _pred(ctx, role)
    return pred if ctx.is_ext_precision() else f"{pred}/z"


def _zip_ext_predicate(ctx, role):
    pred = _pred(ctx, role)
    ext_pred = _ext_pred(ctx, role)
    return f"zip1      {ext_pred}.h, {pred}.h, {pred}.h\n"


def _rdvl_lane_offset(registers, lane):
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
    if lane == 1:
        return f"add          {dst}, {base}, {offset}\n"
    if lane == 2:
        return f"add          {dst}, {base}, {offset}, lsl #1\n"
    code_str = f"add          {dst}, {base}, {offset}\n"
    code_str += f"add          {dst}, {dst}, {offset}, lsl #1\n"
    return code_str


def _emit_ext_contiguous_head(ctx, base, stride, tmp_base, role, dst, low_tmp, high_tmp):
    pred = _pred(ctx, role)
    code_str = f""
    code_str += _zip_ext_predicate(ctx, role)
    code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}]\n"
    code_str += f"add       {tmp_base}, {base}, {stride}, LSL #1\n"
    code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{tmp_base}]\n"
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
    return code_str


def _emit_ext_contiguous_lane(ctx, base, paired_base, role, dst, lane, low_tmp, high_tmp):
    pred = _pred(ctx, role)
    code_str = f""
    code_str += _zip_ext_predicate(ctx, role)
    code_str += _rdvl_lane_offset(ctx.registers, lane)
    code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
    code_str += f"ld1h      {high_tmp}.h, {pred}/z, [{paired_base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
    return code_str


def _emit_ext_contiguous_last_k(ctx, base, role, dst, lane, low_tmp, zero_tmp):
    pred = _pred(ctx, role)
    code_str = f""
    code_str += _zip_ext_predicate(ctx, role)
    if lane > 0:
        code_str += _rdvl_lane_offset(ctx.registers, lane)
        code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}, {ctx.registers.address.TMP_PTR2}, LSL #1]\n"
    else:
        code_str += f"ld1h      {low_tmp}.h, {pred}/z, [{base}]\n"
    code_str += f"mov       {zero_tmp}.h, #0\n"
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {zero_tmp}.h\n"
    return code_str


def _emit_non_ext_contiguous(load_inst, ctx, base, role, dst, lane):
    suffix = get_ld_element_suffix(ctx)
    pred = _load_predicate(ctx, role)
    if lane == 0:
        return f"{load_inst}      {dst}{suffix}, {pred}, [{base}]\n"
    return f"{load_inst}      {dst}{suffix}, {pred}, [{base}, #{lane}, MUL VL]\n"


def _emit_ext_gather_pair(ctx, base, role, index_vec, dst, low_tmp, high_tmp, pair_base):
    code_str = f""
    code_str += _zip_ext_predicate(ctx, role)
    code_str += f"ld1h      {low_tmp}.s, {_ext_pred(ctx, role)}/z, [{base}, {index_vec}.s, UXTW]\n"
    code_str += f"add       {pair_base}, {base}, #2\n"
    code_str += f"ld1h      {high_tmp}.s, {_ext_pred(ctx, role)}/z, [{pair_base}, {index_vec}.s, UXTW]\n"
    code_str += f"uzp1      {low_tmp}.h, {low_tmp}.h, {low_tmp}.h\n"
    code_str += f"uzp1      {high_tmp}.h, {high_tmp}.h, {high_tmp}.h\n"
    code_str += f"zip1      {dst}.h, {low_tmp}.h, {high_tmp}.h\n"
    return code_str


def _emit_ext_gather_last_k(ctx, base, role, index_vec, dst):
    code_str = f""
    code_str += _zip_ext_predicate(ctx, role)
    code_str += f"ld1h      {dst}.s, {_ext_pred(ctx, role)}/z, [{base}, {index_vec}.s, UXTW]\n"
    return code_str


def _emit_non_ext_gather(load_inst, ctx, base, role, index_vec, dst):
    suffix = get_ld_element_suffix(ctx)
    pred = _load_predicate(ctx, role)
    return f"{load_inst}      {dst}{suffix}, {pred}, [{base}, {index_vec}.s, UXTW]\n"


def _emit_a_head(ctx, config, a0, role, ldaopt):
    regs = ctx.registers
    if config.a_mode == LOAD_CONTIGUOUS:
        if ctx.is_ext_precision():
            return _emit_ext_contiguous_head(
                ctx,
                regs.pointers.pA0,
                regs.params.LDA,
                regs.address.TMP_PTR,
                role,
                a0,
                regs.vectors.a_low,
                regs.vectors.pair_high,
            )
        return _emit_non_ext_contiguous(ldaopt, ctx, regs.pointers.pA0, role, a0, 0)
    if ctx.is_ext_precision():
        return _emit_ext_gather_pair(
            ctx,
            regs.pointers.pA0,
            role,
            regs.vectors.a_index,
            a0,
            regs.vectors.a_low,
            regs.vectors.pair_high,
            regs.address.TMP_PTR,
        )
    return _emit_non_ext_gather(ldaopt, ctx, regs.pointers.pA0, role, regs.vectors.a_index, a0)


def _emit_b_head(ctx, config, b0, role, ldopt):
    regs = ctx.registers
    if config.b_mode == LOAD_CONTIGUOUS:
        if ctx.is_ext_precision():
            return _emit_ext_contiguous_head(
                ctx,
                regs.pointers.pB0,
                regs.params.LDB,
                regs.address.TMP_PTR1,
                role,
                b0,
                config.b_low_tmp,
                config.b_high_tmp,
            )
        return _emit_non_ext_contiguous(ldopt, ctx, regs.pointers.pB0, role, b0, 0)
    if ctx.is_ext_precision():
        return _emit_ext_gather_pair(
            ctx,
            regs.pointers.pB0,
            role,
            regs.vectors.b_index,
            b0,
            config.b_low_tmp,
            config.b_high_tmp,
            regs.address.TMP_PTR1,
        )
    return _emit_non_ext_gather(ldopt, ctx, regs.pointers.pB0, role, regs.vectors.b_index, b0)


def _emit_gather_base_setup(target, registers, lane, is_a):
    if lane == 0:
        return ""
    base = registers.pointers.pA0 if is_a else registers.pointers.pB0
    offset = registers.address.pA_OFFSET if is_a else registers.address.pB_OFFSET
    return _scaled_add(target, base, offset, lane)


def _gather_base_register(registers, lane, is_a):
    if lane == 0:
        return registers.pointers.pA0 if is_a else registers.pointers.pB0
    return registers.pointers.pAn if is_a else registers.pointers.pBn


def _emit_a_last_k(ctx, config, a_reg, role, lane):
    regs = ctx.registers
    if config.a_mode == LOAD_CONTIGUOUS:
        return _emit_ext_contiguous_last_k(
            ctx,
            regs.pointers.pA0,
            role,
            a_reg,
            lane,
            regs.vectors.a_low,
            regs.vectors.pair_high,
        )
    code_str = _emit_gather_base_setup(regs.pointers.pAn, regs, lane, True)
    code_str += _emit_ext_gather_last_k(ctx, _gather_base_register(regs, lane, True), role, regs.vectors.a_index, a_reg)
    return code_str


def _emit_b_last_k(ctx, config, b_reg, role, lane):
    regs = ctx.registers
    if config.b_mode == LOAD_CONTIGUOUS:
        return _emit_ext_contiguous_last_k(
            ctx,
            regs.pointers.pB0,
            role,
            b_reg,
            lane,
            config.b_low_tmp,
            config.b_high_tmp,
        )
    code_str = _emit_gather_base_setup(regs.pointers.pBn, regs, lane, False)
    code_str += _emit_ext_gather_last_k(ctx, _gather_base_register(regs, lane, False), role, regs.vectors.b_index, b_reg)
    return code_str


def _emit_gather_base(target, registers, lane, is_a):
    return _emit_gather_base_setup(target, registers, lane, is_a)


def _emit_a_lane(ctx, config, dst, role, lane, ldaopt):
    regs = ctx.registers
    if config.a_mode == LOAD_CONTIGUOUS:
        if ctx.is_ext_precision():
            return _emit_ext_contiguous_lane(
                ctx,
                regs.pointers.pA0,
                regs.address.TMP_PTR,
                role,
                dst,
                lane,
                regs.vectors.a_low,
                regs.vectors.pair_high,
            )
        return _emit_non_ext_contiguous(ldaopt, ctx, regs.pointers.pA0, role, dst, lane)

    code_str = _emit_gather_base(regs.pointers.pAn, regs, lane, True)
    if ctx.is_ext_precision():
        code_str += _emit_ext_gather_pair(
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
    code_str += _emit_non_ext_gather(ldaopt, ctx, regs.pointers.pAn, role, regs.vectors.a_index, dst)
    return code_str


def _emit_b_lane(ctx, config, dst, role, lane, ldopt):
    regs = ctx.registers
    if config.b_mode == LOAD_CONTIGUOUS:
        if ctx.is_ext_precision():
            return _emit_ext_contiguous_lane(
                ctx,
                regs.pointers.pB0,
                regs.address.TMP_PTR1,
                role,
                dst,
                lane,
                config.b_low_tmp,
                config.b_high_tmp,
            )
        return _emit_non_ext_contiguous(ldopt, ctx, regs.pointers.pB0, role, dst, lane)

    code_str = _emit_gather_base(regs.pointers.pBn, regs, lane, False)
    if ctx.is_ext_precision():
        code_str += _emit_ext_gather_pair(
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
    code_str += _emit_non_ext_gather(ldopt, ctx, regs.pointers.pBn, role, regs.vectors.b_index, dst)
    return code_str


def _emit_small_svindex(ctx, config):
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


def _emit_small_n_pre(ctx):
    regs = ctx.registers
    code_str = f""
    code_str += f"mov     {regs.pointers.pAt}, {regs.params.origPA}\n"
    code_str += f"mov     {regs.pointers.pB0}, {regs.pointers.pBt}\n"
    return code_str


def _emit_small_n_post(ctx, config):
    regs = ctx.registers
    shift = get_element_size_shift(ctx)
    if config.b_mode == LOAD_GATHER:
        code_str = f""
        code_str += f"mul     {regs.counters.TMP_CNT}, {regs.params.LDB}, {regs.dims.MIN_N}\n"
        code_str += f"add     {regs.pointers.pBt}, {regs.pointers.pBt}, {regs.counters.TMP_CNT}, lsl #{shift}\n"
        return code_str
    return f"add     {regs.pointers.pBt}, {regs.pointers.pBt}, {regs.dims.MIN_N}, lsl #{shift}\n"


def _emit_small_m_pre(ctx, config):
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


def _emit_small_m_post(ctx, config):
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

    def load_a0b0(self, ctx, a0, a_role, b0, b_role, ldopt, ldaopt):
        code_str = _emit_a_head(ctx, self.config, a0, a_role, ldaopt)
        code_str += _emit_b_head(ctx, self.config, b0, b_role, ldopt)
        return code_str

    def load_a0b0_last_k(self, ctx, a0, a_role, b0, b_role, ldopt, ldaopt):
        code_str = _emit_a_last_k(ctx, self.config, a0, a_role, 0)
        code_str += _emit_b_last_k(ctx, self.config, b0, b_role, 0)
        return code_str

    def load_a1(self, ctx, a1, role, ldopt, ldaopt):
        return _emit_a_lane(ctx, self.config, a1, role, 1, ldaopt)

    def load_a1_last_k(self, ctx, a1, role, ldopt, ldaopt):
        return _emit_a_last_k(ctx, self.config, a1, role, 1)

    def load_a2(self, ctx, a2, role, ldopt, ldaopt):
        return _emit_a_lane(ctx, self.config, a2, role, 2, ldaopt)

    def load_a2_last_k(self, ctx, a2, role, ldopt, ldaopt):
        return _emit_a_last_k(ctx, self.config, a2, role, 2)

    def load_a3(self, ctx, a3, role, ldopt, ldaopt):
        return _emit_a_lane(ctx, self.config, a3, role, 3, ldaopt)

    def load_a3_last_k(self, ctx, a3, role, ldopt, ldaopt):
        return _emit_a_last_k(ctx, self.config, a3, role, 3)

    def load_b1(self, ctx, b1, role, ldopt, ldaopt):
        return _emit_b_lane(ctx, self.config, b1, role, 1, ldopt)

    def load_b1_last_k(self, ctx, b1, role, ldopt, ldaopt):
        return _emit_b_last_k(ctx, self.config, b1, role, 1)

    def load_b2(self, ctx, b2, role, ldopt, ldaopt):
        return _emit_b_lane(ctx, self.config, b2, role, 2, ldopt)

    def load_b2_last_k(self, ctx, b2, role, ldopt, ldaopt):
        return _emit_b_last_k(ctx, self.config, b2, role, 2)

    def load_b3(self, ctx, b3, role, ldopt, ldaopt):
        return _emit_b_lane(ctx, self.config, b3, role, 3, ldopt)

    def load_b3_last_k(self, ctx, b3, role, ldopt, ldaopt):
        return _emit_b_last_k(ctx, self.config, b3, role, 3)

    def set_svindex(self, ctx):
        return _emit_small_svindex(ctx, self.config)

    def kernel_mm_loop_n_pre_func(self, ctx):
        return _emit_small_n_pre(ctx)

    def kernel_mm_loop_n_post_func(self, ctx):
        return _emit_small_n_post(ctx, self.config)

    def kernel_mm_loop_m_pre_func(self, ctx):
        return _emit_small_m_pre(ctx, self.config)

    def kernel_mm_loop_m_post_func(self, ctx):
        return _emit_small_m_post(ctx, self.config)


class GeneralGemmModel:
    def load_a0b0(self, ctx, a0, a_role, b0, b_role, ldopt, ldaopt):
        regs = ctx.registers
        if ctx.is_ext_precision():
            code_str = f""
            code_str += f"ld1h      {regs.vectors.a_low}.h, {_pred(ctx, a_role)}, [{regs.pointers.pA0}]\n"
            code_str += f"ld1h      {regs.vectors.pair_high}.h, {_pred(ctx, a_role)}, [{regs.pointers.pA0}, #1, MUL VL]\n"
            code_str += f"zip1      {a0}.h, {regs.vectors.a_low}.h, {regs.vectors.pair_high}.h\n"
            code_str += f"ld1h      {self._b_low(ctx)}.h, {_pred(ctx, b_role)}, [{regs.pointers.pB0}]\n"
            code_str += f"ld1h      {self._b_high(ctx)}.h, {_pred(ctx, b_role)}, [{regs.pointers.pB0}, #1, MUL VL]\n"
            code_str += f"zip1      {b0}.h, {self._b_low(ctx)}.h, {self._b_high(ctx)}.h\n"
            return code_str
        return (
            f"{ldopt}      {b0}{get_ld_element_suffix(ctx)}, {_load_predicate(ctx, b_role)}, [{regs.pointers.pB0}]\n"
            f"{ldaopt}     {a0}{get_ld_element_suffix(ctx)}, {_load_predicate(ctx, a_role)}, [{regs.pointers.pA0}]\n"
        )

    def _b_low(self, ctx):
        return ctx.registers.vectors.b_low

    def _b_high(self, ctx):
        return ctx.registers.vectors.b_high

    def _emit_general_a(self, ctx, dst, role, lane, ldaopt):
        regs = ctx.registers
        if ctx.is_ext_precision():
            low_offset = lane * 2
            high_offset = low_offset + 1
            code_str = f""
            code_str += f"ld1h     {regs.vectors.a_low}.h, {_pred(ctx, role)}, [{regs.pointers.pA0}, #{low_offset}, MUL VL]\n"
            code_str += f"ld1h     {regs.vectors.pair_high}.h, {_pred(ctx, role)}, [{regs.pointers.pA0}, #{high_offset}, MUL VL]\n"
            code_str += f"zip1     {dst}.h, {regs.vectors.a_low}.h, {regs.vectors.pair_high}.h\n"
            return code_str
        return f"{ldaopt}     {dst}{get_ld_element_suffix(ctx)}, {_load_predicate(ctx, role)}, [{regs.pointers.pA0}, #{lane}, MUL VL]\n"

    def _emit_general_b(self, ctx, dst, role, lane, ldopt):
        regs = ctx.registers
        if ctx.is_ext_precision():
            low_offset = lane * 2
            high_offset = low_offset + 1
            code_str = f""
            code_str += f"ld1h      {self._b_low(ctx)}.h, {_pred(ctx, role)}, [{regs.pointers.pB0}, #{low_offset}, MUL VL]\n"
            code_str += f"ld1h      {self._b_high(ctx)}.h, {_pred(ctx, role)}, [{regs.pointers.pB0}, #{high_offset}, MUL VL]\n"
            code_str += f"zip1      {dst}.h, {self._b_low(ctx)}.h, {self._b_high(ctx)}.h\n"
            return code_str
        return f"{ldopt}      {dst}{get_ld_element_suffix(ctx)}, {_load_predicate(ctx, role)}, [{regs.pointers.pB0}, #{lane}, MUL VL]\n"

    def load_a0b0_last_k(self, ctx, a0, a_role, b0, b_role, ldopt, ldaopt):
        return self.load_a0b0(ctx, a0, a_role, b0, b_role, ldopt, ldaopt)

    def load_a1(self, ctx, a1, role, ldopt, ldaopt):
        return self._emit_general_a(ctx, a1, role, 1, ldaopt)

    def load_a1_last_k(self, ctx, a1, role, ldopt, ldaopt):
        return self.load_a1(ctx, a1, role, ldopt, ldaopt)

    def load_a2(self, ctx, a2, role, ldopt, ldaopt):
        return self._emit_general_a(ctx, a2, role, 2, ldaopt)

    def load_a2_last_k(self, ctx, a2, role, ldopt, ldaopt):
        return self.load_a2(ctx, a2, role, ldopt, ldaopt)

    def load_a3(self, ctx, a3, role, ldopt, ldaopt):
        return self._emit_general_a(ctx, a3, role, 3, ldaopt)

    def load_a3_last_k(self, ctx, a3, role, ldopt, ldaopt):
        return self.load_a3(ctx, a3, role, ldopt, ldaopt)

    def load_b1(self, ctx, b1, role, ldopt, ldaopt):
        return self._emit_general_b(ctx, b1, role, 1, ldopt)

    def load_b1_last_k(self, ctx, b1, role, ldopt, ldaopt):
        return self.load_b1(ctx, b1, role, ldopt, ldaopt)

    def load_b2(self, ctx, b2, role, ldopt, ldaopt):
        return self._emit_general_b(ctx, b2, role, 2, ldopt)

    def load_b2_last_k(self, ctx, b2, role, ldopt, ldaopt):
        return self.load_b2(ctx, b2, role, ldopt, ldaopt)

    def load_b3(self, ctx, b3, role, ldopt, ldaopt):
        return self._emit_general_b(ctx, b3, role, 3, ldopt)

    def load_b3_last_k(self, ctx, b3, role, ldopt, ldaopt):
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


small_nn_model = SmallGemmModel(SmallModelConfig("small_nn", LOAD_CONTIGUOUS, LOAD_GATHER, "z28", "z30"))
small_nt_model = SmallGemmModel(SmallModelConfig("small_nt", LOAD_CONTIGUOUS, LOAD_CONTIGUOUS, "z28", "z30"))
small_tn_model = SmallGemmModel(SmallModelConfig("small_tn", LOAD_GATHER, LOAD_GATHER, "z26", "z29"))
small_tt_model = SmallGemmModel(SmallModelConfig("small_tt", LOAD_GATHER, LOAD_CONTIGUOUS, "z27", "z30"))
general_model = GeneralGemmModel()
