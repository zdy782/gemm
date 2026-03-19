from global_config import *

def get_bf16_load_suffix():
    return ".h"

def gen_zip_pair(dst, src1, src2):
    code_str = f""
    code_str += f"zip1    {dst}.s, {src1}.s, {src2}.s\n"
    return code_str

class small_gemm_nn_def:
    def load_a0b0(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        # NN: A is N (not transposed) -> consecutive load
        # NN: B is N (not transposed) -> gather load with ldb stride
        code_str = f""
        if is_bf16():
            code_str += f"zip1      p4.h, {pga}.h, {pga}.h\n"
            code_str += f"ld1h      z26.h, {pga}/z, [{pA0}]\n"
            code_str += f"add       {TMP_PTR}, {pA0}, {LDA}, LSL #1\n"
            code_str += f"ld1h      z29.h, {pga}/z, [{TMP_PTR}]\n"
            code_str += f"zip1      {a0}.h, z26.h, z29.h\n"

            code_str += f"zip1      p6.h, {pgb}.h, {pgb}.h\n"
            code_str += f"ld1h      z28.s, p6/z, [{pB0}, z27.s, UXTW]\n"
            code_str += f"add       {TMP_PTR1}, {pB0}, #2\n"
            code_str += f"ld1h      z30.s, p6/z, [{TMP_PTR1}, z27.s, UXTW]\n"
            code_str += f"uzp1      z28.h, z28.h, z28.h\n"
            code_str += f"uzp1      z30.h, z30.h, z30.h\n"
            code_str += f"zip1      {b0}.h, z28.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b0}{get_ld_element_suffix()}, {pgb}, [{pB0}, z27.s, UXTW]\n"
            code_str += f"{ldaopt}     {a0}{get_ld_element_suffix()}, {pga}, [{pA0}]\n"
        return code_str

    def load_a0b0_last_k(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        # For last k (odd): only load one bf16 value
        code_str = f""
        code_str += f"zip1      p4.h, {pga}.h, {pga}.h\n"
        code_str += f"ld1h      z26.h, {pga}/z, [{pA0}]\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {a0}.h, z26.h, z30.h\n"

        code_str += f"zip1      p6.h, {pgb}.h, {pgb}.h\n"
        code_str += f"ld1h      {b0}.s, p6/z, [{pB0}, z27.s, UXTW]\n"
        return code_str

    def load_a1(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            if pg == 'p1':
                dst = 'p4'
            else:
                dst = 'p5'
            code_str += f"rdvl      {TMP_PTR2}, #1\n"
            code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
            code_str += f"ld1h      z26.h, {pg}/z, [{pA0}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"ld1h      z29.h, {pg}/z, [{TMP_PTR}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"zip1      {a1}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldaopt}     {a1}{get_ld_element_suffix()}, {pg}, [{pA0}, #1, MUL VL]\n"
        return code_str

    def load_a1_last_k(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if pg == 'p1':
            dst = 'p4'
        else:
            dst = 'p5'
        code_str += f"rdvl      {TMP_PTR2}, #1\n"
        code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
        code_str += f"ld1h      z26.h, {pg}/z, [{pA0}, {TMP_PTR2}, LSL #1]\n"
        code_str += f"mov       z29.h, #0\n"
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"zip1      {a1}.h, z26.h, z29.h\n"
        return code_str

    def load_a2(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            if pg == 'p1':
                dst = 'p4'
            else:
                dst = 'p5'
            code_str += f"rdvl      {TMP_PTR2}, #1\n"
            code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #1\n"
            code_str += f"ld1h      z26.h, {pg}/z, [{pA0}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"ld1h      z29.h, {pg}/z, [{TMP_PTR}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"zip1      {a2}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldaopt}     {a2}{get_ld_element_suffix()}, {pg}, [{pA0}, #2, MUL VL]\n"
        return code_str

    def load_a2_last_k(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if pg == 'p1':
            dst = 'p4'
        else:
            dst = 'p5'
        code_str += f"rdvl      {TMP_PTR2}, #1\n"
        code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #1\n"
        code_str += f"ld1h      z26.h, {pg}/z, [{pA0}, {TMP_PTR2}, LSL #1]\n"
        code_str += f"mov       z29.h, #0\n"
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"zip1      {a2}.h, z26.h, z29.h\n"
        return code_str

    def load_a3(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            if pg == 'p1':
                dst = 'p4'
            else:
                dst = 'p5'
            code_str += f"rdvl      {TMP_PTR2}, #1\n"
            code_str += f"add       {TMP_PTR2}, {TMP_PTR2}, {TMP_PTR2}, LSL #1\n"
            code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
            code_str += f"ld1h      z26.h, {pg}/z, [{pA0}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"ld1h      z29.h, {pg}/z, [{TMP_PTR}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"zip1      {a3}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldaopt}     {a3}{get_ld_element_suffix()}, {pg}, [{pA0}, #3, MUL VL]\n"
        return code_str

    def load_a3_last_k(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if pg == 'p1':
            dst = 'p4'
        else:
            dst = 'p5'
        code_str += f"rdvl      {TMP_PTR2}, #1\n"
        code_str += f"add       {TMP_PTR2}, {TMP_PTR2}, {TMP_PTR2}, LSL #1\n"
        code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
        code_str += f"ld1h      z26.h, {pg}/z, [{pA0}, {TMP_PTR2}, LSL #1]\n"
        code_str += f"mov       z29.h, #0\n"
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"zip1      {a3}.h, z26.h, z29.h\n"
        return code_str

    def load_b1(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}\n"
        if is_bf16():
            if pg == 'p0':
                dst = 'p6'
            else:
                dst = 'p3'
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"ld1h      z28.s, {dst}/z, [{pBn}, z27.s, UXTW]\n"
            code_str += f"add       {TMP_PTR2}, {pBn}, #2\n"
            code_str += f"ld1h      z30.s, {dst}/z, [{TMP_PTR2}, z27.s, UXTW]\n"
            code_str += f"uzp1      z28.h, z28.h, z28.h\n"
            code_str += f"uzp1      z30.h, z30.h, z30.h\n"
            code_str += f"zip1      {b1}.h, z28.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b1}{get_ld_element_suffix()}, {pg}, [{pBn}, z27.s, UXTW]\n"
        return code_str

    def load_b1_last_k(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}\n"
        if pg == 'p0':
            dst = 'p6'
        else:
            dst = 'p3'
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"ld1h      {b1}.s, {dst}/z, [{pBn}, z27.s, UXTW]\n"
        return code_str

    def load_b2(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}, lsl #1\n"
        if is_bf16():
            if pg == 'p0':
                dst = 'p6'
            else:
                dst = 'p3'
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"ld1h      z28.s, {dst}/z, [{pBn}, z27.s, UXTW]\n"
            code_str += f"add       {TMP_PTR2}, {pBn}, #2\n"
            code_str += f"ld1h      z30.s, {dst}/z, [{TMP_PTR2}, z27.s, UXTW]\n"
            code_str += f"uzp1      z28.h, z28.h, z28.h\n"
            code_str += f"uzp1      z30.h, z30.h, z30.h\n"
            code_str += f"zip1      {b2}.h, z28.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b2}{get_ld_element_suffix()}, {pg}, [{pBn}, z27.s, UXTW]\n"
        return code_str

    def load_b2_last_k(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}, lsl #1\n"
        if pg == 'p0':
            dst = 'p6'
        else:
            dst = 'p3'
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"ld1h      {b2}.s, {dst}/z, [{pBn}, z27.s, UXTW]\n"
        return code_str

    def load_b3(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}\n"
        code_str += f"add          {pBn}, {pBn}, {pB_OFFSET}, lsl #1\n"
        if is_bf16():
            if pg == 'p0':
                dst = 'p6'
            else:
                dst = 'p3'
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"ld1h      z28.s, {dst}/z, [{pBn}, z27.s, UXTW]\n"
            code_str += f"add       {TMP_PTR2}, {pBn}, #2\n"
            code_str += f"ld1h      z30.s, {dst}/z, [{TMP_PTR2}, z27.s, UXTW]\n"
            code_str += f"uzp1      z28.h, z28.h, z28.h\n"
            code_str += f"uzp1      z30.h, z30.h, z30.h\n"
            code_str += f"zip1      {b3}.h, z28.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b3}{get_ld_element_suffix()}, {pg}, [{pBn}, z27.s, UXTW]\n"
        return code_str

    def load_b3_last_k(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}\n"
        code_str += f"add          {pBn}, {pBn}, {pB_OFFSET}, lsl #1\n"
        if pg == 'p0':
            dst = 'p6'
        else:
            dst = 'p3'
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"ld1h      {b3}.s, {dst}/z, [{pBn}, z27.s, UXTW]\n"
        return code_str

    def set_svindex():
        # nopackA nopackB
        # B needs gather load with ldb stride
        code_str = f""
        shift = get_element_size_shift()
        code_str += f"lsl     {TMP_CNT}, {LDB}, #{shift}\n"
        code_str += f"mov     z27.s, #0\n"
        code_str += f"index   z27.s, #0, {TMP_CNT_SIN}\n"

        return code_str

    def kernel_mm_loop_n_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov     {pAt}, {origPA}\n"
        code_str += f"mov     {pB0}, {pBt}\n"

        return code_str

    def kernel_mm_loop_n_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mul     {TMP_CNT}, {LDB}, {MIN_N}\n"
        shift = get_element_size_shift()
        code_str += f"add     {pBt}, {pBt}, {TMP_CNT}, lsl #{shift}\n"
        return code_str

    def kernel_mm_loop_m_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov      {pB0}, {pBt}      // KERNEL_MM_LOOP_M_PRE_FUNC\n"
        code_str += f"mov      {pA0}, {pAt}\n"
        shift = get_element_size_shift()
        code_str += f"lsl      {pB_OFFSET}, {LDB}, #{6 + shift - 2}\n"
        code_str += f"mov      {OFFSET_A}, {LDA}\n"
        code_str += f"mov      {OFFSET_B}, #1\n"

        return code_str

    def kernel_mm_loop_m_post_func():
        # nopackA nopackB
        code_str = f""
        shift = get_element_size_shift()
        code_str += f"add      {pAt}, {pAt}, {MIN_M}, lsl #{shift}\n"
        return code_str

class small_gemm_nt_def:
    def load_a0b0(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        # NT: A is N (not transposed) -> consecutive load
        # NT: B is T (transposed) -> consecutive load
        code_str = f""
        if is_bf16():
            code_str += f"zip1      p4.h, {pga}.h, {pga}.h\n"
            code_str += f"ld1h      z26.h, {pga}/z, [{pA0}]\n"
            code_str += f"add       {TMP_PTR}, {pA0}, {LDA}, LSL #1\n"
            code_str += f"ld1h      z29.h, {pga}/z, [{TMP_PTR}]\n"
            code_str += f"zip1      {a0}.h, z26.h, z29.h\n"

            code_str += f"zip1      p6.h, {pgb}.h, {pgb}.h\n"
            code_str += f"ld1h      z28.h, {pgb}/z, [{pB0}]\n"
            code_str += f"add       {TMP_PTR1}, {pB0}, {LDB}, LSL #1\n"
            code_str += f"ld1h      z30.h, {pgb}/z, [{TMP_PTR1}]\n"
            code_str += f"zip1      {b0}.h, z28.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b0}{get_ld_element_suffix()}, {pgb}, [{pB0}]\n"
            code_str += f"{ldaopt}     {a0}{get_ld_element_suffix()}, {pga}, [{pA0}]\n"
        return code_str

    def load_a0b0_last_k(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        # For last k (odd): only load one bf16 value
        code_str = f""
        code_str += f"zip1      p4.h, {pga}.h, {pga}.h\n"
        code_str += f"ld1h      z26.h, {pga}/z, [{pA0}]\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {a0}.h, z26.h, z30.h\n"

        code_str += f"zip1      p6.h, {pgb}.h, {pgb}.h\n"
        code_str += f"ld1h      z28.h, {pgb}/z, [{pB0}]\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {b0}.h, z28.h, z30.h\n"
        return code_str

    def load_a1(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            if pg == 'p1':
                dst = 'p4'
            else:
                dst = 'p5'
            code_str += f"rdvl      {TMP_PTR2}, #1\n"
            code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
            code_str += f"ld1h      z26.h, {pg}/z, [{pA0}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"ld1h      z29.h, {pg}/z, [{TMP_PTR}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"zip1      {a1}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldaopt}     {a1}{get_ld_element_suffix()}, {pg}, [{pA0}, #1, MUL VL]\n"
        return code_str

    def load_a1_last_k(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if pg == 'p1':
            dst = 'p4'
        else:
            dst = 'p5'
        code_str += f"rdvl      {TMP_PTR2}, #1\n"
        code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
        code_str += f"ld1h      z26.h, {pg}/z, [{pA0}, {TMP_PTR2}, LSL #1]\n"
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {a1}.h, z26.h, z30.h\n"
        return code_str

    def load_a2(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            if pg == 'p1':
                dst = 'p4'
            else:
                dst = 'p5'
            code_str += f"rdvl      {TMP_PTR2}, #1\n"
            code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #1\n"
            code_str += f"ld1h      z26.h, {pg}/z, [{pA0}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"ld1h      z29.h, {pg}/z, [{TMP_PTR}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"zip1      {a2}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldaopt}     {a2}{get_ld_element_suffix()}, {pg}, [{pA0}, #2, MUL VL]\n"
        return code_str

    def load_a2_last_k(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if pg == 'p1':
            dst = 'p4'
        else:
            dst = 'p5'
        code_str += f"rdvl      {TMP_PTR2}, #1\n"
        code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #1\n"
        code_str += f"ld1h      z26.h, {pg}/z, [{pA0}, {TMP_PTR2}, LSL #1]\n"
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {a2}.h, z26.h, z30.h\n"
        return code_str

    def load_a3(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            if pg == 'p1':
                dst = 'p4'
            else:
                dst = 'p5'
            code_str += f"rdvl      {TMP_PTR2}, #1\n"
            code_str += f"add       {TMP_PTR2}, {TMP_PTR2}, {TMP_PTR2}, LSL #1\n"
            code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
            code_str += f"ld1h      z26.h, {pg}/z, [{pA0}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"ld1h      z29.h, {pg}/z, [{TMP_PTR}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"zip1      {a3}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldaopt}     {a3}{get_ld_element_suffix()}, {pg}, [{pA0}, #3, MUL VL]\n"
        return code_str

    def load_a3_last_k(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if pg == 'p1':
            dst = 'p4'
        else:
            dst = 'p5'
        code_str += f"rdvl      {TMP_PTR2}, #1\n"
        code_str += f"add       {TMP_PTR2}, {TMP_PTR2}, {TMP_PTR2}, LSL #1\n"
        code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
        code_str += f"ld1h      z26.h, {pg}/z, [{pA0}, {TMP_PTR2}, LSL #1]\n"
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {a3}.h, z26.h, z30.h\n"
        return code_str

    def load_b1(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            if pg == 'p0':
                dst = 'p6'
            else:
                dst = 'p3'
            code_str += f"rdvl      {TMP_PTR2}, #1\n"
            code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
            code_str += f"ld1h      z28.h, {pg}/z, [{pB0}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"ld1h      z30.h, {pg}/z, [{TMP_PTR1}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"zip1      {b1}.h, z28.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b1}{get_ld_element_suffix()}, {pg}, [{pB0}, #1, MUL VL]\n"
        return code_str

    def load_b1_last_k(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if pg == 'p0':
            dst = 'p6'
        else:
            dst = 'p3'
        code_str += f"rdvl      {TMP_PTR2}, #1\n"
        code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
        code_str += f"ld1h      z28.h, {pg}/z, [{pB0}, {TMP_PTR2}, LSL #1]\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"zip1      {b1}.h, z28.h, z30.h\n"
        return code_str

    def load_b2(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            if pg == 'p0':
                dst = 'p6'
            else:
                dst = 'p3'
            code_str += f"rdvl      {TMP_PTR2}, #1\n"
            code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #1\n"
            code_str += f"ld1h      z28.h, {pg}/z, [{pB0}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"ld1h      z30.h, {pg}/z, [{TMP_PTR1}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"zip1      {b2}.h, z28.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b2}{get_ld_element_suffix()}, {pg}, [{pB0}, #2, MUL VL]\n"
        return code_str

    def load_b2_last_k(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if pg == 'p0':
            dst = 'p6'
        else:
            dst = 'p3'
        code_str += f"rdvl      {TMP_PTR2}, #1\n"
        code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #1\n"
        code_str += f"ld1h      z28.h, {pg}/z, [{pB0}, {TMP_PTR2}, LSL #1]\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"zip1      {b2}.h, z28.h, z30.h\n"
        return code_str

    def load_b3(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            if pg == 'p0':
                dst = 'p6'
            else:
                dst = 'p3'
            code_str += f"rdvl      {TMP_PTR2}, #1\n"
            code_str += f"add       {TMP_PTR2}, {TMP_PTR2}, {TMP_PTR2}, LSL #1\n"
            code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
            code_str += f"ld1h      z28.h, {pg}/z, [{pB0}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"ld1h      z30.h, {pg}/z, [{TMP_PTR1}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"zip1      {b3}.h, z28.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b3}{get_ld_element_suffix()}, {pg}, [{pB0}, #3, MUL VL]\n"
        return code_str

    def load_b3_last_k(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if pg == 'p0':
            dst = 'p6'
        else:
            dst = 'p3'
        code_str += f"rdvl      {TMP_PTR2}, #1\n"
        code_str += f"add       {TMP_PTR2}, {TMP_PTR2}, {TMP_PTR2}, LSL #1\n"
        code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
        code_str += f"ld1h      z28.h, {pg}/z, [{pB0}, {TMP_PTR2}, LSL #1]\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"zip1      {b3}.h, z28.h, z30.h\n"
        return code_str

    def set_svindex():
        # nopackA nopackB
        # Both A and B are consecutive load, no index needed
        code_str = f""

        return code_str

    def kernel_mm_loop_n_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov     {pAt}, {origPA}\n"
        code_str += f"mov     {pB0}, {pBt}\n"

        return code_str

    def kernel_mm_loop_n_post_func():
        # nopackA nopackB
        code_str = f""
        shift = get_element_size_shift()
        code_str += f"add     {pBt}, {pBt}, {MIN_N}, lsl #{shift}\n"
        return code_str

    def kernel_mm_loop_m_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov      {pB0}, {pBt}\n"
        code_str += f"mov      {pA0}, {pAt}\n"
        code_str += f"mov      {OFFSET_A}, {LDA}\n"
        code_str += f"mov      {OFFSET_B}, {LDB}\n"

        return code_str

    def kernel_mm_loop_m_post_func():
        # nopackA nopackB
        code_str = f""
        shift = get_element_size_shift()
        code_str += f"add      {pAt}, {pAt}, {MIN_M}, lsl #{shift}\n"
        return code_str

class small_gemm_tn_def:
    def load_a0b0(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        # TN: A is T (transposed) -> gather load with lda stride
        # TN: B is N (not transposed) -> gather load with ldb stride
        code_str = f""
        if is_bf16():
            code_str += f"mov       z26.b, #0\n"
            code_str += f"mov       z29.b, #0\n"
            code_str += f"zip1      p4.h, p1.h, p1.h\n"
            code_str += f"ld1h      z26.s, p4/z, [{pA0}, z28.s, UXTW]\n"
            code_str += f"add       {TMP_PTR}, {pA0}, #2\n"
            code_str += f"ld1h      z29.s, p4/z, [{TMP_PTR}, z28.s, UXTW]\n"
            code_str += f"uzp1      z26.h, z26.h, z26.h\n"
            code_str += f"uzp1      z29.h, z29.h, z29.h\n"
            code_str += f"zip1      {a0}.h, z26.h, z29.h\n"

            code_str += f"mov       z26.b, #0\n"
            code_str += f"mov       z29.b, #0\n"
            code_str += f"zip1      p6.h, {pgb}.h, {pgb}.h\n"
            code_str += f"ld1h      z26.s, p6/z, [{pB0}, z27.s, UXTW]\n"
            code_str += f"add       {TMP_PTR1}, {pB0}, #2\n"
            code_str += f"ld1h      z29.s, p6/z, [{TMP_PTR1}, z27.s, UXTW]\n"
            code_str += f"uzp1      z26.h, z26.h, z26.h\n"
            code_str += f"uzp1      z29.h, z29.h, z29.h\n"
            code_str += f"zip1      {b0}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldopt}      {b0}{get_ld_element_suffix()}, {pgb}, [{pB0}, z27.s, UXTW]\n"
            code_str += f"{ldopt}      {a0}{get_ld_element_suffix()}, {pga}, [{pA0}, z28.s, UXTW]\n"
        return code_str

    def load_a0b0_last_k(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        # TN: A is T (transposed) -> gather load with lda stride
        # TN: B is N (not transposed) -> gather load with ldb stride
        # For last k (odd): only load one bf16 value
        code_str = f""
        code_str += f"zip1      p4.h, p1.h, p1.h\n"
        code_str += f"ld1h      {a0}.s, p4/z, [{pA0}, z28.s, UXTW]\n"
        code_str += f"zip1      p6.h, {pgb}.h, {pgb}.h\n"
        code_str += f"ld1h      {b0}.s, p6/z, [{pB0}, z27.s, UXTW]\n"
        return code_str

    def load_a1(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        if is_bf16():
            if pg == 'p1':
                dst = 'p6'
            else:
                dst = 'p5'
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"ld1h      z26.s, {dst}/z, [{pAn}, z28.s, UXTW]\n"
            code_str += f"add       {TMP_PTR2}, {pAn}, #2\n"
            code_str += f"ld1h      z29.s, {dst}/z, [{TMP_PTR2}, z28.s, UXTW]\n"
            code_str += f"uzp1      z26.h, z26.h, z26.h\n"
            code_str += f"uzp1      z29.h, z29.h, z29.h\n"
            code_str += f"zip1      {a1}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldopt}      {a1}{get_ld_element_suffix()}, {pg}, [{pAn}, z28.s, UXTW]\n"
        return code_str
    
    def load_a1_last_k(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        if pg == 'p1':
            dst = 'p6'
        else:
            dst = 'p5'
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"ld1h      {a1}.s, {dst}/z, [{pAn}, z28.s, UXTW]\n"
        return code_str

    def load_a2(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}, lsl #1\n"
        if is_bf16():
            if pg == 'p1':
                dst = 'p6'
            else:
                dst = 'p5'
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"ld1h      z26.s, {dst}/z, [{pAn}, z28.s, UXTW]\n"
            code_str += f"add       {TMP_PTR2}, {pAn}, #2\n"
            code_str += f"ld1h      z29.s, {dst}/z, [{TMP_PTR2}, z28.s, UXTW]\n"
            code_str += f"uzp1      z26.h, z26.h, z26.h\n"
            code_str += f"uzp1      z29.h, z29.h, z29.h\n"
            code_str += f"zip1      {a2}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldopt}      {a2}{get_ld_element_suffix()}, {pg}, [{pAn}, z28.s, UXTW]\n"
        return code_str

    def load_a2_last_k(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}, lsl #1\n"
        if pg == 'p1':
            dst = 'p6'
        else:
            dst = 'p5'
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"ld1h      {a2}.s, {dst}/z, [{pAn}, z28.s, UXTW]\n"
        return code_str

    def load_a3(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        code_str += f"add          {pAn}, {pAn}, {pA_OFFSET}, lsl #1\n"
        if is_bf16():
            if pg == 'p1':
                dst = 'p6'
            else:
                dst = 'p5'
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"ld1h      z26.s, {dst}/z, [{pAn}, z28.s, UXTW]\n"
            code_str += f"add       {TMP_PTR2}, {pAn}, #2\n"
            code_str += f"ld1h      z29.s, {dst}/z, [{TMP_PTR2}, z28.s, UXTW]\n"
            code_str += f"uzp1      z26.h, z26.h, z26.h\n"
            code_str += f"uzp1      z29.h, z29.h, z29.h\n"
            code_str += f"zip1      {a3}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldopt}      {a3}{get_ld_element_suffix()}, {pg}, [{pAn}, z28.s, UXTW]\n"
        return code_str

    def load_a3_last_k(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        code_str += f"add          {pAn}, {pAn}, {pA_OFFSET}, lsl #1\n"
        if pg == 'p1':
            dst = 'p6'
        else:
            dst = 'p5'
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"ld1h      z26.s, {dst}/z, [{pAn}, z28.s, UXTW]\n"
        code_str += f"add       {TMP_PTR2}, {pAn}, #2\n"
        code_str += f"ld1h      z29.s, {dst}/z, [{TMP_PTR2}, z28.s, UXTW]\n"
        code_str += f"uzp1      z26.h, z26.h, z26.h\n"
        code_str += f"uzp1      z29.h, z29.h, z29.h\n"
        code_str += f"zip1      {a3}.h, z26.h, z29.h\n"
        return code_str

    def load_b1(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}\n"
        if is_bf16():
            if pg == 'p0':
                dst = 'p6'
            else:
                dst = 'p3'
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"ld1h      z26.s, {dst}/z, [{pBn}, z27.s, UXTW]\n"
            code_str += f"add       {TMP_PTR2}, {pBn}, #2\n"
            code_str += f"ld1h      z29.s, {dst}/z, [{TMP_PTR2}, z27.s, UXTW]\n"
            code_str += f"uzp1      z26.h, z26.h, z26.h\n"
            code_str += f"uzp1      z29.h, z29.h, z29.h\n"
            code_str += f"zip1      {b1}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldopt}      {b1}{get_ld_element_suffix()}, {pg}, [{pBn}, z27.s, UXTW]\n"
        return code_str

    def load_b1_last_k(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}\n"
        if pg == 'p0':
            dst = 'p6'
        else:
            dst = 'p3'
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"ld1h      {b1}.s, {dst}/z, [{pBn}, z27.s, UXTW]\n"
        return code_str

    def load_b2(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}, lsl #1\n"
        if is_bf16():
            if pg == 'p0':
                dst = 'p6'
            else:
                dst = 'p3'
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"ld1h      z26.s, {dst}/z, [{pBn}, z27.s, UXTW]\n"
            code_str += f"add       {TMP_PTR2}, {pBn}, #2\n"
            code_str += f"ld1h      z29.s, {dst}/z, [{TMP_PTR2}, z27.s, UXTW]\n"
            code_str += f"uzp1      z26.h, z26.h, z26.h\n"
            code_str += f"uzp1      z29.h, z29.h, z29.h\n"
            code_str += f"zip1      {b2}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldopt}      {b2}{get_ld_element_suffix()}, {pg}, [{pBn}, z27.s, UXTW]\n"
        return code_str

    def load_b2_last_k(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}, lsl #1\n"
        if pg == 'p0':
            dst = 'p6'
        else:
            dst = 'p3'
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"ld1h      z26.s, {dst}/z, [{pBn}, z27.s, UXTW]\n"
        code_str += f"add       {TMP_PTR2}, {pBn}, #2\n"
        code_str += f"ld1h      z29.s, {dst}/z, [{TMP_PTR2}, z27.s, UXTW]\n"
        code_str += f"uzp1      z26.h, z26.h, z26.h\n"
        code_str += f"uzp1      z29.h, z29.h, z29.h\n"
        code_str += f"zip1      {b2}.h, z26.h, z29.h\n"
        return code_str

    def load_b3(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}\n"
        code_str += f"add          {pBn}, {pBn}, {pB_OFFSET}, lsl #1\n"
        if is_bf16():
            if pg == 'p0':
                dst = 'p6'
            else:
                dst = 'p3'
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"ld1h      z26.s, {dst}/z, [{pBn}, z27.s, UXTW]\n"
            code_str += f"add       {TMP_PTR2}, {pBn}, #2\n"
            code_str += f"ld1h      z29.s, {dst}/z, [{TMP_PTR2}, z27.s, UXTW]\n"
            code_str += f"uzp1      z26.h, z26.h, z26.h\n"
            code_str += f"uzp1      z29.h, z29.h, z29.h\n"
            code_str += f"zip1      {b3}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldopt}      {b3}{get_ld_element_suffix()}, {pg}, [{pBn}, z27.s, UXTW]\n"
        return code_str

    def load_b3_last_k(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pBn}, {pB0}, {pB_OFFSET}\n"
        code_str += f"add          {pBn}, {pBn}, {pB_OFFSET}, lsl #1\n"
        if pg == 'p0':
            dst = 'p6'
        else:
            dst = 'p3'
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"ld1h      z26.s, {dst}/z, [{pBn}, z27.s, UXTW]\n"
        code_str += f"add       {TMP_PTR2}, {pBn}, #2\n"
        code_str += f"ld1h      z29.s, {dst}/z, [{TMP_PTR2}, z27.s, UXTW]\n"
        code_str += f"uzp1      z26.h, z26.h, z26.h\n"
        code_str += f"uzp1      z29.h, z29.h, z29.h\n"
        code_str += f"zip1      {b3}.h, z26.h, z29.h\n"
        return code_str

    def set_svindex():
        # nopackA nopackB
        # Both A and B need gather load
        code_str = f""
        shift = get_element_size_shift()
        code_str += f"lsl     {TMP_CNT}, {LDA}, #{shift}\n"
        code_str += f"mov     z28.s, #0\n"
        code_str += f"index   z28.s, #0, {TMP_CNT_SIN}\n"
        code_str += f"lsl     {TMP_CNT}, {LDB}, #{shift}\n"
        code_str += f"mov     z27.s, #0\n"
        code_str += f"index   z27.s, #0, {TMP_CNT_SIN}\n"

        return code_str

    def kernel_mm_loop_n_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov     {pAt}, {origPA}\n"
        code_str += f"mov     {pB0}, {pBt}\n"

        return code_str

    def kernel_mm_loop_n_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mul     {TMP_CNT}, {LDB}, {MIN_N}\n"
        shift = get_element_size_shift()
        code_str += f"add     {pBt}, {pBt}, {TMP_CNT}, lsl #{shift}\n"
        return code_str

    def kernel_mm_loop_m_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov      {pB0}, {pBt}\n"
        code_str += f"mov      {pA0}, {pAt}\n"
        shift = get_element_size_shift()
        code_str += f"lsl      {pA_OFFSET}, {LDA}, #{6 + shift - 2}\n"
        code_str += f"lsl      {pB_OFFSET}, {LDB}, #{6 + shift - 2}\n"
        code_str += f"mov      {OFFSET_A}, #1\n"
        code_str += f"mov      {OFFSET_B}, #1\n"

        return code_str

    def kernel_mm_loop_m_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mul     {TMP_CNT}, {LDA}, {MIN_M}\n"
        shift = get_element_size_shift()
        code_str += f"add      {pAt}, {pAt}, {TMP_CNT}, lsl #{shift}\n"
        return code_str

class small_gemm_tt_def:
    def load_a0b0(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        # TT: A is T (transposed) -> gather load with lda stride
        # TT: B is T (transposed) -> consecutive load
        code_str = f""
        if is_bf16():
            code_str += f"zip1      p4.h, p1.h, p1.h\n"
            code_str += f"ld1h      z26.s, p4/z, [{pA0}, z28.s, UXTW]\n"
            code_str += f"add       {TMP_PTR}, {pA0}, #2\n"
            code_str += f"ld1h      z29.s, p4/z, [{TMP_PTR}, z28.s, UXTW]\n"
            code_str += f"uzp1      z26.h, z26.h, z26.h\n"
            code_str += f"uzp1      z29.h, z29.h, z29.h\n"
            code_str += f"zip1      {a0}.h, z26.h, z29.h\n"

            code_str += f"zip1      p6.h, {pgb}.h, {pgb}.h\n"
            code_str += f"ld1h      z27.h, {pgb}/z, [{pB0}]\n"
            code_str += f"add       {TMP_PTR1}, {pB0}, {LDB}, LSL #1\n"
            code_str += f"ld1h      z30.h, {pgb}/z, [{TMP_PTR1}]\n"
            code_str += f"zip1      {b0}.h, z27.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b0}{get_ld_element_suffix()}, {pgb}, [{pB0}]\n"
            code_str += f"{ldopt}      {a0}{get_ld_element_suffix()}, {pga}, [{pA0}, z28.s, UXTW]\n"
        return code_str
    
    def load_a0b0_last_k(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        # TN: A is T (transposed) -> gather load with lda stride
        # TN: B is N (not transposed) -> gather load with ldb stride
        # For last k (odd): only load one bf16 value
        code_str = f""
        code_str += f"zip1      p4.h, p1.h, p1.h\n"
        code_str += f"ld1h      {a0}.s, p4/z, [{pA0}, z28.s, UXTW]\n"
        code_str += f"zip1      p6.h, {pgb}.h, {pgb}.h\n"
        code_str += f"ld1h      {b0}.h, {pgb}/z, [{pB0}]\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {b0}.h, {b0}.h, z30.h\n"
        return code_str

    def load_a1(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        if is_bf16():
            if pg == 'p1':
                dst = 'p4'
            else:
                dst = 'p5'
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"ld1h      z26.s, {dst}/z, [{pAn}, z28.s, UXTW]\n"
            code_str += f"add       {TMP_PTR2}, {pAn}, #2\n"
            code_str += f"ld1h      z29.s, {dst}/z, [{TMP_PTR2}, z28.s, UXTW]\n"
            code_str += f"uzp1      z26.h, z26.h, z26.h\n"
            code_str += f"uzp1      z29.h, z29.h, z29.h\n"
            code_str += f"zip1      {a1}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldopt}      {a1}{get_ld_element_suffix()}, {pg}, [{pAn}, z28.s, UXTW]\n"
        return code_str
    
    def load_a1_last_k(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        if pg == 'p1':
            dst = 'p4'
        else:
            dst = 'p5'
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"ld1h      {a1}.s, {dst}/z, [{pAn}, z28.s, UXTW]\n"
        return code_str

    def load_a2(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        code_str += f"add          {pAn}, {pAn}, {pA_OFFSET}\n"
        if is_bf16():
            if pg == 'p1':
                dst = 'p4'
            else:
                dst = 'p5'
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"ld1h      z26.s, {dst}/z, [{pAn}, z28.s, UXTW]\n"
            code_str += f"add       {TMP_PTR2}, {pAn}, #2\n"
            code_str += f"ld1h      z29.s, {dst}/z, [{TMP_PTR2}, z28.s, UXTW]\n"
            code_str += f"uzp1      z26.h, z26.h, z26.h\n"
            code_str += f"uzp1      z29.h, z29.h, z29.h\n"
            code_str += f"zip1      {a2}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldopt}      {a2}{get_ld_element_suffix()}, {pg}, [{pAn}, z28.s, UXTW]\n"
        return code_str

    def load_a2_last_k(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        code_str += f"add          {pAn}, {pAn}, {pA_OFFSET}\n"
        if pg == 'p1':
            dst = 'p4'
        else:
            dst = 'p5'
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"ld1h      {a2}.s, {dst}/z, [{pAn}, z28.s, UXTW]\n"
        return code_str

    def load_a3(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        code_str += f"add          {pAn}, {pAn}, {pA_OFFSET}\n"
        code_str += f"add          {pAn}, {pAn}, {pA_OFFSET}\n"
        if is_bf16():
            if pg == 'p1':
                dst = 'p4'
            else:
                dst = 'p5'
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"ld1h      z26.s, {dst}/z, [{pAn}, z28.s, UXTW]\n"
            code_str += f"add       {TMP_PTR2}, {pAn}, #2\n"
            code_str += f"ld1h      z29.s, {dst}/z, [{TMP_PTR2}, z28.s, UXTW]\n"
            code_str += f"uzp1      z26.h, z26.h, z26.h\n"
            code_str += f"uzp1      z29.h, z29.h, z29.h\n"
            code_str += f"zip1      {a3}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldopt}      {a3}{get_ld_element_suffix()}, {pg}, [{pAn}, z28.s, UXTW]\n"
        return code_str
    
    def load_a3_last_k(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        code_str += f"add          {pAn}, {pA0}, {pA_OFFSET}\n"
        code_str += f"add          {pAn}, {pAn}, {pA_OFFSET}\n"
        code_str += f"add          {pAn}, {pAn}, {pA_OFFSET}\n"
        if pg == 'p1':
            dst = 'p4'
        else:
            dst = 'p5'
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"ld1h      {a3}.s, {dst}/z, [{pAn}, z28.s, UXTW]\n"
        return code_str

    def load_b1(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            if pg == 'p0':
                dst = 'p6'
            else:
                dst = 'p3'
            code_str += f"rdvl      {TMP_PTR2}, #1\n"
            code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
            code_str += f"ld1h      z27.h, {pg}/z, [{pB0}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"ld1h      z30.h, {pg}/z, [{TMP_PTR1}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"zip1      {b1}.h, z27.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b1}{get_ld_element_suffix()}, {pg}, [{pB0}, #1, MUL VL]\n"
        return code_str
    
    def load_b1_last_k(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if pg == 'p0':
            dst = 'p6'
        else:
            dst = 'p3'
        code_str += f"rdvl      {TMP_PTR2}, #1\n"
        code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
        code_str += f"ld1h      z27.h, {pg}/z, [{pB0}, {TMP_PTR2}, LSL #1]\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"zip1      {b1}.h, z27.h, z30.h\n"
        return code_str

    def load_b2(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            if pg == 'p0':
                dst = 'p6'
            else:
                dst = 'p3'
            code_str += f"rdvl      {TMP_PTR2}, #1\n"
            code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #1\n"
            code_str += f"ld1h      z27.h, {pg}/z, [{pB0}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"ld1h      z30.h, {pg}/z, [{TMP_PTR1}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"zip1      {b2}.h, z27.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b2}{get_ld_element_suffix()}, {pg}, [{pB0}, #2, MUL VL]\n"
        return code_str

    def load_b2_last_k(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if pg == 'p0':
            dst = 'p6'
        else:
            dst = 'p3'
        code_str += f"rdvl      {TMP_PTR2}, #1\n"
        code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #1\n"
        code_str += f"ld1h      z27.h, {pg}/z, [{pB0}, {TMP_PTR2}, LSL #1]\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"zip1      {b2}.h, z27.h, z30.h\n"
        return code_str

    def load_b3(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            if pg == 'p0':
                dst = 'p6'
            else:
                dst = 'p3'
            code_str += f"rdvl      {TMP_PTR2}, #1\n"
            code_str += f"add       {TMP_PTR2}, {TMP_PTR2}, {TMP_PTR2}, LSL #1\n"
            code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
            code_str += f"ld1h      z27.h, {pg}/z, [{pB0}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"ld1h      z30.h, {pg}/z, [{TMP_PTR1}, {TMP_PTR2}, LSL #1]\n"
            code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
            code_str += f"zip1      {b3}.h, z27.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b3}{get_ld_element_suffix()}, {pg}, [{pB0}, #3, MUL VL]\n"
        return code_str

    def load_b3_last_k(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if pg == 'p0':
            dst = 'p6'
        else:
            dst = 'p3'
        code_str += f"rdvl      {TMP_PTR2}, #1\n"
        code_str += f"add       {TMP_PTR2}, {TMP_PTR2}, {TMP_PTR2}, LSL #1\n"
        code_str += f"lsr       {TMP_PTR2}, {TMP_PTR2}, #2\n"
        code_str += f"ld1h      z27.h, {pg}/z, [{pB0}, {TMP_PTR2}, LSL #1]\n"
        code_str += f"mov       z30.h, #0\n"
        code_str += f"zip1      {dst}.h, {pg}.h, {pg}.h\n"
        code_str += f"zip1      {b3}.h, z27.h, z30.h\n"
        return code_str

    def set_svindex():
        # nopackA nopackB
        # A needs gather load with lda stride
        code_str = f""
        shift = get_element_size_shift()
        code_str += f"lsl     {TMP_CNT}, {LDA}, #{shift}\n"
        code_str += f"mov     z28.s, #0\n"
        code_str += f"index   z28.s, #0, {TMP_CNT_SIN}\n"

        return code_str

    def kernel_mm_loop_n_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov     {pAt}, {origPA}\n"
        code_str += f"mov     {pB0}, {pBt}\n"

        return code_str

    def kernel_mm_loop_n_post_func():
        # nopackA nopackB
        code_str = f""
        shift = get_element_size_shift()
        code_str += f"add     {pBt}, {pBt}, {MIN_N}, lsl #{shift}\n"
        return code_str

    def kernel_mm_loop_m_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov      {pB0}, {pBt}\n"
        code_str += f"mov      {pA0}, {pAt}\n"
        shift = get_element_size_shift()
        code_str += f"lsl      {pA_OFFSET}, {LDA}, #{6 + shift - 2}\n"
        code_str += f"mov      {OFFSET_A}, #1\n"
        code_str += f"mov      {OFFSET_B}, {LDB}\n"

        return code_str

    def kernel_mm_loop_m_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mul     {TMP_CNT}, {LDA}, {MIN_M}\n"
        shift = get_element_size_shift()
        code_str += f"add      {pAt}, {pAt}, {TMP_CNT}, lsl #{shift}\n"
        return code_str

class general_gemm_def:
    def load_a0b0(a0, pga, b0, pgb, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            code_str += f"ld1h      z26.h, {pga}, [{pA0}]\n"
            code_str += f"ld1h      z29.h, {pga}, [{pA0}, #1, MUL VL]\n"
            code_str += f"zip1      {a0}.h, z26.h, z29.h\n"
            code_str += f"ld1h      z26.h, {pgb}, [{pB0}]\n"
            code_str += f"ld1h      z30.h, {pgb}, [{pB0}, #1, MUL VL]\n"
            code_str += f"zip1      {b0}.h, z26.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b0}{get_ld_element_suffix()}, {pgb}, [{pB0}]\n"
            code_str += f"{ldaopt}     {a0}{get_ld_element_suffix()}, {pga}, [{pA0}]\n"
        return code_str

    def load_a1(a1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            code_str += f"ld1h     z26.h, {pg}, [{pA0}, #2, MUL VL]\n"
            code_str += f"ld1h     z29.h, {pg}, [{pA0}, #3, MUL VL]\n"
            code_str += f"zip1     {a1}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldaopt}     {a1}{get_ld_element_suffix()}, {pg}, [{pA0}, #1, MUL VL]\n"
        return code_str

    def load_a2(a2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            code_str += f"ld1h     z26.h, {pg}, [{pA0}, #4, MUL VL]\n"
            code_str += f"ld1h     z29.h, {pg}, [{pA0}, #5, MUL VL]\n"
            code_str += f"zip1     {a2}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldaopt}     {a2}{get_ld_element_suffix()}, {pg}, [{pA0}, #2, MUL VL]\n"
        return code_str

    def load_a3(a3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            code_str += f"ld1h     z26.h, {pg}, [{pA0}, #6, MUL VL]\n"
            code_str += f"ld1h     z29.h, {pg}, [{pA0}, #7, MUL VL]\n"
            code_str += f"zip1     {a3}.h, z26.h, z29.h\n"
        else:
            code_str += f"{ldaopt}     {a3}{get_ld_element_suffix()}, {pg}, [{pA0}, #3, MUL VL]\n"
        return code_str

    def load_b1(b1, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            code_str += f"ld1h      z26.h, {pg}, [{pB0}, #2, MUL VL]\n"
            code_str += f"ld1h      z30.h, {pg}, [{pB0}, #3, MUL VL]\n"
            code_str += f"zip1      {b1}.h, z26.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b1}{get_ld_element_suffix()}, {pg}, [{pB0}, #1, MUL VL]\n"
        return code_str

    def load_b2(b2, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            code_str += f"ld1h      z26.h, {pg}, [{pB0}, #4, MUL VL]\n"
            code_str += f"ld1h      z30.h, {pg}, [{pB0}, #5, MUL VL]\n"
            code_str += f"zip1      {b2}.h, z26.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b2}{get_ld_element_suffix()}, {pg}, [{pB0}, #2, MUL VL]\n"
        return code_str

    def load_b3(b3, pg, ldopt=LD1, ldaopt=LD1):
        # nopackA nopackB
        code_str = f""
        if is_bf16():
            code_str += f"ld1h      z26.h, {pg}, [{pB0}, #6, MUL VL]\n"
            code_str += f"ld1h      z30.h, {pg}, [{pB0}, #7, MUL VL]\n"
            code_str += f"zip1      {b3}.h, z26.h, z30.h\n"
        else:
            code_str += f"{ldopt}      {b3}{get_ld_element_suffix()}, {pg}, [{pB0}, #3, MUL VL]\n"
        return code_str

    def set_svindex():
        # nopackA nopackB
        code_str = f""

        return code_str

    def kernel_mm_loop_n_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov     {pAt}, {origPA}\n"
        code_str += f"mov     {pB0}, {pBt}\n"

        return code_str

    def kernel_mm_loop_n_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov     {pBt}, {pB0}\n"

        return code_str

    def kernel_mm_loop_m_pre_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"mov      {pB0}, {pBt}\n"
        code_str += f"mov      {pA0}, {pAt}\n"

        return code_str

    def kernel_mm_loop_m_post_func():
        # nopackA nopackB
        code_str = f""
        code_str += f"add      {pAt}, {pA0}\n"

        return code_str