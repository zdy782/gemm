from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ParameterRegisters:
    origPA: str = "x0"
    origPB: str = "x1"
    pC: str = "x2"
    LDA: str = "x3"
    LDB: str = "x4"
    LDC: str = "x5"

    @property
    def orig_pa(self) -> str:
        return self.origPA

    @property
    def orig_pb(self) -> str:
        return self.origPB

    @property
    def p_c(self) -> str:
        return self.pC

    @property
    def lda(self) -> str:
        return self.LDA

    @property
    def ldb(self) -> str:
        return self.LDB

    @property
    def ldc(self) -> str:
        return self.LDC


@dataclass(frozen=True)
class DimensionRegisters:
    origM: str = "x6"
    origN: str = "x7"
    origK: str = "x8"
    MIN_N: str = "x9"
    MIN_M: str = "x10"

    @property
    def orig_m(self) -> str:
        return self.origM

    @property
    def orig_n(self) -> str:
        return self.origN

    @property
    def orig_k(self) -> str:
        return self.origK

    @property
    def min_n(self) -> str:
        return self.MIN_N

    @property
    def min_m(self) -> str:
        return self.MIN_M


@dataclass(frozen=True)
class CounterRegisters:
    counterJ: str = "x16"
    counterI: str = "x18"
    TMP_CNT: str = "x14"
    TMP_CNT_SIN: str = "w14"
    wbk: str = "x23"
    TMP_CNT_POST: str = "x23"

    @property
    def loop_j(self) -> str:
        return self.counterJ

    @property
    def loop_i(self) -> str:
        return self.counterI

    @property
    def tmp_count(self) -> str:
        return self.TMP_CNT

    @property
    def tmp_count_word(self) -> str:
        return self.TMP_CNT_SIN

    @property
    def work_blocks(self) -> str:
        return self.wbk

    @property
    def tmp_count_post(self) -> str:
        return self.TMP_CNT_POST


@dataclass(frozen=True)
class PointerRegisters:
    pA0: str = "x24"
    pAt: str = "x29"
    pAn: str = "x26"
    pB0: str = "x19"
    pBn: str = "x15"
    pBt: str = "x11"
    pC0: str = "x20"
    pC1: str = "x21"
    pC2: str = "x22"
    pC3: str = "x23"

    @property
    def a_current(self) -> str:
        return self.pA0

    @property
    def a_tile(self) -> str:
        return self.pAt

    @property
    def a_next(self) -> str:
        return self.pAn

    @property
    def b_current(self) -> str:
        return self.pB0

    @property
    def b_next(self) -> str:
        return self.pBn

    @property
    def b_tile(self) -> str:
        return self.pBt

    @property
    def c_row0(self) -> str:
        return self.pC0

    @property
    def c_row1(self) -> str:
        return self.pC1

    @property
    def c_row2(self) -> str:
        return self.pC2

    @property
    def c_row3(self) -> str:
        return self.pC3


@dataclass(frozen=True)
class AddressRegisters:
    TMP_PTR: str = "x17"
    TMP_PTR1: str = "x21"
    TMP_PTR2: str = "x28"
    OFFSET_A: str = "x12"
    OFFSET_B: str = "x13"
    pA_OFFSET: str = "x25"
    pB_OFFSET: str = "x27"

    @property
    def tmp_ptr(self) -> str:
        return self.TMP_PTR

    @property
    def tmp_ptr1(self) -> str:
        return self.TMP_PTR1

    @property
    def tmp_ptr2(self) -> str:
        return self.TMP_PTR2

    @property
    def offset_a(self) -> str:
        return self.OFFSET_A

    @property
    def offset_b(self) -> str:
        return self.OFFSET_B

    @property
    def a_offset(self) -> str:
        return self.pA_OFFSET

    @property
    def b_offset(self) -> str:
        return self.pB_OFFSET


@dataclass(frozen=True)
class PredicateRegisters:
    n_main: str = "p0"
    m_main: str = "p1"
    m_tail: str = "p2"
    n_tail: str = "p3"
    ext_m_main: str = "p4"
    ext_m_tail: str = "p5"
    ext_n_main: str = "p6"
    ext_n_tail: str = "p7"
    false_all: str = "p15"


@dataclass(frozen=True)
class VectorRegisters:
    b_index: str = "z27"
    a_index: str = "z28"
    a_low: str = "z26"
    pair_high: str = "z29"
    b_low: str = "z28"
    b_high: str = "z29"
    b_contiguous_low: str = "z27"
    save_tmp: str = "z25"
    save_tmp1: str = "z26"


@dataclass(frozen=True)
class SaveRegisters:
    base_indices: Tuple[str, str, str, str] = ("w12", "w13", "w14", "w15")


@dataclass(frozen=True)
class KernelVariantRegisters:
    a_regs: Tuple[str, str, str, str]
    b_regs: Tuple[str, str, str, str]


@dataclass(frozen=True)
class RegisterPlan:
    params: ParameterRegisters = field(default_factory=ParameterRegisters)
    dims: DimensionRegisters = field(default_factory=DimensionRegisters)
    counters: CounterRegisters = field(default_factory=CounterRegisters)
    pointers: PointerRegisters = field(default_factory=PointerRegisters)
    address: AddressRegisters = field(default_factory=AddressRegisters)
    predicates: PredicateRegisters = field(default_factory=PredicateRegisters)
    vectors: VectorRegisters = field(default_factory=VectorRegisters)
    save: SaveRegisters = field(default_factory=SaveRegisters)
    kernel_variants: Tuple[KernelVariantRegisters, ...] = (
        KernelVariantRegisters(("z0", "z1", "z2", "z3"), ("z16", "z17", "z24", "z25")),
        KernelVariantRegisters(("z4", "z5", "z6", "z7"), ("z18", "z19", "z30", "z31")),
        KernelVariantRegisters(("z8", "z9", "z10", "z11"), ("z20", "z21", "z24", "z25")),
        KernelVariantRegisters(("z12", "z13", "z14", "z15"), ("z22", "z23", "z30", "z31")),
    )

    def logical_predicate(self, role: str) -> str:
        return getattr(self.predicates, role)

    def ext_predicate(self, role: str) -> str:
        mapping = {
            "m_main": self.predicates.ext_m_main,
            "m_tail": self.predicates.ext_m_tail,
            "n_main": self.predicates.ext_n_main,
            "n_tail": self.predicates.ext_n_tail,
        }
        return mapping[role]

    def kernel_variant(self, idx: int) -> KernelVariantRegisters:
        return self.kernel_variants[idx]


DEFAULT_REGISTER_PLAN = RegisterPlan()
