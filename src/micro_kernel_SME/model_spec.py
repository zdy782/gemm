from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

# `KernelSpec` is the generator's immutable input contract. Everything below
# the CLI layer should read shape, transpose, precision, and tile choices from
# this object instead of carrying ad-hoc argument lists.

class Precision(str, Enum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"


class GemmType(str, Enum):
    SMALL = "small"
    GENERAL = "general"


class Transpose(str, Enum):
    NORMAL = "N"
    TRANSPOSED = "T"


@dataclass(frozen=True)
class TileShape:
    m_vl: int
    n_vl: int


@dataclass(frozen=True)
class KernelSpec:
    M: int
    N: int
    K: int
    lda: int
    ldb: int
    ldc: int
    gemm_type: GemmType
    trans_a: Transpose
    trans_b: Transpose
    precision: Precision
    tile: TileShape
    pack_a: bool = False
    pack_b: bool = False

    @classmethod
    def from_args(
        cls,
        M: int,
        N: int,
        K: int,
        lda: int,
        ldb: int,
        ldc: int,
        gemm_type: str,
        transA: str,
        transB: str,
        data_type: str,
        m_vl: int,
        n_vl: int,
        pack_a: bool = False,
        pack_b: bool = False,
    ) -> "KernelSpec":
        return cls(
            M=M,
            N=N,
            K=K,
            lda=lda,
            ldb=ldb,
            ldc=ldc,
            gemm_type=GemmType(gemm_type),
            trans_a=Transpose(transA),
            trans_b=Transpose(transB),
            precision=Precision(data_type),
            tile=TileShape(m_vl=m_vl, n_vl=n_vl),
            pack_a=pack_a,
            pack_b=pack_b,
        )

    @property
    def data_type(self) -> str:
        return self.precision.value

    @property
    def transA(self) -> str:
        return self.trans_a.value

    @property
    def transB(self) -> str:
        return self.trans_b.value

    def is_fp32(self) -> bool:
        return self.precision is Precision.FP32

    def is_bf16(self) -> bool:
        return self.precision is Precision.BF16

    def is_fp16(self) -> bool:
        return self.precision is Precision.FP16

    def is_ext_precision(self) -> bool:
        return self.precision in (Precision.BF16, Precision.FP16)

    def gemm_prefix(self) -> str:
        if self.is_bf16():
            return "sbgemm"
        if self.is_fp16():
            return "shgemm"
        return "sgemm"

    def pack_suffix(self) -> str:
        if self.pack_a and self.pack_b:
            return "packab"
        if self.pack_a:
            return "packa"
        if self.pack_b:
            return "packb"
        return "nopack"

    def effective_a_contiguous(self) -> bool:
        return self.pack_a or self.trans_a is Transpose.NORMAL

    def effective_b_contiguous(self) -> bool:
        return self.pack_b or self.trans_b is Transpose.TRANSPOSED

    def kernel_view_spec(self) -> "KernelSpec":
        if self.gemm_type is not GemmType.SMALL:
            return self
        trans_a = Transpose.NORMAL if self.pack_a else self.trans_a
        trans_b = Transpose.TRANSPOSED if self.pack_b else self.trans_b
        return replace(
            self,
            trans_a=trans_a,
            trans_b=trans_b,
        )

@dataclass(frozen=True)
class GenerationContext:
    spec: KernelSpec
    registers: Any
    model: Any

    # The context keeps the three axes of codegen state together:
    # - `spec`: what kernel to generate
    # - `registers`: which architectural names each layer should use
    # - `model`: how A/B are loaded for the selected transpose family

    def is_fp32(self) -> bool:
        return self.spec.is_fp32()

    def is_bf16(self) -> bool:
        return self.spec.is_bf16()

    def is_fp16(self) -> bool:
        return self.spec.is_fp16()

    def is_ext_precision(self) -> bool:
        return self.spec.is_ext_precision()

    def use_ext_paired_fast_path(self) -> bool:
        # The unified zip1+zip2 path is only meaningful for small ext-precision
        # kernels. `fp32` and `general` keep their own loading strategy.
        return self.spec.gemm_type is GemmType.SMALL and self.spec.is_ext_precision()
