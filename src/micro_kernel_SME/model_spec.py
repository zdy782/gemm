from dataclasses import dataclass
from enum import Enum
from typing import Any


class Precision(str, Enum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"


class GemmType(str, Enum):
    SMALL = "small"
    GENERAL = "general"


class ExtLoadStrategy(str, Enum):
    LEGACY_HALF_VL = "legacy_half_vl"
    EXPERIMENTAL = "experimental"


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
    ext_load_strategy: ExtLoadStrategy = ExtLoadStrategy.LEGACY_HALF_VL

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
        ext_load_strategy: str = "legacy_half_vl",
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
            ext_load_strategy=ExtLoadStrategy(ext_load_strategy),
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

    def is_legacy_ext_load(self) -> bool:
        return self.ext_load_strategy is ExtLoadStrategy.LEGACY_HALF_VL

    def is_experimental_ext_load(self) -> bool:
        return self.ext_load_strategy is ExtLoadStrategy.EXPERIMENTAL


@dataclass(frozen=True)
class GenerationContext:
    spec: KernelSpec
    registers: Any
    model: Any

    def is_fp32(self) -> bool:
        return self.spec.is_fp32()

    def is_bf16(self) -> bool:
        return self.spec.is_bf16()

    def is_fp16(self) -> bool:
        return self.spec.is_fp16()

    def is_ext_precision(self) -> bool:
        return self.spec.is_ext_precision()

    def is_legacy_ext_load(self) -> bool:
        return self.spec.is_legacy_ext_load()

    def is_experimental_ext_load(self) -> bool:
        return self.spec.is_experimental_ext_load()
