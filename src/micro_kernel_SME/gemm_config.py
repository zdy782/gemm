from dataclasses import dataclass
from typing import Dict, Tuple

from gemm_type_impl import (
    general_model,
    small_nn_model,
    small_nt_model,
    small_tn_model,
    small_tt_model,
)
from model_spec import GemmType, KernelSpec, Transpose


@dataclass(frozen=True)
class ModelRegistry:
    registry: Dict[Tuple[GemmType, Transpose, Transpose], object]
    general_model: object

    def resolve(self, spec: KernelSpec):
        if spec.gemm_type is GemmType.GENERAL:
            return self.general_model
        key = (spec.gemm_type, spec.trans_a, spec.trans_b)
        if key not in self.registry:
            raise ValueError(
                f"Unsupported GEMM config: type={spec.gemm_type.value}, "
                f"transA={spec.transA}, transB={spec.transB}"
            )
        return self.registry[key]


DEFAULT_MODEL_REGISTRY = ModelRegistry(
    registry={
        (GemmType.SMALL, Transpose.NORMAL, Transpose.NORMAL): small_nn_model,
        (GemmType.SMALL, Transpose.NORMAL, Transpose.TRANSPOSED): small_nt_model,
        (GemmType.SMALL, Transpose.TRANSPOSED, Transpose.NORMAL): small_tn_model,
        (GemmType.SMALL, Transpose.TRANSPOSED, Transpose.TRANSPOSED): small_tt_model,
    },
    general_model=general_model,
)


def resolve_model(spec: KernelSpec):
    if spec.gemm_type is GemmType.SMALL:
        if spec.effective_a_contiguous() and spec.effective_b_contiguous():
            return small_nt_model
        if spec.effective_a_contiguous():
            return small_nn_model
        if spec.effective_b_contiguous():
            return small_tt_model
        return small_tn_model
    return DEFAULT_MODEL_REGISTRY.resolve(spec)
