from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from micro_kernel_SME.half.selector.features import (
    SelectorResult,
    build_features,
    combo_label,
    combo_to_pack_flags,
    combo_to_tile_vl,
)
from micro_kernel_SME.half.selector.tree import predict_classification


MODEL_PATH = Path(__file__).resolve().with_name("rules.py")


def _load_rules_module():
    spec = importlib.util.spec_from_file_location("micro_kernel_SME.half.selector.rules", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"selector rules file not found: {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("micro_kernel_SME.half.selector.rules", module)
    spec.loader.exec_module(module)
    return module


def load_model() -> Dict[str, object]:
    module = _load_rules_module()
    if not hasattr(module, "MODEL"):
        raise ValueError(f"selector rules file missing MODEL: {MODEL_PATH}")
    return module.MODEL


def predict_bf16_combo(
    M: int,
    N: int,
    K: int,
    transA: str,
    transB: str,
    model: Dict[str, object] | None = None,
) -> SelectorResult:
    model = load_model() if model is None else model
    features = build_features(M, N, K, transA, transB)
    pack_path: List[str] = []
    pack = predict_classification(model["pack_tree"], features, pack_path)
    tile_tree = model["tile_trees"].get(pack)
    tile_path: List[str] = []
    if tile_tree is None:
        tile = model["default_tile_by_pack"][pack]
        tile_path.append(f"leaf:{tile}")
    else:
        tile = predict_classification(tile_tree, features, tile_path)
    pack_flags = combo_to_pack_flags(pack)
    m_vl, n_vl = combo_to_tile_vl(tile)
    combo = combo_label(pack, tile)
    path = tuple([f"pack:{entry}" for entry in pack_path] + [f"tile:{entry}" for entry in tile_path])
    return SelectorResult(
        pack=pack,
        tile=tile,
        pack_a=pack_flags["pack_a"],
        pack_b=pack_flags["pack_b"],
        m_vl=m_vl,
        n_vl=n_vl,
        combo=combo,
        path=path,
    )


def predict_combo_flags(
    M: int,
    N: int,
    K: int,
    transA: str,
    transB: str,
    model: Dict[str, object] | None = None,
) -> tuple[bool, bool, int, int]:
    result = predict_bf16_combo(M, N, K, transA, transB, model=model)
    return result.pack_a, result.pack_b, result.m_vl, result.n_vl


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Predict the BF16 pack/tile combo for one GEMM shape.")
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--transA", type=str, required=True, choices=["N", "T", "n", "t"])
    parser.add_argument("--transB", type=str, required=True, choices=["N", "T", "n", "t"])
    parser.add_argument("--json", action="store_true", help="Print the prediction as JSON.")
    parser.add_argument("--show-path", action="store_true", help="Print the tree decision path.")
    return parser.parse_args()


def _result_dict(result: SelectorResult) -> Dict[str, object]:
    return {
        "pack": result.pack,
        "tile": result.tile,
        "pack_a": result.pack_a,
        "pack_b": result.pack_b,
        "m_vl": result.m_vl,
        "n_vl": result.n_vl,
        "combo": result.combo,
        "path": list(result.path),
    }


def main() -> None:
    args = _parse_args()
    result = predict_bf16_combo(
        args.M,
        args.N,
        args.K,
        args.transA.upper(),
        args.transB.upper(),
    )
    if args.json:
        print(json.dumps(_result_dict(result), indent=2))
        return
    print(f"combo: {result.combo}")
    print(f"pack: {result.pack} (pack_a={str(result.pack_a).lower()}, pack_b={str(result.pack_b).lower()})")
    print(f"tile: {result.tile} (m_vl={result.m_vl}, n_vl={result.n_vl})")
    if args.show_path and result.path:
        print(f"path: {' | '.join(result.path)}")


if __name__ == "__main__":
    main()
