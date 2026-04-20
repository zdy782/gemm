from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

PACK_LABELS = ("nopack", "packa", "packb", "packab")
LONG_TILE_LABELS = ("1VLx4VL", "2VLx2VL", "4VLx1VL")
SHORT_TILE_LABELS = ("1x4", "2x2", "4x1")
TRANS_PAIRS = ("NN", "NT", "TN", "TT")
SHAPE_TAGS = ("SLS", "LSS", "LLS", "verybigN", "verybigM", "verybigK")

LONG_TO_SHORT_TILE = {
    "1VLx4VL": "1x4",
    "2VLx2VL": "2x2",
    "4VLx1VL": "4x1",
}
SHORT_TO_LONG_TILE = {value: key for key, value in LONG_TO_SHORT_TILE.items()}
SHORT_TILE_TO_VL = {
    "1x4": (1, 4),
    "2x2": (2, 2),
    "4x1": (4, 1),
}
PACK_TO_FLAGS = {
    "nopack": {"pack_a": False, "pack_b": False},
    "packa": {"pack_a": True, "pack_b": False},
    "packb": {"pack_a": False, "pack_b": True},
    "packab": {"pack_a": True, "pack_b": True},
}


def normalize_trans(trans: str) -> str:
    value = str(trans).strip().upper()
    if value not in {"N", "T"}:
        raise ValueError(f"invalid transpose flag: {trans}")
    return value


def shape_tag(m: int, n: int, k: int) -> str:
    if k >= 256:
        return "verybigK"
    if m >= 500:
        return "verybigM"
    if n >= 500:
        return "verybigN"
    if m <= 48:
        return "SLS"
    if n <= 128:
        return "LSS"
    return "LLS"


def one_hot(prefix: str, value: str, values: Iterable[str]) -> Dict[str, int]:
    return {f"{prefix}_{item}": int(value == item) for item in values}


def build_features(m: int, n: int, k: int, trans_a: str, trans_b: str) -> Dict[str, float]:
    ta = normalize_trans(trans_a)
    tb = normalize_trans(trans_b)
    tag = shape_tag(m, n, k)
    features: Dict[str, float] = {
        "M": float(m),
        "N": float(n),
        "K": float(k),
        "taT": float(ta == "T"),
        "tbT": float(tb == "T"),
        "pair": float((ta == "T") * 2 + (tb == "T")),
        "M_over_N": float(m) / float(n),
        "N_over_M": float(n) / float(m),
        "M_ge_N": float(m >= n),
        "M_ge_2N": float(m >= 2 * n),
        "N_ge_2M": float(n >= 2 * m),
        "M_ge_4N": float(m >= 4 * n),
        "N_ge_4M": float(n >= 4 * m),
        "K_ge_64": float(k >= 64),
        "K_ge_128": float(k >= 128),
        "K_ge_256": float(k >= 256),
        "verybigN": float(n >= 500),
        "verybigM": float(m >= 500),
        "verybigK": float(k >= 256),
        "shape_code": float(SHAPE_TAGS.index(tag)),
    }
    features.update(one_hot("shape", tag, SHAPE_TAGS))
    return features


def combo_label(pack: str, tile: str) -> str:
    if pack not in PACK_LABELS:
        raise ValueError(f"unsupported pack label: {pack}")
    if tile not in SHORT_TILE_LABELS:
        raise ValueError(f"unsupported tile label: {tile}")
    return f"{pack}_{tile}"


def combo_to_parts(combo: str) -> tuple[str, str]:
    pack, tile = combo.split("_", 1)
    if pack not in PACK_LABELS:
        raise ValueError(f"invalid pack label in combo: {combo}")
    if tile not in SHORT_TILE_LABELS:
        raise ValueError(f"invalid tile label in combo: {combo}")
    return pack, tile


def metric_column_name(pack: str, tile_long: str) -> str:
    if pack == "nopack":
        return f"Mflops_autogemm_nopacking_{tile_long}"
    return f"Mflops_sbgemm_{pack}_{tile_long}"


def improvement_column_name(pack: str, tile_long: str) -> str:
    if pack == "nopack":
        return f"Improve_blas/autogemm_nopacking_{tile_long}"
    return f"Improve_blas/sbgemm_{pack}_{tile_long}"


def combo_to_pack_flags(pack: str) -> Dict[str, bool]:
    if pack not in PACK_TO_FLAGS:
        raise ValueError(f"unsupported pack label: {pack}")
    return dict(PACK_TO_FLAGS[pack])


def combo_to_tile_vl(tile: str) -> tuple[int, int]:
    if tile not in SHORT_TILE_TO_VL:
        raise ValueError(f"unsupported tile label: {tile}")
    return SHORT_TILE_TO_VL[tile]


@dataclass(frozen=True)
class SelectorResult:
    pack: str
    tile: str
    pack_a: bool
    pack_b: bool
    m_vl: int
    n_vl: int
    combo: str
    path: tuple[str, ...]
