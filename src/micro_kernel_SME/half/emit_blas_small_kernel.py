"""Emit BLAS small-kernel base assembly files for SME BF16 nopack kernels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from micro_kernel_SME.half.generate_sme_test import generate_sme_asm


PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_DIR.parents[3]
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "blas" / "source" / "kernel" / "arm64" / "sme" / "gen"
PLACEHOLDER_DIM = 256

TILE_SPECS = {
    "1vlx4vl": (1, 4),
    "2vlx2vl": (2, 2),
    "4vlx1vl": (4, 1),
}
TRANSPOSE_PAIRS = (
    ("N", "N"),
    ("N", "T"),
    ("T", "N"),
    ("T", "T"),
)

BLAS_SMALL_PROLOGUE = (
    "#define ASSEMBLER\n"
    "#include \"common.h\"\n\n"
    "PROLOGUE\n"
    "#ifdef BETA0\n"
    "fmov    s1, wzr\n"
    "#endif\n"
    "ldr     x14, [sp]\n"
    "mov     x15, x0\n"
    "mov     x16, x1\n"
    "mov     x17, x2\n"
    "mov     x0, x3\n"
    "mov     x1, x5\n"
    "mov     x2, x7\n"
    "mov     x3, x4\n"
    "mov     x4, x6\n"
    "mov     x5, x14\n"
    "mov     x6, x15\n"
    "mov     x7, x16\n"
    "mov     x8, x17\n"
)


def _normalize_generated_asm(asm_text: str) -> str:
    lines = asm_text.splitlines()
    if len(lines) < 19:
        raise ValueError("generated assembly is unexpectedly short")
    if lines[18] != ".align 5":
        raise ValueError("generated assembly prologue layout changed unexpectedly")
    return BLAS_SMALL_PROLOGUE + "\n".join(lines[18:]) + "\n"


def _emit_one(output_root: Path, tile_dir: str, trans_a: str, trans_b: str) -> Path:
    m_vl, n_vl = TILE_SPECS[tile_dir]
    generated = generate_sme_asm(
        PLACEHOLDER_DIM,
        PLACEHOLDER_DIM,
        PLACEHOLDER_DIM,
        PLACEHOLDER_DIM,
        PLACEHOLDER_DIM,
        PLACEHOLDER_DIM,
        "small",
        trans_a,
        trans_b,
        "blas_small_emit",
        data_type="bf16",
        m_vl=m_vl,
        n_vl=n_vl,
        pack_a=False,
        pack_b=False,
    )
    normalized = _normalize_generated_asm(generated)
    suffix = f"{trans_a.lower()}{trans_b.lower()}"
    output_path = output_root / tile_dir / suffix / f"sbgemm_small_kernel_sme_{tile_dir}_{suffix}.S"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(normalized, encoding="utf-8")
    return output_path


def emit_blas_small_kernels(output_root: Path, tiles: Iterable[str]) -> list[Path]:
    emitted: list[Path] = []
    for tile_dir in tiles:
        if tile_dir not in TILE_SPECS:
            raise ValueError(f"unsupported tile directory: {tile_dir}")
        for trans_a, trans_b in TRANSPOSE_PAIRS:
            emitted.append(_emit_one(output_root, tile_dir, trans_a, trans_b))
    return emitted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit BLAS small-kernel SME BF16 base assembly files.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for checked-in BLAS SME generated kernels.",
    )
    parser.add_argument(
        "--tiles",
        nargs="+",
        default=("1vlx4vl", "4vlx1vl"),
        choices=tuple(TILE_SPECS.keys()),
        help="Tile directories to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    emitted = emit_blas_small_kernels(args.output_root.resolve(), args.tiles)
    for path in emitted:
        print(path)


if __name__ == "__main__":
    main()
