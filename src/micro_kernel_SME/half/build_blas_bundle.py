"""Build BLAS-style shared library bundles for fixed or selector BF16 kernels.

This entrypoint generates a variant-specific directory containing:

- lib/libautogemm_half.so
- bin/sbgemm.goto
- bin/shgemm.goto

Fixed bundles keep pack and tile frozen at build time. Selector bundles compile
all 12 BF16 pack/tile combinations and dispatch at runtime with the generated
decision tree, while FP16 stays on one fixed backend.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from micro_kernel_SME.half.selector.codegen import generate_cpp_selector
from micro_kernel_SME.half.selector.predict import load_model
from micro_kernel_SME.half.generate_sme_test import build_symbol_names, generate_sme_asm, generate_sme_driver_cpp
from micro_kernel_SME.half.global_config import assert_valid_tile_combo
from micro_kernel_SME.half.model_spec import KernelSpec


TRANSPOSE_PAIRS: Tuple[Tuple[str, str], ...] = (("N", "N"), ("N", "T"), ("T", "N"), ("T", "T"))
PRECISIONS: Tuple[str, ...] = ("bf16", "fp16")
PACK_CHOICES: Tuple[str, ...] = ("nopack", "packa", "packb", "packab")
BF16_SELECTOR_TILES: Tuple[Tuple[str, int, int], ...] = (("1x4", 1, 4), ("2x2", 2, 2), ("4x1", 4, 1))
PLACEHOLDER_DIM = 256
SHARED_LIB_NAME = "libautogemm_half.so"
BUNDLE_LAYOUT_VERSION = "direct-driver-benchmark-v5"
CC = os.environ.get("CC", "clang")
CXX = os.environ.get("CXX", "clang++")
AUTOGEMM_TARGET_TRIPLE = os.environ.get("AUTOGEMM_TARGET_TRIPLE", "")
AUTOGEMM_SYSROOT = os.environ.get("AUTOGEMM_SYSROOT", "")
COMMON_CXX_FLAGS = [
    "-O3",
    "-std=c++17",
    "-fPIC",
    "-Wno-implicit-int-float-conversion",
    "-Wno-asm-operand-widths",
    "-Wno-inline-asm",
]
COMMON_ASM_FLAGS = ["-O3", "-fPIC"]
MARCH_FLAGS = {
    "bf16": "-march=armv9-a+sme+bf16",
    "fp16": "-march=armv9-a+sme",
}


@dataclass(frozen=True)
class BackendSpec:
    """Bundle backend identity and generated symbol names."""

    data_type: str
    trans_a: str
    trans_b: str
    pack_name: str
    pack_a: bool
    pack_b: bool
    m_vl: int
    n_vl: int
    driver_name: str
    kernel_name: str
    uniq_id: str

    @property
    def tile_label(self) -> str:
        return f"{self.m_vl}x{self.n_vl}"

    @property
    def stem(self) -> str:
        return f"{self.data_type}_{self.pack_name}_{self.tile_label}_{self.trans_a}{self.trans_b}"


def _pack_flags(pack: str) -> Tuple[bool, bool]:
    if pack == "nopack":
        return False, False
    if pack == "packa":
        return True, False
    if pack == "packb":
        return False, True
    if pack == "packab":
        return True, True
    raise ValueError(f"Unsupported pack mode: {pack}")


def _variant_dir(output_dir: Path, pack: str, m_vl: int, n_vl: int) -> Path:
    return output_dir / f"{pack}_{m_vl}x{n_vl}"


def _selector_variant_dir(output_dir: Path, fp16_pack: str, fp16_m_vl: int, fp16_n_vl: int) -> Path:
    return output_dir / f"bf16_selector_fp16_{fp16_pack}_{fp16_m_vl}x{fp16_n_vl}"


def _placeholder_lda(_trans_a: str) -> int:
    return PLACEHOLDER_DIM


def _placeholder_ldb(_trans_b: str) -> int:
    return PLACEHOLDER_DIM


def _backend_spec(
    data_type: str,
    trans_a: str,
    trans_b: str,
    pack_name: str,
    m_vl: int,
    n_vl: int,
) -> BackendSpec:
    pack_a, pack_b = _pack_flags(pack_name)
    uniq_id = f"bundle_{data_type}_{pack_name}_{trans_a.lower()}{trans_b.lower()}_{m_vl}x{n_vl}"
    spec = KernelSpec.from_args(
        PLACEHOLDER_DIM,
        PLACEHOLDER_DIM,
        PLACEHOLDER_DIM,
        _placeholder_lda(trans_a),
        _placeholder_ldb(trans_b),
        PLACEHOLDER_DIM,
        "small",
        trans_a,
        trans_b,
        data_type,
        m_vl,
        n_vl,
        pack_a,
        pack_b,
    )
    kernel_name, driver_name = build_symbol_names(spec, uniq_id)
    return BackendSpec(
        data_type=data_type,
        trans_a=trans_a,
        trans_b=trans_b,
        pack_name=pack_name,
        pack_a=pack_a,
        pack_b=pack_b,
        m_vl=m_vl,
        n_vl=n_vl,
        driver_name=driver_name,
        kernel_name=kernel_name,
        uniq_id=uniq_id,
    )


def _generate_backend_sources(backend: BackendSpec) -> Tuple[str, str]:
    asm_code = generate_sme_asm(
        PLACEHOLDER_DIM,
        PLACEHOLDER_DIM,
        PLACEHOLDER_DIM,
        _placeholder_lda(backend.trans_a),
        _placeholder_ldb(backend.trans_b),
        PLACEHOLDER_DIM,
        "small",
        backend.trans_a,
        backend.trans_b,
        backend.uniq_id,
        backend.data_type,
        backend.m_vl,
        backend.n_vl,
        backend.pack_a,
        backend.pack_b,
    )
    driver_code = generate_sme_driver_cpp(
        PLACEHOLDER_DIM,
        PLACEHOLDER_DIM,
        PLACEHOLDER_DIM,
        _placeholder_lda(backend.trans_a),
        _placeholder_ldb(backend.trans_b),
        PLACEHOLDER_DIM,
        "small",
        backend.trans_a,
        backend.trans_b,
        backend.uniq_id,
        backend.data_type,
        backend.m_vl,
        backend.n_vl,
        backend.pack_a,
        backend.pack_b,
    )
    return asm_code, driver_code


def _normalize_precision_input_type(data_type: str) -> Tuple[str, str]:
    if data_type == "bf16":
        return "__bf16", "#include <arm_bf16.h>\n"
    if data_type == "fp16":
        return "__fp16", ""
    raise ValueError(f"Unsupported precision: {data_type}")


def _hosted_linux_aarch64_available() -> bool:
    return platform.system() == "Linux" and platform.machine() in {"aarch64", "arm64"}


def _common_target_flags() -> List[str]:
    flags: List[str] = []
    if AUTOGEMM_TARGET_TRIPLE:
        flags.append(f"--target={AUTOGEMM_TARGET_TRIPLE}")
    if AUTOGEMM_SYSROOT:
        flags.append(f"--sysroot={AUTOGEMM_SYSROOT}")
    return flags


def _validate_hosted_toolchain() -> None:
    if _hosted_linux_aarch64_available():
        return
    if not AUTOGEMM_TARGET_TRIPLE:
        raise RuntimeError(
            "Hosted bundle builds require either a native Linux aarch64 host "
            "or AUTOGEMM_TARGET_TRIPLE for cross compilation."
        )
    if not AUTOGEMM_SYSROOT:
        raise RuntimeError(
            "Hosted cross builds require AUTOGEMM_SYSROOT so clang can link "
            "the Linux shared library and benchmark frontends."
        )


def _decl_lines(selected: Sequence[BackendSpec]) -> str:
    lines = []
    for backend in selected:
        input_type, _ = _normalize_precision_input_type(backend.data_type)
        lines.append(
            "extern \"C\" int "
            f"{backend.driver_name}(const long M, const long N, const long K, const float alpha, "
            f"const {input_type} *A, const {input_type} *B, const float beta, float *C, "
            "const long lda, const long ldb, const long ldc);"
        )
    return "\n".join(lines)


def _emit_wrapper_dispatch(selected: Sequence[BackendSpec], selector_model: Dict[str, object] | None) -> str:
    if selector_model is None:
        lines = []
        for backend in selected:
            lines.append(
                f"    if (ta == '{backend.trans_a}' && tb == '{backend.trans_b}') {{\n"
                f"        return {backend.driver_name}(M, N, K, alpha, A, B, beta, C, lda, ldb, ldc);\n"
                "    }\n"
            )
        lines.append('    std::fprintf(stderr, "autogemm: unsupported transpose pair %c%c\\n", ta, tb);\n')
        lines.append("    return -1;\n")
        return "".join(lines)

    lines = ["    const AutoGemmBf16Choice choice = autogemm_predict_bf16_choice(M, N, K, ta, tb);\n"]
    for backend in selected:
        lines.append(
            "    if "
            f"(ta == '{backend.trans_a}' && tb == '{backend.trans_b}' && "
            f"std::strcmp(choice.pack, \"{backend.pack_name}\") == 0 && "
            f"std::strcmp(choice.tile, \"{backend.tile_label}\") == 0) {{\n"
            f"        return {backend.driver_name}(M, N, K, alpha, A, B, beta, C, lda, ldb, ldc);\n"
            "    }\n"
        )
    lines.append(
        '    std::fprintf(stderr, "autogemm: selector produced unsupported combo %s_%s for %c%c\\n", '
        "choice.pack, choice.tile, ta, tb);\n"
    )
    lines.append("    return -1;\n")
    return "".join(lines)


def _emit_benchmark_driver_lookup(selected: Sequence[BackendSpec], selector_model: Dict[str, object] | None) -> str:
    if selector_model is None:
        lines = []
        for backend in selected:
            lines.append(
                f"    if (ta == '{backend.trans_a}' && tb == '{backend.trans_b}') {{\n"
                f"        return {backend.driver_name};\n"
                "    }\n"
            )
        lines.append("    return nullptr;\n")
        return "".join(lines)

    lines = ["    const AutoGemmBf16Choice choice = autogemm_predict_bf16_choice(M, N, K, ta, tb);\n"]
    for backend in selected:
        lines.append(
            "    if "
            f"(ta == '{backend.trans_a}' && tb == '{backend.trans_b}' && "
            f"std::strcmp(choice.pack, \"{backend.pack_name}\") == 0 && "
            f"std::strcmp(choice.tile, \"{backend.tile_label}\") == 0) {{\n"
            f"        return {backend.driver_name};\n"
            "    }\n"
        )
    lines.append("    return nullptr;\n")
    return "".join(lines)


def _generate_wrapper_cpp(
    backends: Sequence[BackendSpec],
    bf16_selector_model: Dict[str, object] | None = None,
) -> str:
    bf16_backends = [backend for backend in backends if backend.data_type == "bf16"]
    fp16_backends = [backend for backend in backends if backend.data_type == "fp16"]
    bf16_input, bf16_include = _normalize_precision_input_type("bf16")
    fp16_input, _ = _normalize_precision_input_type("fp16")
    selector_code = "" if bf16_selector_model is None else generate_cpp_selector(bf16_selector_model)

    return f"""
#include <arm_sve.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
{bf16_include}

enum CBLAS_ORDER {{
    CblasRowMajor = 101,
    CblasColMajor = 102,
}};

enum CBLAS_TRANSPOSE {{
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113,
}};

{_decl_lines(bf16_backends)}
{_decl_lines(fp16_backends)}

static inline char autogemm_normalize_transpose_char(char value) {{
    const char normalized = static_cast<char>(std::toupper(static_cast<unsigned char>(value)));
    if (normalized == 'N' || normalized == 'R') {{
        return 'N';
    }}
    if (normalized == 'T' || normalized == 'C') {{
        return 'T';
    }}
    return '\\0';
}}

static inline char autogemm_cblas_transpose_char(int value) {{
    if (value == CblasNoTrans) {{
        return 'N';
    }}
    if (value == CblasTrans || value == CblasConjTrans) {{
        return 'T';
    }}
    return '\\0';
}}

static inline void autogemm_scale_c_col_major(long M, long N, float beta, float *C, long ldc) {{
    if (beta == 1.0f) {{
        return;
    }}
    if (beta == 0.0f) {{
        for (long j = 0; j < N; ++j) {{
            for (long i = 0; i < M; ++i) {{
                C[i + j * ldc] = 0.0f;
            }}
        }}
        return;
    }}
    for (long j = 0; j < N; ++j) {{
        for (long i = 0; i < M; ++i) {{
            C[i + j * ldc] *= beta;
        }}
    }}
}}

static inline void autogemm_report_status(const char *func_name, int status) {{
    if (status != 0) {{
        std::fprintf(stderr, "autogemm: %s failed with status=%d\\n", func_name, status);
    }}
}}

{selector_code}
static int autogemm_dispatch_sbgemm(
    char transa,
    char transb,
    long M,
    long N,
    long K,
    float alpha,
    const {bf16_input} *A,
    long lda,
    const {bf16_input} *B,
    long ldb,
    float beta,
    float *C,
    long ldc
) {{
    const char ta = autogemm_normalize_transpose_char(transa);
    const char tb = autogemm_normalize_transpose_char(transb);
    if (ta == '\\0' || tb == '\\0') {{
        std::fprintf(stderr, "autogemm: invalid transpose pair %c %c\\n", transa, transb);
        return -1;
    }}
    if (M <= 0 || N <= 0) {{
        return 0;
    }}
    if (K <= 0 || alpha == 0.0f) {{
        autogemm_scale_c_col_major(M, N, beta, C, ldc);
        return 0;
    }}
{_emit_wrapper_dispatch(bf16_backends, bf16_selector_model)}
}}

static int autogemm_dispatch_shgemm(
    char transa,
    char transb,
    long M,
    long N,
    long K,
    float alpha,
    const {fp16_input} *A,
    long lda,
    const {fp16_input} *B,
    long ldb,
    float beta,
    float *C,
    long ldc
) {{
    const char ta = autogemm_normalize_transpose_char(transa);
    const char tb = autogemm_normalize_transpose_char(transb);
    if (ta == '\\0' || tb == '\\0') {{
        std::fprintf(stderr, "autogemm: invalid transpose pair %c %c\\n", transa, transb);
        return -1;
    }}
    if (M <= 0 || N <= 0) {{
        return 0;
    }}
    if (K <= 0 || alpha == 0.0f) {{
        autogemm_scale_c_col_major(M, N, beta, C, ldc);
        return 0;
    }}
{_emit_wrapper_dispatch(fp16_backends, None)}
}}

extern "C" void sbgemm_(
    const char *transa,
    const char *transb,
    const int *M,
    const int *N,
    const int *K,
    const float *alpha,
    const {bf16_input} *A,
    const int *lda,
    const {bf16_input} *B,
    const int *ldb,
    const float *beta,
    float *C,
    const int *ldc
) {{
    autogemm_report_status(
        "sbgemm_",
        autogemm_dispatch_sbgemm(*transa, *transb, *M, *N, *K, *alpha, A, *lda, B, *ldb, *beta, C, *ldc)
    );
}}

extern "C" void sbgemm(
    const char *transa,
    const char *transb,
    const int *M,
    const int *N,
    const int *K,
    const float *alpha,
    const {bf16_input} *A,
    const int *lda,
    const {bf16_input} *B,
    const int *ldb,
    const float *beta,
    float *C,
    const int *ldc
) {{
    sbgemm_(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}}

extern "C" void shgemm_(
    const char *transa,
    const char *transb,
    const int *M,
    const int *N,
    const int *K,
    const float *alpha,
    const {fp16_input} *A,
    const int *lda,
    const {fp16_input} *B,
    const int *ldb,
    const float *beta,
    float *C,
    const int *ldc
) {{
    autogemm_report_status(
        "shgemm_",
        autogemm_dispatch_shgemm(*transa, *transb, *M, *N, *K, *alpha, A, *lda, B, *ldb, *beta, C, *ldc)
    );
}}

extern "C" void shgemm(
    const char *transa,
    const char *transb,
    const int *M,
    const int *N,
    const int *K,
    const float *alpha,
    const {fp16_input} *A,
    const int *lda,
    const {fp16_input} *B,
    const int *ldb,
    const float *beta,
    float *C,
    const int *ldc
) {{
    shgemm_(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}}

extern "C" void cblas_sbgemm(
    const int order,
    const int transa,
    const int transb,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const {bf16_input} *A,
    const int lda,
    const {bf16_input} *B,
    const int ldb,
    const float beta,
    float *C,
    const int ldc
) {{
    if (order == CblasColMajor) {{
        autogemm_report_status(
            "cblas_sbgemm",
            autogemm_dispatch_sbgemm(
                autogemm_cblas_transpose_char(transa),
                autogemm_cblas_transpose_char(transb),
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc
            )
        );
        return;
    }}
    if (order == CblasRowMajor) {{
        autogemm_report_status(
            "cblas_sbgemm",
            autogemm_dispatch_sbgemm(
                autogemm_cblas_transpose_char(transb),
                autogemm_cblas_transpose_char(transa),
                N, M, K, alpha, B, ldb, A, lda, beta, C, ldc
            )
        );
        return;
    }}
    std::fprintf(stderr, "autogemm: invalid CBLAS order=%d\\n", order);
}}

extern "C" void cblas_shgemm(
    const int order,
    const int transa,
    const int transb,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const {fp16_input} *A,
    const int lda,
    const {fp16_input} *B,
    const int ldb,
    const float beta,
    float *C,
    const int ldc
) {{
    if (order == CblasColMajor) {{
        autogemm_report_status(
            "cblas_shgemm",
            autogemm_dispatch_shgemm(
                autogemm_cblas_transpose_char(transa),
                autogemm_cblas_transpose_char(transb),
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc
            )
        );
        return;
    }}
    if (order == CblasRowMajor) {{
        autogemm_report_status(
            "cblas_shgemm",
            autogemm_dispatch_shgemm(
                autogemm_cblas_transpose_char(transb),
                autogemm_cblas_transpose_char(transa),
                N, M, K, alpha, B, ldb, A, lda, beta, C, ldc
            )
        );
        return;
    }}
    std::fprintf(stderr, "autogemm: invalid CBLAS order=%d\\n", order);
}}
"""


def _generate_benchmark_cpp(
    selected_backends: Sequence[BackendSpec],
    data_type: str,
    bf16_selector_model: Dict[str, object] | None = None,
) -> str:
    input_type, include_block = _normalize_precision_input_type(data_type)
    selector_code = ""
    if data_type == "bf16" and bf16_selector_model is not None:
        selector_code = generate_cpp_selector(bf16_selector_model)
    dispatch_code = _emit_benchmark_driver_lookup(
        selected_backends,
        bf16_selector_model if data_type == "bf16" else None,
    )
    return rf"""
#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <string>

{include_block}using InputType = {input_type};
using DriverFn = int (*)(
    const long M,
    const long N,
    const long K,
    const float alpha,
    const InputType *A,
    const InputType *B,
    const float beta,
    float *C,
    const long lda,
    const long ldb,
    const long ldc
);
{_decl_lines(selected_backends)}

struct BenchArgs {{
    int m = 256;
    int n = 256;
    int k = 256;
    int lda = -1;
    int ldb = -1;
    int ldc = -1;
    bool has_m = false;
    bool has_n = false;
    bool has_k = false;
    bool has_lda = false;
    bool has_ldb = false;
    bool has_ldc = false;
    int from = 256;
    int to = 256;
    int step = 1;
    int loops = 1;
    int inner_loops = 64;
    int nthreads = 1;
    float alpha_r = 1.0f;
    float alpha_i = 0.0f;
    float beta_r = 1.0f;
    float beta_i = 0.0f;
    char api = 'F';
    char order = 'C';
    char transa = 'N';
    char transb = 'N';
}};

struct ResolvedCall {{
    DriverFn driver = nullptr;
    const InputType *A = nullptr;
    const InputType *B = nullptr;
    float *C = nullptr;
    long m = 0;
    long n = 0;
    long k = 0;
    long lda = 0;
    long ldb = 0;
    long ldc = 0;
}};

#ifdef AUTOGEMM_BF16
static inline InputType float_to_input(float value) {{
    uint32_t float_bits = 0;
    uint16_t bf16_bits = 0;
    InputType result;
    std::memcpy(&float_bits, &value, sizeof(float_bits));
    bf16_bits = static_cast<uint16_t>(float_bits >> 16);
    std::memcpy(&result, &bf16_bits, sizeof(result));
    return result;
}}
#else
static inline InputType float_to_input(float value) {{
    return static_cast<InputType>(value);
}}
#endif

template <typename T>
static T* alloc_aligned(std::size_t count) {{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 64, sizeof(T) * count) != 0) {{
        return nullptr;
    }}
    return static_cast<T*>(ptr);
}}

static inline void fill_input(InputType *buffer, std::size_t count) {{
    for (std::size_t idx = 0; idx < count; ++idx) {{
        float value = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        buffer[idx] = float_to_input(value);
    }}
}}

static inline void fill_output(float *buffer, std::size_t count) {{
    for (std::size_t idx = 0; idx < count; ++idx) {{
        buffer[idx] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }}
}}

static inline char normalize_char(char value) {{
    return static_cast<char>(std::toupper(static_cast<unsigned char>(value)));
}}

static inline bool is_transposed(char value) {{
    const char normalized = normalize_char(value);
    return normalized != 'N' && normalized != 'R';
}}

static inline std::size_t matrix_a_elements(const BenchArgs &args) {{
    const bool is_col_major = (args.api == 'F' || args.order == 'C');
    const int opposite_dim = (is_transposed(args.transa) == is_col_major) ? args.m : args.k;
    return static_cast<std::size_t>(opposite_dim) * static_cast<std::size_t>(args.lda);
}}

static inline std::size_t matrix_b_elements(const BenchArgs &args) {{
    const bool is_col_major = (args.api == 'F' || args.order == 'C');
    const int opposite_dim = (is_transposed(args.transb) == is_col_major) ? args.k : args.n;
    return static_cast<std::size_t>(opposite_dim) * static_cast<std::size_t>(args.ldb);
}}

static inline std::size_t matrix_c_elements(const BenchArgs &args) {{
    const bool is_col_major = (args.api == 'F' || args.order == 'C');
    const int opposite_dim = is_col_major ? args.n : args.m;
    return static_cast<std::size_t>(opposite_dim) * static_cast<std::size_t>(args.ldc);
}}

static void resolve_leading_dimensions(BenchArgs *args) {{
    const bool is_col_major = (args->api == 'F' || args->order == 'C');
    if (!args->has_lda) {{
        args->lda = (is_transposed(args->transa) == is_col_major) ? args->k : args->m;
    }}
    if (!args->has_ldb) {{
        args->ldb = (is_transposed(args->transb) == is_col_major) ? args->n : args->k;
    }}
    if (!args->has_ldc) {{
        args->ldc = is_col_major ? args->m : args->n;
    }}
}}

static void parse_args(int argc, char **argv, BenchArgs *args) {{
    for (int idx = 1; idx < argc; ++idx) {{
        std::string option = argv[idx];
        auto require_value = [&](const char *name) -> const char* {{
            if (idx + 1 >= argc) {{
                std::fprintf(stderr, "Missing value for %s\n", name);
                std::exit(1);
            }}
            return argv[++idx];
        }};

        if (option == "-m") {{
            args->m = std::atoi(require_value("-m"));
            args->has_m = true;
        }} else if (option == "-n") {{
            args->n = std::atoi(require_value("-n"));
            args->has_n = true;
        }} else if (option == "-k") {{
            args->k = std::atoi(require_value("-k"));
            args->has_k = true;
        }} else if (option == "-lda") {{
            args->lda = std::atoi(require_value("-lda"));
            args->has_lda = true;
        }} else if (option == "-ldb") {{
            args->ldb = std::atoi(require_value("-ldb"));
            args->has_ldb = true;
        }} else if (option == "-ldc") {{
            args->ldc = std::atoi(require_value("-ldc"));
            args->has_ldc = true;
        }} else if (option == "-from") {{
            args->from = std::atoi(require_value("-from"));
        }} else if (option == "-to") {{
            args->to = std::atoi(require_value("-to"));
        }} else if (option == "-step") {{
            args->step = std::atoi(require_value("-step"));
        }} else if (option == "-loops") {{
            args->loops = std::atoi(require_value("-loops"));
        }} else if (option == "-innerLoops") {{
            args->inner_loops = std::atoi(require_value("-innerLoops"));
        }} else if (option == "-nthreads") {{
            args->nthreads = std::atoi(require_value("-nthreads"));
        }} else if (option == "-alphaR") {{
            args->alpha_r = std::strtof(require_value("-alphaR"), nullptr);
        }} else if (option == "-alphaI") {{
            args->alpha_i = std::strtof(require_value("-alphaI"), nullptr);
        }} else if (option == "-betaR") {{
            args->beta_r = std::strtof(require_value("-betaR"), nullptr);
        }} else if (option == "-betaI") {{
            args->beta_i = std::strtof(require_value("-betaI"), nullptr);
        }} else if (option == "-api") {{
            args->api = normalize_char(*require_value("-api"));
        }} else if (option == "-order") {{
            args->order = normalize_char(*require_value("-order"));
        }} else if (option == "-transa") {{
            args->transa = normalize_char(*require_value("-transa"));
        }} else if (option == "-transb") {{
            args->transb = normalize_char(*require_value("-transb"));
        }} else {{
            std::fprintf(stderr, "Unknown option: %s\n", option.c_str());
            std::exit(1);
        }}
    }}
}}

static inline double now_seconds() {{
    timespec ts = {{0, 0}};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1.0e-9;
}}

{selector_code}
static DriverFn resolve_driver(long M, long N, long K, char ta, char tb) {{
{dispatch_code}
}}

static ResolvedCall resolve_call(const BenchArgs &args, const InputType *a, const InputType *b, float *c) {{
    ResolvedCall resolved;
    const char ta = normalize_char(args.transa);
    const char tb = normalize_char(args.transb);
    const bool row_major_cblas = (args.api == 'C' && args.order == 'R');
    if (row_major_cblas) {{
        resolved.driver = resolve_driver(args.n, args.m, args.k, tb, ta);
        resolved.A = b;
        resolved.B = a;
        resolved.C = c;
        resolved.m = args.n;
        resolved.n = args.m;
        resolved.k = args.k;
        resolved.lda = args.ldb;
        resolved.ldb = args.lda;
        resolved.ldc = args.ldc;
        return resolved;
    }}

    resolved.driver = resolve_driver(args.m, args.n, args.k, ta, tb);
    resolved.A = a;
    resolved.B = b;
    resolved.C = c;
    resolved.m = args.m;
    resolved.n = args.n;
    resolved.k = args.k;
    resolved.lda = args.lda;
    resolved.ldb = args.ldb;
    resolved.ldc = args.ldc;
    return resolved;
}}

static void run_single_case(const BenchArgs &args) {{
    BenchArgs resolved = args;
    resolve_leading_dimensions(&resolved);

    InputType *a = alloc_aligned<InputType>(matrix_a_elements(resolved));
    InputType *b = alloc_aligned<InputType>(matrix_b_elements(resolved));
    float *c = alloc_aligned<float>(matrix_c_elements(resolved));
    float *c_initial = alloc_aligned<float>(matrix_c_elements(resolved));
    if (a == nullptr || b == nullptr || c == nullptr || c_initial == nullptr) {{
        std::fprintf(stderr, "Allocation failure\n");
        std::free(a);
        std::free(b);
        std::free(c);
        std::free(c_initial);
        std::exit(1);
    }}

    fill_input(a, matrix_a_elements(resolved));
    fill_input(b, matrix_b_elements(resolved));
    fill_output(c_initial, matrix_c_elements(resolved));
    const ResolvedCall call = resolve_call(resolved, a, b, c);
    if (call.driver == nullptr) {{
        std::fprintf(
            stderr,
            "autogemm benchmark: unsupported transpose pair transa=%c transb=%c api=%c order=%c\n",
            resolved.transa,
            resolved.transb,
            resolved.api,
            resolved.order
        );
        std::free(a);
        std::free(b);
        std::free(c);
        std::free(c_initial);
        std::exit(1);
    }}

    const double flop_count = 2.0 * static_cast<double>(resolved.m) * static_cast<double>(resolved.n) * static_cast<double>(resolved.k);
    double total_time = 0.0;
    constexpr int warmup_loops = 20;

    for (int loop = 0; loop < resolved.loops; ++loop) {{
        std::memcpy(c, c_initial, matrix_c_elements(resolved) * sizeof(float));
        for (int warmup = 0; warmup < warmup_loops; ++warmup) {{
            const int status = call.driver(
                call.m, call.n, call.k,
                resolved.alpha_r,
                call.A,
                call.B,
                resolved.beta_r,
                call.C,
                call.lda,
                call.ldb,
                call.ldc
            );
            if (status != 0) {{
                std::fprintf(stderr, "autogemm benchmark: warmup driver failed with status=%d\n", status);
                std::free(a);
                std::free(b);
                std::free(c);
                std::free(c_initial);
                std::exit(1);
            }}
        }}
        const double start = now_seconds();
        for (int inner = 0; inner < resolved.inner_loops; ++inner) {{
            const int status = call.driver(
                call.m, call.n, call.k,
                resolved.alpha_r,
                call.A,
                call.B,
                resolved.beta_r,
                call.C,
                call.lda,
                call.ldb,
                call.ldc
            );
            if (status != 0) {{
                std::fprintf(stderr, "autogemm benchmark: timed driver failed with status=%d\n", status);
                std::free(a);
                std::free(b);
                std::free(c);
                std::free(c_initial);
                std::exit(1);
            }}
        }}
        total_time += (now_seconds() - start) / static_cast<double>(resolved.inner_loops);
    }}

    const double avg_time = total_time / static_cast<double>(resolved.loops);
    const double avg_mflops = flop_count / avg_time / 1.0e6;
    std::printf(
        "MFlops_Effi_Time_avg:[ %.2f MFlops ] Time_avg:[ %.8f s ] M=%d N=%d K=%d transa=%c transb=%c api=%c order=%c\n",
        avg_mflops,
        avg_time,
        resolved.m,
        resolved.n,
        resolved.k,
        resolved.transa,
        resolved.transb,
        resolved.api,
        resolved.order
    );

    std::free(a);
    std::free(b);
    std::free(c);
    std::free(c_initial);
}}

int main(int argc, char **argv) {{
    BenchArgs args;
    parse_args(argc, argv, &args);

    if (args.alpha_i != 0.0f || args.beta_i != 0.0f) {{
        std::fprintf(stderr, "Complex alpha or beta is unsupported for real half GEMM\n");
        return 1;
    }}
    if (args.nthreads != 1) {{
        std::fprintf(stderr, "Warning: -nthreads=%d requested, but this benchmark runs single-threaded\n", args.nthreads);
    }}
    if (args.api != 'F' && args.api != 'C') {{
        std::fprintf(stderr, "Invalid api=%c\n", args.api);
        return 1;
    }}
    if (args.order != 'C' && args.order != 'R') {{
        std::fprintf(stderr, "Invalid order=%c\n", args.order);
        return 1;
    }}
    if (args.step <= 0) {{
        std::fprintf(stderr, "Invalid step=%d\n", args.step);
        return 1;
    }}

    std::srand(0);
    for (int current = args.from; current <= args.to; current += args.step) {{
        BenchArgs run_args = args;
        if (!run_args.has_m) {{
            run_args.m = current;
        }}
        if (!run_args.has_n) {{
            run_args.n = current;
        }}
        if (!run_args.has_k) {{
            run_args.k = current;
        }}
        run_single_case(run_args);
    }}
    return 0;
}}
"""


def _write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _run_command(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def _compile_backend_objects(variant_dir: Path, backend: BackendSpec) -> List[Path]:
    asm_code, driver_code = _generate_backend_sources(backend)
    gen_dir = variant_dir / "gen"
    obj_dir = variant_dir / "obj"
    asm_path = gen_dir / f"{backend.stem}_kernel.S"
    driver_path = gen_dir / f"{backend.stem}_driver.cpp"
    asm_obj = obj_dir / f"{backend.stem}_kernel.o"
    driver_obj = obj_dir / f"{backend.stem}_driver.o"
    _write_file(asm_path, asm_code)
    _write_file(driver_path, driver_code)

    march_flag = MARCH_FLAGS[backend.data_type]
    _run_command(
        [CC, *_common_target_flags(), march_flag, *COMMON_ASM_FLAGS, "-c", str(asm_path), "-o", str(asm_obj)],
        variant_dir,
    )
    _run_command(
        [CXX, *_common_target_flags(), march_flag, *COMMON_CXX_FLAGS, "-c", str(driver_path), "-o", str(driver_obj)],
        variant_dir,
    )
    return [asm_obj, driver_obj]


def _compile_wrapper_and_library(
    variant_dir: Path,
    backends: Sequence[BackendSpec],
    objects: Sequence[Path],
    bf16_selector_model: Dict[str, object] | None = None,
) -> Path:
    gen_dir = variant_dir / "gen"
    obj_dir = variant_dir / "obj"
    lib_dir = variant_dir / "lib"
    wrapper_src = gen_dir / "autogemm_wrapper.cpp"
    wrapper_obj = obj_dir / "autogemm_wrapper.o"
    lib_path = lib_dir / SHARED_LIB_NAME

    _write_file(wrapper_src, _generate_wrapper_cpp(backends, bf16_selector_model))
    _run_command(
        [CXX, *_common_target_flags(), MARCH_FLAGS["bf16"], *COMMON_CXX_FLAGS, "-c", str(wrapper_src), "-o", str(wrapper_obj)],
        variant_dir,
    )
    _run_command(
        [
            CXX,
            *_common_target_flags(),
            MARCH_FLAGS["bf16"],
            "-shared",
            "-Wl,-soname," + SHARED_LIB_NAME,
            "-o",
            str(lib_path),
            str(wrapper_obj),
            *[str(path) for path in objects],
        ],
        variant_dir,
    )
    return lib_path


def _compile_benchmark_executable(
    variant_dir: Path,
    data_type: str,
    backends: Sequence[BackendSpec],
    bf16_selector_model: Dict[str, object] | None = None,
) -> Path:
    gen_dir = variant_dir / "gen"
    bin_dir = variant_dir / "bin"
    bench_src = gen_dir / ("benchmark_sbgemm.cpp" if data_type == "bf16" else "benchmark_shgemm.cpp")
    bench_name = "sbgemm.goto" if data_type == "bf16" else "shgemm.goto"
    bench_path = bin_dir / bench_name
    define_flag = "-DAUTOGEMM_BF16" if data_type == "bf16" else "-DAUTOGEMM_FP16"
    march_flag = MARCH_FLAGS["bf16"] if data_type == "bf16" else MARCH_FLAGS["fp16"]
    selected_backends = [backend for backend in backends if backend.data_type == data_type]

    _write_file(
        bench_src,
        _generate_benchmark_cpp(
            selected_backends,
            data_type,
            bf16_selector_model if data_type == "bf16" else None,
        ),
    )

    _run_command(
        [
            CXX,
            *_common_target_flags(),
            march_flag,
            "-O3",
            "-std=c++17",
            define_flag,
            str(bench_src),
            "-L",
            str(variant_dir / "lib"),
            "-lautogemm_half",
            "-Wl,-rpath,$ORIGIN/../lib",
            "-o",
            str(bench_path),
        ],
        variant_dir,
    )
    return bench_path


def _build_backends(pack: str, m_vl: int, n_vl: int) -> List[BackendSpec]:
    backends: List[BackendSpec] = []
    for data_type in PRECISIONS:
        for trans_a, trans_b in TRANSPOSE_PAIRS:
            backends.append(_backend_spec(data_type, trans_a, trans_b, pack, m_vl, n_vl))
    return backends


def _build_selector_backends(fp16_pack: str, fp16_m_vl: int, fp16_n_vl: int) -> List[BackendSpec]:
    backends: List[BackendSpec] = []
    for pack_name in PACK_CHOICES:
        for _tile_name, m_vl, n_vl in BF16_SELECTOR_TILES:
            for trans_a, trans_b in TRANSPOSE_PAIRS:
                backends.append(_backend_spec("bf16", trans_a, trans_b, pack_name, m_vl, n_vl))
    for trans_a, trans_b in TRANSPOSE_PAIRS:
        backends.append(_backend_spec("fp16", trans_a, trans_b, fp16_pack, fp16_m_vl, fp16_n_vl))
    return backends


def _prepare_variant_dir(variant_dir: Path) -> None:
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    for subdir in ("gen", "obj", "lib", "bin"):
        (variant_dir / subdir).mkdir(parents=True, exist_ok=True)


def build_bundle(pack: str, m_vl: int, n_vl: int, output_dir: Path) -> Path:
    """Generate and compile a fixed pack or VL bundle."""
    _validate_hosted_toolchain()
    assert_valid_tile_combo(m_vl, n_vl)
    variant_dir = _variant_dir(output_dir.resolve(), pack, m_vl, n_vl)
    _prepare_variant_dir(variant_dir)

    backends = _build_backends(pack, m_vl, n_vl)
    objects: List[Path] = []
    for backend in backends:
        objects.extend(_compile_backend_objects(variant_dir, backend))

    _compile_wrapper_and_library(variant_dir, backends, objects)
    _compile_benchmark_executable(variant_dir, "bf16", backends)
    _compile_benchmark_executable(variant_dir, "fp16", backends)
    _write_file(variant_dir / "bundle_version.txt", BUNDLE_LAYOUT_VERSION + "\n")
    return variant_dir


def build_bf16_selector_bundle(
    fp16_pack: str,
    fp16_m_vl: int,
    fp16_n_vl: int,
    output_dir: Path,
) -> Path:
    """Generate and compile a BF16 selector-aware bundle."""
    _validate_hosted_toolchain()
    assert_valid_tile_combo(fp16_m_vl, fp16_n_vl)
    try:
        model = load_model()
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(
            "BF16 selector bundle requires generated rules.py. "
            "Run `python3 src/micro_kernel_SME/half/selector/train.py` first."
        ) from exc

    variant_dir = _selector_variant_dir(output_dir.resolve(), fp16_pack, fp16_m_vl, fp16_n_vl)
    _prepare_variant_dir(variant_dir)

    backends = _build_selector_backends(fp16_pack, fp16_m_vl, fp16_n_vl)
    objects: List[Path] = []
    for backend in backends:
        objects.extend(_compile_backend_objects(variant_dir, backend))

    _compile_wrapper_and_library(variant_dir, backends, objects, bf16_selector_model=model)
    _compile_benchmark_executable(variant_dir, "bf16", backends, bf16_selector_model=model)
    _compile_benchmark_executable(variant_dir, "fp16", backends)
    _write_file(variant_dir / "bundle_version.txt", BUNDLE_LAYOUT_VERSION + "\n")
    return variant_dir


def main() -> None:
    """CLI entrypoint for fixed or selector-aware bundle builds."""
    parser = argparse.ArgumentParser(description="Build a fixed pack/VL or BF16 selector-aware BLAS-style shared library bundle.")
    parser.add_argument("--pack", choices=PACK_CHOICES)
    parser.add_argument("--m-vl", type=int)
    parser.add_argument("--n-vl", type=int)
    parser.add_argument("--bf16-selector", action="store_true", help="Build a selector-aware BF16 bundle with all 12 pack/tile variants.")
    parser.add_argument("--fp16-pack", choices=PACK_CHOICES, default="nopack")
    parser.add_argument("--fp16-m-vl", type=int, default=1)
    parser.add_argument("--fp16-n-vl", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    args = parser.parse_args()

    if args.bf16_selector:
        variant_dir = build_bf16_selector_bundle(
            args.fp16_pack,
            args.fp16_m_vl,
            args.fp16_n_vl,
            args.output_dir,
        )
    else:
        missing = [
            flag_name
            for flag_name, flag_value in (("--pack", args.pack), ("--m-vl", args.m_vl), ("--n-vl", args.n_vl))
            if flag_value is None
        ]
        if missing:
            parser.error(f"{', '.join(missing)} are required unless --bf16-selector is set")
        variant_dir = build_bundle(args.pack, args.m_vl, args.n_vl, args.output_dir)

    print(f"[BUILD] Generated bundle at {variant_dir}")
    print(f"[BUILD] Shared library: {variant_dir / 'lib' / SHARED_LIB_NAME}")
    print(f"[BUILD] Benchmark binary: {variant_dir / 'bin' / 'sbgemm.goto'}")
    print(f"[BUILD] Benchmark binary: {variant_dir / 'bin' / 'shgemm.goto'}")


if __name__ == "__main__":
    main()
