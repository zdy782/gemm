# gemm

This repository generates and tests ARM SME GEMM micro-kernels.

It is oriented to real SME-capable ARMv9 hardware. Local VM, container, or emulator experiments are treated as local-only workflows and are not part of the repository interface.

## Overview

- Code generation is implemented in Python under `src/micro_kernel_SME/`.
- Test drivers live under `test/micro_kernel_SME_tester/`.
- The test flow generates `kernel_asm.S`, `test.cpp`, and a `Makefile`, then builds and runs the case with `clang` and `make`.
- Supported data types in the current generator and test path:
  - `fp32`
  - `bf16`
  - `fp16`

## Layout

- `src/micro_kernel_SME/`
  - SME assembly generation logic
  - data-type configuration
  - GEMM variant selection
- `test/micro_kernel_SME_tester/`
  - single-case runner
  - CSV-based batch test runner
  - C++ reference helpers

## Requirements

- Python 3
- `clang`
- `clang++`
- `make`
- An ARMv9 platform with SME support
- For `bf16`, toolchain support for `-march=armv9-a+sme+bf16`

Python code imports `loguru`, so install it in your active environment before running the generators.

## Quick Start

Run one test case:

```bash
python test/micro_kernel_SME_tester/test_single.py \
  --M 16 --N 64 --K 16 \
  --gemm_type small \
  --transA N --transB N \
  --data_type fp32
```

Run the CSV batch suite:

```bash
python test/micro_kernel_SME_tester/test_ut.py
```

## Notes

- The generated build flags come from `src/micro_kernel_SME/generate_makefile.py`.
- Temporary generated test directories are created under `test/micro_kernel_SME_tester/tmp/` during runs.
- This repository currently assumes an SME execution target. Apple M-series machines can be used for generator development, but not for native SME execution.
