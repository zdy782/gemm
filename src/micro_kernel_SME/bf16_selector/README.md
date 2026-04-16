# BF16 Selector

This package trains and serves a small, inspectable BF16 selector that chooses
the best `pack + tile` combination among the 12 generated kernel variants.

## Data Flow

- Source truth comes from:
  - [bf16_nopack.csv](/Users/wuyihao/Desktop/gemm/ref/data/bf16_nopack.csv)
  - [bf16_packa.csv](/Users/wuyihao/Desktop/gemm/ref/data/bf16_packa.csv)
  - [bf16_packb.csv](/Users/wuyihao/Desktop/gemm/ref/data/bf16_packb.csv)
  - [bf16_packab.csv](/Users/wuyihao/Desktop/gemm/ref/data/bf16_packab.csv)
- `train.py` merges those four files into a standardized wide table at
  [bf16_all.csv](/Users/wuyihao/Desktop/gemm/ref/data/bf16_all.csv).
- The same training run emits [rules.py](/Users/wuyihao/Desktop/gemm/src/micro_kernel_SME/bf16_selector/rules.py),
  which is the only runtime dependency for prediction.

## Labels

- `pack`: `nopack`, `packa`, `packb`, `packab`
- `tile`: `1x4`, `2x2`, `4x1`
- `BestImplementation` is defined over the 12 generated BF16 kernel variants
  only. BLAS is kept as a reference metric in the synthesized CSV but is not a
  selector leaf in this version.

## Usage

Train the model and regenerate artifacts:

```bash
python3 /Users/wuyihao/Desktop/gemm/src/micro_kernel_SME/bf16_selector/train.py
```

Predict one shape:

```bash
python3 /Users/wuyihao/Desktop/gemm/src/micro_kernel_SME/bf16_selector/predict.py \
  --M 64 --N 256 --K 32 --transA T --transB N --show-path
```
