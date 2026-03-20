#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

python3 "$SCRIPT_DIR/test_perf.py" \
  --data_type bf16 \
  --output "$SCRIPT_DIR/sme_kernel_perf_bf16_${STAMP}.csv" \
  "$@"

python3 "$SCRIPT_DIR/test_perf.py" \
  --data_type fp16 \
  --output "$SCRIPT_DIR/sme_kernel_perf_fp16_${STAMP}.csv" \
  "$@"
