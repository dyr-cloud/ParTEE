#!/usr/bin/env bash
set -euo pipefail

: "${PARTEE_MODEL_PATH:?Set PARTEE_MODEL_PATH}"
: "${PARTEE_GPU_MANIFEST_PATH:?Set PARTEE_GPU_MANIFEST_PATH}"
: "${PARTEE_GPU_SHARED_DIR:?Set PARTEE_GPU_SHARED_DIR}"

python -m partee.gpu_host
