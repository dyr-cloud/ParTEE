#!/usr/bin/env bash
set -euo pipefail

: "${PARTEE_MODEL_PATH:?Set PARTEE_MODEL_PATH}"
: "${PARTEE_DATASET_PATH:?Set PARTEE_DATASET_PATH}"
: "${PARTEE_OFFLINE_ROOT:?Set PARTEE_OFFLINE_ROOT}"

python -m partee.prepare_multistep \
  --model-path "${PARTEE_MODEL_PATH}" \
  --dataset-path "${PARTEE_DATASET_PATH}" \
  --output-dir "${PARTEE_OFFLINE_ROOT}" \
  "$@"
