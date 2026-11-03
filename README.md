# ParTEE: Llama Fine-tuning with GPU/TEE Collaboration

ParTEE is a GPU/TEE collaborative system for privacy-preserving Llama
fine-tuning. This artifact provides a two-host Llama-3.1-8B-Instruct workflow
using the GLUE RTE task.

## Architecture

ParTEE separates execution across three environments:

- **Trusted preprocessing environment:** partitions the model and prepares
  AES-256-GCM-protected Forward and Backward OTP records.
- **GPU host:** stores quantized public frozen matrices and performs outsourced
  linear computation over masked integer tensors.
- **Occlum enclave:** owns plaintext activations, labels, loss computation,
  selected attention-head rows, LoRA parameters, gradients, optimizer state,
  and validation.

Attention Confidence selects protected Llama GQA groups. Remaining frozen
linear operations are outsourced under independent Forward and Backward OTP
masks. Each material record is bound to its run, phase, epoch, batch, layer,
direction, and invocation. The enclave maintains a persistent consumption
ledger, while the GPU validates invocation identifiers and rejects duplicate
material use.

## Requirements

- Linux GPU host with two CUDA-capable GPUs.
- Intel SGX and Occlum.
- Python 3.
- A local Llama-3.1-8B-Instruct snapshot.
- A local Hugging Face `datasets` copy of GLUE RTE.

Install the Python dependencies on the host and in the Occlum Python image:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

## Configuration

Create a local deployment configuration:

```bash
cp configs/llama_rte.env.example configs/llama_rte.env
set -a
source configs/llama_rte.env
set +a
```

Set these host-side paths in `configs/llama_rte.env`:

- `PARTEE_MODEL_PATH`: Llama snapshot or Hugging Face snapshot root.
- `PARTEE_DATASET_PATH`: local RTE dataset directory.
- `PARTEE_OFFLINE_ROOT`: trusted preprocessing output directory.
- `PARTEE_GPU_MANIFEST_PATH`: GPU-visible copy of `gpu_manifest.pt`.
- `PARTEE_GPU_SHARED_DIR`: GPU-host view of the shared RPC directory.
- `PARTEE_GPU_DEVICES`: comma-separated CUDA devices.
- `PARTEE_CRITICAL_HEADS`: protected `layer:GQA-group` pairs.

All processes participating in one run must use the same
`PARTEE_CRITICAL_HEADS` value.

## Attention Confidence selection

The default configuration uses GQA group `2:5`. To recompute the selection:

```bash
python -m partee.select_heads \
  --model-path "$PARTEE_MODEL_PATH" \
  --dataset-path "$PARTEE_DATASET_PATH" \
  --output head_selection.json \
  --critical-group-count 1
```

Set `PARTEE_CRITICAL_HEADS` to the `layer:group` value written to
`head_selection.json` before preprocessing and runtime startup.

## Trusted OTP preprocessing

Run preprocessing in the trusted environment. For a small multi-step run:

```bash
bash scripts/prepare_otp.sh \
  --epochs 2 \
  --train-batches 2 \
  --validation-batches 1 \
  --batch-size 1 \
  --sequence-length 128 \
  --device cuda:0
```

The preprocessor creates encrypted TEE state, encrypted input batches,
independent Forward and Backward OTP records, and separate TEE/GPU manifests.
Use a new empty `PARTEE_OFFLINE_ROOT` for each run. Keep the complete directory
in the trusted deployment path and copy only `gpu_manifest.pt` to the GPU host.

## Occlum image configuration

Place the runtime components in the Occlum image as follows:

1. Copy `PARTEE_OFFLINE_ROOT` to
   `occlum_instance/image/two_host_otp`.
2. Package the `partee` Python module and its dependencies into the image.
3. Expose the host RPC directory at
   `occlum_instance/image/shared_folder_llamarte/two_host_rpc`.
4. Build the Occlum image using the deployment's `Occlum.json` and normal
   `occlum build` procedure.

The corresponding enclave paths are:

```text
/two_host_otp
/two_host_otp/otp_aes_key.bin
/host/image/shared_folder_llamarte/two_host_rpc
```

Host-mounted paths accessed inside the enclave must retain the standard
`/host/image/` prefix.

## Start the GPU host

On the GPU host, load the deployment configuration and start the masked-GEMM
service:

```bash
set -a
source configs/llama_rte.env
set +a
bash scripts/run_gpu.sh
```

The service loads the public Llama checkpoint, retains quantized Forward and
transposed Backward matrices, and waits for enclave requests.

## Run training inside Occlum

Start the configured Occlum instance, then run inside the enclave:

```bash
export PYTHONPATH=/path/in/image/to/artifact
bash scripts/run_tee.sh
```

The trainer processes the prepared epochs and batches, computes protected loss,
executes independent Forward and Backward OTP operations, updates TEE-owned
LoRA and task-head parameters, runs validation, and updates model parameters
during training.

Optional enclave path overrides are available through:

```text
PARTEE_TEE_SHARED_DIR
PARTEE_TEE_OFFLINE_ROOT
PARTEE_TEE_KEY_PATH
```

The defaults use the standard paths shown above.
