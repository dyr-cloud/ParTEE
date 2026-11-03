#!/usr/bin/env python3
"""GPU-host frozen-GEMM service for secure multi-step Llama training.

Run this program before starting ``partee.train_tee`` inside Occlum. GPU 0 stores
layers 0--15 and GPU 1 stores layers 16--31.  The service accepts masked
integer tensors only; it never receives input_ids, labels, or plaintext model
activations.
"""

from __future__ import annotations

import gc
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

from .common import (
    CRITICAL_HEADS,
    OTP_BITS,
    OTP_MODULUS,
    complement_rows,
    serve_gpu_requests,
    torch_load_cpu,
)


GPU_MODEL_PATH = os.environ.get("PARTEE_MODEL_PATH", "")
SCRIPT_DIR = Path(__file__).resolve().parent
GPU_MANIFEST_PATH = Path(
    os.environ.get("PARTEE_GPU_MANIFEST_PATH", str(SCRIPT_DIR / "gpu_manifest.pt"))
)
GPU_SHARED_DIR = Path(
    os.environ.get("PARTEE_GPU_SHARED_DIR", str(SCRIPT_DIR / "two_host_rpc"))
)
GPU_DEVICES = tuple(
    item.strip()
    for item in os.environ.get("PARTEE_GPU_DEVICES", "cuda:0,cuda:1").split(",")
    if item.strip()
)
GEMM_OUTPUT_CHUNK_ROWS = 256
QUANTIZE_CHUNK_ROWS = 256
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_snapshot(path_text: str) -> Path:
    candidate = Path(path_text)
    if (candidate / "config.json").is_file():
        return candidate
    snapshots = candidate / "snapshots"
    if snapshots.is_dir():
        matches = sorted(path.parent for path in snapshots.glob("*/config.json"))
        if matches:
            return matches[-1]
    raise FileNotFoundError("No Llama config.json found under {}".format(candidate))


def validate_devices() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on the GPU host")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "Secure Llama timing requires cuda:0 and cuda:1, but only {} GPU(s) "
            "are visible".format(torch.cuda.device_count())
        )


@torch.no_grad()
def quantize_matrix_to_device(
    matrix: torch.Tensor,
    scales: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    qmax = (1 << (OTP_BITS - 1)) - 1
    quantized = torch.empty(
        matrix.shape[0], matrix.shape[1], dtype=torch.int8, device=device
    )
    for start in range(0, matrix.shape[0], QUANTIZE_CHUNK_ROWS):
        stop = min(start + QUANTIZE_CHUNK_ROWS, matrix.shape[0])
        source = matrix[start:stop].detach().float().to(device)
        scale = scales[start:stop].to(device)
        quantized[start:stop] = (
            (source / scale.unsqueeze(1))
            .round()
            .clamp(-qmax, qmax)
            .to(torch.int8)
        )
        del source, scale
    return quantized


class DualGPUFrozenGEMMService:
    def __init__(self, model_path: Path, manifest_path: Path) -> None:
        validate_devices()
        self.devices = tuple(torch.device(name) for name in GPU_DEVICES)
        self.manifest = torch_load_cpu(manifest_path)
        if int(self.manifest.get("version", -1)) != 3:
            raise RuntimeError("Expected GPU manifest version 3")
        if [tuple(item) for item in self.manifest["critical_heads"]] != [
            tuple(item) for item in CRITICAL_HEADS
        ]:
            raise RuntimeError(
                "PARTEE_CRITICAL_HEADS differs from the public GPU manifest"
            )
        self.run_id = str(self.manifest["run_id"])
        self.authorized_materials = {}
        for step_id, step in self.manifest["steps"].items():
            for layer_name, directions in step["layers"].items():
                for direction, records in directions.items():
                    for invocation, record in enumerate(records):
                        material_id = str(record["material_id"])
                        self.authorized_materials[material_id] = (
                            str(step_id), layer_name, direction, invocation
                        )
        self.seen_materials = set()
        print("Loading public Llama on CPU; only quantized frozen matrices remain afterward...")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            num_labels=2,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).cpu()
        self.weights: Dict[str, Dict[str, torch.Tensor]] = {}
        self.weight_devices: Dict[str, torch.device] = {}
        expected_names = set(self.manifest["layers"])
        prepared_names = set()
        for layer_index, layer in enumerate(model.model.layers):
            device = self.devices[0 if layer_index < 16 else 1]
            candidates = {
                "model.layers.{}.self_attn.q_proj".format(layer_index): layer.self_attn.q_proj,
                "model.layers.{}.self_attn.k_proj".format(layer_index): layer.self_attn.k_proj,
                "model.layers.{}.self_attn.v_proj".format(layer_index): layer.self_attn.v_proj,
                "model.layers.{}.self_attn.o_proj".format(layer_index): layer.self_attn.o_proj,
                "model.layers.{}.mlp.gate_proj".format(layer_index): layer.mlp.gate_proj,
                "model.layers.{}.mlp.up_proj".format(layer_index): layer.mlp.up_proj,
                "model.layers.{}.mlp.down_proj".format(layer_index): layer.mlp.down_proj,
            }
            for layer_name, linear in candidates.items():
                record = self.manifest["layers"][layer_name]
                critical_rows = record["critical_rows"].to(torch.long)
                public_rows = complement_rows(linear.out_features, critical_rows)
                public_weight = linear.weight.detach().cpu().index_select(0, public_rows)
                forward = quantize_matrix_to_device(
                    public_weight,
                    record["forward_weight_scale"],
                    device,
                )
                backward = quantize_matrix_to_device(
                    public_weight.transpose(0, 1),
                    record["backward_weight_scale"],
                    device,
                )
                self.weights[layer_name] = {
                    "forward": forward,
                    "backward": backward,
                }
                self.weight_devices[layer_name] = device
                prepared_names.add(layer_name)
                print(
                    "Quantized {:3d}/{:3d} on {}: {}".format(
                        len(prepared_names), len(expected_names), device, layer_name
                    ),
                    flush=True,
                )
                del public_weight
        if prepared_names != expected_names:
            raise RuntimeError(
                "GPU matrix set mismatch; missing={}, unexpected={}".format(
                    sorted(expected_names - prepared_names),
                    sorted(prepared_names - expected_names),
                )
            )
        del model
        gc.collect()
        for device in self.devices:
            torch.cuda.empty_cache()
        self.calls = 0
        self.compute_seconds = 0.0
        print("Public Llama modules released; GPU service retains INT8 matrices only.")
        for device in self.devices:
            allocated = torch.cuda.memory_allocated(device) / 2 ** 30
            print("{} persistent public matrix memory: {:.3f} GiB".format(device, allocated))

    def status(self) -> Dict:
        per_device_bytes = {
            str(device): int(
                sum(
                    matrix.numel() * matrix.element_size()
                    for layer_name, directions in self.weights.items()
                    if self.weight_devices[layer_name] == device
                    for matrix in directions.values()
                )
            )
            for device in self.devices
        }
        return {
            "status": "masked_gemm_only",
            "run_id": self.run_id,
            "matrix_count": len(self.weights),
            "authorized_material_count": len(self.authorized_materials),
            "consumed_material_count": len(self.seen_materials),
            "directions_per_matrix": 2,
            "calls": self.calls,
            "compute_seconds": self.compute_seconds,
            "persistent_quantized_matrix_bytes_by_device": per_device_bytes,
            "persistent_quantized_matrix_bytes": int(sum(per_device_bytes.values())),
        }

    @torch.no_grad()
    def gemm(
        self,
        run_id: str,
        step_id: str,
        material_id: str,
        invocation: int,
        layer_name: str,
        direction: str,
        masked_tensor: torch.Tensor,
    ) -> Dict:
        if str(run_id) != self.run_id:
            raise RuntimeError("GPU/TEE run identifier mismatch")
        material_id = str(material_id)
        expected = self.authorized_materials.get(material_id)
        received = (str(step_id), layer_name, direction, int(invocation))
        if expected != received:
            raise RuntimeError(
                "Unauthorized or desynchronized OTP invocation: expected={}, received={}".format(
                    expected, received
                )
            )
        if material_id in self.seen_materials:
            raise RuntimeError("GPU rejected repeated OTP material: {}".format(material_id))
        self.seen_materials.add(material_id)
        if layer_name not in self.weights:
            raise KeyError("Unknown public matrix {}".format(layer_name))
        if direction not in {"forward", "backward"}:
            raise ValueError("direction must be forward or backward")
        matrix = self.weights[layer_name][direction]
        device = self.weight_devices[layer_name]
        if masked_tensor.shape[-1] != matrix.shape[1]:
            raise RuntimeError(
                "Masked input width {} != {} matrix width {} for {}".format(
                    masked_tensor.shape[-1], direction, matrix.shape[1], layer_name
                )
            )
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        masked_device = masked_tensor.to(device=device, dtype=torch.float64)
        output = torch.empty(
            *masked_tensor.shape[:-1], matrix.shape[0], dtype=torch.int64
        )
        for start in range(0, matrix.shape[0], GEMM_OUTPUT_CHUNK_ROWS):
            stop = min(start + GEMM_OUTPUT_CHUNK_ROWS, matrix.shape[0])
            partial = F.linear(
                masked_device,
                matrix[start:stop].to(torch.float64),
            )
            partial = torch.remainder(partial, OTP_MODULUS)
            output[..., start:stop] = partial.round().to(torch.int64).cpu()
            del partial
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start_time
        self.calls += 1
        self.compute_seconds += seconds
        return {
            "encrypted_result": output,
            "compute_seconds": seconds,
            "device": str(device),
        }


def main() -> None:
    set_seed(SEED)
    if not GPU_MODEL_PATH:
        raise RuntimeError("Set PARTEE_MODEL_PATH to the public Llama snapshot")
    model_path = resolve_snapshot(GPU_MODEL_PATH)
    if not GPU_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "GPU manifest not found: {}. Run the trusted OTP preprocessor first.".format(
                GPU_MANIFEST_PATH
            )
        )
    print("GPU shared RPC directory: {}".format(GPU_SHARED_DIR))
    print("Layer placement: layers 0-15 -> cuda:0; layers 16-31 -> cuda:1")
    service = DualGPUFrozenGEMMService(model_path, GPU_MANIFEST_PATH)
    serve_gpu_requests(service, GPU_SHARED_DIR)


if __name__ == "__main__":
    main()
