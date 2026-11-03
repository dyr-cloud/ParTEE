#!/usr/bin/env python3
"""Common secure two-host runtime for multi-step Llama-3.1-8B training.

The TEE owns every plaintext activation and the autograd graph.  The GPU host
is only a public frozen-GEMM service and receives uniformly masked integer
tensors.  No RPC response contains a decrypted activation.
"""

from __future__ import annotations

import io
import json
import math
import os
import secrets
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _critical_heads_from_environment() -> Sequence[Tuple[int, int]]:
    value = os.environ.get("PARTEE_CRITICAL_HEADS", "2:5").strip()
    if not value:
        return []
    result = []
    for item in value.split(","):
        layer, group = item.strip().split(":", 1)
        result.append((int(layer), int(group)))
    return result


CRITICAL_HEADS = list(_critical_heads_from_environment())
EXPERIMENT_NAME = "partee_ours" if CRITICAL_HEADS else "all_frozen_otp"
OTP_BITS = 8
OTP_MODULUS = 2_147_483_647
LORA_RANK = 8
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
RPC_POLL_SECONDS = 0.002
RPC_TIMEOUT_SECONDS = 7_200.0


def torch_load_cpu(source):
    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(source, map_location="cpu")


def atomic_torch_save(payload, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        ".{}.{}.{}.tmp".format(destination.name, os.getpid(), time.time_ns())
    )
    torch.save(payload, temporary)
    os.replace(str(temporary), str(destination))


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def payload_tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return tensor_bytes(value)
    if isinstance(value, dict):
        return sum(payload_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(payload_tensor_bytes(item) for item in value)
    return 0


def layer_and_module(layer_name: str) -> Tuple[Optional[int], str]:
    parts = layer_name.split(".")
    layer_index = None
    if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
        try:
            layer_index = int(parts[2])
        except ValueError:
            layer_index = None
    return layer_index, parts[-1]


def complement_rows(out_features: int, excluded: torch.Tensor) -> torch.Tensor:
    keep = torch.ones(out_features, dtype=torch.bool)
    if excluded.numel():
        keep[excluded.to(dtype=torch.long, device="cpu")] = False
    return torch.arange(out_features, dtype=torch.long)[keep]


def rows_for_projection(
    layer_index: int,
    critical_groups: Sequence[Tuple[int, int]],
    kind: str,
    head_dim: int,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> torch.Tensor:
    groups = [group for layer, group in critical_groups if layer == layer_index]
    if not groups:
        return torch.empty(0, dtype=torch.long)
    query_per_group = num_attention_heads // num_key_value_heads
    rows = []
    for group in groups:
        if not 0 <= group < num_key_value_heads:
            raise ValueError("Invalid GQA group {} at layer {}".format(group, layer_index))
        if kind == "q_proj":
            first_head = group * query_per_group
            first_row = first_head * head_dim
            count = query_per_group * head_dim
        elif kind in {"k_proj", "v_proj"}:
            first_row = group * head_dim
            count = head_dim
        else:
            raise ValueError("Critical rows only apply to q/k/v projections")
        rows.extend(range(first_row, first_row + count))
    return torch.tensor(sorted(set(rows)), dtype=torch.long)


class FileRPCClient:
    """TEE-side client for the GPU-only frozen GEMM service."""

    def __init__(self, shared_dir: Path) -> None:
        self.shared_dir = Path(shared_dir)
        self.request_dir = self.shared_dir / "gpu_requests"
        self.response_dir = self.shared_dir / "gpu_responses"
        self.ready_path = self.shared_dir / "gpu_ready.pt"
        self.request_dir.mkdir(parents=True, exist_ok=True)
        self.response_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.calls = 0
        self.wait_seconds = 0.0
        self.gpu_compute_seconds = 0.0
        self.rpc_events = []

    def wait_until_ready(self) -> None:
        start = time.perf_counter()
        while not self.ready_path.is_file():
            if time.perf_counter() - start > RPC_TIMEOUT_SECONDS:
                raise TimeoutError("GPU RPC server did not become ready: {}".format(self.ready_path))
            time.sleep(RPC_POLL_SECONDS)

    def call(self, operation: str, **payload):
        call_start = time.perf_counter()
        self.counter += 1
        self.calls += 1
        request_id = "{:08d}_{:08d}".format(os.getpid(), self.counter)
        request_path = self.request_dir / ("request_{}.pt".format(request_id))
        response_path = self.response_dir / ("response_{}.pt".format(request_id))
        request_write_start = time.perf_counter()
        atomic_torch_save(
            {"request_id": request_id, "operation": operation, "payload": payload},
            request_path,
        )
        request_write_seconds = time.perf_counter() - request_write_start
        serialized_request_bytes = request_path.stat().st_size
        start = time.perf_counter()
        while not response_path.is_file():
            if time.perf_counter() - start > RPC_TIMEOUT_SECONDS:
                raise TimeoutError("Timed out waiting for {}".format(response_path))
            time.sleep(RPC_POLL_SECONDS)
        response_wait_seconds = time.perf_counter() - start
        serialized_response_bytes = response_path.stat().st_size
        response_read_start = time.perf_counter()
        response = torch_load_cpu(response_path)
        response_read_seconds = time.perf_counter() - response_read_start
        response_path.unlink(missing_ok=True)
        self.wait_seconds += response_wait_seconds
        if not response.get("ok", False):
            raise RuntimeError(
                "GPU operation {!r} failed:\n{}".format(
                    operation, response.get("error", "unknown error")
                )
            )
        result = response.get("result")
        if isinstance(result, dict):
            self.gpu_compute_seconds += float(result.get("compute_seconds", 0.0))
        event = {
            "operation": operation,
            "request_payload_tensor_bytes": payload_tensor_bytes(payload),
            "response_payload_tensor_bytes": payload_tensor_bytes(result),
            "serialized_request_bytes": int(serialized_request_bytes),
            "serialized_response_bytes": int(serialized_response_bytes),
            "tee_to_gpu_request_write_seconds": request_write_seconds,
            "response_wait_seconds": response_wait_seconds,
            "gpu_to_tee_response_read_seconds": response_read_seconds,
            "rpc_total_seconds": time.perf_counter() - call_start,
        }
        if operation == "gemm":
            event["layer_name"] = payload["layer_name"]
            event["direction"] = payload["direction"]
            event["masked_tensor_bytes"] = tensor_bytes(payload["masked_tensor"])
            if isinstance(result, dict):
                for key in (
                    "compute_seconds",
                    "device",
                    "matrix_shape",
                    "matrix_elements",
                    "matrix_bytes",
                    "masked_input_shape",
                    "masked_input_bytes",
                    "encrypted_output_shape",
                    "encrypted_output_bytes",
                ):
                    if key in result:
                        event[key] = result[key]
        self.rpc_events.append(event)
        return result


def clean_gpu_rpc_directory(shared_dir: Path) -> None:
    shared_dir.mkdir(parents=True, exist_ok=True)
    for name in ("gpu_requests", "gpu_responses"):
        directory = shared_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.pt"):
            path.unlink(missing_ok=True)
    (shared_dir / "gpu_ready.pt").unlink(missing_ok=True)


def serve_gpu_requests(service, shared_dir: Path) -> None:
    """Run on the untrusted GPU host; accept masked GEMM requests only."""
    clean_gpu_rpc_directory(shared_dir)
    request_dir = shared_dir / "gpu_requests"
    response_dir = shared_dir / "gpu_responses"
    atomic_torch_save(
        {
            "pid": os.getpid(),
            "status": "masked_gemm_only",
            "forbidden_payload_fields": [
                "plaintext", "input_ids", "labels", "decrypted_output"
            ],
        },
        shared_dir / "gpu_ready.pt",
    )
    print("GPU masked-GEMM RPC server listening in {}".format(shared_dir), flush=True)
    running = True
    while running:
        requests = sorted(request_dir.glob("request_*.pt"))
        if not requests:
            time.sleep(RPC_POLL_SECONDS)
            continue
        request_path = requests[0]
        request = torch_load_cpu(request_path)
        request_path.unlink(missing_ok=True)
        request_id = request["request_id"]
        operation = request["operation"]
        payload = request["payload"]
        response_path = response_dir / ("response_{}.pt".format(request_id))
        try:
            forbidden = {"plaintext", "input_ids", "labels", "decrypted_output"}
            overlap = forbidden.intersection(payload)
            if overlap:
                raise RuntimeError("Security violation: forbidden GPU payload fields {}".format(sorted(overlap)))
            if operation == "ping":
                result = service.status()
            elif operation == "gemm":
                required = {
                    "run_id",
                    "step_id",
                    "material_id",
                    "invocation",
                    "layer_name",
                    "direction",
                    "masked_tensor",
                }
                if set(payload) != required:
                    raise RuntimeError(
                        "GEMM payload fields differ from the version-3 protocol; "
                        "received {}".format(sorted(payload))
                    )
                result = service.gemm(**payload)
            elif operation == "stop":
                result = service.status()
                running = False
            else:
                raise KeyError("Unknown GPU RPC operation: {}".format(operation))
            response = {"ok": True, "result": result}
        except Exception:
            response = {"ok": False, "error": traceback.format_exc()}
        atomic_torch_save(response, response_path)
    (shared_dir / "gpu_ready.pt").unlink(missing_ok=True)
    print("GPU masked-GEMM RPC server stopped.", flush=True)


def _decrypt_file_profiled(aesgcm, path: Path, aad: str):
    read_start = time.perf_counter()
    payload = path.read_bytes()
    read_seconds = time.perf_counter() - read_start
    if len(payload) < 29:
        raise RuntimeError("Truncated encrypted file: {}".format(path))
    decrypt_start = time.perf_counter()
    raw = aesgcm.decrypt(payload[:12], payload[12:], aad.encode("utf-8"))
    decrypt_seconds = time.perf_counter() - decrypt_start
    deserialize_start = time.perf_counter()
    value = torch_load_cpu(io.BytesIO(raw))
    deserialize_seconds = time.perf_counter() - deserialize_start
    return value, {
        "encrypted_file_bytes": int(len(payload)),
        "decrypted_serialized_bytes": int(len(raw)),
        "material_read_seconds": read_seconds,
        "aes_decrypt_seconds": decrypt_seconds,
        "material_deserialize_seconds": deserialize_seconds,
    }


def _decrypt_file(aesgcm, path: Path, aad: str):
    value, _ = _decrypt_file_profiled(aesgcm, path, aad)
    return value


def quantize_activation(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    qmax = (1 << (OTP_BITS - 1)) - 1
    source = tensor.detach().float().cpu()
    maximum = source.abs().amax().clamp_min(1e-12)
    scale = maximum / qmax
    quantized = (source / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return quantized, scale


class SecureLinearEngine:
    """TEE-resident OTP state; plaintext tensors never leave this object."""

    def __init__(
        self,
        offline_root: Path,
        key_path: Path,
        client: FileRPCClient,
        ledger_path: Optional[Path] = None,
    ):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise RuntimeError("The TEE Python image needs the cryptography package.") from exc
        key = Path(key_path).read_bytes()
        if len(key) != 32:
            raise RuntimeError("OTP AES key must contain exactly 32 bytes")
        self.aesgcm = AESGCM(key)
        self.offline_root = Path(offline_root)
        self.material_dir = self.offline_root / "otp_material"
        self.manifest = torch_load_cpu(self.offline_root / "manifest.pt")
        if int(self.manifest.get("version", -1)) != 3:
            raise RuntimeError("Expected multi-step secure OTP manifest version 3")
        if [tuple(item) for item in self.manifest["critical_heads"]] != [
            tuple(item) for item in CRITICAL_HEADS
        ]:
            raise RuntimeError(
                "PARTEE_CRITICAL_HEADS differs from the prepared OTP manifest"
            )
        self.client = client
        self.run_id = str(self.manifest["run_id"])
        self.ledger_path = Path(
            ledger_path
            if ledger_path is not None
            else self.offline_root / "consumption_ledger.bin"
        )
        self.active_step = None
        self.completed_steps = set()
        self.consumed = set()
        self._step_invocations = {}
        self._step_reserved = set()
        self._load_or_initialize_ledger()
        self.otp_forward_calls = 0
        self.otp_backward_calls = 0
        self.otp_bytes = 0
        self.otp_events = []
        self.tee_events = []

    def _ledger_payload(self) -> Dict:
        return {
            "version": 1,
            "run_id": self.run_id,
            "manifest_id": str(self.manifest["manifest_id"]),
            "active_step": self.active_step,
            "completed_steps": sorted(self.completed_steps),
            "consumed_material_ids": sorted(self.consumed),
        }

    def _persist_ledger(self) -> None:
        plaintext = json.dumps(
            self._ledger_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        nonce = secrets.token_bytes(12)
        ciphertext = self.aesgcm.encrypt(
            nonce, plaintext, b"partee-otp-consumption-ledger-v1"
        )
        destination = self.ledger_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            ".{}.{}.{}.tmp".format(destination.name, os.getpid(), time.time_ns())
        )
        with temporary.open("wb") as handle:
            handle.write(nonce)
            handle.write(ciphertext)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(destination))

    def _load_or_initialize_ledger(self) -> None:
        if not self.ledger_path.is_file():
            self._persist_ledger()
            return
        payload = self.ledger_path.read_bytes()
        if len(payload) < 29:
            raise RuntimeError("Truncated OTP consumption ledger")
        try:
            plaintext = self.aesgcm.decrypt(
                payload[:12],
                payload[12:],
                b"partee-otp-consumption-ledger-v1",
            )
            state = json.loads(plaintext.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("OTP consumption ledger authentication failed") from exc
        if state.get("run_id") != self.run_id:
            raise RuntimeError(
                "OTP ledger belongs to run {!r}, not {!r}".format(
                    state.get("run_id"), self.run_id
                )
            )
        if state.get("manifest_id") != str(self.manifest["manifest_id"]):
            raise RuntimeError("OTP ledger manifest identifier mismatch")
        self.consumed = set(state.get("consumed_material_ids", []))
        self.completed_steps = set(state.get("completed_steps", []))
        interrupted = state.get("active_step")
        if interrupted is not None:
            raise RuntimeError(
                "Interrupted OTP step {!r} is recorded in the durable ledger. "
                "The implementation fails closed rather than reusing its material; "
                "prepare a fresh run package to continue.".format(interrupted)
            )

    def begin_step(self, step_id: str) -> None:
        step_id = str(step_id)
        if self.active_step is not None:
            raise RuntimeError("OTP step already active: {}".format(self.active_step))
        if step_id in self.completed_steps:
            raise RuntimeError("OTP step already completed: {}".format(step_id))
        if step_id not in self.manifest["steps"]:
            raise KeyError("Unknown OTP step: {}".format(step_id))
        self.active_step = step_id
        self._step_invocations = {}
        self._step_reserved = set()
        self._persist_ledger()

    def complete_step(self) -> None:
        if self.active_step is None:
            raise RuntimeError("No OTP step is active")
        expected = int(self.manifest["steps"][self.active_step]["material_count"])
        if len(self._step_reserved) != expected:
            raise RuntimeError(
                "OTP step {} consumed {} of {} expected records".format(
                    self.active_step, len(self._step_reserved), expected
                )
            )
        self.completed_steps.add(self.active_step)
        self.active_step = None
        self._step_invocations = {}
        self._step_reserved = set()
        self._persist_ledger()

    def record_tee_event(
        self,
        category: str,
        layer_name: str,
        direction: str,
        seconds: float,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        extra: Optional[Dict] = None,
    ) -> None:
        layer_index, module_type = layer_and_module(layer_name)
        event = {
            "category": category,
            "layer": layer_index,
            "layer_name": layer_name,
            "module": module_type,
            "direction": direction,
            "seconds": float(seconds),
            "input_shape": list(input_tensor.shape),
            "output_shape": list(output_tensor.shape),
            "input_bytes": tensor_bytes(input_tensor),
            "output_bytes": tensor_bytes(output_tensor),
        }
        if extra:
            event.update(extra)
        self.tee_events.append(event)

    def _material(self, layer_name: str, direction: str) -> Dict:
        if self.active_step is None:
            raise RuntimeError("begin_step() must be called before an OTP operation")
        token = (layer_name, direction)
        invocation = int(self._step_invocations.get(token, 0))
        records = self.manifest["steps"][self.active_step]["layers"][layer_name][direction]
        if invocation >= len(records):
            raise RuntimeError(
                "No OTP record for step={} layer={} direction={} invocation={}".format(
                    self.active_step, layer_name, direction, invocation
                )
            )
        record = records[invocation]
        material_id = str(record["material_id"])
        if material_id in self.consumed:
            raise RuntimeError("OTP material already consumed: {}".format(material_id))
        self._step_invocations[token] = invocation + 1
        self.consumed.add(material_id)
        self._step_reserved.add(material_id)
        # Reserve durably before decryption or RPC. A crash therefore wastes a
        # record and fails closed instead of making reuse possible.
        self._persist_ledger()
        path = self.material_dir / record["file"]
        material, timing = _decrypt_file_profiled(
            self.aesgcm, path, str(record["aad"])
        )
        if str(material.get("material_id")) != material_id:
            raise RuntimeError("OTP material identifier mismatch in {}".format(path))
        return material, timing, record, invocation

    def run(self, layer_name: str, direction: str, plaintext: torch.Tensor) -> torch.Tensor:
        total_start = time.perf_counter()
        material, material_timing, invocation_record, invocation = self._material(
            layer_name, direction
        )
        quantize_start = time.perf_counter()
        quantized, input_scale = quantize_activation(plaintext)
        quantize_seconds = time.perf_counter() - quantize_start
        expected_shape = tuple(int(v) for v in material["input_shape"])
        if tuple(quantized.shape) != expected_shape:
            raise RuntimeError(
                "OTP shape mismatch for {} {}: expected {}, received {}".format(
                    layer_name, direction, expected_shape, tuple(quantized.shape)
                )
            )
        mask_generation_start = time.perf_counter()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(material["mask_seed"]))
        mask = torch.randint(
            0, OTP_MODULUS, expected_shape, dtype=torch.int64, generator=generator
        )
        mask_generation_seconds = time.perf_counter() - mask_generation_start
        masking_start = time.perf_counter()
        masked = torch.remainder(quantized.to(torch.int64) + mask, OTP_MODULUS)
        masking_seconds = time.perf_counter() - masking_start
        rpc_start = time.perf_counter()
        result = self.client.call(
            "gemm",
            run_id=self.run_id,
            step_id=self.active_step,
            material_id=invocation_record["material_id"],
            invocation=invocation,
            layer_name=layer_name,
            direction=direction,
            masked_tensor=masked,
        )
        rpc_seconds = time.perf_counter() - rpc_start
        encrypted_result = result["encrypted_result"]
        reconstruction_start = time.perf_counter()
        recovered = torch.remainder(
            encrypted_result - material["mask_product"], OTP_MODULUS
        )
        half = OTP_MODULUS // 2
        centered = torch.where(recovered > half, recovered - OTP_MODULUS, recovered)
        reconstruction_seconds = time.perf_counter() - reconstruction_start
        dequantize_start = time.perf_counter()
        output = centered.to(torch.float32)
        output = output * input_scale * material["weight_scale"]
        if direction == "forward" and material.get("bias") is not None:
            output = output + material["bias"]
        dequantize_seconds = time.perf_counter() - dequantize_start
        self.otp_bytes += 8 * (masked.numel() + encrypted_result.numel())
        if direction == "forward":
            self.otp_forward_calls += 1
        else:
            self.otp_backward_calls += 1
        layer_index, module_type = layer_and_module(layer_name)
        static_record = self.manifest["layers"][layer_name]
        direction_record = invocation_record
        rpc_event = self.client.rpc_events[-1]
        mask_product_bytes = tensor_bytes(material["mask_product"])
        weight_scale_bytes = tensor_bytes(material["weight_scale"])
        bias_bytes = (
            0 if material.get("bias") is None else tensor_bytes(material["bias"])
        )
        self.otp_events.append(
            {
                "category": "otp",
                "layer": layer_index,
                "layer_name": layer_name,
                "module": module_type,
                "direction": direction,
                "run_id": self.run_id,
                "step_id": self.active_step,
                "material_id": invocation_record["material_id"],
                "invocation": invocation,
                "input_shape": list(plaintext.shape),
                "output_shape": list(output.shape),
                "input_plaintext_bytes": tensor_bytes(plaintext),
                "output_plaintext_bytes": tensor_bytes(output),
                "quantized_input_bytes": tensor_bytes(quantized),
                "mask_bytes": tensor_bytes(mask),
                "masked_tensor_bytes": tensor_bytes(masked),
                "encrypted_result_bytes": tensor_bytes(encrypted_result),
                "mask_product_bytes": mask_product_bytes,
                "weight_scale_bytes": weight_scale_bytes,
                "bias_bytes": bias_bytes,
                "encrypted_material_file_bytes": material_timing["encrypted_file_bytes"],
                "original_weight_shape": static_record.get("original_weight_shape"),
                "original_weight_elements": static_record.get("original_weight_elements"),
                "original_weight_bfloat16_bytes": static_record.get(
                    "original_weight_bfloat16_bytes"
                ),
                "public_weight_shape": static_record.get("public_weight_shape"),
                "public_weight_elements": static_record.get("public_weight_elements"),
                "public_quantized_weight_bytes": static_record.get(
                    "public_quantized_weight_bytes",
                    direction_record.get("quantized_matrix_bytes"),
                ),
                "critical_row_count": static_record.get("critical_row_count", 0),
                "critical_weight_float32_bytes": static_record.get(
                    "critical_weight_float32_bytes", 0
                ),
                "gpu_device": result.get("device"),
                "gpu_matrix_shape": direction_record.get("matrix_shape"),
                "material_read_seconds": material_timing["material_read_seconds"],
                "aes_decrypt_seconds": material_timing["aes_decrypt_seconds"],
                "material_deserialize_seconds": material_timing[
                    "material_deserialize_seconds"
                ],
                "activation_quantize_seconds": quantize_seconds,
                "mask_generation_seconds": mask_generation_seconds,
                "mask_apply_seconds": masking_seconds,
                "tee_to_gpu_request_write_seconds": rpc_event[
                    "tee_to_gpu_request_write_seconds"
                ],
                "gpu_response_wait_seconds": rpc_event["response_wait_seconds"],
                "gpu_to_tee_response_read_seconds": rpc_event[
                    "gpu_to_tee_response_read_seconds"
                ],
                "rpc_total_seconds": rpc_seconds,
                "gpu_masked_gemm_seconds": float(result.get("compute_seconds", 0.0)),
                "reconstruction_seconds": reconstruction_seconds,
                "dequantize_seconds": dequantize_seconds,
                "otp_total_seconds": time.perf_counter() - total_start,
                "serialized_request_bytes": rpc_event["serialized_request_bytes"],
                "serialized_response_bytes": rpc_event["serialized_response_bytes"],
            }
        )
        return output


class _SecureOTPLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, engine, layer_name):
        ctx.engine = engine
        ctx.layer_name = layer_name
        return engine.run(layer_name, "forward", tensor)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = ctx.engine.run(ctx.layer_name, "backward", grad_output)
        return grad_input, None, None


class SecureOTPLinear(nn.Module):
    def __init__(self, engine: SecureLinearEngine, layer_name: str) -> None:
        super().__init__()
        object.__setattr__(self, "_engine", engine)
        self.layer_name = layer_name

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return _SecureOTPLinearFunction.apply(tensor, self._engine, self.layer_name)


class _TimedLoRAFunction(torch.autograd.Function):
    """LoRA with an explicitly timed, complete analytical backward pass."""

    @staticmethod
    def forward(
        ctx,
        tensor,
        weight_a,
        weight_b,
        dropout_probability,
        training,
        scaling,
        engine,
        layer_name,
    ):
        dropout_probability = float(dropout_probability)
        training = bool(training)
        scaling = float(scaling)

        dropout_forward_start = time.perf_counter()
        if training and dropout_probability > 0.0:
            # This is the same inverted-dropout rule used by nn.Dropout, but
            # retaining the mask lets the custom backward compute dX exactly.
            keep_probability = 1.0 - dropout_probability
            dropout_mask = (
                torch.rand_like(tensor).lt(keep_probability).to(tensor.dtype)
                / keep_probability
            )
            dropped = tensor * dropout_mask
        else:
            dropout_mask = torch.empty(
                0, dtype=tensor.dtype, device=tensor.device
            )
            dropped = tensor
        dropout_forward_seconds = (
            time.perf_counter() - dropout_forward_start
        )

        # LoRA forward timing intentionally excludes mask generation and the
        # dropout application. It covers only A/B projections and scaling.
        forward_start = time.perf_counter()
        rank_features = F.linear(dropped, weight_a)
        output = F.linear(rank_features, weight_b) * scaling
        forward_seconds = time.perf_counter() - forward_start

        ctx.save_for_backward(
            dropped,
            rank_features,
            weight_a,
            weight_b,
            dropout_mask,
        )
        ctx.scaling = scaling
        ctx.engine = engine
        ctx.layer_name = layer_name

        engine.record_tee_event(
            "lora",
            layer_name,
            "forward",
            forward_seconds,
            tensor,
            output,
            {
                "lora_a_shape": list(weight_a.shape),
                "lora_b_shape": list(weight_b.shape),
                "lora_parameter_bytes": (
                    tensor_bytes(weight_a) + tensor_bytes(weight_b)
                ),
                "dropout_forward_seconds": dropout_forward_seconds,
                "timing_scope": "a_b_projections_and_scaling_only",
            },
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # The LoRA timer encloses all matrix-gradient work (dB, dA and dX),
        # but deliberately excludes the separate dropout backward operation.
        backward_start = time.perf_counter()
        dropped, rank_features, weight_a, weight_b, dropout_mask = (
            ctx.saved_tensors
        )

        grad_scaled = grad_output * ctx.scaling
        grad_scaled_2d = grad_scaled.reshape(-1, grad_scaled.shape[-1])
        rank_features_2d = rank_features.reshape(
            -1, rank_features.shape[-1]
        )
        dropped_2d = dropped.reshape(-1, dropped.shape[-1])

        # y = scaling * (dropout(x) A^T) B^T
        grad_weight_b = grad_scaled_2d.transpose(0, 1).matmul(
            rank_features_2d
        )
        grad_rank = grad_scaled.matmul(weight_b)
        grad_rank_2d = grad_rank.reshape(-1, grad_rank.shape[-1])
        grad_weight_a = grad_rank_2d.transpose(0, 1).matmul(dropped_2d)
        grad_dropped = grad_rank.matmul(weight_a)
        backward_seconds = time.perf_counter() - backward_start

        dropout_backward_start = time.perf_counter()
        if dropout_mask.numel():
            grad_input = grad_dropped * dropout_mask
        else:
            grad_input = grad_dropped
        dropout_backward_seconds = (
            time.perf_counter() - dropout_backward_start
        )
        ctx.engine.record_tee_event(
            "lora",
            ctx.layer_name,
            "backward",
            backward_seconds,
            grad_output,
            grad_input,
            {
                "lora_a_shape": list(weight_a.shape),
                "lora_b_shape": list(weight_b.shape),
                "lora_parameter_bytes": (
                    tensor_bytes(weight_a) + tensor_bytes(weight_b)
                ),
                "dropout_backward_seconds": dropout_backward_seconds,
                "timing_scope": "dx_da_db_excluding_dropout_backward",
            },
        )

        return (
            grad_input,
            grad_weight_a,
            grad_weight_b,
            None,
            None,
            None,
            None,
            None,
        )


class LoRAAdapter(nn.Module):
    def __init__(
        self,
        engine: SecureLinearEngine,
        layer_name: str,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_engine", engine)
        self.layer_name = layer_name
        self.lora_a = nn.Linear(in_features, LORA_RANK, bias=False)
        self.lora_b = nn.Linear(LORA_RANK, out_features, bias=False)
        self.dropout = nn.Dropout(LORA_DROPOUT)
        self.scaling = LORA_ALPHA / LORA_RANK
        nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return _TimedLoRAFunction.apply(
            tensor,
            self.lora_a.weight,
            self.lora_b.weight,
            self.dropout.p,
            self.training,
            self.scaling,
            self._engine,
            self.layer_name,
        )


class SecureSplitProjection(nn.Module):
    def __init__(
        self,
        engine: SecureLinearEngine,
        layer_name: str,
        in_features: int,
        out_features: int,
        public_rows: torch.Tensor,
        critical_rows: torch.Tensor,
        critical_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.public = SecureOTPLinear(engine, layer_name)
        self.lora = LoRAAdapter(engine, layer_name, in_features, out_features)
        self.out_features = int(out_features)
        self.register_buffer("public_rows", public_rows.to(torch.long))
        self.register_buffer("critical_rows", critical_rows.to(torch.long))
        self.register_buffer("critical_weight", critical_weight.float())
        object.__setattr__(self, "_engine", engine)
        self.layer_name = layer_name

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        compact = self.public(tensor)
        output = compact.new_zeros(*compact.shape[:-1], self.out_features)
        output = output.index_copy(-1, self.public_rows, compact)
        if self.critical_rows.numel():
            critical_start = time.perf_counter()
            critical = F.linear(tensor, self.critical_weight)
            self._engine.record_tee_event(
                "critical_head",
                self.layer_name,
                "forward",
                time.perf_counter() - critical_start,
                tensor,
                critical,
                {
                    "critical_weight_shape": list(self.critical_weight.shape),
                    "critical_weight_bytes": tensor_bytes(self.critical_weight),
                },
            )
            output = output.index_copy(-1, self.critical_rows, critical)
        return output + self.lora(tensor)


class SecureRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.register_buffer("weight", weight.float())
        self.eps = float(eps)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        variance = tensor.float().pow(2).mean(-1, keepdim=True)
        return tensor.float() * torch.rsqrt(variance + self.eps) * self.weight


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class SecureRotaryEmbedding(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        dim = int(config["head_dim"])
        base = float(config.get("rope_theta", 500000.0))
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        scaling = config.get("rope_scaling")
        if isinstance(scaling, dict) and scaling:
            rope_type = scaling.get("rope_type", scaling.get("type", ""))
            if rope_type == "llama3":
                factor = float(scaling["factor"])
                low_factor = float(scaling["low_freq_factor"])
                high_factor = float(scaling["high_freq_factor"])
                old_length = float(scaling["original_max_position_embeddings"])
                wavelength = 2.0 * math.pi / inv_freq
                low_wavelength = old_length / low_factor
                high_wavelength = old_length / high_factor
                scaled = torch.where(wavelength > low_wavelength, inv_freq / factor, inv_freq)
                smooth = (old_length / wavelength - low_factor) / (high_factor - low_factor)
                smoothed = (1.0 - smooth) * (inv_freq / factor) + smooth * inv_freq
                medium = (wavelength <= low_wavelength) & (wavelength >= high_wavelength)
                inv_freq = torch.where(medium, smoothed, scaled)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.einsum(
            "bs,d->bsd", position_ids.float(), self.inv_freq
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().unsqueeze(1), embedding.sin().unsqueeze(1)


class SecureLlamaAttention(nn.Module):
    def __init__(self, engine: SecureLinearEngine, state: Dict, layer_index: int) -> None:
        super().__init__()
        config = state["config"]
        hidden = int(config["hidden_size"])
        num_heads = int(config["num_attention_heads"])
        num_kv = int(config["num_key_value_heads"])
        head_dim = int(config["head_dim"])
        self.num_heads = num_heads
        self.num_kv = num_kv
        self.head_dim = head_dim
        self.kv_repeat = num_heads // num_kv
        self.scale = head_dim ** -0.5
        projections = {}
        for kind, out_features in (
            ("q_proj", num_heads * head_dim),
            ("k_proj", num_kv * head_dim),
            ("v_proj", num_kv * head_dim),
        ):
            key = "layer_{}_{}".format(layer_index, kind)
            protected = state["critical_projections"][key]
            projections[kind] = SecureSplitProjection(
                engine,
                "model.layers.{}.self_attn.{}".format(layer_index, kind),
                hidden,
                out_features,
                protected["public_rows"],
                protected["critical_rows"],
                protected["critical_weight"],
            )
        self.q_proj = projections["q_proj"]
        self.k_proj = projections["k_proj"]
        self.v_proj = projections["v_proj"]
        self.o_proj = SecureOTPLinear(
            engine, "model.layers.{}.self_attn.o_proj".format(layer_index)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch, sequence, self.num_kv, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, sequence, self.num_kv, self.head_dim
        ).transpose(1, 2)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        key = key.repeat_interleave(self.kv_repeat, dim=1)
        value = value.repeat_interleave(self.kv_repeat, dim=1)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        causal = torch.ones(sequence, sequence, dtype=torch.bool).tril()
        allowed = causal.view(1, 1, sequence, sequence)
        allowed = allowed & attention_mask.to(torch.bool).view(batch, 1, 1, sequence)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.float(), dim=-1)
        attended = torch.matmul(probabilities, value.float())
        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, self.num_heads * self.head_dim
        )
        return self.o_proj(attended)


class SecureLlamaMLP(nn.Module):
    def __init__(self, engine: SecureLinearEngine, layer_index: int) -> None:
        super().__init__()
        prefix = "model.layers.{}.mlp.".format(layer_index)
        self.gate_proj = SecureOTPLinear(engine, prefix + "gate_proj")
        self.up_proj = SecureOTPLinear(engine, prefix + "up_proj")
        self.down_proj = SecureOTPLinear(engine, prefix + "down_proj")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(tensor)) * self.up_proj(tensor))


class SecureLlamaLayer(nn.Module):
    def __init__(self, engine: SecureLinearEngine, state: Dict, layer_index: int) -> None:
        super().__init__()
        norms = state["layer_norms"][layer_index]
        eps = float(state["config"]["rms_norm_eps"])
        self.input_layernorm = SecureRMSNorm(norms["input"], eps)
        self.post_attention_layernorm = SecureRMSNorm(norms["post_attention"], eps)
        self.self_attn = SecureLlamaAttention(engine, state, layer_index)
        self.mlp = SecureLlamaMLP(engine, layer_index)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states, attention_mask, cos, sin
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class SecureLlamaForSequenceClassification(nn.Module):
    """Small TEE graph: no embedding table and no frozen dense matrices."""

    def __init__(self, engine: SecureLinearEngine, state: Dict) -> None:
        super().__init__()
        self.config = state["config"]
        self.rotary = SecureRotaryEmbedding(self.config)
        self.layers = nn.ModuleList(
            [
                SecureLlamaLayer(engine, state, index)
                for index in range(int(self.config["num_hidden_layers"]))
            ]
        )
        self.norm = SecureRMSNorm(
            state["final_norm"], float(self.config["rms_norm_eps"])
        )
        score_weight = state["score_weight"].float()
        self.score = nn.Linear(score_weight.shape[1], score_weight.shape[0], bias=False)
        with torch.no_grad():
            self.score.weight.copy_(score_weight)

    def forward(
        self,
        initial_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = initial_hidden_states.float()
        cos, sin = self.rotary(position_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask, cos, sin)
        hidden = self.norm(hidden)
        sequence_index = attention_mask.to(torch.long).sum(dim=-1).sub(1).clamp_min(0)
        pooled = hidden[
            torch.arange(hidden.shape[0], dtype=torch.long), sequence_index
        ]
        return self.score(pooled)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if ".lora." in name or name.startswith("score."):
                yield parameter


def load_encrypted_tee_state(offline_root: Path, key_path: Path) -> Dict:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise RuntimeError("The TEE Python image needs the cryptography package.") from exc
    key = Path(key_path).read_bytes()
    if len(key) != 32:
        raise RuntimeError("TEE AES key must contain exactly 32 bytes")
    return _decrypt_file(
        AESGCM(key),
        Path(offline_root) / "tee_state.bin",
        "partee-tee-state-v3",
    )


def load_encrypted_object(path: Path, key_path: Path, associated_data: str):
    """Load one AES-256-GCM protected runtime object inside the enclave."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise RuntimeError("The TEE Python image needs the cryptography package.") from exc
    key = Path(key_path).read_bytes()
    if len(key) != 32:
        raise RuntimeError("TEE AES key must contain exactly 32 bytes")
    return _decrypt_file(AESGCM(key), Path(path), associated_data)
