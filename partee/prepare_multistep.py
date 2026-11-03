#!/usr/bin/env python3
"""Trusted offline preparation for multi-step two-host ParTEE training.

This extends the original one-step preprocessor without changing its masking,
quantization, matrix partitioning, or AES-256-GCM record format.  It creates an
independent record for every run/epoch/batch/layer/direction/invocation tuple.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import random
import secrets
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import (
    CRITICAL_HEADS,
    OTP_BITS,
    OTP_MODULUS,
    atomic_torch_save,
    complement_rows,
    rows_for_projection,
)


EXPECTED_PUBLIC_LINEARS = 224
OUTPUT_CHUNK_ROWS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--validation-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_snapshot(path_text: str) -> Path:
    candidate = Path(path_text).expanduser().resolve()
    if (candidate / "config.json").is_file():
        return candidate
    snapshots = candidate / "snapshots"
    if snapshots.is_dir():
        matches = sorted(path.parent for path in snapshots.glob("*/config.json"))
        if matches:
            return matches[-1]
    raise FileNotFoundError("No Llama config.json found under {}".format(candidate))


def validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "train_batches", "validation_batches", "batch_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError("--{} must be positive".format(name.replace("_", "-")))
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Trusted preprocessing requires a CUDA device")
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise RuntimeError("Requested {}, but only {} GPUs are visible".format(
            device, torch.cuda.device_count()
        ))


def encrypt_object(
    aesgcm: AESGCM, value, destination: Path, associated_data: str
) -> None:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    nonce = secrets.token_bytes(12)
    ciphertext = aesgcm.encrypt(
        nonce, buffer.getvalue(), associated_data.encode("utf-8")
    )
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


@torch.no_grad()
def matrix_scales(matrix: torch.Tensor) -> torch.Tensor:
    qmax = (1 << (OTP_BITS - 1)) - 1
    scales = []
    for start in range(0, matrix.shape[0], OUTPUT_CHUNK_ROWS):
        stop = min(start + OUTPUT_CHUNK_ROWS, matrix.shape[0])
        source = matrix[start:stop].detach().float()
        scales.append(source.abs().amax(dim=1).clamp_min(1e-12).div(qmax).cpu())
    return torch.cat(scales)


@torch.no_grad()
def prepare_direction(
    aesgcm: AESGCM,
    material_id: str,
    matrix: torch.Tensor,
    shape: Tuple[int, ...],
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    destination: Path,
    device: torch.device,
) -> Dict:
    mask_seed = secrets.randbits(63)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(mask_seed)
    mask = torch.randint(
        0, OTP_MODULUS, shape, dtype=torch.int64, generator=generator
    )
    mask_device = mask.to(device=device, dtype=torch.float64)
    product = torch.empty(
        *shape[:-1], matrix.shape[0], dtype=torch.int64, device="cpu"
    )
    qmax = (1 << (OTP_BITS - 1)) - 1
    for start in range(0, matrix.shape[0], OUTPUT_CHUNK_ROWS):
        stop = min(start + OUTPUT_CHUNK_ROWS, matrix.shape[0])
        source = matrix[start:stop].detach().float().to(device)
        scale = weight_scale[start:stop].to(device)
        quantized = (
            (source / scale.unsqueeze(1)).round().clamp(-qmax, qmax).to(torch.int8)
        )
        partial = F.linear(mask_device, quantized.to(torch.float64))
        partial = torch.remainder(partial, OTP_MODULUS)
        product[..., start:stop] = partial.round().to(torch.int64).cpu()
        del source, scale, quantized, partial
    material = {
        "material_id": material_id,
        "mask_seed": mask_seed,
        "mask_product": product,
        "weight_scale": weight_scale.float().cpu(),
        "bias": None if bias is None else bias.float().cpu(),
        "input_shape": shape,
        "output_shape": tuple(product.shape),
    }
    aad = "partee-otp:{}".format(material_id)
    encrypt_object(aesgcm, material, destination, aad)
    record = {
        "material_id": material_id,
        "file": destination.name,
        "aad": aad,
        "input_shape": list(shape),
        "output_shape": list(product.shape),
        "matrix_shape": list(matrix.shape),
        "quantized_matrix_bytes": int(matrix.numel()),
        "mask_bytes": int(np.prod(shape)) * 8,
        "mask_product_bytes": int(product.numel() * product.element_size()),
        "weight_scale_bytes": int(weight_scale.numel() * weight_scale.element_size()),
        "bias_bytes": 0 if bias is None else int(bias.numel() * bias.element_size()),
        "encrypted_material_file_bytes": destination.stat().st_size,
    }
    del mask, mask_device, product, material
    return record


def tokenized_batch(tokenizer, split, indices: Sequence[int], sequence_length: int):
    examples = split.select(list(indices))
    eos = tokenizer.eos_token or ""
    texts = [
        "Premise: {}\nHypothesis: {}{}".format(premise, hypothesis, eos)
        for premise, hypothesis in zip(examples["sentence1"], examples["sentence2"])
    ]
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=sequence_length,
        add_special_tokens=True,
        return_tensors="pt",
    )
    labels = torch.tensor(examples["label"], dtype=torch.long)
    return dict(encoded), labels


def batch_indices(length: int, batch_size: int, count: int, seed: int) -> Iterable[Sequence[int]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(length, generator=generator).tolist()
    required = batch_size * count
    if required > length:
        raise ValueError(
            "Requested {} examples from a split containing {}".format(required, length)
        )
    for start in range(0, required, batch_size):
        yield order[start : start + batch_size]


def material_filename(material_id: str) -> str:
    digest = hashlib.sha256(material_id.encode("utf-8")).hexdigest()
    return "otp_{}.bin".format(digest[:32])


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    model_path = resolve_snapshot(args.model_path)
    tokenizer_path = resolve_snapshot(args.tokenizer_path or args.model_path)
    output_root = Path(args.output_dir).expanduser().resolve()
    material_dir = output_root / "otp_material"
    batch_dir = output_root / "batches"
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            "Output directory must be absent or empty to prevent accidental material reuse: {}".format(
                output_root
            )
        )
    material_dir.mkdir(parents=True)
    batch_dir.mkdir(parents=True)
    run_id = args.run_id or secrets.token_hex(16)
    manifest_id = secrets.token_hex(16)
    key = secrets.token_bytes(32)
    key_path = output_root / "otp_aes_key.bin"
    key_path.write_bytes(key)
    os.chmod(str(key_path), 0o600)
    aesgcm = AESGCM(key)

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = load_dataset(args.dataset_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=2,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).cpu()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()
    config = model.config
    head_dim = int(
        getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    )
    static_state = {
        "version": 3,
        "run_id": run_id,
        "config": {
            "hidden_size": int(config.hidden_size),
            "intermediate_size": int(config.intermediate_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_dim": head_dim,
            "rms_norm_eps": float(config.rms_norm_eps),
            "rope_theta": float(getattr(config, "rope_theta", 500000.0)),
            "rope_scaling": getattr(config, "rope_scaling", None),
            "num_labels": 2,
        },
        "layer_norms": [],
        "final_norm": model.model.norm.weight.detach().float().cpu(),
        "score_weight": model.score.weight.detach().float().cpu(),
        "critical_projections": {},
    }

    matrices = {}
    static_layers = {}
    gpu_layers = {}
    for layer_index, layer in enumerate(model.model.layers):
        static_state["layer_norms"].append({
            "input": layer.input_layernorm.weight.detach().float().cpu(),
            "post_attention": layer.post_attention_layernorm.weight.detach().float().cpu(),
        })
        attention = layer.self_attn
        candidates = [
            ("self_attn.q_proj", attention.q_proj, "q_proj"),
            ("self_attn.k_proj", attention.k_proj, "k_proj"),
            ("self_attn.v_proj", attention.v_proj, "v_proj"),
            ("self_attn.o_proj", attention.o_proj, None),
            ("mlp.gate_proj", layer.mlp.gate_proj, None),
            ("mlp.up_proj", layer.mlp.up_proj, None),
            ("mlp.down_proj", layer.mlp.down_proj, None),
        ]
        for suffix, linear, critical_kind in candidates:
            layer_name = "model.layers.{}.{}".format(layer_index, suffix)
            critical_rows = (
                rows_for_projection(
                    layer_index,
                    CRITICAL_HEADS,
                    critical_kind,
                    head_dim,
                    int(config.num_attention_heads),
                    int(config.num_key_value_heads),
                )
                if critical_kind is not None
                else torch.empty(0, dtype=torch.long)
            )
            public_rows = complement_rows(linear.out_features, critical_rows)
            public_weight = linear.weight.detach().cpu().index_select(0, public_rows)
            public_bias = (
                None
                if linear.bias is None
                else linear.bias.detach().cpu().index_select(0, public_rows)
            )
            forward_scale = matrix_scales(public_weight)
            # Keep a transposed view instead of a second model-sized BF16 copy.
            # prepare_direction still quantizes the same W^T rows as the
            # original one-step implementation.
            backward_weight = public_weight.transpose(0, 1)
            backward_scale = matrix_scales(backward_weight)
            matrices[layer_name] = {
                "forward": public_weight,
                "backward": backward_weight,
                "forward_scale": forward_scale,
                "backward_scale": backward_scale,
                "bias": public_bias,
            }
            static_layers[layer_name] = {
                "original_weight_shape": list(linear.weight.shape),
                "original_weight_elements": int(linear.weight.numel()),
                "original_weight_bfloat16_bytes": int(linear.weight.numel() * 2),
                "public_weight_shape": list(public_weight.shape),
                "public_weight_elements": int(public_weight.numel()),
                "public_quantized_weight_bytes": int(public_weight.numel()),
                "critical_row_count": int(critical_rows.numel()),
                "critical_weight_float32_bytes": int(
                    critical_rows.numel() * linear.in_features * 4
                ),
            }
            gpu_layers[layer_name] = {
                "critical_rows": critical_rows,
                "forward_weight_scale": forward_scale,
                "backward_weight_scale": backward_scale,
                "weight_shape": tuple(linear.weight.shape),
            }
            if critical_kind is not None:
                state_key = "layer_{}_{}".format(layer_index, critical_kind)
                static_state["critical_projections"][state_key] = {
                    "public_rows": public_rows,
                    "critical_rows": critical_rows,
                    "critical_weight": linear.weight.detach().float().cpu().index_select(
                        0, critical_rows
                    ),
                }
    if len(matrices) != EXPECTED_PUBLIC_LINEARS:
        raise RuntimeError("Expected {} public linears, found {}".format(
            EXPECTED_PUBLIC_LINEARS, len(matrices)
        ))

    steps = {}
    gpu_steps = {}
    ordered_steps = []
    position_ids = torch.arange(args.sequence_length, dtype=torch.long).unsqueeze(0)
    preprocess_start = time.perf_counter()
    for epoch in range(args.epochs):
        phase_specs = (
            ("train", dataset["train"], args.train_batches, args.seed + epoch),
            ("validation", dataset["validation"], args.validation_batches, args.seed),
        )
        for phase, split, count, shuffle_seed in phase_specs:
            for batch_index, indices in enumerate(
                batch_indices(len(split), args.batch_size, count, shuffle_seed)
            ):
                step_id = "run={}|phase={}|epoch={:04d}|batch={:06d}".format(
                    run_id, phase, epoch, batch_index
                )
                encoded, labels = tokenized_batch(
                    tokenizer, split, indices, args.sequence_length
                )
                with torch.no_grad():
                    hidden = model.model.embed_tokens(encoded["input_ids"]).float().cpu()
                batch_state = {
                    "run_id": run_id,
                    "step_id": step_id,
                    "phase": phase,
                    "epoch": epoch,
                    "batch": batch_index,
                    "initial_hidden_states": hidden,
                    "attention_mask": encoded["attention_mask"].to(torch.long).cpu(),
                    "position_ids": position_ids.expand(args.batch_size, -1).clone(),
                    "labels": labels,
                }
                batch_file = "batch_{:06d}.bin".format(len(ordered_steps))
                batch_aad = "partee-batch:{}".format(step_id)
                encrypt_object(aesgcm, batch_state, batch_dir / batch_file, batch_aad)
                step_layers = {}
                gpu_step_layers = {}
                material_count = 0
                for layer_name, values in matrices.items():
                    step_layers[layer_name] = {"forward": [], "backward": []}
                    gpu_step_layers[layer_name] = {"forward": [], "backward": []}
                    directions = ("forward", "backward") if phase == "train" else ("forward",)
                    for direction in directions:
                        matrix = values[direction]
                        input_width = matrix.shape[1]
                        shape = (args.batch_size, args.sequence_length, input_width)
                        material_id = (
                            "{}|layer={}|direction={}|invocation=0000".format(
                                step_id, layer_name, direction
                            )
                        )
                        filename = material_filename(material_id)
                        record = prepare_direction(
                            aesgcm,
                            material_id,
                            matrix,
                            shape,
                            values[direction + "_scale"],
                            values["bias"] if direction == "forward" else None,
                            material_dir / filename,
                            device,
                        )
                        step_layers[layer_name][direction].append(record)
                        gpu_step_layers[layer_name][direction].append({
                            "material_id": material_id
                        })
                        material_count += 1
                steps[step_id] = {
                    "phase": phase,
                    "epoch": epoch,
                    "batch": batch_index,
                    "batch_file": batch_file,
                    "batch_aad": batch_aad,
                    "material_count": material_count,
                    "layers": step_layers,
                }
                gpu_steps[step_id] = {"layers": gpu_step_layers}
                ordered_steps.append(step_id)
                print("Prepared {} with {} OTP records".format(step_id, material_count), flush=True)

    preprocessing_seconds = time.perf_counter() - preprocess_start
    manifest = {
        "version": 3,
        "run_id": run_id,
        "manifest_id": manifest_id,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "epochs": args.epochs,
        "train_batches": args.train_batches,
        "validation_batches": args.validation_batches,
        "otp_bits": OTP_BITS,
        "otp_modulus": OTP_MODULUS,
        "critical_heads": list(CRITICAL_HEADS),
        "preprocessing_seconds": preprocessing_seconds,
        "ordered_steps": ordered_steps,
        "layers": static_layers,
        "steps": steps,
    }
    atomic_torch_save(manifest, output_root / "manifest.pt")
    atomic_torch_save({
        "version": 3,
        "run_id": run_id,
        "manifest_id": manifest_id,
        "model_path_hint": model_path.name,
        "critical_heads": list(CRITICAL_HEADS),
        "layers": gpu_layers,
        "steps": gpu_steps,
    }, output_root / "gpu_manifest.pt")
    encrypt_object(
        aesgcm, static_state, output_root / "tee_state.bin", "partee-tee-state-v3"
    )
    print("Prepared run {} in {}".format(run_id, output_root))
    print("The AES key is runtime material; do not commit this directory.")


if __name__ == "__main__":
    main()
