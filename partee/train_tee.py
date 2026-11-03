#!/usr/bin/env python3
"""Occlum-side multi-step ParTEE trainer for the prepared Llama/RTE run."""

from __future__ import annotations

import argparse
import io
import os
import random
import secrets
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .common import (
    CRITICAL_HEADS,
    FileRPCClient,
    SecureLlamaForSequenceClassification,
    SecureLinearEngine,
    load_encrypted_object,
    load_encrypted_tee_state,
    torch_load_cpu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shared-dir",
        default=os.environ.get(
            "PARTEE_TEE_SHARED_DIR",
            "/host/image/shared_folder_llamarte/two_host_rpc",
        ),
    )
    parser.add_argument(
        "--offline-root",
        default=os.environ.get("PARTEE_TEE_OFFLINE_ROOT", "/two_host_otp"),
    )
    parser.add_argument(
        "--key-path",
        default=os.environ.get("PARTEE_TEE_KEY_PATH", "/two_host_otp/otp_aes_key.bin"),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def trainable_state(model) -> dict:
    return {
        name: value.detach().cpu()
        for name, value in model.named_parameters()
        if ".lora." in name or name.startswith("score.")
    }


def restore_trainable_state(model, state: dict) -> None:
    parameters = dict(model.named_parameters())
    if set(state) != {
        name
        for name in parameters
        if ".lora." in name or name.startswith("score.")
    }:
        raise RuntimeError("Checkpoint trainable parameter set does not match the TEE model")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].dtype))


def save_encrypted_checkpoint(value, destination: Path, key_path: Path) -> None:
    key = key_path.read_bytes()
    if len(key) != 32:
        raise RuntimeError("TEE AES key must contain exactly 32 bytes")
    buffer = io.BytesIO()
    torch.save(value, buffer)
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(key).encrypt(
        nonce, buffer.getvalue(), b"partee-training-checkpoint-v1"
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


def load_checkpoint(path: Path, key_path: Path):
    return load_encrypted_object(
        path, key_path, "partee-training-checkpoint-v1"
    )


def checkpoint_after_step(
    engine,
    model,
    optimizer,
    pending_path: Path,
    latest_path: Path,
    next_step_index: int,
    step_id: str,
    key_path: Path,
) -> None:
    payload = {
        "version": 1,
        "run_id": engine.run_id,
        "manifest_id": engine.manifest["manifest_id"],
        "next_step_index": next_step_index,
        "last_completed_step_id": step_id,
        "trainable_state": trainable_state(model),
        "optimizer_state": optimizer.state_dict(),
    }
    save_encrypted_checkpoint(payload, pending_path, key_path)
    engine.complete_step()
    os.replace(str(pending_path), str(latest_path))


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    set_seed(args.seed)
    shared_dir = Path(args.shared_dir)
    offline_root = Path(args.offline_root)
    key_path = Path(args.key_path)
    if not key_path.is_file():
        raise FileNotFoundError("TEE AES key not found: {}".format(key_path))

    state = load_encrypted_tee_state(offline_root, key_path)
    if int(state.get("version", -1)) != 3:
        raise RuntimeError("Expected encrypted TEE state version 3")
    client = FileRPCClient(shared_dir)
    print("Waiting for GPU masked-GEMM service in {}".format(shared_dir), flush=True)
    client.wait_until_ready()
    gpu_status = client.call("ping")
    print("GPU service ready: {}".format(gpu_status), flush=True)
    ledger_path = shared_dir / "tee_consumption_ledger.bin"
    engine = SecureLinearEngine(offline_root, key_path, client, ledger_path)
    if state.get("run_id") != engine.run_id:
        raise RuntimeError("TEE state and OTP manifest run identifiers differ")
    if gpu_status.get("run_id") != engine.run_id:
        raise RuntimeError("GPU service and TEE OTP run identifiers differ")
    model = SecureLlamaForSequenceClassification(engine, state)
    trainable = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=0.0
    )
    optimizer.zero_grad(set_to_none=True)

    checkpoint_dir = shared_dir / "checkpoints"
    latest_path = checkpoint_dir / "latest_checkpoint.bin"
    pending_path = checkpoint_dir / "pending_checkpoint.bin"
    start_index = 0
    if pending_path.is_file():
        pending = load_checkpoint(pending_path, key_path)
        last_step = pending["last_completed_step_id"]
        if last_step not in engine.completed_steps:
            raise RuntimeError(
                "A pending checkpoint exists but its OTP step is not durably complete"
            )
        os.replace(str(pending_path), str(latest_path))
    if latest_path.is_file() and not args.no_resume:
        checkpoint = load_checkpoint(latest_path, key_path)
        if checkpoint["run_id"] != engine.run_id:
            raise RuntimeError("Checkpoint run identifier mismatch")
        if checkpoint["manifest_id"] != engine.manifest["manifest_id"]:
            raise RuntimeError("Checkpoint manifest identifier mismatch")
        restore_trainable_state(model, checkpoint["trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_index = int(checkpoint["next_step_index"])
        expected_completed = set(engine.manifest["ordered_steps"][:start_index])
        if engine.completed_steps != expected_completed:
            raise RuntimeError("Checkpoint and OTP ledger completed-step sets differ")
        print("Resuming at prepared step {}".format(start_index), flush=True)
    elif engine.completed_steps:
        raise RuntimeError(
            "OTP ledger contains completed steps but no checkpoint was resumed"
        )

    ordered_steps = engine.manifest["ordered_steps"]
    initial_parameters = [parameter.detach().clone() for parameter in trainable]
    training_steps = 0
    for step_index in range(start_index, len(ordered_steps)):
        step_id = ordered_steps[step_index]
        step = engine.manifest["steps"][step_id]
        batch = load_encrypted_object(
            offline_root / "batches" / step["batch_file"],
            key_path,
            step["batch_aad"],
        )
        if batch["step_id"] != step_id or batch["run_id"] != engine.run_id:
            raise RuntimeError("Encrypted batch identity mismatch")
        phase = step["phase"]
        engine.begin_step(step_id)
        hidden = batch["initial_hidden_states"].float()
        attention_mask = batch["attention_mask"].to(torch.long)
        position_ids = batch["position_ids"].to(torch.long)
        labels = batch["labels"].to(torch.long)
        if phase == "train":
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(hidden, attention_mask, position_ids)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, args.max_grad_norm
            )
            optimizer.step()
            training_steps += 1
            print(
                "train epoch={} batch={} loss={:.6f} grad_norm={:.6f}".format(
                    step["epoch"], step["batch"], float(loss.detach()), float(gradient_norm)
                ),
                flush=True,
            )
        elif phase == "validation":
            model.eval()
            with torch.no_grad():
                logits = model(hidden, attention_mask, position_ids)
                loss = F.cross_entropy(logits, labels)
                accuracy = logits.argmax(dim=-1).eq(labels).float().mean()
            print(
                "validation epoch={} batch={} loss={:.6f} accuracy={:.6f}".format(
                    step["epoch"], step["batch"], float(loss), float(accuracy)
                ),
                flush=True,
            )
        else:
            raise RuntimeError("Unknown prepared phase: {}".format(phase))
        checkpoint_after_step(
            engine,
            model,
            optimizer,
            pending_path,
            latest_path,
            step_index + 1,
            step_id,
            key_path,
        )

    changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(initial_parameters, trainable)
    )
    if training_steps and not changed:
        raise RuntimeError("No TEE-owned trainable parameter changed")
    client.call("stop")
    print(
        "Completed {} prepared steps; forward OTP calls={}; backward OTP calls={}; "
        "TEE trainable parameters updated={}.".format(
            len(ordered_steps) - start_index,
            engine.otp_forward_calls,
            engine.otp_backward_calls,
            changed,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
