#!/usr/bin/env python3
"""Attention Confidence-based GQA-group selection for Llama on RTE.

The scoring and aggregation are the existing ParTEE Eq. (6)-(7)
implementation, separated from the one-host utility simulation so its output
can configure trusted preprocessing through ``PARTEE_CRITICAL_HEADS``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def score_attention_heads(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eos_token_id: Optional[int],
    max_batches: Optional[int],
) -> torch.Tensor:
    """Compute Eq. (6)-(7), averaging max attention over valid tokens."""
    model.eval()
    score_sums = None
    token_counts = None
    progress = tqdm(loader, desc="Scoring frozen attention heads", leave=False)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        attentions = outputs.attentions
        if attentions is None:
            raise RuntimeError("The model did not return attention matrices.")
        if score_sums is None:
            score_sums = torch.zeros(
                len(attentions), attentions[0].shape[1], device=device
            )
            token_counts = torch.zeros_like(score_sums)
        valid_queries = attention_mask.bool()
        valid_keys = attention_mask.bool()
        if eos_token_id is not None:
            valid_keys = valid_keys & input_ids.ne(eos_token_id)
        for layer_index, attention in enumerate(attentions):
            key_mask = valid_keys[:, None, None, :]
            masked_attention = attention.masked_fill(~key_mask, -torch.inf)
            alpha = masked_attention.amax(dim=-1)
            query_mask = valid_queries[:, None, :]
            finite = torch.isfinite(alpha) & query_mask
            score_sums[layer_index] += torch.where(
                finite, alpha, torch.zeros_like(alpha)
            ).sum(dim=(0, 2))
            token_counts[layer_index] += finite.sum(dim=(0, 2))
    if score_sums is None or token_counts is None:
        raise RuntimeError("No batches were available for head scoring.")
    return (score_sums / token_counts.clamp_min(1)).detach().cpu()


def aggregate_gqa_group_scores(
    query_head_scores: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> torch.Tensor:
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
    if query_head_scores.shape[1] != num_attention_heads:
        raise ValueError("Attention-score shape does not match the Llama config.")
    queries_per_group = num_attention_heads // num_key_value_heads
    return query_head_scores.reshape(
        query_head_scores.shape[0], num_key_value_heads, queries_per_group
    ).mean(dim=-1)


def select_critical_groups(scores: torch.Tensor, count: int):
    total_groups = scores.numel()
    if not 0 <= count <= total_groups:
        raise ValueError(
            "critical_head_count must be in [0, {}], got {}.".format(
                total_groups, count
            )
        )
    if count == 0:
        return []
    flat_indices = torch.topk(scores.flatten(), k=count).indices.tolist()
    groups_per_layer = scores.shape[1]
    return [
        (flat_index // groups_per_layer, flat_index % groups_per_layer)
        for flat_index in flat_indices
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--critical-group-count", type=int, default=1)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(args.dataset_path)["train"]

    def encode(batch):
        eos = tokenizer.eos_token or ""
        texts = [
            "Premise: {}\nHypothesis: {}{}".format(a, b, eos)
            for a, b in zip(batch["sentence1"], batch["sentence2"])
        ]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized = dataset.map(encode, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.config.output_attentions = True
    scores = score_attention_heads(
        model, loader, device, tokenizer.eos_token_id, args.max_batches
    )
    grouped = aggregate_gqa_group_scores(
        scores,
        int(model.config.num_attention_heads),
        int(model.config.num_key_value_heads),
    )
    selected = select_critical_groups(grouped, args.critical_group_count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "critical_groups": [list(item) for item in selected],
                "PARTEE_CRITICAL_HEADS": ",".join(
                    "{}:{}".format(layer, group) for layer, group in selected
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Selected critical groups: {}".format(selected))


if __name__ == "__main__":
    main()
