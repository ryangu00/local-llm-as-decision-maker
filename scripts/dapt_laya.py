#!/usr/bin/env python3
"""Offline MLM DAPT for the encoder inside a Laya agent."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="convaiinnovations/laya")
    parser.add_argument("--subfolder", default="multilingual")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _allowed_path(path: Path, label: str) -> Path:
    """Keep script-owned reads/writes inside the current checkout."""
    resolved = path.expanduser().resolve()
    roots = [Path.cwd().resolve()]
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError(f"{label} must be inside the current directory: {path}")
    return resolved


def _load_agent(base: str, subfolder: str, device: str) -> Any:
    # Offline mode prevents laya/Hugging Face helpers from attempting a cache fill.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from laya_ft import _load_agent

    return _load_agent(base, subfolder, device)


def _agent_config(agent: Any, model: Any) -> dict[str, Any]:
    from laya_ft import _get_agent_config

    return _get_agent_config(agent, model)


def _encoder_reference(agent: Any, cfg: dict[str, Any], base: str) -> str:
    """Prefer a local encoder directory, then the config's encoder model name."""
    candidates: list[Path] = []
    for obj in (agent, getattr(agent, "model", None)):
        for attr in ("model_dir", "_model_dir", "checkpoint_dir", "_checkpoint_dir"):
            value = getattr(obj, attr, None)
            if value:
                candidates.append(Path(value) / "encoder")
    base_path = Path(base).expanduser()
    if base_path.is_dir():
        candidates.extend((base_path / "encoder", base_path / "multilingual" / "encoder"))
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return str(candidate)

    encoder_cfg = cfg.get("encoder")
    if isinstance(encoder_cfg, str) and encoder_cfg:
        return encoder_cfg
    if isinstance(encoder_cfg, dict):
        for key in ("name", "_name_or_path", "model_name", "path"):
            value = encoder_cfg.get(key)
            if isinstance(value, str) and value:
                return value
    encoder = getattr(getattr(agent, "model", None), "encoder", None)
    config = getattr(encoder, "config", None)
    value = getattr(config, "_name_or_path", None)
    if isinstance(value, str) and value:
        return value
    raise RuntimeError("could not determine the encoder source from rl_agent_config.json")


def _load_mlm(source: str, agent_encoder: Any) -> tuple[Any, Any]:
    from transformers import AutoConfig, AutoModelForMaskedLM

    try:
        mlm = AutoModelForMaskedLM.from_pretrained(source, local_files_only=True)
        print(f"MLM source: {source} (from_pretrained, local_only=True)")
    except Exception as exc:
        # An exported Laya directory contains encoder/config.json but deliberately
        # no standalone encoder checkpoint. Its config still gives us the exact
        # architecture; agent_encoder supplies the actual weights below.
        print(f"MLM pretrained load unavailable: {type(exc).__name__}: {exc}")
        config = AutoConfig.from_pretrained(source, local_files_only=True)
        mlm = AutoModelForMaskedLM.from_config(config)
        print(f"MLM source: {source} (from_config fallback, local_only=True)")

    encoder = getattr(mlm, "base_model", None)
    if encoder is None or not hasattr(encoder, "load_state_dict"):
        raise RuntimeError("AutoModelForMaskedLM did not expose a base encoder")
    report = encoder.load_state_dict(agent_encoder.state_dict(), strict=False)
    print(f"encoder load missing_keys={list(report.missing_keys)}")
    print(f"encoder load unexpected_keys={list(report.unexpected_keys)}")
    return mlm, encoder


def _pack_corpus(text: str, tokenizer: Any, max_len: int, pad_id: int) -> list[dict[str, list[int]]]:
    segments = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not segments:
        raise ValueError("corpus has no non-empty paragraphs")
    all_ids: list[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for segment in segments:
        encoded = tokenizer(segment, add_special_tokens=True, truncation=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if not ids:
            continue
        all_ids.extend(int(token_id) for token_id in ids)
        if eos_id is not None and all_ids[-1] != eos_id:
            all_ids.append(int(eos_id))
    if not all_ids:
        raise ValueError("corpus tokenized to zero tokens")

    blocks: list[dict[str, list[int]]] = []
    for start in range(0, len(all_ids), max_len):
        ids = all_ids[start : start + max_len]
        attention = [1] * len(ids)
        if len(ids) < max_len:
            ids = ids + [pad_id] * (max_len - len(ids))
            attention += [0] * (max_len - len(attention))
        blocks.append({"input_ids": ids, "attention_mask": attention})
    return blocks


def _mask_batch(batch: list[dict[str, list[int]]], tokenizer: Any, mask_prob: float, torch: Any, device: Any) -> tuple[Any, Any, Any]:
    input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
    attention = torch.tensor([row["attention_mask"] for row in batch], dtype=torch.long, device=device)
    special_rows = [
        tokenizer.get_special_tokens_mask(row["input_ids"], already_has_special_tokens=True)
        for row in batch
    ]
    special = torch.tensor(special_rows, dtype=torch.bool, device=device)
    eligible = attention.bool() & ~special
    masked = torch.bernoulli(torch.full(input_ids.shape, mask_prob, device=device)).bool() & eligible
    if not bool(masked.any()):
        available = eligible.nonzero(as_tuple=False)
        if len(available) == 0:
            raise ValueError("batch contains no non-special tokens to mask")
        masked[tuple(available[0].tolist())] = True

    labels = input_ids.clone()
    labels[~masked] = -100
    draw = torch.rand(input_ids.shape, device=device)
    replace_mask = masked & (draw < 0.8)
    random_mask = masked & (draw >= 0.8) & (draw < 0.9)
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("tokenizer has no mask_token_id")
    input_ids[replace_mask] = int(mask_id)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer)
    vocab_size = int(vocab_size)
    input_ids[random_mask] = torch.randint(vocab_size, (int(random_mask.sum().item()),), device=device)
    return input_ids, attention, labels


def _save_agent(agent: Any, cfg: dict[str, Any], tokenizer: Any, out: Path) -> None:
    import torch
    from safetensors.torch import save_file

    model = getattr(agent, "model", None)
    encoder = getattr(model, "encoder", None)
    if model is None or encoder is None or not hasattr(encoder, "config"):
        raise RuntimeError("loaded agent must expose model.encoder.config for export")
    out.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().to("cpu").half().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(out / "model.safetensors"))
    encoder.config.save_pretrained(str(out / "encoder"))
    tokenizer.save_pretrained(str(out / "tokenizer"))
    (out / "rl_agent_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Read back the load-bearing files before claiming export success.
    for required in (out / "model.safetensors", out / "encoder" / "config.json", out / "tokenizer"):
        if not required.exists():
            raise RuntimeError(f"export missing {required}")


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.lr <= 0 or args.max_len < 2 or args.batch < 1:
        raise ValueError("epochs, lr, max-len and batch must be positive")
    if not 0 < args.mask_prob < 1:
        raise ValueError("mask-prob must be between 0 and 1")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive when supplied")
    corpus = _allowed_path(args.corpus, "corpus")
    out = _allowed_path(args.out, "out")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    agent = _load_agent(args.base, args.subfolder, args.device)
    model = getattr(agent, "model", None)
    tokenizer = getattr(agent, "tokenizer", None) or getattr(agent, "tok", None)
    if model is None or tokenizer is None or not hasattr(model, "encoder"):
        raise RuntimeError("laya.load agent must expose agent.model, agent.model.encoder and agent.tok/tokenizer")
    cfg = _agent_config(agent, model)
    source = _encoder_reference(agent, cfg, args.base)
    mlm, mlm_encoder = _load_mlm(source, model.encoder)
    mlm.to(device)
    for parameter in mlm.parameters():
        parameter.requires_grad = False
    for parameter in mlm_encoder.parameters():
        parameter.requires_grad = True
    trainable = [parameter for parameter in mlm_encoder.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("MLM encoder has no trainable parameters")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    blocks = _pack_corpus(corpus.read_text(encoding="utf-8"), tokenizer, args.max_len, int(pad_id))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    mlm.train()
    steps = 0
    last_loss = float("nan")
    for epoch in range(args.epochs):
        order = list(range(len(blocks)))
        random.Random(args.seed + epoch).shuffle(order)
        for start in range(0, len(order), args.batch):
            if args.max_steps is not None and steps >= args.max_steps:
                break
            chunk = [blocks[index] for index in order[start : start + args.batch]]
            inputs, attention, labels = _mask_batch(chunk, tokenizer, args.mask_prob, torch, device)
            optimizer.zero_grad(set_to_none=True)
            output = mlm(input_ids=inputs, attention_mask=attention, labels=labels)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite MLM loss at step {steps + 1}: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            steps += 1
            last_loss = float(loss.detach().cpu())
            if steps % 100 == 0:
                print(f"step {steps} mlm_loss={last_loss:.6f}", flush=True)
        if args.max_steps is not None and steps >= args.max_steps:
            break

    model.encoder.load_state_dict(mlm_encoder.state_dict(), strict=True)
    _save_agent(agent, cfg, tokenizer, out)
    print(f"mlm total_steps={steps} last_loss={last_loss:.6f}", flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    train(parse_args())
