#!/usr/bin/env python3
"""Fine-tune a Laya checkpoint on local typed-decision JSONL files.

Adapted from the public Laya Kaggle fine-tuning notebook (Apache-2.0,
per the study record); exact notebook URL and revision were not recorded.
Public model credit: https://huggingface.co/convaiinnovations/laya

The training step intentionally follows notebook_code.py: RLCD proper-scoring
rewards plus the notebook's GRPO-style policy-gradient term and soft CE term.
The data format here only supplies labels, so labels become one-hot targets.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--heldout", required=True, type=Path)
    p.add_argument("--base", default="convaiinnovations/laya")
    p.add_argument("--subfolder", default="multilingual")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2.5e-5)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("state"), str):
                raise ValueError(f"{path}:{line_no}: state must be a string")
            if not isinstance(row.get("questions"), dict) or not row["questions"]:
                raise ValueError(f"{path}:{line_no}: questions must be a non-empty object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no JSONL rows")
    return rows


def _target_for_question(q: dict[str, Any]) -> list[float]:
    qtype = q.get("type")
    if qtype == "choice":
        criteria = q.get("criteria")
        label = q.get("label")
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice criteria must be a non-empty object")
        keys = list(criteria)
        if label not in criteria:
            raise ValueError(f"choice label {label!r} is not in criteria")
        return [1.0 if key == label else 0.0 for key in keys]
    if qtype == "noul":
        label = q.get("label")
        if isinstance(label, str):
            label = label.lower()
        if label not in (True, False, "true", "false"):
            raise ValueError("noul label must be true or false")
        return [0.0, 1.0] if label is True or label == "true" else [1.0, 0.0]
    if qtype == "score":
        criteria = q.get("criteria")
        label = q.get("label")
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("score criteria must be a non-empty list")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(criteria):
            raise ValueError("score label must be an in-range integer")
        return [1.0 if i == label else 0.0 for i in range(len(criteria))]
    raise ValueError(f"unsupported question type: {qtype!r}")


def _question_for_laya(q: dict[str, Any]) -> dict[str, Any]:
    return {"t": q["type"], "ins": q.get("instructions", ""), "crit": q.get("criteria", {})}


def make_items(rows: list[dict[str, Any]], tokenizer: Any, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Tokenize rows using the same build_sequence path as the notebook."""
    from laya.common import QTYPES, build_sequence, render_options

    max_len = int(cfg.get("max_len", 1024))
    head_max_len = int(cfg.get("head_max_len", 256))
    items: list[dict[str, Any]] = []
    for row in rows:
        for qid, q in row["questions"].items():
            if not isinstance(q, dict):
                raise ValueError(f"question {qid!r} must be an object")
            target = _target_for_question(q)
            seq, markers = build_sequence(
                tokenizer,
                row["state"],
                _question_for_laya(q),
                max_len,
                head_max_len,
            )
            expected_markers = len(render_options(_question_for_laya(q)))
            if len(markers) != expected_markers or len(target) != len(markers):
                raise ValueError(
                    f"question {qid!r}: marker/target mismatch "
                    f"({len(markers)} markers, {len(target)} targets)"
                )
            items.append(
                {
                    "ids": seq,
                    "markers": markers,
                    "qtype": int(QTYPES[q["type"]]),
                    "target": target,
                    "label": int(max(range(len(target)), key=target.__getitem__)),
                    "type": q["type"],
                }
            )
    if not items:
        raise ValueError("dataset produced no training items")
    return items


def collate_train_batch(items: list[dict[str, Any]], pad_id: int, torch: Any) -> dict[str, Any]:
    n = len(items)
    length = max(len(item["ids"]) for item in items)
    kmax = max(len(item["markers"]) for item in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    attention = torch.zeros((n, length), dtype=torch.long)
    marker_pos = torch.zeros((n, kmax), dtype=torch.long)
    marker_mask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, item in enumerate(items):
        ids[i, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
        attention[i, : len(item["ids"])] = 1
        k = len(item["markers"])
        marker_pos[i, :k] = torch.tensor(item["markers"], dtype=torch.long)
        marker_mask[i, :k] = True
        target[i, : len(item["target"])] = torch.tensor(item["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": torch.tensor([item["qtype"] for item in items], dtype=torch.long),
    }


def _model_forward(model: Any, batch: dict[str, Any], device: Any) -> Any:
    logits, _ = model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["marker_pos"].to(device),
        batch["marker_mask"].to(device),
        batch["qtype"].to(device),
    )
    return logits.float()


def _proper_reward_cpu_fallback(proper_reward: Any, q: Any, target: Any, qtype: Any, mask: Any) -> Any:
    """Compute the reward on CPU if the selected device lacks a primitive."""
    try:
        return proper_reward(q, target, qtype, mask, w_sph=0.75, w_rps=1.0)
    except RuntimeError as exc:
        if "NOT IMPLEMENTED" not in str(exc).upper() and "NOT SUPPORTED" not in str(exc).upper():
            raise
        reward = proper_reward(q.cpu(), target.cpu(), qtype.cpu(), mask.cpu(), w_sph=0.75, w_rps=1.0)
        return reward.to(q.device)


def fit_one_temp(selections: list[tuple[Any, list[float]]], torch: Any) -> float:
    """Exact scalar temperature fit used by notebook_code.py, on CPU."""
    if len(selections) < 10:
        return 1.0
    kmax = max(len(logits) for logits, _ in selections)
    logits = torch.full((len(selections), kmax), -1e4, dtype=torch.float32)
    targets = torch.zeros((len(selections), kmax), dtype=torch.float32)
    for i, (row_logits, row_target) in enumerate(selections):
        logits[i, : len(row_logits)] = torch.tensor(row_logits, dtype=torch.float32)
        targets[i, : len(row_target)] = torch.tensor(row_target, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure() -> Any:
        optimizer.zero_grad()
        loss = -(targets * torch.log_softmax(logits / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def evaluate(model: Any, items: list[dict[str, Any]], tokenizer: Any, device: Any, torch: Any) -> tuple[dict[str, float], list[tuple[int, list[float], list[float]]]]:
    model.eval()
    correct = {"choice": [], "noul": [], "score": []}
    choice_briers: list[float] = []
    calibration: list[tuple[int, list[float], list[float]]] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    with torch.no_grad():
        for start in range(0, len(items), 16):
            batch_items = items[start : start + 16]
            batch = collate_train_batch(batch_items, pad_id, torch)
            logits = _model_forward(model, batch, device).cpu()
            for i, item in enumerate(batch_items):
                k = len(item["markers"])
                row_logits = logits[i, :k]
                probs = torch.softmax(row_logits, -1).tolist()
                predicted = int(max(range(k), key=probs.__getitem__))
                qtype = item["type"]
                correct[qtype].append(float(predicted == item["label"]))
                if qtype == "choice":
                    choice_briers.append(sum((p - t) ** 2 for p, t in zip(probs, item["target"])))
                calibration.append((item["qtype"], row_logits.tolist(), item["target"]))
    metrics = {
        "choice_acc": _mean(correct["choice"]),
        "choice_top1": _mean(correct["choice"]),
        "choice_brier": _mean(choice_briers),
        "noul_acc": _mean(correct["noul"]),
        "score_acc": _mean(correct["score"]),
    }
    model.train()
    return metrics, calibration


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _get_agent_config(agent: Any, model: Any) -> dict[str, Any]:
    for obj in (agent, model):
        for name in ("cfg", "config", "_config"):
            value = getattr(obj, name, None)
            if isinstance(value, dict):
                return copy.deepcopy(value)
            if hasattr(value, "to_dict"):
                converted = value.to_dict()
                if isinstance(converted, dict):
                    return copy.deepcopy(converted)
    for obj in (agent, model):
        for name in ("model_dir", "_model_dir", "checkpoint_dir", "_checkpoint_dir"):
            value = getattr(obj, name, None)
            if value:
                config_path = Path(value) / "rl_agent_config.json"
                if config_path.is_file():
                    return json.loads(config_path.read_text(encoding="utf-8"))
    raise RuntimeError("laya.load agent did not expose its rl_agent_config.json or config dict")


def _load_agent(base: str, subfolder: str, device: str) -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import laya

    kwargs = {"device": device, "subfolder": subfolder, "fast": False}
    try:
        return laya.load(base, **kwargs)
    except TypeError:
        kwargs.pop("fast")
        return laya.load(base, **kwargs)


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.lr <= 0 or args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("epochs, lr, batch-size and grad-accum must be positive")
    import torch
    from laya.common import proper_reward

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    agent = _load_agent(args.base, args.subfolder, args.device)
    model = getattr(agent, "model", None)
    # Laya 0.3.20's Agent names this field `tok`; accept `tokenizer` too for
    # compatible builds without changing the official loaded-model path.
    tokenizer = getattr(agent, "tokenizer", None) or getattr(agent, "tok", None)
    if model is None or tokenizer is None:
        raise RuntimeError("laya.load agent must expose agent.model and agent.tok/tokenizer")
    cfg = _get_agent_config(agent, model)
    # Match notebook_code.py:160-163 rather than silently inheriting a
    # checkpoint-specific head budget.
    cfg["gradient_checkpointing"] = True
    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256
    train_items = make_items(read_jsonl(args.train), tokenizer, cfg)
    heldout_items = make_items(read_jsonl(args.heldout), tokenizer, cfg)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    model.to(device)
    if hasattr(model, "encoder") and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "head_checkpointing"):
        model.head_checkpointing = True
    model.train()
    encoder_params = [p for name, p in model.named_parameters() if "encoder." in name]
    head_params = [p for name, p in model.named_parameters() if "encoder." not in name]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr},
            # notebook_code.py uses 1e-4 vs 2.5e-5; keep that 4x head rate.
            {"params": head_params, "lr": args.lr * 4.0},
        ],
        weight_decay=0.01,
    )
    total_updates = max(1, math.ceil(len(train_items) / args.batch_size / args.grad_accum) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)

    for epoch in range(args.epochs):
        order = list(range(len(train_items)))
        random.Random(args.seed + epoch).shuffle(order)
        sigma = 0.4 + (0.1 - 0.4) * (epoch / max(1, args.epochs - 1))
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        n_batches = 0
        for batch_no, start in enumerate(range(0, len(order), args.batch_size), 1):
            chunk = [train_items[i] for i in order[start : start + args.batch_size]]
            batch = collate_train_batch(chunk, pad_id, torch)
            logits = _model_forward(model, batch, device)
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            qtype = batch["qtype"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            group_size = 4
            eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                reward = _proper_reward_cpu_fallback(proper_reward, q, target.unsqueeze(0), qtype, mask)
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(advantage * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / args.grad_accum + 0.0 * logits.sum()
            loss.backward()
            total_loss += float(loss.detach().cpu()) * args.grad_accum
            n_batches += 1
            if batch_no % args.grad_accum == 0 or start + args.batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        metrics, _ = evaluate(model, heldout_items, tokenizer, device, torch)
        print(
            f"epoch {epoch + 1}/{args.epochs} train_loss={total_loss / max(1, n_batches):.6f} "
            f"heldout choice_acc={metrics['choice_acc']:.4f} "
            f"choice_top1={metrics['choice_top1']:.4f} "
            f"choice_brier={metrics['choice_brier']:.4f} "
            f"noul_acc={metrics['noul_acc']:.4f} score_acc={metrics['score_acc']:.4f}"
        )

    _, calibration = evaluate(model, heldout_items, tokenizer, device, torch)
    qtype_values = {"choice": 0, "score": 1, "noul": 2}
    # QTYPES is used for the model call above; these names are only used to order
    # the exported per-type temperatures as the notebook does.
    reverse_qtypes = {}
    for item in heldout_items:
        reverse_qtypes[item["qtype"]] = item["type"]
    temperatures = [1.2, 1.2, 1.2]
    for model_qtype, qname in reverse_qtypes.items():
        selected = [(logits, target) for qt, logits, target in calibration if qt == model_qtype]
        index = qtype_values[qname]
        temperatures[index] = fit_one_temp(selected, torch)

    args.out.mkdir(parents=True, exist_ok=True)
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "config"):
        raise RuntimeError("loaded model does not expose model.encoder.config")
    from safetensors.torch import save_file

    state = {key: value.detach().to("cpu").half().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(args.out / "model.safetensors"))
    encoder.config.save_pretrained(str(args.out / "encoder"))
    tokenizer.save_pretrained(str(args.out / "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-local-finetuned"
    cfg["temperature"] = temperatures
    cfg.pop("temperature_by_options", None)
    (args.out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {args.out} | temperatures(choice, score, noul)={ [round(x, 4) for x in temperatures] }")


if __name__ == "__main__":
    train(parse_args())
