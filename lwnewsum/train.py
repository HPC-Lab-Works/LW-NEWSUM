"""Training and generation loops shared by every neural system."""
from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import Example


class Collator:
    def __init__(self, tokenizer, max_source_len: int, max_target_len: int, prefix: str):
        self.tok, self.ms, self.mt, self.prefix = tokenizer, max_source_len, max_target_len, prefix

    def __call__(self, batch: list[Example]):
        enc = self.tok([self.prefix + b.source for b in batch], max_length=self.ms, truncation=True,
                       padding=True, return_tensors="pt")
        tgt = self.tok(text_target=[b.target for b in batch], max_length=self.mt, truncation=True,
                       padding=True, return_tensors="pt")["input_ids"]
        tgt[tgt == self.tok.pad_token_id] = -100
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": tgt,
                "uids": [b.uid for b in batch]}


def _autocast(device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _param_groups(model, spec, tcfg):
    """Adapter/module parameters use lr_module; backbone parameters (when trainable) use lr_full."""
    module_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_module = n.startswith("module.") or "lora_" in n or "ia3_" in n
        (module_params if is_module else backbone_params).append(p)
    groups = []
    if module_params:
        groups.append({"params": module_params, "lr": tcfg["lr_module"]})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": tcfg["lr_full"]})
    return groups


@torch.no_grad()
def validation_loss(model, loader, device, bf16) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        with _autocast(device, bf16):
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
        k = b["input_ids"].size(0)
        tot += out.loss.float().item() * k
        n += k
    model.train()
    return tot / max(n, 1)


def train_model(model, spec, tok, splits, dcfg, tcfg, out_dir: Path, device, prefix: str, seed: int):
    from transformers import get_linear_schedule_with_warmup

    from .systems import trainable_state_dict

    coll = Collator(tok, dcfg["max_source_len"], dcfg["max_target_len"], prefix)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(splits["train"], batch_size=tcfg["per_device_batch_size"], shuffle=True,
                              collate_fn=coll, generator=g, num_workers=tcfg["num_workers"])
    val = splits["validation"][: tcfg["max_val_samples"]]
    val_loader = DataLoader(val, batch_size=tcfg["per_device_batch_size"] * 2, shuffle=False,
                            collate_fn=coll, num_workers=tcfg["num_workers"])

    model.to(device).train()
    opt = torch.optim.AdamW(_param_groups(model, spec, tcfg), weight_decay=tcfg["weight_decay"])
    steps_per_epoch = math.ceil(len(train_loader) / tcfg["grad_accum"])
    total = steps_per_epoch * tcfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(tcfg["warmup_ratio"] * total), total)
    eval_every = min(tcfg["eval_every_steps"], steps_per_epoch)

    log_path = out_dir / "train_log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w")
    best, best_state, bad, step, t0 = float("inf"), None, 0, 0, time.time()
    initial = validation_loss(model, val_loader, device, tcfg["bf16"])
    logf.write(json.dumps({"step": 0, "epoch": 0, "val_loss": initial}) + "\n")
    running, nb, stop = 0.0, 0, False
    for epoch in range(tcfg["epochs"]):
        for i, b in enumerate(train_loader):
            with _autocast(device, tcfg["bf16"]):
                out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device),
                            labels=b["labels"].to(device))
            (out.loss / tcfg["grad_accum"]).backward()
            running += out.loss.float().item()
            nb += 1
            if (i + 1) % tcfg["grad_accum"] != 0 and (i + 1) != len(train_loader):
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], tcfg["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % eval_every == 0 or step == total:
                vl = validation_loss(model, val_loader, device, tcfg["bf16"])
                rec = {"step": step, "epoch": epoch + (i + 1) / len(train_loader), "train_loss": running / nb,
                       "val_loss": vl, "elapsed_s": time.time() - t0}
                alpha = getattr(getattr(model, "module", None), "alpha", None)
                if alpha is not None:
                    rec["alpha"] = float(alpha)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                print(f"[train] {rec}")
                running, nb = 0.0, 0
                if vl < best - 1e-4:
                    best, best_state, bad = vl, trainable_state_dict(model), 0
                else:
                    bad += 1
                    if bad >= tcfg["early_stopping_patience"]:
                        stop = True
                        break
        if stop:
            break
    logf.close()
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return {"best_val_loss": best, "initial_val_loss": initial, "optimizer_steps": step,
            "train_time_s": time.time() - t0, "stopped_early": stop}


@torch.no_grad()
def generate_predictions(model, tok, examples, dcfg, batch_size, device, prefix: str, bf16: bool) -> list[dict]:
    coll = Collator(tok, dcfg["max_source_len"], dcfg["max_target_len"], prefix)
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=coll)
    model.to(device).eval()
    by_uid = {e.uid: e for e in examples}
    rows = []
    gen = dict(dcfg["gen"])
    gen["early_stopping"] = True
    def _gen(input_ids, attention_mask):
        """Generate; on CUDA out-of-memory, split the batch in half and retry (results are unchanged)."""
        try:
            with _autocast(device, bf16):
                ids = model.generate(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), **gen)
            return tok.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except torch.OutOfMemoryError:
            if input_ids.size(0) == 1:
                raise
            torch.cuda.empty_cache()
            h = input_ids.size(0) // 2
            print(f"[generate] out of memory at batch size {input_ids.size(0)}; retrying as two halves", flush=True)
            return _gen(input_ids[:h], attention_mask[:h]) + _gen(input_ids[h:], attention_mask[h:])

    for bi, b in enumerate(loader):
        preds = _gen(b["input_ids"], b["attention_mask"])
        for uid, p in zip(b["uids"], preds):
            e = by_uid[uid]
            rows.append({"uid": uid, "prediction": p.strip(), "reference": e.target})
        if bi % 20 == 0:
            print(f"[generate] {bi + 1}/{len(loader)} batches")
    return rows
