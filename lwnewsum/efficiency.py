"""Computational characteristics (Section 12): parameters, size, FLOPs, latency, memory.

Latency is measured with CUDA synchronisation, 3 warm-up batches, identical
decoding settings and identical documents for every system, at batch size 1
(interactive use) and batch size 32 (throughput). Peak memory is the maximum
allocated CUDA memory during batch-32 generation. FLOPs are counted for one
teacher-forced forward pass with a 512-token source and 128-token target.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .modeling import count_parameters


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def forward_flops(model, device, src_len=512, tgt_len=128) -> float | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    ids = torch.full((1, src_len), 5, dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    labels = torch.full((1, tgt_len), 5, dtype=torch.long, device=device)
    # nn.TransformerEncoderLayer uses a fused "fast path" kernel in eval mode that FlopCounterMode does not
    # count; train mode (dropout is irrelevant for counting) forces the standard, countable operators.
    was_training = model.training
    model.train()
    fc = FlopCounterMode(display=False)
    try:
        with fc:
            model(input_ids=ids, attention_mask=mask, labels=labels)
    finally:
        model.train(was_training)
    return float(fc.get_total_flops())


@torch.no_grad()
def latency(model, tok, texts, gen, device, prefix, max_len, batch_size, bf16=True, warmup=3):
    model.to(device).eval()
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    times, peak, lens = [], 0.0, []
    ac = torch.autocast("cuda", dtype=torch.bfloat16) if (bf16 and device.type == "cuda") else torch.autocast("cpu", enabled=False)
    warmup = min(warmup, len(batches))
    for i, bt in enumerate(batches[:warmup] + batches):
        enc = tok([prefix + t for t in bt], max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(device)
        if device.type == "cuda" and i == warmup:
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        t0 = time.perf_counter()
        with ac:
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **gen)
        _sync(device)
        if i >= warmup:
            times.append((time.perf_counter() - t0) / len(bt))
            lens.append(int(out.shape[-1]))  # decoder steps actually run for this batch
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 2**20
    return {"ms_per_doc_mean": 1000 * float(np.mean(times)), "ms_per_doc_std": 1000 * float(np.std(times)),
            "peak_mem_mb": peak, "decoder_steps": float(np.mean(lens))}


def profile(model, tok, texts, gen, device, prefix, max_len, bf16=True, trained=True) -> dict:
    model.to(device).eval()
    p = count_parameters(model)
    if not trained:  # pretrained reference used as-is: nothing is trained in this study
        p["trainable"], p["trainable_pct"] = 0, 0.0
    out = {"params_total_m": p["total"] / 1e6, "params_trainable_m": p["trainable"] / 1e6,
           "trainable_pct": p["trainable_pct"], "fp32_size_mb": p["total"] * 4 / 2**20}
    out["gflops_forward"] = (forward_flops(model, device) or float("nan")) / 1e9
    b1 = latency(model, tok, texts[:50], gen, device, prefix, max_len, 1, bf16)
    b32 = latency(model, tok, texts, gen, device, prefix, max_len, 32, bf16)
    out.update({"ms_per_doc_bs1": b1["ms_per_doc_mean"], "ms_per_doc_bs1_std": b1["ms_per_doc_std"],
                "ms_per_doc_bs32": b32["ms_per_doc_mean"], "peak_mem_mb_bs32": b32["peak_mem_mb"],
                "decoder_steps_bs1": b1["decoder_steps"], "decoder_steps_bs32": b32["decoder_steps"],
                "param_dtype": str(next(model.parameters()).dtype)})
    return out
