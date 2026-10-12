"""Zero-shot prompting of small instruction-tuned decoder-only LLMs (reference systems).

These models are not trained here. Each receives the same instruction per
dataset (cfg["datasets"][d]["slm_instruction"]), the article truncated to
``slm_max_article_tokens`` tokens, greedy decoding, and the model's own chat
template. They follow the evaluation protocol of recent small-language-model
news-summarisation studies (e.g. Xu et al., NAACL 2025).
"""
from __future__ import annotations

import re

import torch


def load_causal(name: str, cache_dir=None, device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if (device is not None and device.type == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir, dtype=dtype)
    for p in model.parameters():
        p.requires_grad = False
    return model, tok


def _truncate(tok, text: str, max_tokens: int) -> str:
    ids = tok(text, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
    return tok.decode(ids, skip_special_tokens=True)


def _prompt(tok, instruction: str, article: str) -> str:
    msgs = [{"role": "user", "content": f"{instruction}\n\nArticle:\n{article}\n\nSummary:"}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return msgs[0]["content"] + " "


def _clean(text: str) -> str:
    text = re.sub(r"^\s*(summary|here is (a|the) summary[^:]*)\s*:\s*", "", text.strip(), flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


@torch.no_grad()
def generate_causal(model, tok, examples, dcfg, device, batch_size=16, max_article_tokens=1024):
    model.to(device).eval()
    instr = dcfg["slm_instruction"]
    rows = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        prompts = [_prompt(tok, instr, _truncate(tok, e.source, max_article_tokens)) for e in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=dcfg["gen"]["max_new_tokens"], do_sample=False,
                             pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        rows += [{"uid": e.uid, "prediction": _clean(t), "reference": e.target} for e, t in zip(batch, texts)]
        if (i // batch_size) % 20 == 0:
            print(f"[slm] {i + len(batch)}/{len(examples)}")
    return rows
