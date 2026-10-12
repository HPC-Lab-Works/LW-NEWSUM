"""NLI-based factual-consistency estimates (Section 11 of the manuscript).

Two complementary document-level scores are computed on the same seeded subset
of test documents for every system:

1. FCS / HR (chunked NLI). The source is split into sentence-aligned chunks of at
   most ``nli_chunk_tokens`` tokens, so no part of the article is truncated away.
   A summary sentence s_i is *supported* if max_c P(entail | chunk_c, s_i) >= tau.
   FCS(doc) = 100 * (#supported / #summary sentences); HR = 100 - FCS.
2. SummaC-ZS (Laban et al., 2022). For each summary sentence, the maximum over
   source sentences of P(entail) - P(contradict); averaged over summary sentences.

Both are automatic proxies, not human fact verification.
"""
from __future__ import annotations

import numpy as np
import torch

from .utils import split_sentences


class NLIScorer:
    def __init__(self, model_name: str, device, batch_size: int = 64, max_len: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        if device.type == "cuda":
            self.model.half()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent = labels["entailment"]
        self.con = labels["contradiction"]
        self.device, self.bs, self.max_len = device, batch_size, max_len

    @torch.no_grad()
    def probs(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(premises), self.bs):
            enc = self.tok(premises[i:i + self.bs], hypotheses[i:i + self.bs], truncation="longest_first",
                           max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))

    def chunk(self, text: str, max_tokens: int) -> list[str]:
        chunks, cur, cur_len = [], [], 0
        for s in split_sentences(text):
            n = len(self.tok.tokenize(s))
            if cur and cur_len + n > max_tokens:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            cur.append(s)
            cur_len += n
        if cur:
            chunks.append(" ".join(cur))
        return chunks or [text]


def factuality(scorer: NLIScorer, sources: list[str], preds: list[str], tau: float, chunk_tokens: int):
    fcs, mean_ent, summac = [], [], []
    for src, pred in zip(sources, preds):
        sents = split_sentences(pred) or [pred]
        # (1) chunked NLI
        chunks = scorer.chunk(src, chunk_tokens)
        prem = [c for _ in sents for c in chunks]
        hyp = [s for s in sents for _ in chunks]
        p = scorer.probs(prem, hyp)[:, scorer.ent].reshape(len(sents), len(chunks)).max(axis=1)
        fcs.append(100.0 * float((p >= tau).mean()))
        mean_ent.append(100.0 * float(p.mean()))
        # (2) SummaC-ZS with sentence-level premises
        src_s = split_sentences(src) or [src]
        prem = [c for _ in sents for c in src_s]
        hyp = [s for s in sents for _ in src_s]
        pr = scorer.probs(prem, hyp)
        z = (pr[:, scorer.ent] - pr[:, scorer.con]).reshape(len(sents), len(src_s)).max(axis=1)
        summac.append(100.0 * float(z.mean()))
    return {"fcs": fcs, "hr": [100.0 - x for x in fcs], "mean_entailment": mean_ent, "summac_zs": summac}


def factuality_subset_ids(uids: list[str], n: int, seed: int) -> list[str]:
    if n is None or n >= len(uids):
        return list(uids)
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(uids), size=n, replace=False).tolist())
    return [uids[i] for i in idx]
