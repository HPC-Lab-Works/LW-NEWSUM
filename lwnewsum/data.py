"""Dataset loading, cleaning and descriptive statistics.

* CNN/DailyMail and XSum use the OFFICIAL train/validation/test splits from the
  Hugging Face hub, so results are comparable with the literature.
* Indian News is a small scraped corpus; it is cleaned (boilerplate removal,
  de-duplication, length filters) and split with a fixed seed. Every filtering
  step is logged so the manuscript can report it.
"""
from __future__ import annotations

import collections
import hashlib
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .utils import read_jsonl, split_sentences, write_json


@dataclass
class Example:
    uid: str
    source: str
    target: str


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_splits(cfg: dict, dataset: str, seed_for_split: int = 42) -> dict[str, list[Example]]:
    dcfg = cfg["datasets"][dataset]
    if dataset == "indian":
        splits = _load_indian(cfg, dcfg, seed_for_split)
    else:
        splits = _load_hf(cfg, dcfg)
    # optional sub-sampling (debug or compute-limited runs); always seeded, always reported
    rng = np.random.default_rng(seed_for_split)
    if dcfg.get("max_train_samples"):
        splits["train"] = _subsample(splits["train"], dcfg["max_train_samples"], rng)
        splits["validation"] = _subsample(splits["validation"], max(1, dcfg["max_train_samples"] // 10), rng)
    if dcfg.get("max_eval_samples"):
        splits["test"] = _subsample(splits["test"], dcfg["max_eval_samples"], rng)
    return splits


def _subsample(rows: list[Example], n: int, rng) -> list[Example]:
    if n >= len(rows):
        return rows
    idx = sorted(rng.choice(len(rows), size=n, replace=False).tolist())
    return [rows[i] for i in idx]


# --------------------------------------------------------------------------- #
# Hugging Face datasets (CNN/DailyMail 3.0.0, XSum)
# --------------------------------------------------------------------------- #
def _load_local_parallel(local_dir: str) -> dict[str, list[dict]]:
    """Official Multi-News release layout: {train,val,test}[.truncate].src/.tgt (one example per line)."""
    from pathlib import Path

    d, out = Path(local_dir), {}
    for split, stem in (("train", "train"), ("validation", "val"), ("test", "test")):
        src = sorted(d.glob(f"{stem}*.src*"))[0]
        tgt = sorted(d.glob(f"{stem}*.tgt*"))[0]
        with open(src, encoding="utf-8") as fs, open(tgt, encoding="utf-8") as ft:
            out[split] = [{"document": a.replace("NEWLINE_CHAR", " "), "summary": b.lstrip("– -")}
                          for a, b in zip(fs, ft)]
    return out


def _load_hf(cfg: dict, dcfg: dict) -> dict[str, list[Example]]:
    if dcfg.get("local_dir"):
        ds = _load_local_parallel(dcfg["local_dir"])
    else:
        from datasets import load_dataset

        cache = cfg["paths"].get("cache_dir")
        kw = {"revision": dcfg["hf_revision"]} if dcfg.get("hf_revision") else {}
        if dcfg.get("hf_data_dir"):
            # parquet conversion of a multi-config script dataset: one sub-folder per config
            try:
                ds = load_dataset(dcfg["hf_name"], data_dir=dcfg["hf_data_dir"], cache_dir=cache, **kw)
            except Exception as exc:  # fall back to explicit parquet file patterns
                print(f"[data] data_dir loading failed ({exc}); trying explicit parquet files", flush=True)
                from urllib.parse import quote

                base = f"hf://datasets/{dcfg['hf_name']}@{quote(dcfg.get('hf_revision', 'main'), safe='')}"
                files = {sp: f"{base}/{dcfg['hf_data_dir']}/{sp}/*.parquet" for sp in ("train", "validation", "test")}
                ds = load_dataset("parquet", data_files=files, cache_dir=cache)
        else:
            ds = load_dataset(dcfg["hf_name"], dcfg.get("hf_config"), cache_dir=cache, **kw)
    out = {}
    for split in ("train", "validation", "test"):
        rows, dropped = [], 0
        for i, r in enumerate(ds[split]):
            raw_src = r[dcfg["source_field"]]
            if dcfg.get("source_sep"):
                raw_src = raw_src.replace(dcfg["source_sep"], " ||||| ")  # keep document boundaries visible
            src = _normalise_ws(raw_src)
            tgt = _normalise_ws(r[dcfg["target_field"]])
            # Only empty pairs are removed from the test split (keeps it comparable).
            if not src or not tgt or (split == "train" and len(src.split()) < 30):
                dropped += 1
                continue
            uid = r.get("id") or f"{split}-{i}"
            rows.append(Example(str(uid), src, tgt))
        out[split] = rows
        print(f"[data] {dcfg['hf_name']} {split}: kept {len(rows)}, dropped {dropped}")
    return out


# --------------------------------------------------------------------------- #
# Indian News (local JSONL)
# --------------------------------------------------------------------------- #
_BOILERPLATE = [
    r"Story continues below this ad",
    r"Also Read\s*\|[^.]*\.?",
    r"Advertisement",
    r"^summarize:\s*",
]
_NON_NEWS_SOURCE = re.compile(r"timeslearn|masterclass|/web-series/|course", re.I)


def clean_indian_records(records: list[dict], min_source_words: int, max_compression: float):
    log = collections.OrderedDict(raw=len(records))
    rows = []
    for r in records:
        src = str(r.get("input_text", ""))
        tgt = str(r.get("target_text", ""))
        for pat in _BOILERPLATE:
            src = re.sub(pat, " ", src, flags=re.I | re.M)
            tgt = re.sub(pat, " ", tgt, flags=re.I | re.M)
        rows.append(dict(source=_normalise_ws(src), target=_normalise_ws(tgt), url=str(r.get("source", ""))))

    rows = [r for r in rows if not _NON_NEWS_SOURCE.search(r["url"])]
    log["after_non_news_source_filter"] = len(rows)

    seen, uniq = set(), []
    for r in rows:
        h = hashlib.md5(r["source"].lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq.append(r)
    rows = uniq
    log["after_deduplication"] = len(rows)

    rows = [r for r in rows if len(r["source"].split()) >= min_source_words and r["target"]]
    log["after_min_source_words"] = len(rows)

    rows = [r for r in rows if len(r["target"].split()) < max_compression * len(r["source"].split())]
    log["after_compression_filter"] = len(rows)
    return rows, log


def _load_indian(cfg: dict, dcfg: dict, seed: int) -> dict[str, list[Example]]:
    raw = read_jsonl(cfg["paths"]["indian_news_jsonl"])
    rows, log = clean_indian_records(raw, dcfg["min_source_words"], dcfg["max_compression"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows)).tolist()
    n = len(rows)
    n_tr = int(round(dcfg["split"][0] * n))
    n_va = int(round(dcfg["split"][1] * n))
    parts = {"train": perm[:n_tr], "validation": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    out = {k: [Example(f"indian-{i}", rows[i]["source"], rows[i]["target"]) for i in v] for k, v in parts.items()}
    log.update({f"n_{k}": len(v) for k, v in out.items()})
    write_json(f"{cfg['paths']['output_dir']}/indian/cleaning_log.json", log)
    print(f"[data] Indian News cleaning: {dict(log)}")
    return out


def _normalise_ws(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# --------------------------------------------------------------------------- #
# Descriptive statistics (Table: dataset statistics)
# --------------------------------------------------------------------------- #
def _ngrams(tokens: list[str], n: int) -> set:
    return set(zip(*[tokens[i:] for i in range(n)]))


def dataset_statistics(splits: dict[str, list[Example]]) -> dict:
    stats = {f"n_{k}": len(v) for k, v in splits.items()}
    rows = splits["train"] + splits["validation"] + splits["test"]
    tok = lambda s: re.findall(r"\w+", s.lower())  # noqa: E731
    src_w = np.array([len(tok(r.source)) for r in rows])
    tgt_w = np.array([len(tok(r.target)) for r in rows])
    src_s = np.array([len(split_sentences(r.source)) for r in rows])
    tgt_s = np.array([len(split_sentences(r.target)) for r in rows])
    novel = {1: [], 2: []}
    for r in rows:
        s, t = tok(r.source), tok(r.target)
        for n in (1, 2):
            tg = _ngrams(t, n)
            if tg:
                novel[n].append(1 - len(tg & _ngrams(s, n)) / len(tg))
    stats.update(
        avg_source_words=float(src_w.mean()),
        avg_target_words=float(tgt_w.mean()),
        avg_source_sents=float(src_s.mean()),
        avg_target_sents=float(tgt_s.mean()),
        compression_ratio=float((src_w / np.maximum(tgt_w, 1)).mean()),
        novel_unigram_pct=100 * float(np.mean(novel[1])),
        novel_bigram_pct=100 * float(np.mean(novel[2])),
        pct_source_over_512_t5_tokens_est=100 * float(np.mean(src_w * 1.3 > 512)),
    )
    return stats
