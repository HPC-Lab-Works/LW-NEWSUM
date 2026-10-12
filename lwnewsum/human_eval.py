"""Blind human evaluation (Section 15.9).

build_sheet: samples N test documents (seeded), shuffles the system outputs per
document into anonymous slots A-D, and writes
  outputs/<dataset>/human_eval/annotation_sheet.csv   (give to each annotator)
  outputs/<dataset>/human_eval/key.json               (keep private)
Annotators rate each summary 1-5 on informativeness, faithfulness, fluency and
conciseness (guidelines in paper/appendix). Save each completed sheet as
annotator_<id>.csv in the same folder, then run analyze().

analyze: mean scores per system, Krippendorff's alpha (interval) per criterion,
and Wilcoxon signed-rank tests of LW-NEWSUM against each system.
"""
from __future__ import annotations

import csv
import string
from pathlib import Path

import numpy as np

from .data import load_splits
from .utils import read_json, read_jsonl, run_dir, write_json

# The question for human judges is whether the plug-in gains (and the lower NLI scores) reflect real changes:
# the fine-tuned backbone, both plug-ins on top of it, and DistilBART as a strong reference.
SYSTEMS_FOR_HUMAN = ["t5_ft", "lwnewsum_on_ft", "mlp_on_ft", "distilbart"]
REFERENCE_SYSTEM = "lwnewsum_on_ft"
UNSEEDED = ("lead3", "textrank", "distilbart", "t5_zeroshot")
CRITERIA = ["informativeness", "faithfulness", "fluency", "conciseness"]


def build_sheet(cfg: dict, dataset: str, n_docs: int = 50, seed: int = 2026):
    splits = load_splits(cfg, dataset, 42)
    src = {e.uid: e.source for e in splits["test"]}
    preds = {}
    for s in SYSTEMS_FOR_HUMAN:
        d = run_dir(cfg, dataset, s, cfg["seeds"][0] if s not in UNSEEDED else None)
        preds[s] = {r["uid"]: r["prediction"] for r in read_jsonl(d / "predictions.jsonl")}
    common = sorted(set.intersection(*[set(p) for p in preds.values()]))
    rng = np.random.default_rng(seed)
    docs = [common[i] for i in rng.choice(len(common), size=min(n_docs, len(common)), replace=False)]
    out = Path(cfg["paths"]["output_dir"]) / dataset / "human_eval"
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    with open(out / "annotation_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "article", "slot", "summary"] + CRITERIA)
        for uid in docs:
            order = list(rng.permutation(SYSTEMS_FOR_HUMAN))
            key[uid] = {string.ascii_uppercase[i]: s for i, s in enumerate(order)}
            for i, s in enumerate(order):
                w.writerow([uid, src[uid] if i == 0 else "", string.ascii_uppercase[i], preds[s][uid]] + [""] * 4)
    write_json(out / "key.json", key)
    print(f"[human_eval] sheet with {len(docs)} documents written to {out}")


def krippendorff_alpha_interval(data: np.ndarray) -> float:
    """data: annotators x units, NaN for missing. Interval metric."""
    units = [data[:, j][~np.isnan(data[:, j])] for j in range(data.shape[1])]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    n = sum(len(u) for u in units)
    do = sum(((u[:, None] - u[None, :]) ** 2).sum() / (len(u) - 1) for u in units) / n
    allv = np.concatenate(units)
    de = ((allv[:, None] - allv[None, :]) ** 2).sum() / (n * (n - 1))
    return float(1 - do / de) if de > 0 else float("nan")


def analyze(cfg: dict, dataset: str):
    from scipy.stats import wilcoxon

    out = Path(cfg["paths"]["output_dir"]) / dataset / "human_eval"
    key = read_json(out / "key.json")
    sheets = sorted(out.glob("annotator_*.csv"))
    ratings = {}  # (annotator, uid, system, criterion) -> score
    for a, path in enumerate(sheets):
        cur = None
        # utf-8-sig: Excel's "CSV UTF-8" adds a byte-order mark that would otherwise corrupt the first header
        for row in csv.DictReader(open(path, encoding="utf-8-sig", newline="")):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            if not row.get("slot"):
                continue
            cur = row.get("doc_id") or cur
            sysname = key[cur][row["slot"].upper()]
            for c in CRITERIA:
                val = row.get(c, "")
                if val:
                    score = float(val)
                    if not 1 <= score <= 5:
                        raise ValueError(f"{path.name}: {c} score {val} for document {cur}, slot {row['slot']} is outside 1-5")
                    ratings[(a, cur, sysname, c)] = score
    res = {"n_annotators": len(sheets), "n_docs": len(key), "mean": {}, "alpha": {}, "wilcoxon_vs_reference": {}, "reference": REFERENCE_SYSTEM}
    items = [(u, s) for u in key for s in SYSTEMS_FOR_HUMAN]
    for c in CRITERIA:
        mat = np.array([[ratings.get((a, u, s, c), np.nan) for (u, s) in items] for a in range(len(sheets))])
        res["alpha"][c] = krippendorff_alpha_interval(mat)
        per_item = np.nanmean(mat, axis=0)
        for s in SYSTEMS_FOR_HUMAN:
            idx = [i for i, (_, ss) in enumerate(items) if ss == s]
            res["mean"].setdefault(s, {})[c] = float(np.nanmean(per_item[idx]))
        lw = np.array([per_item[i] for i, (_, s) in enumerate(items) if s == REFERENCE_SYSTEM])
        for s in SYSTEMS_FOR_HUMAN:
            if s == REFERENCE_SYSTEM:
                continue
            other = np.array([per_item[i] for i, (_, ss) in enumerate(items) if ss == s])
            try:
                p = float(wilcoxon(lw, other).pvalue)
            except ValueError:
                p = float("nan")
            res["wilcoxon_vs_reference"].setdefault(s, {})[c] = p
    expected = len(sheets) * len(items) * len(CRITERIA)
    res["n_ratings"], res["n_expected"] = len(ratings), expected
    if len(ratings) < expected:
        print(f"[human_eval] warning: {expected - len(ratings)} of {expected} ratings are empty")
    write_json(out / "human_eval_results.json", res)
    print(res)
    return res
