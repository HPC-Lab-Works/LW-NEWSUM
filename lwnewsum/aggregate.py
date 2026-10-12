"""Collect every run into results/summary.json (seed means, std, significance)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .stats import holm, mean_std, paired_bootstrap
from .systems import SYSTEMS
from .utils import read_json, read_jsonl, write_json

MAIN_METRICS = ["rouge1", "rouge2", "rougeLsum", "bleu", "meteor", "bertscore_f1", "summary_words",
                "novel_bigram_pct"]
SIG_METRICS = ["rouge1", "rouge2", "rougeLsum", "bertscore_f1"]
FACT_METRICS = ["fcs", "hr", "summac_zs", "mean_entailment"]
PLUGIN_PAIRS = [("lwnewsum_on_ft", "t5_ft"), ("mlp_on_ft", "t5_ft"), ("lwnewsum_on_ft", "mlp_on_ft"),
                ("lwnewsum_on_ft_bart", "bartbase_ft"), ("mlp_on_ft_bart", "bartbase_ft"),
                ("lwnewsum_on_ft_bart", "mlp_on_ft_bart")]
PROPOSED = ["lwnewsum", "lwnewsum_on_ft", "lwnewsum_on_ft_bart", "lwnewsum_flant5", "lwnewsum_bartbase"]


def _runs(root: Path, dataset: str, system: str) -> list[Path]:
    d = root / dataset / system
    return sorted(p for p in d.glob("*") if (p / "metrics.json").exists()) if d.exists() else []


def _per_example_mean(runs: list[Path], fname: str, key: str, metric: str):
    """Average per-example scores across seeds, aligned by uid."""
    tables = []
    for r in runs:
        f = r / fname
        if not f.exists():
            continue
        obj = read_json(f)
        per = obj[key] if key else obj
        if metric not in per:
            continue
        tables.append(dict(zip(per["uid"], per[metric])))
    if not tables:
        return None
    uids = sorted(set.intersection(*[set(t) for t in tables]))
    return uids, np.array([[t[u] for t in tables] for u in uids]).mean(axis=1)


def aggregate(cfg: dict) -> dict:
    root = Path(cfg["paths"]["output_dir"])
    ecfg = cfg["eval"]
    summary = {}
    for dataset in cfg["datasets"]:
        if not (root / dataset).exists():
            continue
        ds = {"systems": {}, "significance": {}}
        if (root / dataset / "dataset_stats.json").exists():
            ds["stats"] = read_json(root / dataset / "dataset_stats.json")
        if (root / dataset / "efficiency.json").exists():
            ds["efficiency"] = read_json(root / dataset / "efficiency.json")
        if (root / dataset / "cleaning_log.json").exists():
            ds["cleaning_log"] = read_json(root / dataset / "cleaning_log.json")
        print(f"[aggregate] {dataset}: collecting metrics", flush=True)
        for name, spec in SYSTEMS.items():
            runs = _runs(root, dataset, name)
            if not runs:
                continue
            infos = [read_json(r / "metrics.json") for r in runs]
            entry = {"label": spec.label, "group": spec.group, "n_runs": len(runs), "metrics": {}, "factuality": {}}
            for m in MAIN_METRICS:
                vals = [i["metrics"][m] for i in infos if m in i.get("metrics", {})]
                if vals:
                    entry["metrics"][m] = mean_std(vals)
            facts = [read_json(r / "factuality.json")["summary"] for r in runs if (r / "factuality.json").exists()]
            for m in FACT_METRICS:
                vals = [f[m] for f in facts if m in f]
                if vals:
                    entry["factuality"][m] = mean_std(vals)
            tr = [i["train"] for i in infos if "train" in i]
            if tr:
                entry["train"] = {k: mean_std([t[k] for t in tr]) for k in ("best_val_loss", "train_time_s",
                                                                             "optimizer_steps")}
            if "params" in infos[0]:
                entry["params"] = infos[0]["params"]
            ds["systems"][name] = entry

        # significance: each proposed system vs every other non-ablation system
        for prop in PROPOSED:
            prop_runs = _runs(root, dataset, prop)
            if not prop_runs:
                continue
            res = {}
            for metric in SIG_METRICS + FACT_METRICS[:1] + ["summac_zs"]:
                print(f"[aggregate] {dataset}: bootstrap {prop} on {metric}", flush=True)
                fname, key = ("factuality.json", "per_example") if metric in FACT_METRICS else ("per_example.json", None)
                a = _per_example_mean(prop_runs, fname, key, metric)
                if a is None:
                    continue
                rows = []
                for other, spec in SYSTEMS.items():
                    if other == prop or spec.group == "ablation":
                        continue
                    b = _per_example_mean(_runs(root, dataset, other), fname, key, metric)
                    if b is None:
                        continue
                    common = sorted(set(a[0]) & set(b[0]))
                    ia = {u: i for i, u in enumerate(a[0])}
                    ib = {u: i for i, u in enumerate(b[0])}
                    r = paired_bootstrap(a[1][[ia[u] for u in common]], b[1][[ib[u] for u in common]],
                                         ecfg["bootstrap_samples"], ecfg["bootstrap_seed"])
                    r["vs"] = other
                    rows.append(r)
                for r, ph in zip(rows, holm([r["p"] for r in rows])):
                    r["p_holm"] = ph
                res[metric] = rows
            ds["significance"][prop] = res

        # plug-in tests: each plug-in vs its own fine-tuned base, and attention vs MLP plug-in
        ds["plugin_tests"] = {}
        for metric in ("rougeLsum", "bertscore_f1", "fcs", "summac_zs"):
            fname, key = ("factuality.json", "per_example") if metric in FACT_METRICS else ("per_example.json", None)
            rows = []
            for a_sys, b_sys in PLUGIN_PAIRS:
                a = _per_example_mean(_runs(root, dataset, a_sys), fname, key, metric)
                b = _per_example_mean(_runs(root, dataset, b_sys), fname, key, metric)
                if a is None or b is None:
                    continue
                common = sorted(set(a[0]) & set(b[0]))
                ia = {u: i for i, u in enumerate(a[0])}
                ib = {u: i for i, u in enumerate(b[0])}
                r = paired_bootstrap(a[1][[ia[u] for u in common]], b[1][[ib[u] for u in common]],
                                     ecfg["bootstrap_samples"], ecfg["bootstrap_seed"])
                r.update(a=a_sys, b=b_sys)
                rows.append(r)
            for r, ph in zip(rows, holm([r["p"] for r in rows])):
                r["p_holm"] = ph
            ds["plugin_tests"][metric] = rows
            print(f"[aggregate] {dataset}: plug-in tests on {metric} ({len(rows)} comparisons)", flush=True)

        # qualitative examples (seeded, first seed of each trained system)
        ds["examples"] = _examples(root, dataset, cfg)
        summary[dataset] = ds
    write_json(root / "summary.json", summary)
    print(f"[aggregate] wrote {root / 'summary.json'}")
    return summary


def _examples(root: Path, dataset: str, cfg: dict, n: int = 2) -> list[dict]:
    wanted = ["lead3", "t5_ft", "distilbart", "lwnewsum"]
    preds = {}
    for s in wanted:
        runs = _runs(root, dataset, s)
        if runs:
            preds[s] = {r["uid"]: r["prediction"] for r in read_jsonl(runs[0] / "predictions.jsonl")}
    if len(preds) < 2:
        return []
    common = sorted(set.intersection(*[set(p) for p in preds.values()]))
    rng = np.random.default_rng(cfg["eval"]["bootstrap_seed"])
    pick = rng.choice(len(common), size=min(n, len(common)), replace=False)
    ref_run = _runs(root, dataset, next(iter(preds)))[0]
    refs = {r["uid"]: r["reference"] for r in read_jsonl(ref_run / "predictions.jsonl")}
    return [{"uid": common[i], "reference": refs[common[i]], **{s: preds[s][common[i]] for s in preds}} for i in pick]
