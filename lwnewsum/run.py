"""Command-line entry point.

Examples
--------
python -m lwnewsum.run stats      --dataset cnndm
python -m lwnewsum.run run        --dataset cnndm --system lwnewsum --seed 13
python -m lwnewsum.run factuality --dataset cnndm --system lwnewsum --seed 13
python -m lwnewsum.run efficiency --dataset cnndm
python -m lwnewsum.run aggregate
python -m lwnewsum.run export     --paper_dir paper
python -m lwnewsum.run human_eval --dataset cnndm
Every command accepts ``--config`` and repeated ``--set key=value`` overrides.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from . import extractive
from .data import dataset_statistics, load_splits
from .systems import SYSTEMS, build_model, input_prefix, load_trained, save_trained
from .utils import load_config, read_json, read_jsonl, run_dir, set_seed, write_json, write_jsonl

SPLIT_SEED = 42


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _env() -> dict:
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


# --------------------------------------------------------------------------- #
def cmd_stats(cfg, a):
    splits = load_splits(cfg, a.dataset, SPLIT_SEED)
    st = dataset_statistics(splits)
    write_json(Path(cfg["paths"]["output_dir"]) / a.dataset / "dataset_stats.json", st)
    print(json.dumps(st, indent=2))


def cmd_run(cfg, a):
    spec = SYSTEMS[a.system]
    seed = a.seed if spec.seeded else None
    out = run_dir(cfg, a.dataset, a.system, seed)
    if (out / "metrics.json").exists() and not a.force:
        print(f"[run] {out} already complete; use --force to redo.")
        return
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed if seed is not None else SPLIT_SEED)
    dcfg = cfg["datasets"][a.dataset]
    splits = load_splits(cfg, a.dataset, SPLIT_SEED)
    test = splits["test"]
    device = _device()
    info = {"dataset": a.dataset, "system": a.system, "seed": seed, "env": _env(),
            "n_train": len(splits["train"]), "n_val": len(splits["validation"]), "n_test": len(test)}

    if spec.kind == "extractive":
        k = extractive.k_for_dataset(a.dataset)
        fn = extractive.lead_k if a.system == "lead3" else extractive.textrank
        t0 = time.time()
        rows = [{"uid": e.uid, "prediction": fn(e.source, k), "reference": e.target} for e in test]
        info["gen_time_s"] = time.time() - t0
    elif spec.kind == "causal":
        from .slm import generate_causal, load_causal
        from .systems import backbone_name

        name = backbone_name(cfg, a.dataset, spec)
        model, tok = load_causal(name, cfg["paths"].get("cache_dir"), device)
        info["model_name"] = name
        info["params"] = {"total": sum(p.numel() for p in model.parameters()), "trainable": 0, "trainable_pct": 0.0}
        t0 = time.time()
        rows = generate_causal(model, tok, test, dcfg, device, cfg["eval"]["slm_batch_size"],
                               cfg["eval"]["slm_max_article_tokens"])
        info["gen_time_s"] = time.time() - t0
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        model, tok, name = build_model(cfg, a.dataset, a.system, seed)
        prefix = input_prefix(cfg, a.dataset, a.system)
        from .modeling import count_parameters

        info["model_name"] = name
        info["params"] = count_parameters(model)
        if not spec.trains:
            info["params"].update(trainable=0, trainable_pct=0.0)
        if spec.trains:
            if (out / "trainable_state.pt").exists() and not a.force:
                load_trained(model, out)
                info["train"] = read_json(out / "train_summary.json")
            else:
                from .train import train_model

                info["train"] = train_model(model, spec, tok, splits, dcfg, cfg["train"], out, device, prefix, seed)
                save_trained(model, a.system, out)
                write_json(out / "train_summary.json", info["train"])
        from .train import generate_predictions

        t0 = time.time()
        rows = generate_predictions(model, tok, test, dcfg, dcfg.get("gen_batch_size", cfg["eval"]["gen_batch_size"]), device, prefix,
                                    cfg["train"]["bf16"])
        info["gen_time_s"] = time.time() - t0
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_jsonl(out / "predictions.jsonl", rows)
    _score(cfg, a.dataset, out, rows, {e.uid: e.source for e in test}, device, info, not a.no_bertscore)


def _score(cfg, dataset, out, rows, sources, device, info, with_bertscore=True):
    from .metrics import compute_all

    preds = [r["prediction"] for r in rows]
    refs = [r["reference"] for r in rows]
    srcs = [sources[r["uid"]] for r in rows]
    corpus, per = compute_all(preds, refs, srcs, cfg, device=str(device), with_bertscore=with_bertscore)
    per["uid"] = [r["uid"] for r in rows]
    write_json(out / "per_example.json", per)
    info["metrics"] = corpus
    write_json(out / "metrics.json", info)
    print(f"[score] {out}: " + ", ".join(f"{k}={v:.2f}" for k, v in corpus.items() if isinstance(v, float)))


def cmd_score(cfg, a):
    spec = SYSTEMS[a.system]
    seed = a.seed if spec.seeded else None
    out = run_dir(cfg, a.dataset, a.system, seed)
    rows = read_jsonl(out / "predictions.jsonl")
    splits = load_splits(cfg, a.dataset, SPLIT_SEED)
    info = read_json(out / "metrics.json") if (out / "metrics.json").exists() else {}
    _score(cfg, a.dataset, out, rows, {e.uid: e.source for e in splits["test"]}, _device(), info, not a.no_bertscore)


def cmd_factuality(cfg, a):
    from .factuality import NLIScorer, factuality, factuality_subset_ids

    spec = SYSTEMS[a.system]
    seed = a.seed if spec.seeded else None
    out = run_dir(cfg, a.dataset, a.system, seed)
    if (out / "factuality.json").exists() and not a.force:
        print(f"[factuality] {out} done.")
        return
    ecfg = cfg["eval"]
    splits = load_splits(cfg, a.dataset, SPLIT_SEED)
    src = {e.uid: e.source for e in splits["test"]}
    rows = {r["uid"]: r for r in read_jsonl(out / "predictions.jsonl")}
    ids = factuality_subset_ids([e.uid for e in splits["test"]], ecfg["factuality_subset"], ecfg["bootstrap_seed"])
    scorer = NLIScorer(ecfg["nli_model"], _device())
    res = factuality(scorer, [src[i] for i in ids], [rows[i]["prediction"] for i in ids], ecfg["nli_threshold"],
                     ecfg["nli_chunk_tokens"])
    res["uid"] = ids
    summary = {k: float(np.mean(v)) for k, v in res.items() if k != "uid"}
    write_json(out / "factuality.json", {"summary": summary, "per_example": res, "nli_model": ecfg["nli_model"]})
    print(f"[factuality] {out}: {summary}")


def cmd_efficiency(cfg, a):
    from .efficiency import profile

    dcfg = cfg["datasets"][a.dataset]
    splits = load_splits(cfg, a.dataset, SPLIT_SEED)
    texts = [e.source for e in splits["test"][: cfg["eval"]["efficiency_docs"]]]
    device = _device()
    gen = dict(dcfg["gen"], early_stopping=True)
    res = {}
    for name in a.systems.split(","):
        spec = SYSTEMS[name]
        seed = cfg["seeds"][0] if spec.seeded else None
        if spec.kind == "causal":
            res[name] = _efficiency_causal(cfg, a.dataset, spec, splits["test"][: cfg["eval"]["efficiency_docs"]], device)
            print(f"[efficiency] {name}: {res[name]}")
            continue
        model, tok, _ = build_model(cfg, a.dataset, name, seed)
        if spec.trains:
            load_trained(model, run_dir(cfg, a.dataset, name, seed))
        if a.wrap_plain and spec.kind == "seq2seq":
            from .modeling import EncoderAdaptedSeq2Seq, IdentityModule

            if not isinstance(model, EncoderAdaptedSeq2Seq):
                # same generation path as the module variants: encoder -> (no-op) -> generate(encoder_outputs=...)
                model = EncoderAdaptedSeq2Seq(model, IdentityModule(), freeze_backbone=False)
                print(f"[efficiency] {name}: timed through the encoder-adapted generation path (no-op module)")
        prefix = input_prefix(cfg, a.dataset, name)
        res[name] = profile(model, tok, texts, gen, device, prefix, dcfg["max_source_len"], cfg["train"]["bf16"],
                            trained=spec.trains)
        print(f"[efficiency] {name}: {res[name]}")
        if a.module_off and hasattr(model, "module") and hasattr(model, "backbone"):
            # same object, same weights, same generation path: only the module is replaced by a no-op
            from .modeling import IdentityModule

            model.module = IdentityModule()
            res[f"{name}__module_off"] = profile(model, tok, texts, gen, device, prefix, dcfg["max_source_len"],
                                                 cfg["train"]["bf16"], trained=spec.trains)
            print(f"[efficiency] {name}__module_off: {res[f'{name}__module_off']}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    res["_env"] = _env()
    name = f"efficiency_{a.tag}.json" if a.tag else "efficiency.json"
    write_json(Path(cfg["paths"]["output_dir"]) / a.dataset / name, res)


def _efficiency_causal(cfg, dataset, spec, examples, device):
    from .slm import generate_causal, load_causal
    from .systems import backbone_name

    model, tok = load_causal(backbone_name(cfg, dataset, spec), cfg["paths"].get("cache_dir"), device)
    total = sum(p.numel() for p in model.parameters())
    dcfg = cfg["datasets"][dataset]
    res = {"params_total_m": total / 1e6, "params_trainable_m": 0.0, "fp32_size_mb": total * 4 / 2**20,
           "gflops_forward": None}
    for bs, key in ((1, "ms_per_doc_bs1"), (32, "ms_per_doc_bs32")):
        docs = examples[:50] if bs == 1 else examples
        generate_causal(model, tok, examples[:bs], dcfg, device, bs, cfg["eval"]["slm_max_article_tokens"])  # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        generate_causal(model, tok, docs, dcfg, device, bs, cfg["eval"]["slm_max_article_tokens"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        res[key] = 1000 * (time.perf_counter() - t0) / len(docs)
    res["peak_mem_mb_bs32"] = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    del model
    return res


def cmd_aggregate(cfg, a):
    from .aggregate import aggregate

    aggregate(cfg)


def cmd_export(cfg, a):
    from .export_tables import export

    export(cfg, a.paper_dir)


def cmd_human_eval(cfg, a):
    from .human_eval import build_sheet

    build_sheet(cfg, a.dataset, a.n_docs)


def cmd_human_eval_analyze(cfg, a):
    from .human_eval import analyze

    analyze(cfg, a.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["stats", "run", "score", "factuality", "efficiency", "aggregate", "export",
                                        "human_eval", "human_eval_analyze"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.epochs=1")
    ap.add_argument("--dataset", default="cnndm", choices=["cnndm", "xsum", "xlsum", "multinews", "indian"])
    ap.add_argument("--system", default="lwnewsum", choices=list(SYSTEMS))
    ap.add_argument("--systems", default="t5_ft,flant5_ft,bartbase_ft,t5_lora,t5_dora,t5_ia3,t5_mlp_adapter,lwnewsum,"
                    "lwnewsum_flant5,lwnewsum_bartbase,distilbart,qwen25_05b,qwen25_15b,smollm2_17b")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no_bertscore", action="store_true")
    ap.add_argument("--paper_dir", default="paper")
    ap.add_argument("--n_docs", type=int, default=50)
    ap.add_argument("--module_off", action="store_true",
                    help="efficiency: also time each module variant with its module replaced by a no-op")
    ap.add_argument("--wrap_plain", action="store_true",
                    help="efficiency: time plain fine-tuned models through the same generation path as the module variants")
    ap.add_argument("--tag", default="", help="efficiency: write efficiency_<tag>.json instead of efficiency.json")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    globals()[f"cmd_{a.command}"](cfg, a)


if __name__ == "__main__":
    main()
