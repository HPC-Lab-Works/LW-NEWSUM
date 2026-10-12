#!/usr/bin/env python3
"""Run the full experimental grid on a multi-GPU node (e.g. NVIDIA DGX H200, 8 GPUs).

Each job (dataset x system x seed) trains if needed, generates test predictions,
scores them and runs the NLI factuality evaluation, on one GPU. Jobs whose
outputs already exist are skipped, so the script can be re-started safely.

    python scripts/launch_dgx.py                          # everything
    python scripts/launch_dgx.py --datasets cnndm,xsum    # a subset
    python scripts/launch_dgx.py --gpus 0,1,2,3 --dry_run
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lwnewsum.systems import SYSTEMS  # noqa: E402
from lwnewsum.utils import load_config, run_dir  # noqa: E402

ABLATION_DATASETS = {"cnndm", "xsum"}


def jobs_for(cfg, datasets, systems):
    jobs = []
    for ds, name in itertools.product(datasets, systems):
        spec = SYSTEMS[name]
        if spec.group == "ablation" and ds not in ABLATION_DATASETS:
            continue
        for seed in (cfg["seeds"] if spec.seeded else [None]):
            out = run_dir(cfg, ds, name, seed)
            dep = None
            if spec.depends_on:
                dep = run_dir(cfg, ds, spec.depends_on, seed) / "metrics.json"
            jobs.append({"ds": ds, "sys": name, "seed": seed, "out": out, "dep": dep,
                         "heavy": spec.kind == "causal" or not spec.freeze_backbone})
    # heavy jobs first so the long ones start early
    return sorted(jobs, key=lambda j: (not j["heavy"], j["dep"] is not None))


def done(j):
    return (j["out"] / "metrics.json").exists() and (j["out"] / "factuality.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--datasets", default="cnndm,xsum,xlsum,multinews,indian")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: all visible)")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_bertscore", action="store_true", help="skip BERTScore (offline testing only)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.gpus is None:
        import torch

        gpus = [str(i) for i in range(torch.cuda.device_count())] or ["cpu"]
    else:
        gpus = a.gpus.split(",")
    pending = [j for j in jobs_for(cfg, a.datasets.split(","), a.systems.split(",")) if not done(j)]
    print(f"{len(pending)} jobs pending on GPUs {gpus}")
    if a.dry_run:
        for j in pending:
            print(j["ds"], j["sys"], j["seed"])
        return
    logs = Path(cfg["paths"]["output_dir"]) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    progress = open(Path(cfg["paths"]["output_dir"]) / "progress.log", "a", buffering=1)
    total, n_done, durations, t_start = len(pending), 0, [], time.time()
    running = {}  # gpu -> (proc, job)
    failed = []

    def report(msg):
        eta = ""
        if durations:
            remaining = (total - n_done) * (sum(durations) / len(durations)) / max(len(gpus), 1)
            eta = f" | ETA ~{remaining / 3600:.1f} h"
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{n_done}/{total} done, {len(failed)} failed, "
                f"{len(running)} running{eta}] {msg}")
        print(line, flush=True)
        progress.write(line + "\n")

    report(f"launcher started on GPUs {gpus}")
    extra = sum((["--set", s] for s in a.set), []) + (["--no_bertscore"] if a.no_bertscore else [])
    while pending or running:
        for g, (p, j) in list(running.items()):
            if p.poll() is not None:
                del running[g]
                mins = (time.time() - j["t0"]) / 60
                if p.returncode != 0:
                    failed.append(j)
                    report(f"FAIL  {j['ds']} {j['sys']} seed={j['seed']} after {mins:.1f} min "
                           f"(see logs/{j['ds']}_{j['sys']}_{j['seed']}.log)")
                else:
                    n_done += 1
                    durations.append(time.time() - j["t0"])
                    report(f"DONE  {j['ds']} {j['sys']} seed={j['seed']} in {mins:.1f} min")
        for g in gpus:
            if g in running or not pending:
                continue
            ready = [j for j in pending if j["dep"] is None or j["dep"].exists()]
            if not ready:
                break
            j = ready[0]
            pending.remove(j)
            base = [sys.executable, "-m", "lwnewsum.run", "--config", a.config, "--dataset", j["ds"],
                    "--system", j["sys"]] + (["--seed", str(j["seed"])] if j["seed"] is not None else []) + extra
            cmd = " && ".join(" ".join(map(str, base[:3] + [c] + base[3:])) for c in ("run", "factuality"))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if g == "cpu" else g, TOKENIZERS_PARALLELISM="false",
                       PYTHONUNBUFFERED="1")
            log = open(logs / f"{j['ds']}_{j['sys']}_{j['seed']}.log", "w")
            j["t0"] = time.time()
            running[g] = (subprocess.Popen(cmd, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT), j)
            report(f"START {j['ds']} {j['sys']} seed={j['seed']} on GPU {g}")
        if pending and not running and not any(j["dep"] is None or j["dep"].exists() for j in pending):
            print("Remaining jobs wait on failed dependencies:", [(j["ds"], j["sys"], j["seed"]) for j in pending])
            break
        time.sleep(20)
    report(f"FINISHED in {(time.time() - t_start) / 3600:.1f} h; {len(failed)} failed")
    for j in failed:
        print("  failed:", j["ds"], j["sys"], j["seed"])


if __name__ == "__main__":
    main()
