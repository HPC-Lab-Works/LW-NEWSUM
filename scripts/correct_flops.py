"""Correct GFLOPs of LW-NEWSUM systems in an existing efficiency.json and summary.json.

Earlier versions of lwnewsum/efficiency.py counted FLOPs in eval mode, where nn.TransformerEncoderLayer runs a
fused kernel that torch's FlopCounterMode does not see. Only the module's attention layers were missed, so the
missing amount is a constant that does not depend on the backbone weights. This script measures it (module in
train mode minus eval mode, 512 source tokens) and adds it to every LW-NEWSUM entry. FLOP counts are
hardware-independent, so this can run on any machine. Re-running `efficiency` with the fixed code gives the
same values.

Usage: python scripts/correct_flops.py [outputs]
"""
import inspect
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lwnewsum.modeling import LocalGlobalModule  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
cfg = yaml.safe_load(open("configs/default.yaml"))["lwnewsum"]
kw = {k: v for k, v in cfg.items() if k in inspect.signature(LocalGlobalModule.__init__).parameters}
D_MODEL = {"lwnewsum": 512, "lwnewsum_on_ft": 512, "lwnewsum_joint": 512, "lwnewsum_flant5": 512,
           "lwnewsum_bartbase": 768, "lwnewsum_on_ft_bart": 768}


def missing_gflops(d):
    m, h, mask, res = LocalGlobalModule(d, **kw), torch.randn(1, 512, d), torch.ones(1, 512, dtype=torch.bool), {}
    for mode in ("eval", "train"):
        getattr(m, mode)()
        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            with fc:
                m(h, mask)
        res[mode] = fc.get_total_flops() / 1e9
    return res["train"] - res["eval"]


delta = {d: missing_gflops(d) for d in set(D_MODEL.values())}
print("[correct_flops] missing GFLOPs per width:", delta)


def fix(eff):
    for name, d in D_MODEL.items():
        e = eff.get(name)
        if e and e.get("gflops_forward") and not e.get("gflops_corrected"):
            e["gflops_forward"] += delta[d]
            e["gflops_corrected"] = True
            print(f"[correct_flops] {name}: {e['gflops_forward']:.2f} GFLOPs")


for ds_dir in out.iterdir():
    f = ds_dir / "efficiency.json"
    if f.exists():
        eff = json.load(open(f))
        fix(eff)
        json.dump(eff, open(f, "w"), indent=2)
s = out / "summary.json"
if s.exists():
    summ = json.load(open(s))
    for ds in summ.values():
        if isinstance(ds, dict) and "efficiency" in ds:
            fix(ds["efficiency"])
    json.dump(summ, open(s, "w"), indent=2)
