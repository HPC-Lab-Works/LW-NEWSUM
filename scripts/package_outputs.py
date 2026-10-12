"""Bundle the released system outputs (no model checkpoints) into one archive.

Includes, for every run: predictions.jsonl, metrics.json, per_example.json, factuality.json, train_log.jsonl and
train_summary.json; plus outputs/summary.json, every efficiency*.json, dataset_stats.json and the Indian News
cleaning log. Excludes trainable_state.pt, hf_model/ and human-evaluation keys or sheets.

Usage: python scripts/package_outputs.py [outputs] [lw-newsum-outputs.tar.gz]
"""
import sys
import tarfile
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "lw-newsum-outputs.tar.gz")
KEEP = {"predictions.jsonl", "metrics.json", "per_example.json", "factuality.json", "train_log.jsonl",
        "train_summary.json", "summary.json", "dataset_stats.json", "cleaning_log.json"}
SKIP_DIRS = {"hf_model", "human_eval", "logs"}

n, size = 0, 0
with tarfile.open(dst, "w:gz") as tar:
    for f in sorted(src.rglob("*")):
        if not f.is_file() or SKIP_DIRS & set(f.relative_to(src).parts):
            continue
        if f.name in KEEP or (f.name.startswith("efficiency") and f.suffix == ".json"):
            tar.add(f, arcname=str(Path("outputs") / f.relative_to(src)))
            n += 1
            size += f.stat().st_size
print(f"[package] {n} files, {size / 2**20:.1f} MB uncompressed -> {dst} ({dst.stat().st_size / 2**20:.1f} MB)")
