"""Shared helpers: configuration, seeding, sentence splitting, I/O."""
from __future__ import annotations

import copy
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply dotted overrides such as ``train.epochs=1``."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Sentence splitter: NLTK Punkt when available, regex fallback otherwise."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        from nltk.tokenize import sent_tokenize

        sents = sent_tokenize(text)
    except Exception:  # punkt not downloaded or nltk missing
        sents = _SENT_RE.split(text)
    return [s.strip() for s in sents if len(s.strip()) > 3]


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | os.PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_dir(cfg: dict, dataset: str, system: str, seed: int | None) -> Path:
    tag = f"seed{seed}" if seed is not None else "deterministic"
    return Path(cfg["paths"]["output_dir"]) / dataset / system / tag
