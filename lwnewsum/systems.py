"""Registry of all compared systems (Section 6 of the manuscript).

Groups
------
extractive : Lead-k, TextRank
reference  : pretrained checkpoints used as-is (T5-Small zero-shot, DistilBART, small
             instruction-tuned LLMs prompted zero-shot)
baseline   : trained here on the same splits (full fine-tuning of T5-Small, Flan-T5-Small
             and BART-base; parameter-efficient LoRA, DoRA, IA3 and an MLP adapter)
proposed   : LW-NEWSUM on frozen T5-Small, on a fine-tuned T5-Small, jointly fine-tuned,
             and on frozen Flan-T5-Small and BART-base backbones (generality)
ablation   : LW-NEWSUM variants, each re-trained from scratch
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .utils import deep_update


@dataclass
class SystemSpec:
    name: str
    label: str                      # name used in the paper tables
    kind: str                       # extractive | seq2seq | causal
    group: str
    trains: bool = False
    backbone_key: str = "t5"        # key into cfg["backbones"]
    adapter: Optional[str] = None   # lora | dora | ia3 | mlp | lwnewsum | None (full model)
    freeze_backbone: bool = True
    lw_overrides: dict = field(default_factory=dict)
    depends_on: Optional[str] = None
    seeded: bool = True


SYSTEMS: dict[str, SystemSpec] = {}


def _add(spec: SystemSpec):
    SYSTEMS[spec.name] = spec


# Extractive -------------------------------------------------------------------
_add(SystemSpec("lead3", "Lead-3", "extractive", "extractive", seeded=False))
_add(SystemSpec("textrank", "TextRank", "extractive", "extractive", seeded=False))
# Pretrained references (not trained here) -------------------------------------
_add(SystemSpec("t5_zeroshot", "T5-Small (zero-shot)", "seq2seq", "reference", seeded=False))
_add(SystemSpec("distilbart", "DistilBART-12-6", "seq2seq", "reference", backbone_key="distilbart", seeded=False))
_add(SystemSpec("qwen25_05b", "Qwen2.5-0.5B-Instruct", "causal", "reference", backbone_key="qwen25_05b", seeded=False))
_add(SystemSpec("qwen25_15b", "Qwen2.5-1.5B-Instruct", "causal", "reference", backbone_key="qwen25_15b", seeded=False))
_add(SystemSpec("smollm2_17b", "SmolLM2-1.7B-Instruct", "causal", "reference", backbone_key="smollm2_17b", seeded=False))
_add(SystemSpec("llama32_1b", "Llama-3.2-1B-Instruct", "causal", "reference", backbone_key="llama32_1b", seeded=False))
# Trained baselines ------------------------------------------------------------
_add(SystemSpec("t5_ft", "T5-Small (full FT)", "seq2seq", "baseline", trains=True, freeze_backbone=False))
_add(SystemSpec("flant5_ft", "Flan-T5-Small (full FT)", "seq2seq", "baseline", trains=True, backbone_key="flant5",
                freeze_backbone=False))
_add(SystemSpec("bartbase_ft", "BART-base (full FT)", "seq2seq", "baseline", trains=True, backbone_key="bartbase",
                freeze_backbone=False))
_add(SystemSpec("t5_lora", "T5-Small + LoRA", "seq2seq", "baseline", trains=True, adapter="lora"))
_add(SystemSpec("t5_dora", "T5-Small + DoRA", "seq2seq", "baseline", trains=True, adapter="dora"))
_add(SystemSpec("t5_ia3", "T5-Small + (IA)$^3$", "seq2seq", "baseline", trains=True, adapter="ia3"))
_add(SystemSpec("t5_mlp_adapter", "T5-Small + MLP adapter", "seq2seq", "baseline", trains=True, adapter="mlp"))
# Proposed ---------------------------------------------------------------------
_add(SystemSpec("lwnewsum", "LW-NEWSUM", "seq2seq", "proposed", trains=True, adapter="lwnewsum"))
_add(SystemSpec("lwnewsum_on_ft", "LW-NEWSUM (on FT T5)", "seq2seq", "proposed", trains=True, adapter="lwnewsum",
                depends_on="t5_ft"))
# Plug-in mode: module trained on top of an already fine-tuned (then frozen) backbone, with
# parameter-matched MLP-adapter controls that isolate the value of contextual attention.
_add(SystemSpec("mlp_on_ft", "MLP adapter (on FT T5)", "seq2seq", "baseline", trains=True, adapter="mlp",
                depends_on="t5_ft"))
_add(SystemSpec("lwnewsum_on_ft_bart", "LW-NEWSUM (on FT BART-base)", "seq2seq", "proposed", trains=True,
                adapter="lwnewsum", backbone_key="bartbase", depends_on="bartbase_ft"))
_add(SystemSpec("mlp_on_ft_bart", "MLP adapter (on FT BART-base)", "seq2seq", "baseline", trains=True,
                adapter="mlp", backbone_key="bartbase", depends_on="bartbase_ft"))
_add(SystemSpec("lwnewsum_joint", "LW-NEWSUM (joint FT)", "seq2seq", "proposed", trains=True, adapter="lwnewsum",
                freeze_backbone=False))
_add(SystemSpec("lwnewsum_flant5", "LW-NEWSUM (Flan-T5-Small)", "seq2seq", "proposed", trains=True,
                adapter="lwnewsum", backbone_key="flant5"))
_add(SystemSpec("lwnewsum_bartbase", "LW-NEWSUM (BART-base)", "seq2seq", "proposed", trains=True,
                adapter="lwnewsum", backbone_key="bartbase"))
# Ablations: each variant is RE-TRAINED from scratch with the same budget --------
_ABL = {
    "abl_no_local": ("w/o local encoder", {"local_layers": 0}),
    "abl_no_global": ("w/o global encoder", {"global_layers": 0, "fusion": "local_only"}),
    "abl_no_fusion": ("w/o semantic fusion (G only)", {"fusion": "global_only"}),
    "abl_mean_fusion": ("mean fusion", {"fusion": "mean"}),
    "abl_fixed_alpha": ("fixed residual ($\\alpha{=}1$)", {"residual": "fixed"}),
    "abl_no_window": ("full-attention local encoder", {"local_window": 0}),
    "abl_no_pe": ("w/o positional encoding", {"positional_encoding": False}),
    "abl_parallel": ("parallel topology", {"topology": "parallel"}),
}
for k, (lab, ov) in _ABL.items():
    _add(SystemSpec(k, lab, "seq2seq", "ablation", trains=True, adapter="lwnewsum", lw_overrides=ov))


def backbone_name(cfg: dict, dataset: str, spec: SystemSpec) -> str:
    b = cfg["backbones"][spec.backbone_key]
    return b[dataset] if isinstance(b, dict) else b


def input_prefix(cfg: dict, dataset: str, system: str) -> str:
    return "summarize: " if SYSTEMS[system].backbone_key in ("t5", "flant5") else ""


def build_model(cfg: dict, dataset: str, system: str, seed: Optional[int]):
    """Return (model, tokenizer, model_name) ready for training or generation (seq2seq systems)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from .modeling import EncoderAdaptedSeq2Seq, LocalGlobalModule, MLPAdapter
    from .utils import run_dir

    spec = SYSTEMS[system]
    assert spec.kind == "seq2seq", system
    cache = cfg["paths"].get("cache_dir")
    name = backbone_name(cfg, dataset, spec)
    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache)
    if spec.depends_on:
        dep_dir = run_dir(cfg, dataset, spec.depends_on, seed)
        if (dep_dir / "hf_model").exists():
            backbone = AutoModelForSeq2SeqLM.from_pretrained(dep_dir / "hf_model")
        elif (dep_dir / "trainable_state.pt").exists():
            # full fine-tuning runs store every trained weight in trainable_state.pt
            import torch

            backbone = AutoModelForSeq2SeqLM.from_pretrained(name, cache_dir=cache)
            _, unexpected = backbone.load_state_dict(torch.load(dep_dir / "trainable_state.pt", map_location="cpu"),
                                                     strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected keys restoring {dep_dir}: {unexpected[:5]}")
        else:
            raise FileNotFoundError(f"{system} needs the {spec.depends_on} checkpoint in {dep_dir}; run it first.")
    else:
        backbone = AutoModelForSeq2SeqLM.from_pretrained(name, cache_dir=cache)
    d_model = backbone.config.d_model

    if spec.adapter is None:
        if not spec.trains:
            for p in backbone.parameters():
                p.requires_grad = False
        return backbone, tok, name
    if spec.adapter in ("lora", "dora", "ia3"):
        from peft import IA3Config, LoraConfig, TaskType, get_peft_model

        if spec.adapter == "ia3":
            ic = cfg["ia3"]
            pcfg = IA3Config(task_type=TaskType.SEQ_2_SEQ_LM, target_modules=list(ic["target_modules"]),
                             feedforward_modules=list(ic["feedforward_modules"]))
        else:
            lc = cfg["lora"]
            pcfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=lc["r"], lora_alpha=lc["alpha"],
                              lora_dropout=lc["dropout"], target_modules=list(lc["target_modules"]),
                              use_dora=spec.adapter == "dora")
        return get_peft_model(backbone, pcfg), tok, name
    if spec.adapter == "mlp":
        # width chosen so the adapter matches the LW-NEWSUM module size for this d_model,
        # rounded to a multiple of 64 (gives 1,920 for T5-Small and 1,344 for BART-base)
        mcfg = dict(cfg["mlp_adapter"])
        if mcfg.pop("match_lwnewsum", True):
            target = sum(p.numel() for p in LocalGlobalModule(d_model, **cfg["lwnewsum"]).parameters())
            mcfg["bottleneck_dim"] = max(64, int(round((target - 3 * d_model) / (2 * d_model + 1) / 64)) * 64)
        mod = MLPAdapter(d_model, **mcfg)
        return EncoderAdaptedSeq2Seq(backbone, mod, freeze_backbone=True), tok, name
    lw_cfg = deep_update(cfg["lwnewsum"], spec.lw_overrides)
    mod = LocalGlobalModule(d_model, **lw_cfg)
    return EncoderAdaptedSeq2Seq(backbone, mod, freeze_backbone=spec.freeze_backbone), tok, name


def trainable_state_dict(model) -> dict:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def save_trained(model, system: str, out: Path):
    import torch

    out.mkdir(parents=True, exist_ok=True)
    if system in ("t5_ft", "bartbase_ft"):
        model.save_pretrained(out / "hf_model")      # re-used by lwnewsum_on_ft
    torch.save(trainable_state_dict(model), out / "trainable_state.pt")


def load_trained(model, out: Path):
    import torch

    sd = torch.load(out / "trainable_state.pt", map_location="cpu")
    _, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when restoring {out}: {unexpected[:5]}")
    return model
