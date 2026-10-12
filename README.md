# LW-NEWSUM

Code and configurations for

> **Post-hoc Encoder Refinement of Fine-Tuned News Summarizers: A Controlled Study of the Lightweight Local–Global Module LW-NEWSUM**
> Himanshu Kumar, V. K. Jain and Vivek Kumar. Manuscript under review, 2026.

LW-NEWSUM is a small module (1.98M parameters on T5-Small) that sits between the encoder and the decoder of a frozen encoder–decoder summarizer. It projects the encoder states into a 256-dimensional bottleneck, models neighbourhood context with windowed self-attention and document-level context with full self-attention, fuses both views and adds the result back through a learnable residual scale. It can be trained on a frozen pretrained model or as a plug-in on top of an already fine-tuned model, and removed again without changing that model.

## Main findings

All results are means over three seeds on the official test splits; significance uses a paired bootstrap (10,000 resamples) with Holm correction.

| ROUGE-Lsum | CNN/DailyMail | XSum | XL-Sum (En) | Multi-News |
|---|---|---|---|---|
| T5-Small, fully fine-tuned | 38.73 | 27.21 | 27.34 | 39.99 |
| + MLP adapter plug-in (parameter-matched) | 39.09 | 27.61 | 27.56 | 40.63 |
| + LW-NEWSUM plug-in | 38.99 | 27.35 | 27.43 | 40.51 |
| BART-base, fully fine-tuned | 39.30 | 32.72 | 31.78 | 41.73 |
| + MLP adapter plug-in | 39.45 | 32.44 | 31.59 | 42.34 |
| + LW-NEWSUM plug-in | 39.56 | 32.49 | 31.70 | 42.25 |

- A small residual module trained on top of a fine-tuned summarizer gives small but significant gains in ROUGE and BERTScore on most corpora.
- On T5-Small, a parameter-matched feed-forward adapter without attention achieves the same or slightly larger gains, so the improvement comes from post-hoc refinement rather than from the local–global attention design. LW-NEWSUM is better than the adapter only on BART-base for CNN/DailyMail and XL-Sum.
- All plug-ins make outputs more abstractive and lower NLI-based factual-consistency estimates on the abstractive corpora.

The complete results, including the extractive baselines, LoRA, DoRA, (IA)³, DistilBART, zero-shot small language models, factual consistency, ablations and computational cost, are reported in the article.

## Repository layout

```
configs/default.yaml       every hyper-parameter reported in the paper
lwnewsum/                  Python package
  data.py                  official splits (CNN/DailyMail, XSum, XL-Sum, Multi-News) and Indian News cleaning
  modeling.py              LW-NEWSUM module, MLP adapter, encoder-adapted wrapper
  systems.py               registry of all compared systems and the re-trained ablations
  train.py                 training loop (AdamW, warm-up, bfloat16, early stopping) and generation
  slm.py                   zero-shot prompting of small decoder-only language models
  metrics.py               ROUGE-1/2/Lsum, sacreBLEU, METEOR, BERTScore, abstractiveness
  factuality.py            chunked NLI (FCS, HR) without truncation, and SummaC-ZS
  stats.py                 paired bootstrap and Holm-Bonferroni correction
  efficiency.py            parameters, FLOPs, latency (batch sizes 1 and 32), peak memory
  aggregate.py             seeds -> mean and standard deviation, significance -> outputs/summary.json
  export_tables.py         writes every result table as LaTeX (default folder: paper/tables/)
  human_eval.py            blind annotation sheet, Krippendorff's alpha, Wilcoxon tests
scripts/launch_dgx.py      runs the whole grid across all GPUs; restartable, with a progress log
scripts/smoke_test.sh      offline CPU test of every code path with tiny random models
scripts/package_outputs.py bundles the system outputs (no checkpoints) for release
data/raw/                  Indian News corpus (raw), cleaned at load time
```

## Installation

Python 3.10 or later and a CUDA GPU are required for the full experiments; the smoke test runs on CPU.

```bash
git clone https://github.com/HPC-Lab-Works/LW-NEWSUM.git
cd LW-NEWSUM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4')]"
```

Check the environment with the offline smoke test (resumable; progress in `/tmp/lwnewsum_smoke/smoke_progress.log`):

```bash
bash scripts/smoke_test.sh
```

## Reproducing the experiments

```bash
# dataset statistics
for d in cnndm xsum xlsum multinews indian; do python -m lwnewsum.run stats --dataset $d; done

# full grid (dataset x system x seed): train, predict, score, factuality
# one job per GPU; finished jobs are skipped, so it can be restarted at any time
mkdir -p outputs && nohup python scripts/launch_dgx.py > outputs/launch.log 2>&1 &
tail -f outputs/progress.log

# computational cost, on one otherwise idle GPU
CUDA_VISIBLE_DEVICES=0 python -m lwnewsum.run efficiency --dataset cnndm

# aggregation, significance tests and all result tables (LaTeX) plus the full significance CSV
python -m lwnewsum.run aggregate
python -m lwnewsum.run export --paper_dir paper

# human evaluation
python -m lwnewsum.run human_eval --dataset cnndm --n_docs 50
python -m lwnewsum.run human_eval_analyze --dataset cnndm
```

The complete grid takes roughly 110 to 140 GPU-hours on NVIDIA H200 GPUs. A single system can be run with, for example:

```bash
python -m lwnewsum.run run --dataset xsum --system lwnewsum_on_ft --seed 13
```

Plug-in systems (`lwnewsum_on_ft`, `mlp_on_ft`, `lwnewsum_on_ft_bart`, `mlp_on_ft_bart`) require the corresponding fully fine-tuned run (`t5_ft` or `bartbase_ft`) with the same seed.

## System outputs

Predictions, per-document metric and factuality scores, training logs and the aggregated `summary.json` for every run are distributed as a release archive (see the Releases page of this repository). The archive is created with `python scripts/package_outputs.py` and contains no model checkpoints.

## Data and licences

- The code is released under the MIT licence (`LICENSE`).
- CNN/DailyMail, XSum, XL-Sum and Multi-News are downloaded from the Hugging Face Hub and remain under their own licences; Multi-News is for non-commercial research use only.
- The Indian News articles remain the property of their publishers and are included for research reproducibility only.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`). The reference will be updated once the article is published.
