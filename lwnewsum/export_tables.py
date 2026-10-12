"""Write LaTeX tables and value macros for the Overleaf project from outputs/summary.json.

If results do not exist yet, every cell is written as "--" so the paper still compiles.
In the manuscript, individual values are available as \\res{dataset}{system}{metric}
(e.g. \\res{cnndm}{lwnewsum}{rougeLsum}) via tables/results_macros.tex.
"""
from __future__ import annotations

import re
from pathlib import Path

from .systems import SYSTEMS
from .utils import read_json

DATASETS = [("cnndm", "CNN/DailyMail"), ("xsum", "XSum"), ("xlsum", "XL-Sum (En)"), ("multinews", "Multi-News"),
            ("indian", "Indian News")]
MAIN_ROWS = ["lead3", "textrank",
             "t5_zeroshot", "distilbart", "qwen25_05b", "qwen25_15b", "smollm2_17b", "llama32_1b",
             "t5_ft", "flant5_ft", "bartbase_ft", "t5_lora", "t5_dora", "t5_ia3", "t5_mlp_adapter",
             "mlp_on_ft", "mlp_on_ft_bart",
             "lwnewsum", "lwnewsum_on_ft", "lwnewsum_on_ft_bart", "lwnewsum_joint", "lwnewsum_flant5",
             "lwnewsum_bartbase"]
ABL_ROWS = ["lwnewsum", "abl_no_local", "abl_no_global", "abl_no_fusion", "abl_mean_fusion", "abl_fixed_alpha",
            "abl_no_window", "abl_no_pe", "abl_parallel"]
GROUP_TITLES = {"extractive": "Extractive references",
                "reference": "Pretrained references used as released (no training in this study)",
                "baseline": "Baselines trained in this study", "proposed": "Proposed"}
NA = "--"


def tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return re.sub(r"[\\&%$#_{}~^]", lambda m: rep[m.group()], s)


def _fmt(ms, digits=2, std=True):
    if ms is None:
        return NA
    m, s = ms
    if std and s:
        return f"{m:.{digits}f}\\,{{\\scriptsize$\\pm${s:.{digits}f}}}"
    return f"{m:.{digits}f}"


def _get(summary, ds, sysname, metric, section="metrics"):
    try:
        return summary[ds]["systems"][sysname][section][metric]
    except (KeyError, TypeError):
        return None


def _best(summary, ds, rows, metric, section="metrics", higher=True):
    vals = [(_get(summary, ds, r, metric, section) or [None])[0] for r in rows]
    vals = [v for v in vals if v is not None]
    return (max(vals) if higher else min(vals)) if vals else None


def _cell(summary, ds, r, m, best, section="metrics", digits=2):
    v = _get(summary, ds, r, m, section)
    txt = _fmt(v, digits)
    if v is not None and best is not None and abs(v[0] - best) < 1e-9:
        txt = f"\\textbf{{{txt}}}"
    return txt


def _params(summary, r):
    for ds, _ in DATASETS:
        p = (summary.get(ds, {}).get("systems", {}).get(r, {}) or {}).get("params")
        if p:
            return f"{p['trainable'] / 1e6:.2f}"
    return "0" if SYSTEMS[r].group in ("extractive", "reference") else NA


def table_main(summary, ds, title) -> str:
    cols = ["rouge1", "rouge2", "rougeLsum", "bleu", "meteor", "bertscore_f1"]
    best = {c: _best(summary, ds, MAIN_ROWS, c) for c in cols}
    lines = [r"\begin{table*}[!htbp]", r"\centering\small",
             rf"\caption{{Test-set results on {title} (mean $\pm$ standard deviation over three seeds for trained "
             r"systems; deterministic systems have no deviation). R-L is ROUGE-Lsum; BERTScore is rescaled F1. "
             r"Trainable parameters are in millions. Best value per column in bold.}",
             rf"\label{{tab:main-{ds}}}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccccc}", r"\toprule",
             r"System & R-1 & R-2 & R-L & BLEU & METEOR & BERTScore & Train. params (M) \\", r"\midrule"]
    group = None
    have_any = bool(summary.get(ds, {}).get("systems"))
    for r in MAIN_ROWS:
        if have_any and r not in summary[ds]["systems"]:
            continue
        g = SYSTEMS[r].group
        if g != group:
            if group is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{GROUP_TITLES[g]}}}}} \\")
            group = g
        cells = [_cell(summary, ds, r, c, best[c]) for c in cols]
        lines.append(f"{SYSTEMS[r].label} & " + " & ".join(cells) + f" & {_params(summary, r)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def table_datasets(summary) -> str:
    lines = [r"\begin{table*}[!htbp]", r"\centering\small",
             r"\caption{Dataset statistics after preprocessing. Words are counted with a regular-expression "
             r"tokeniser; novel bigrams are the percentage of reference bigrams absent from the source.}",
             r"\label{tab:datasets}", r"\begin{tabular}{l" + "r" * len(DATASETS) + "}", r"\toprule",
             "Statistic & " + " & ".join(t for _, t in DATASETS) + r" \\", r"\midrule"]
    rows = [("Train", "n_train", 0), ("Validation", "n_validation", 0), ("Test", "n_test", 0),
            ("Avg. source words", "avg_source_words", 1), ("Avg. summary words", "avg_target_words", 1),
            ("Avg. summary sentences", "avg_target_sents", 1), ("Compression ratio", "compression_ratio", 1),
            ("Novel summary bigrams (\\%)", "novel_bigram_pct", 1),
            ("Sources $>$512 tokens (\\%, est.)", "pct_source_over_512_t5_tokens_est", 1)]
    for lab, key, d in rows:
        vals = []
        for ds, _ in DATASETS:
            v = summary.get(ds, {}).get("stats", {}).get(key)
            vals.append(NA if v is None else (f"{v:,.0f}" if d == 0 else f"{v:.{d}f}"))
        lines.append(f"{lab} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def table_factuality(summary) -> str:
    rows = [r for r in MAIN_ROWS]
    DATASETS = [d for d in globals()["DATASETS"] if summary.get(d[0], {}).get("systems")] or globals()["DATASETS"]
    lines = [r"\begin{table*}[!htbp]", r"\centering\small",
             r"\caption{NLI-based factual-consistency estimates on a fixed random subset of test documents. FCS: "
             r"percentage of summary sentences entailed (probability $\geq 0.5$) by at least one source chunk; "
             r"HR $=100-$FCS; SummaC-ZS: mean over summary sentences of the maximum entailment minus "
             r"contradiction probability over source sentences ($\times 100$). Extractive outputs are copied from "
             r"the source and therefore bound these scores from above.}",
             r"\label{tab:factuality}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "cc" * len(DATASETS) + "}", r"\toprule",
             "& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{t}}}" for _, t in DATASETS) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DATASETS))),
             "System & " + " & ".join(["FCS$\\uparrow$ & SummaC$\\uparrow$"] * len(DATASETS)) + r" \\",
             r"\midrule"]
    for r in rows:
        if not any(_get(summary, ds, r, "fcs", "factuality") for ds, _ in DATASETS) and summary:
            continue
        cells = []
        for ds, _ in DATASETS:
            for m in ("fcs", "summac_zs"):
                cells.append(_fmt(_get(summary, ds, r, m, "factuality"), 1, std=False))
        lines.append(f"{SYSTEMS[r].label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def table_efficiency(summary) -> str:
    eff = summary.get("cnndm", {}).get("efficiency", {})
    rows = ["t5_ft", "flant5_ft", "bartbase_ft", "t5_lora", "t5_dora", "t5_ia3", "t5_mlp_adapter", "lwnewsum",
            "mlp_on_ft", "lwnewsum_on_ft", "lwnewsum_flant5", "lwnewsum_bartbase", "mlp_on_ft_bart",
            "lwnewsum_on_ft_bart", "distilbart", "qwen25_05b", "qwen25_15b", "smollm2_17b"]
    rows = [r for r in rows if eff.get(r)]  # systems that were not profiled are omitted
    lines = [r"\begin{table*}[!htbp]", r"\centering\small",
             r"\caption{Computational characteristics on CNN/DailyMail (single NVIDIA H200, bfloat16, beam size 4). "
             r"GFLOPs: one teacher-forced forward pass (512 source and 128 target tokens). Latency is per document; "
             r"peak memory is measured during batch-32 generation. Plug-in variants have the same architecture as the "
             r"corresponding frozen-backbone variants and differ only in weights. GFLOPs are not reported for the decoder-only "
             r"models, whose prompt length varies with the instruction template. Latencies come from one run on a shared GPU; "
             r"repeated runs of the same system varied by up to about 30\%, so smaller latency differences are not meaningful. "
             r"The batch-32 latency of BART-base (full FT) was higher than that of the same weights loaded for the plug-ins in "
             r"every repetition, with identical precision and number of decoding steps.}",
             r"\label{tab:efficiency}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"System & Total (M) & Trainable (M) & FP32 (MB) & GFLOPs & ms/doc (bs=1) & ms/doc (bs=32) & Peak mem. (MB) \\",
             r"\midrule"]
    for r in rows:
        e = eff.get(r)
        if not e:
            cells = [NA] * 7
        else:
            g = lambda k, d: NA if e.get(k) is None or e.get(k) != e.get(k) else f"{e[k]:.{d}f}"  # noqa: E731
            cells = [g("params_total_m", 1), g("params_trainable_m", 2), g("fp32_size_mb", 0), g("gflops_forward", 1),
                     g("ms_per_doc_bs1", 1), g("ms_per_doc_bs32", 1), g("peak_mem_mb_bs32", 0)]
        lines.append(f"{SYSTEMS[r].label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def table_ablation(summary) -> str:
    dss = [("cnndm", "CNN/DailyMail"), ("xsum", "XSum")]
    cols = ["rouge1", "rouge2", "rougeLsum", "bertscore_f1"]
    lines = [r"\begin{table*}[!htbp]", r"\centering\small",
             r"\caption{Ablation study. Every variant is re-trained from scratch with the same data, budget and "
             r"three seeds; values are means with the change relative to the full model in parentheses.}",
             r"\label{tab:ablation}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "cccc" * len(dss) + "}", r"\toprule",
             "& " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{t}}}" for _, t in dss) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(dss))),
             "Variant & " + " & ".join(["R-1 & R-2 & R-L & BERTScore"] * len(dss)) + r" \\", r"\midrule"]
    for r in ABL_ROWS:
        cells = []
        for ds, _ in dss:
            full = {c: (_get(summary, ds, "lwnewsum", c) or [None])[0] for c in cols}
            for c in cols:
                v = _get(summary, ds, r, c)
                if v is None:
                    cells.append(NA)
                elif r == "lwnewsum" or full[c] is None:
                    cells.append(f"{v[0]:.2f}")
                else:
                    cells.append(f"{v[0]:.2f} ({v[0] - full[c]:+.2f})")
        label = "Full LW-NEWSUM" if r == "lwnewsum" else SYSTEMS[r].label
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


SIG_COMPACT_ROWS = ["t5_ft", "t5_lora", "t5_dora", "t5_ia3", "t5_mlp_adapter", "bartbase_ft", "distilbart"]


def _stars(r):
    return "" if r is None else ("$^{**}$" if r["p_holm"] < 0.01 else "$^{*}$" if r["p_holm"] < 0.05 else "")


def table_significance(summary) -> str:
    """Compact main-text table: frozen-backbone LW-NEWSUM against the trained comparison systems, all corpora."""
    dss = [d for d in DATASETS if summary.get(d[0], {}).get("significance", {}).get("lwnewsum")] or DATASETS[:2]
    lines = [r"\begin{table*}[!htbp]", r"\centering\small",
             r"\caption{Paired bootstrap comparison of frozen-backbone LW-NEWSUM (T5-Small) with the trained comparison "
             r"systems (seed-averaged per-document scores, 10{,}000 resamples). Entries are LW-NEWSUM minus the comparison "
             r"system; $^{*}$: $p<0.05$, $^{**}$: $p<0.01$, Holm-Bonferroni adjusted within each dataset and metric. "
             r"Intervals and all remaining comparisons are given in Supplementary File~S1; plug-in comparisons "
             r"are marked in \cref{tab:plugin,tab:plugin-bart}.}",
             r"\label{tab:significance}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * (2 * len(dss)) + "}", r"\toprule",
             r"& \multicolumn{" + str(len(dss)) + r"}{c}{$\Delta$ ROUGE-Lsum} & \multicolumn{" + str(len(dss))
             + r"}{c}{$\Delta$ BERTScore} \\",
             rf"\cmidrule(lr){{2-{1 + len(dss)}}}\cmidrule(lr){{{2 + len(dss)}-{1 + 2 * len(dss)}}}",
             "Comparison & " + " & ".join([t for _, t in dss] * 2) + r" \\", r"\midrule"]
    for other in SIG_COMPACT_ROWS:
        cells = []
        for metric in ("rougeLsum", "bertscore_f1"):
            for ds, _ in dss:
                rows = {r["vs"]: r for r in summary.get(ds, {}).get("significance", {}).get("lwnewsum", {}).get(metric, [])}
                r = rows.get(other)
                cells.append(NA if r is None else f"{r['diff']:+.2f}{_stars(r)}")
        lines.append(f"vs.\\ {SYSTEMS[other].label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def significance_csv(summary) -> str:
    """Supplementary file: every LW-NEWSUM variant against every comparison system, all metrics, with intervals."""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["dataset", "variant", "comparison", "metric", "diff", "ci_low", "ci_high", "p", "p_holm", "n"])
    for ds, _ in DATASETS:
        for v, per_metric in summary.get(ds, {}).get("significance", {}).items():
            for metric, rows in per_metric.items():
                for r in rows:
                    w.writerow([ds, v, r["vs"], metric, f"{r['diff']:.4f}", f"{r['ci_low']:.4f}", f"{r['ci_high']:.4f}",
                                f"{r['p']:.5f}", f"{r['p_holm']:.5f}", r["n"]])
        for metric, rows in summary.get(ds, {}).get("plugin_tests", {}).items():
            for r in rows:
                w.writerow([ds, r["a"], r["b"], metric, f"{r['diff']:.4f}", f"{r['ci_low']:.4f}", f"{r['ci_high']:.4f}",
                            f"{r['p']:.5f}", f"{r['p_holm']:.5f}", r["n"]])
    return buf.getvalue()


def table_examples(summary) -> str:
    ex = summary.get("cnndm", {}).get("examples", [])
    order = [("reference", "Reference"), ("lead3", "Lead-3"), ("t5_ft", "T5-Small (full FT)"),
             ("distilbart", "DistilBART-12-6"), ("lwnewsum", "LW-NEWSUM")]
    lines = [r"\begin{table*}[!htbp]", r"\centering\footnotesize",
             r"\caption{Randomly selected (seeded) CNN/DailyMail test examples. Outputs are reproduced verbatim from "
             r"the systems; no example was edited or hand-picked.}", r"\label{tab:examples}",
             r"\begin{tabular}{p{0.17\linewidth}p{0.78\linewidth}}", r"\toprule"]
    if not ex:
        lines.append(r"\multicolumn{2}{c}{\textit{Examples are inserted automatically after the experiments}} \\")
    for i, e in enumerate(ex):
        lines.append(rf"\multicolumn{{2}}{{l}}{{\textbf{{Example {i + 1}}} (id: \texttt{{{tex_escape(e['uid'][:16])}}})}} \\")
        for k, lab in order:
            if k in e:
                lines.append(f"{lab} & {tex_escape(e[k])} \\\\")
        if i < len(ex) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def table_plugin(summary, backbone="t5") -> str:
    """Plug-in mode for one backbone: rows dataset x {fine-tuned, + MLP, + LW-NEWSUM}; columns quality and NLI scores."""
    base, mlp, lw, bname, label, sizes = {
        "t5": ("t5_ft", "mlp_on_ft", "lwnewsum_on_ft", "T5-Small", "tab:plugin", "1.97M (MLP) and 1.98M (LW-NEWSUM)"),
        "bart": ("bartbase_ft", "mlp_on_ft_bart", "lwnewsum_on_ft_bart", "BART-base", "tab:plugin-bart",
                 "2.07M (MLP) and 2.11M (LW-NEWSUM)")}[backbone]
    cols = [("rougeLsum", "metrics", "R-L"), ("bertscore_f1", "metrics", "BERTScore"),
            ("fcs", "factuality", "FCS"), ("summac_zs", "factuality", "SummaC-ZS")]
    dss = [d for d in DATASETS if summary.get(d[0], {}).get("systems")] or DATASETS[:2]
    lines = [r"\begin{table*}[!htbp]", r"\centering\small",
             rf"\caption{{Plug-in mode on a fully fine-tuned {bname}. The fine-tuned backbone is frozen and either a "
             rf"parameter-matched MLP adapter without attention or the LW-NEWSUM module is trained on top of it "
             rf"({sizes} trainable parameters). Values are means over three seeds; changes relative to the fine-tuned "
             r"backbone are in parentheses. $^{*}$: $p<0.05$ and $^{**}$: $p<0.01$ against the fine-tuned backbone "
             r"(paired bootstrap, Holm-corrected). $^{\dagger}$: LW-NEWSUM is significantly better than the MLP plug-in; "
             r"$^{\ddagger}$: significantly worse ($p<0.05$).}",
             rf"\label{{{label}}}", r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
             r"Dataset & Variant & " + " & ".join(c[2] for c in cols) + r" \\", r"\midrule"]

    def test(ds, metric, a, b):
        for r in summary.get(ds, {}).get("plugin_tests", {}).get(metric, []):
            if r["a"] == a and r["b"] == b:
                return r
        return None

    for di, (ds, title) in enumerate(dss):
        for vi, (sysname, vlabel) in enumerate([(base, "fine-tuned"), (mlp, "+ MLP adapter"), (lw, "+ LW-NEWSUM")]):
            cells = []
            for metric, sec, _lab in cols:
                v = _get(summary, ds, sysname, metric, sec)
                b = _get(summary, ds, base, metric, sec)
                if v is None:
                    cells.append(NA)
                    continue
                txt = f"{v[0]:.2f}"
                if sysname != base and b is not None:
                    t = test(ds, metric, sysname, base)
                    star = "" if t is None else ("$^{**}$" if t["p_holm"] < 0.01 else "$^{*}$" if t["p_holm"] < 0.05 else "")
                    dag = ""
                    if sysname == lw:
                        t2 = test(ds, metric, lw, mlp)
                        if t2 and t2["p_holm"] < 0.05:
                            dag = r"$^{\dagger}$" if t2["diff"] > 0 else r"$^{\ddagger}$"
                    txt += f" ({v[0] - b[0]:+.2f}){star}{dag}"
                cells.append(txt)
            lines.append(f"{title if vi == 0 else ''} & {vlabel} & " + " & ".join(cells) + r" \\")
        if di < len(dss) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def macros(summary) -> str:
    out = [r"% Auto-generated. Usage: \res{cnndm}{lwnewsum}{rougeLsum}",
           r"\providecommand{\res}[3]{\ifcsname res@#1@#2@#3\endcsname\csname res@#1@#2@#3\endcsname\else--\fi}"]
    for ds, _ in DATASETS:
        for name, entry in summary.get(ds, {}).get("systems", {}).items():
            for sec in ("metrics", "factuality"):
                for m, (mean, _) in entry.get(sec, {}).items():
                    out.append(rf"\expandafter\def\csname res@{ds}@{name}@{m}\endcsname{{{mean:.2f}}}")
            if "params" in entry:
                out.append(rf"\expandafter\def\csname res@{ds}@{name}@trainableM\endcsname"
                           rf"{{{entry['params']['trainable'] / 1e6:.2f}}}")
        cl = summary.get(ds, {}).get("cleaning_log")
        if cl:
            for k, v in cl.items():
                out.append(rf"\expandafter\def\csname res@{ds}@clean@{k.replace('_', '')}\endcsname{{{v}}}")
    return "\n".join(out)


def export(cfg: dict, paper_dir: str) -> None:
    path = Path(cfg["paths"]["output_dir"]) / "summary.json"
    summary = read_json(path) if path.exists() else {}
    tdir = Path(paper_dir) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    files = {"tab_datasets.tex": table_datasets(summary), "tab_factuality.tex": table_factuality(summary),
             "tab_efficiency.tex": table_efficiency(summary), "tab_ablation.tex": table_ablation(summary),
             "tab_significance.tex": table_significance(summary),
             "tab_examples.tex": table_examples(summary),
             "tab_plugin.tex": table_plugin(summary, "t5"), "tab_plugin_bart.tex": table_plugin(summary, "bart"),
             "results_macros.tex": macros(summary)}
    for ds, title in DATASETS:
        has = bool(summary.get(ds, {}).get("systems"))
        files[f"tab_main_{ds}.tex"] = table_main(summary, ds, title) if has or not summary else \
            f"% No results for {title} yet; this table is filled automatically after its runs finish."
    for f, txt in files.items():
        (tdir / f).write_text(txt + "\n", encoding="utf-8")
    sup = Path(paper_dir) / "supplementary"
    sup.mkdir(parents=True, exist_ok=True)
    (sup / "S1_significance_tests.csv").write_text(significance_csv(summary), encoding="utf-8")
    _learning_curves(cfg, Path(paper_dir) / "figures")
    print(f"[export] wrote {len(files)} files to {tdir} and S1_significance_tests.csv to {sup}")


def _learning_curves(cfg, fig_dir: Path):
    """Validation-loss curves (first seed) for the trained T5-Small systems on CNN/DailyMail."""
    import json

    root = Path(cfg["paths"]["output_dir"]) / "cnndm"
    series = {}
    for s in ("t5_ft", "t5_lora", "t5_mlp_adapter", "lwnewsum"):
        runs = sorted((root / s).glob("seed*/train_log.jsonl")) if (root / s).exists() else []
        if runs:
            series[SYSTEMS[s].label] = [json.loads(l) for l in open(runs[0])]
    if not series:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for lab, rec in series.items():
        ax.plot([r["step"] for r in rec], [r["val_loss"] for r in rec], marker="o", ms=3, label=lab)
    ax.set_xlabel("Optimiser step")
    ax.set_ylabel("Validation loss")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves.pdf")
    plt.close(fig)
