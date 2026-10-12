"""Reference-based metrics with per-example scores (needed for paired bootstrap).

ROUGE-1/2/Lsum: rouge-score with Porter stemming; ROUGE-Lsum uses newline-separated
sentences, the standard convention for multi-sentence CNN/DailyMail summaries.
BLEU: sacreBLEU corpus score (plus sentence-level scores for testing).
METEOR: NLTK implementation.  BERTScore: roberta-large, rescaled with baseline.
"""
from __future__ import annotations

import re

import numpy as np

from .utils import split_sentences


def _nl(text: str) -> str:
    return "\n".join(split_sentences(text)) or text


def rouge_scores(preds: list[str], refs: list[str]) -> dict[str, list[float]]:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    out = {"rouge1": [], "rouge2": [], "rougeLsum": []}
    for p, r in zip(preds, refs):
        s = sc.score(_nl(r), _nl(p))
        for k in out:
            out[k].append(100 * s[k].fmeasure)
    return out


def bleu_scores(preds, refs) -> tuple[float, list[float]]:
    import sacrebleu

    corpus = sacrebleu.corpus_bleu(preds, [refs]).score
    sent = [sacrebleu.sentence_bleu(p, [r]).score for p, r in zip(preds, refs)]
    return corpus, sent


def meteor_scores(preds, refs) -> list[float] | None:
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score

        for pkg in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
        return [100 * meteor_score([r.split()], p.split()) for p, r in zip(preds, refs)]
    except Exception as exc:  # pragma: no cover
        print(f"[metrics] METEOR unavailable: {exc}")
        return None


def bertscore_scores(preds, refs, model_type="roberta-large", rescale=True, device=None, batch_size=32):
    from bert_score import score

    p, r, f = score(preds, refs, model_type=model_type, lang="en", rescale_with_baseline=rescale,
                    device=device, batch_size=batch_size, verbose=False)
    return {"bertscore_p": (100 * p).tolist(), "bertscore_r": (100 * r).tolist(), "bertscore_f1": (100 * f).tolist()}


def _ngrams(tokens, n):
    return set(zip(*[tokens[i:] for i in range(n)]))


def abstractiveness(preds, sources) -> dict[str, list[float]]:
    """Summary length and % novel n-grams w.r.t. the source (copying behaviour)."""
    tok = lambda s: re.findall(r"\w+", s.lower())  # noqa: E731
    length, nov1, nov2 = [], [], []
    for p, s in zip(preds, sources):
        tp, ts = tok(p), tok(s)
        length.append(len(tp))
        for n, store in ((1, nov1), (2, nov2)):
            g = _ngrams(tp, n)
            store.append(100 * (1 - len(g & _ngrams(ts, n)) / len(g)) if g else 0.0)
    return {"summary_words": length, "novel_unigram_pct": nov1, "novel_bigram_pct": nov2}


def compute_all(preds, refs, sources, cfg, device=None, with_bertscore=True) -> tuple[dict, dict]:
    """Return (corpus_level, per_example) dictionaries."""
    per = {}
    per.update(rouge_scores(preds, refs))
    corpus_bleu, per["bleu_sent"] = bleu_scores(preds, refs)
    met = meteor_scores(preds, refs)
    if met is not None:
        per["meteor"] = met
    if with_bertscore:
        per.update(bertscore_scores(preds, refs, cfg["eval"]["bertscore_model"], cfg["eval"]["bertscore_rescale"],
                                    device=device))
    per.update(abstractiveness(preds, sources))
    corpus = {k: float(np.mean(v)) for k, v in per.items()}
    corpus["bleu"] = corpus_bleu
    corpus["n"] = len(preds)
    return corpus, per
