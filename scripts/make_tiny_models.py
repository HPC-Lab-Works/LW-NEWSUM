"""Build tiny random models + a local SentencePiece tokenizer for offline smoke tests.

Only used to check that every code path runs end to end on CPU; the numbers are
meaningless. Real experiments use t5-small, DistilBART and bart-large-mnli from the hub.
"""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from transformers import (LlamaConfig, LlamaForCausalLM, BartConfig, BartForConditionalGeneration, BertConfig, BertForSequenceClassification,
                          PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration)

out = Path(sys.argv[1] if len(sys.argv) > 1 else "tiny_models")
out.mkdir(parents=True, exist_ok=True)
corpus = out / "corpus.txt"
with open(corpus, "w") as f:
    for line in open("data/raw/indian-news-2024-2025-summarization.jsonl"):
        r = json.loads(line)
        f.write(r["input_text"].replace("\n", " ") + "\n" + r["target_text"].replace("\n", " ") + "\n")
raw = Tokenizer(models.Unigram())
raw.pre_tokenizer = pre_tokenizers.Metaspace()
raw.decoder = decoders.Metaspace()
raw.train([str(corpus)], trainers.UnigramTrainer(vocab_size=2000, special_tokens=["<pad>", "</s>", "<unk>"],
                                                 unk_token="<unk>"))
raw.post_processor = processors.TemplateProcessing(single="$A </s>", pair="$A </s> $B </s>",
                                                   special_tokens=[("</s>", 1)])
tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
V = len(tok)

t5 = T5ForConditionalGeneration(T5Config(vocab_size=V, d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
                                         decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
t5.save_pretrained(out / "t5"); tok.save_pretrained(out / "t5")
bart = BartForConditionalGeneration(BartConfig(vocab_size=V, d_model=64, encoder_layers=2, decoder_layers=2,
                                               encoder_attention_heads=4, decoder_attention_heads=4,
                                               encoder_ffn_dim=128, decoder_ffn_dim=128, pad_token_id=0,
                                               eos_token_id=1, bos_token_id=0, decoder_start_token_id=0,
                                               forced_bos_token_id=None, max_position_embeddings=1024))
bart.save_pretrained(out / "bart"); tok.save_pretrained(out / "bart")
nli = BertForSequenceClassification(BertConfig(vocab_size=V, hidden_size=64, num_hidden_layers=2,
                                               num_attention_heads=4, intermediate_size=128, num_labels=3,
                                               id2label={0: "contradiction", 1: "neutral", 2: "entailment"},
                                               label2id={"contradiction": 0, "neutral": 1, "entailment": 2}))
nli.save_pretrained(out / "nli"); tok.save_pretrained(out / "nli")
llm = LlamaForCausalLM(LlamaConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4, pad_token_id=0, eos_token_id=1,
                                   bos_token_id=1))
llm.save_pretrained(out / "llm")
ctok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
ctok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}{% if add_generation_prompt %} Answer: {% endif %}"
ctok.save_pretrained(out / "llm")
print("tiny models written to", out)
