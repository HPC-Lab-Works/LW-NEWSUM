#!/usr/bin/env bash
# Offline end-to-end smoke test with tiny random models (numbers are meaningless).
#
# RESUMABLE: steps that already finished are skipped instantly, so after a fix you
# only re-run what failed. Progress is written to $OUT/smoke_progress.log.
#
#   bash scripts/smoke_test.sh                 # resume (default)
#   SMOKE_FRESH=1 bash scripts/smoke_test.sh   # wipe previous smoke outputs and start over
#   tail -f /tmp/lwnewsum_smoke/smoke_progress.log   # watch progress from another shell
set -uo pipefail
TINY=${1:-/tmp/lwnewsum_tiny}
OUT=${SMOKE_OUT:-/tmp/lwnewsum_smoke}
[ "${SMOKE_FRESH:-0}" = "1" ] && rm -rf "$OUT"
mkdir -p "$OUT/.done"
LOG="$OUT/smoke_progress.log"
[ -d "$TINY/t5" ] || python3 scripts/make_tiny_models.py "$TINY"

COMMON=(--dataset indian
  --set paths.output_dir=$OUT --set backbones.t5=$TINY/t5 --set backbones.flant5=$TINY/t5
  --set backbones.bartbase=$TINY/bart --set backbones.distilbart=$TINY/bart --set backbones.qwen25_05b=$TINY/llm
  --set seeds=[13,42]
  --set datasets.indian.max_train_samples=40 --set datasets.indian.max_eval_samples=12
  --set datasets.indian.gen.max_new_tokens=16 --set datasets.indian.gen.min_new_tokens=2 --set datasets.indian.gen.num_beams=2
  --set train.epochs=1 --set train.per_device_batch_size=4 --set train.grad_accum=2 --set train.bf16=false
  --set train.eval_every_steps=2 --set train.num_workers=0
  --set lwnewsum.bottleneck_dim=32 --set lwnewsum.ffn_dim=64 --set lwnewsum.local_window=8 --set mlp_adapter.bottleneck_dim=32
  --set lora.r=4 --set eval.gen_batch_size=4 --set eval.factuality_subset=6 --set eval.nli_model=$TINY/nli
  --set eval.bootstrap_samples=500 --set eval.efficiency_docs=8 --no_bertscore)

REFS="lead3 textrank t5_zeroshot distilbart qwen25_05b"
TRAINED="t5_ft flant5_ft bartbase_ft t5_lora t5_dora t5_ia3 t5_mlp_adapter lwnewsum lwnewsum_on_ft lwnewsum_joint
         lwnewsum_flant5 lwnewsum_bartbase mlp_on_ft lwnewsum_on_ft_bart mlp_on_ft_bart abl_no_local abl_no_global abl_no_fusion abl_mean_fusion abl_fixed_alpha
         abl_no_window abl_no_pe abl_parallel"
SEEDS=${SMOKE_SEEDS:-13 42}
N_TRAINED=$(echo $TRAINED | wc -w); N_SEEDS=$(echo $SEEDS | wc -w)
TOTAL=$(( 1 + 2 * $(echo $REFS | wc -w) + 2 * N_TRAINED * N_SEEDS + 4 ))
I=0
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(ts) $*" | tee -a "$LOG"; }

# step <name> <done-file> <command...>: skip if done-file exists, else run and log timing.
step() {
  local name=$1 done_file=$2; shift 2
  I=$((I + 1))
  if [ -e "$done_file" ]; then log "[$I/$TOTAL] SKIP  $name (already done)"; return 0; fi
  log "[$I/$TOTAL] START $name"
  local t0=$(date +%s)
  if "$@" >> "$OUT/smoke_detail.log" 2>&1; then
    log "[$I/$TOTAL] DONE  $name ($(( $(date +%s) - t0 ))s)"
  else
    log "[$I/$TOTAL] FAIL  $name -> last lines of $OUT/smoke_detail.log:"
    tail -n 25 "$OUT/smoke_detail.log" | tee -a "$LOG"
    log "Fix the error, then run the same command again: finished steps will be skipped."
    exit 1
  fi
}
rd() { echo "$OUT/indian/$1/$2"; }   # run directory

log "==== smoke test (resume mode; SMOKE_FRESH=1 to restart) ===="
step "dataset stats" "$OUT/indian/dataset_stats.json" python3 -m lwnewsum.run stats "${COMMON[@]}"
for S in $REFS; do
  step "run $S" "$(rd $S deterministic)/metrics.json" python3 -m lwnewsum.run run --system $S "${COMMON[@]}"
  step "factuality $S" "$(rd $S deterministic)/factuality.json" python3 -m lwnewsum.run factuality --system $S "${COMMON[@]}"
done
for SEED in $SEEDS; do
  for S in $TRAINED; do
    step "run $S seed$SEED" "$(rd $S seed$SEED)/metrics.json" \
      python3 -m lwnewsum.run run --system $S --seed $SEED "${COMMON[@]}"
    step "factuality $S seed$SEED" "$(rd $S seed$SEED)/factuality.json" \
      python3 -m lwnewsum.run factuality --system $S --seed $SEED "${COMMON[@]}"
  done
done
step "efficiency" "$OUT/indian/efficiency.json" python3 -m lwnewsum.run efficiency \
  --systems t5_ft,bartbase_ft,t5_lora,t5_dora,t5_ia3,t5_mlp_adapter,lwnewsum,lwnewsum_bartbase,distilbart,qwen25_05b "${COMMON[@]}"
step "aggregate" "$OUT/.done/aggregate" bash -c "python3 -m lwnewsum.run aggregate $(printf '%q ' "${COMMON[@]}") && touch $OUT/.done/aggregate"
step "export tables" "$OUT/.done/export" bash -c "python3 -m lwnewsum.run export --paper_dir $OUT/paper $(printf '%q ' "${COMMON[@]}") && touch $OUT/.done/export"
step "human-eval sheet" "$OUT/indian/human_eval/annotation_sheet.csv" python3 -m lwnewsum.run human_eval --n_docs 5 "${COMMON[@]}"
log "SMOKE TEST PASSED -> $OUT"
