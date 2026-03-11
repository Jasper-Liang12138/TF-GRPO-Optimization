#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/apulis-dev/userdata/TF-GRPO-Optimization/TF-GRPO-Opt-Baseline"
ENV_RUNNER="/home/apulis-dev/userdata/TF-GRPO-Optimization/run_in_env.sh"
MODEL_DIR="/home/apulis-dev/userdata/models/Qwen2.5-Math-7B-Instruct"
DATA_DIR="/home/apulis-dev/userdata/datasets/gsm8k"
OUT_BASE="${OUT_BASE:-/home/apulis-dev/outputs/tf_grpo_baseline_7b}"

SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
GROUP_SIZE="${GROUP_SIZE:-5}"
EPOCHS="${EPOCHS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
DEVICE="${DEVICE:-npu:0}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/gsm8k.json}"
TEST_DATA="${TEST_DATA:-$DATA_DIR/test.json}"
RUN_TAG="${RUN_TAG:-default}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$OUT_BASE/runs/${RUN_TAG}_${RUN_STAMP}}"
EXP_DIR="$RUN_DIR/experience_bank"
RESULT_DIR="$RUN_DIR/results"
LOG_DIR="$RUN_DIR/logs"
REPORT_DIR="$RUN_DIR/reports"

mkdir -p "$EXP_DIR" "$RESULT_DIR" "$LOG_DIR" "$REPORT_DIR"
MASTER_LOG="$LOG_DIR/run_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "[INFO] run dir: $RUN_DIR" | tee -a "$MASTER_LOG"

action() {
  local name="$1"
  shift
  local step_log="$LOG_DIR/${name}_${RUN_TAG}.log"
  echo "[STEP] $name" | tee -a "$MASTER_LOG"
  echo "[CMD] $*" | tee -a "$MASTER_LOG"
  (
    cd "$PROJECT_ROOT"
    "$@"
  ) 2>&1 | tee "$step_log"
  local rc=${PIPESTATUS[0]}
  echo "[STEP-END] $name rc=$rc log=$step_log" | tee -a "$MASTER_LOG"
  return $rc
}

EXP_BANK_PATH="$EXP_DIR/experience_bank_epoch_${EPOCHS}.json"
TFGRPO_RESULT="$RESULT_DIR/gsm8k_tfgrpo_${RUN_TAG}.json"
ZEROSHOT_RESULT="$RESULT_DIR/gsm8k_zeroshot_${RUN_TAG}.json"
REPORT_PATH="$REPORT_DIR/tf_grpo_baseline_${RUN_TAG}.md"

action build_experience \
  bash "$ENV_RUNNER" python -u build_experience.py \
    --model "$MODEL_DIR" \
    --data_path "$TRAIN_DATA" \
    --sample_size "$SAMPLE_SIZE" \
    --group_size "$GROUP_SIZE" \
    --epochs "$EPOCHS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" \
    --output_dir "$EXP_DIR"

action eval_tfgrpo \
  bash "$ENV_RUNNER" python -u math_inference.py \
    --model "$MODEL_DIR" \
    --dataset gsm8k \
    --data_path "$TEST_DATA" \
    --save_path "$TFGRPO_RESULT" \
    --experience_bank_path "$EXP_BANK_PATH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" \
    --IF_TF_GRPO_MODE

action eval_zeroshot \
  bash "$ENV_RUNNER" python -u math_inference.py \
    --model "$MODEL_DIR" \
    --dataset gsm8k \
    --data_path "$TEST_DATA" \
    --save_path "$ZEROSHOT_RESULT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE"

action make_report \
  bash "$ENV_RUNNER" python -u generate_baseline_report.py \
    --report_path "$REPORT_PATH" \
    --run_tag "$RUN_TAG" \
    --model_path "$MODEL_DIR" \
    --train_data "$TRAIN_DATA" \
    --test_data "$TEST_DATA" \
    --sample_size "$SAMPLE_SIZE" \
    --group_size "$GROUP_SIZE" \
    --epochs "$EPOCHS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" \
    --exp_bank_path "$EXP_BANK_PATH" \
    --tfgrpo_result "$TFGRPO_RESULT" \
    --zeroshot_result "$ZEROSHOT_RESULT" \
    --master_log "$MASTER_LOG" \
    --build_log "$LOG_DIR/build_experience_${RUN_TAG}.log" \
    --tfgrpo_log "$LOG_DIR/eval_tfgrpo_${RUN_TAG}.log" \
    --zeroshot_log "$LOG_DIR/eval_zeroshot_${RUN_TAG}.log"

echo "[DONE] all steps completed" | tee -a "$MASTER_LOG"
echo "[REPORT] $REPORT_PATH" | tee -a "$MASTER_LOG"
echo "[RUN_DIR] $RUN_DIR" | tee -a "$MASTER_LOG"
