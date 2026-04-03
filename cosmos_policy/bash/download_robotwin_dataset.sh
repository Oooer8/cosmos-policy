#!/usr/bin/env bash
set -euo pipefail

# ── 镜像设置 ─────────────────────────────────────────────
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

RAW_ROOT="${RAW_ROOT:-./raw_data}"
REPO="TianxingChen/RoboTwin2.0"
MAX_JOBS=4

tasks=(
  adjust_bottle
  click_alarmclock
  grab_roller
  handover_block
  lift_pot
  open_microwave
  pick_diverse_bottles
  pick_dual_bottles
  place_bread_basket
  place_bread_skillet
  place_empty_cup
  place_phone_stand
  press_stapler
  put_object_cabinet
  scan_object
  shake_bottle
  stack_blocks_three
  stack_blocks_two
  stack_bowls_two
  stamp_seal
)

# ── 颜色输出 ──────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ── 日志目录 ──────────────────────────────────────────────
LOG_DIR="${RAW_ROOT}/_logs"
mkdir -p "$LOG_DIR"

# ── 下载单个 task 的单个 config ───────────────────────────
# 每次只下一个文件，彻底规避 --include 多参数 bug
download_one() {
  local task="$1"
  local config="$2"
  local log="${LOG_DIR}/${task}_${config}.log"
  local cache_dir="${RAW_ROOT}/.cache/${task}_${config}"

  # 已存在则跳过
  if [ -f "${RAW_ROOT}/dataset/${task}/${config}" ]; then
    echo -e "${GREEN}[SKIP]${NC}   ${task}/${config} (already exists)"
    return 0
  fi

  echo -e "${YELLOW}[START]${NC}  ${task}/${config}"
  if huggingface-cli download "$REPO" \
      --repo-type dataset \
      --include "dataset/${task}/${config}" \
      --local-dir "$RAW_ROOT" \
      --cache-dir "$cache_dir" \
      >"$log" 2>&1; then
    rm -rf "$cache_dir"
    echo -e "${GREEN}[DONE]${NC}   ${task}/${config}"
  else
    echo -e "${RED}[FAILED]${NC} ${task}/${config}  (see ${log})"
    return 1
  fi
}

export -f download_one
export RAW_ROOT REPO LOG_DIR GREEN RED YELLOW NC

# ── 构造任务列表：task x config 笛卡尔积 ─────────────────
CONFIGS=(
  "aloha-agilex_clean_50.zip"
  "aloha-agilex_randomized_500.zip"
)

job_list=()
for task in "${tasks[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    job_list+=("${task}|||${cfg}")
  done
done

# ── 并行执行 ──────────────────────────────────────────────
echo "========================================"
echo " Repo      : $REPO"
echo " Endpoint  : $HF_ENDPOINT"
echo " Local dir : $RAW_ROOT"
echo " Tasks     : ${#tasks[@]}"
echo " Configs   : ${CONFIGS[*]}"
echo " Total jobs: ${#job_list[@]}"
echo " Max jobs  : $MAX_JOBS"
echo "========================================"

printf '%s\n' "${job_list[@]}" \
  | xargs -P "$MAX_JOBS" -I {} bash -c '
      item="{}"
      task="${item%%|||*}"
      cfg="${item##*|||}"
      download_one "$task" "$cfg"
    '

# ── 汇总 ─────────────────────────────────────────────────
echo ""
echo "========================================"
FAILED=()
for task in "${tasks[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    log="${LOG_DIR}/${task}_${cfg}.log"
    if grep -qi "error\|failed\|traceback" "$log" 2>/dev/null; then
      FAILED+=("${task}/${cfg}")
    fi
  done
done

if [ ${#FAILED[@]} -eq 0 ]; then
  echo -e "${GREEN}All jobs downloaded successfully.${NC}"
else
  echo -e "${RED}${#FAILED[@]} job(s) failed:${NC}"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
echo "========================================"