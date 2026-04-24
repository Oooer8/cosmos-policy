#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Parallel RoboTwin -> Cosmos Policy converter (task-sharded).

This script:
1. Splits conversion work by task_name.
2. Runs one converter process per task shard with bounded parallelism.
3. Merges all shard outputs into one OUTPUT_ROOT/preprocessed tree.
4. Saves one metadata file per task and a merged metadata summary.

Required environment variables:
  INPUT_ROOT   Root directory containing extracted RoboTwin tasks.
  OUTPUT_ROOT  Final merged output root.

Optional environment variables:
  PYTHON_BIN               Python executable to use. Default: python
  MAX_JOBS                 Max parallel task conversions. Default: 2
  IMG_RESIZE_SIZE          Passed to converter. Default: 256
  PERCENT_VAL              Passed to converter. Default: 0.05
  MAX_EPISODES_PER_CONFIG  Passed to converter if > 0. Default: 0
  EMBODIMENT_PREFIX        Passed to converter. Default: aloha-agilex
  INSTRUCTIONS_ROOT        Passed to converter if set.
  OVERWRITE                If 1, passes --overwrite to converter. Default: 0
  SHARD_ROOT               Temp per-task output root. Default: ${OUTPUT_ROOT}__shards
  LOG_ROOT                 Per-task logs root. Default: ${SHARD_ROOT}/logs
  MERGE_MODE               move (default) or copy

Usage:
  INPUT_ROOT=/path/to/dataset \
  OUTPUT_ROOT=/path/to/RoboTwin-Agilex-Cosmos-Policy \
  MAX_JOBS=4 \
  bash bin/parallel_convert_robotwin_by_task.sh [task_name ...]

If no task names are passed, tasks are auto-discovered under INPUT_ROOT by finding
top-level directories that contain at least one matching embodiment config folder.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

: "${INPUT_ROOT:?Please set INPUT_ROOT to the extracted RoboTwin dataset root.}"
: "${OUTPUT_ROOT:?Please set OUTPUT_ROOT to the merged conversion output root.}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_JOBS="${MAX_JOBS:-2}"
IMG_RESIZE_SIZE="${IMG_RESIZE_SIZE:-256}"
PERCENT_VAL="${PERCENT_VAL:-0.05}"
MAX_EPISODES_PER_CONFIG="${MAX_EPISODES_PER_CONFIG:-0}"
EMBODIMENT_PREFIX="${EMBODIMENT_PREFIX:-aloha-agilex}"
INSTRUCTIONS_ROOT="${INSTRUCTIONS_ROOT:-}"
OVERWRITE="${OVERWRITE:-0}"
SHARD_ROOT="${SHARD_ROOT:-${OUTPUT_ROOT%/}__shards}"
LOG_ROOT="${LOG_ROOT:-${SHARD_ROOT}/logs}"
MERGE_MODE="${MERGE_MODE:-move}"
STATUS_ROOT="${LOG_ROOT}/status"

if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "INPUT_ROOT does not exist: $INPUT_ROOT" >&2
  exit 1
fi

if [[ "$MERGE_MODE" != "move" && "$MERGE_MODE" != "copy" ]]; then
  echo "MERGE_MODE must be either 'move' or 'copy', got: $MERGE_MODE" >&2
  exit 1
fi

mkdir -p "$SHARD_ROOT" "$LOG_ROOT" "$STATUS_ROOT"

declare -a TASKS
if [[ "$#" -gt 0 ]]; then
  TASKS=("$@")
else
  mapfile -t TASKS < <(
    find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d | while read -r task_dir; do
      if find "$task_dir" -mindepth 1 -maxdepth 1 -type d -name "${EMBODIMENT_PREFIX}*" | grep -q .; then
        basename "$task_dir"
      fi
    done | LC_ALL=C sort
  )
fi

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "No task names found under INPUT_ROOT=$INPUT_ROOT for embodiment prefix ${EMBODIMENT_PREFIX}" >&2
  exit 1
fi

echo "Discovered ${#TASKS[@]} task(s):"
printf '  - %s\n' "${TASKS[@]}"
echo "Shard root: $SHARD_ROOT"
echo "Merged output root: $OUTPUT_ROOT"
echo "Max parallel jobs: $MAX_JOBS"
echo "Merge mode: $MERGE_MODE"

run_one_task() {
  local task_name="$1"
  local shard_out="${SHARD_ROOT}/${task_name}"
  local log_path="${LOG_ROOT}/${task_name}.log"

  mkdir -p "$shard_out"

  local -a cmd=(
    "$PYTHON_BIN" -m cosmos_policy.experiments.robot.aloha.convert_robotwin_agilex_to_aloha
    --input_root "$INPUT_ROOT"
    --output_root "$shard_out"
    --task_names "$task_name"
    --percent_val "$PERCENT_VAL"
    --img_resize_size "$IMG_RESIZE_SIZE"
    --embodiment_prefix "$EMBODIMENT_PREFIX"
  )

  if [[ -n "$INSTRUCTIONS_ROOT" ]]; then
    cmd+=(--instructions_root "$INSTRUCTIONS_ROOT")
  fi
  if [[ "$MAX_EPISODES_PER_CONFIG" != "0" ]]; then
    cmd+=(--max_episodes_per_config "$MAX_EPISODES_PER_CONFIG")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  {
    echo "=== task: ${task_name} ==="
    printf 'Command: '
    printf '%q ' "${cmd[@]}"
    echo
    "${cmd[@]}"
  } >"$log_path" 2>&1
}

active_jobs=0
failed_jobs=0

launch_one_task() {
  local task_name="$1"
  local status_path="${STATUS_ROOT}/${task_name}.status"

  echo "Launching task: $task_name"
  printf 'running\n' >"$status_path"
  (
    set +e
    run_one_task "$task_name"
    status=$?
    printf '%s\n' "$status" >"$status_path"
    exit "$status"
  ) &
  active_jobs=$((active_jobs + 1))
}

wait_for_one_task() {
  if wait -n; then
    :
  else
    failed_jobs=$((failed_jobs + 1))
  fi
  active_jobs=$((active_jobs - 1))
}

for task_name in "${TASKS[@]}"; do
  while [[ "$active_jobs" -ge "$MAX_JOBS" ]]; do
    wait_for_one_task
  done

  launch_one_task "$task_name"
done

while [[ "$active_jobs" -gt 0 ]]; do
  wait_for_one_task
done

declare -a FAILED_TASKS=()
for task_name in "${TASKS[@]}"; do
  status_path="${STATUS_ROOT}/${task_name}.status"
  if [[ ! -f "$status_path" ]]; then
    FAILED_TASKS+=("${task_name} (missing status)")
    continue
  fi

  status="$(<"$status_path")"
  if [[ "$status" != "0" ]]; then
    FAILED_TASKS+=("${task_name} (exit ${status})")
  fi
done

if [[ "${#FAILED_TASKS[@]}" -gt 0 ]]; then
  echo "One or more task conversions failed; skipping merge." >&2
  printf '  - %s\n' "${FAILED_TASKS[@]}" >&2
  echo "Inspect per-task logs under: ${LOG_ROOT}/<task_name>.log" >&2
  exit 1
fi

echo "All task conversions completed. Starting merge."

merged_preprocessed="${OUTPUT_ROOT%/}/preprocessed"
merged_train="${merged_preprocessed}/train"
merged_val="${merged_preprocessed}/val"
merged_meta_dir="${merged_preprocessed}/conversion_metadata"

mkdir -p "$merged_train" "$merged_val" "$merged_meta_dir"

merge_path() {
  local src="$1"
  local dst_parent="$2"

  if [[ "$MERGE_MODE" == "move" ]]; then
    mv "$src" "$dst_parent/"
  else
    cp -R "$src" "$dst_parent/"
  fi
}

for task_name in "${TASKS[@]}"; do
  shard_preprocessed="${SHARD_ROOT}/${task_name}/preprocessed"
  shard_train_task="${shard_preprocessed}/train/${task_name}"
  shard_val_task="${shard_preprocessed}/val/${task_name}"
  shard_meta="${shard_preprocessed}/conversion_metadata.json"

  if [[ ! -f "$shard_meta" ]]; then
    echo "Missing shard metadata for task ${task_name}: ${shard_meta}" >&2
    exit 1
  fi

  if [[ -d "$shard_train_task" ]]; then
    if [[ -e "${merged_train}/${task_name}" ]]; then
      echo "Refusing to overwrite existing merged train task directory: ${merged_train}/${task_name}" >&2
      exit 1
    fi
    merge_path "$shard_train_task" "$merged_train"
  fi

  if [[ -d "$shard_val_task" ]]; then
    if [[ -e "${merged_val}/${task_name}" ]]; then
      echo "Refusing to overwrite existing merged val task directory: ${merged_val}/${task_name}" >&2
      exit 1
    fi
    merge_path "$shard_val_task" "$merged_val"
  fi

  if [[ "$MERGE_MODE" == "move" ]]; then
    mv "$shard_meta" "${merged_meta_dir}/${task_name}.json"
  else
    cp "$shard_meta" "${merged_meta_dir}/${task_name}.json"
  fi
done

"$PYTHON_BIN" - "$merged_meta_dir" "$merged_preprocessed/conversion_metadata_merged.json" <<'PY'
import json
import os
import sys

meta_dir = sys.argv[1]
merged_path = sys.argv[2]

task_files = sorted(
    os.path.join(meta_dir, name)
    for name in os.listdir(meta_dir)
    if name.endswith(".json")
)

dataset_infos = []
episode_mapping = []
summary = {
    "num_task_files": len(task_files),
    "num_task_configs": 0,
    "num_train_episodes": 0,
    "num_val_episodes": 0,
}

for path in task_files:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    dataset_info = dict(payload.get("dataset_info", {}))
    dataset_info["task_metadata_file"] = os.path.basename(path)
    dataset_infos.append(dataset_info)
    episode_mapping.extend(payload.get("episode_mapping", []))
    summary["num_task_configs"] += int(dataset_info.get("num_task_configs", 0))
    summary["num_train_episodes"] += int(dataset_info.get("num_train_episodes", 0))
    summary["num_val_episodes"] += int(dataset_info.get("num_val_episodes", 0))

merged = {
    "summary": summary,
    "dataset_infos": dataset_infos,
    "episode_mapping": episode_mapping,
}

with open(merged_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)
PY

cat <<EOF

Merge complete.

Final directory layout:
  ${OUTPUT_ROOT%/}/
    preprocessed/
      train/
        <task_name>/
          <task_config>/
            episode*.hdf5
      val/
        <task_name>/
          <task_config>/
            episode*.hdf5
      conversion_metadata/
        <task_name>.json
      conversion_metadata_merged.json

Per-task converter logs:
  ${LOG_ROOT}/<task_name>.log

Notes:
  - This script intentionally keeps per-task metadata separate to avoid shard overwrite.
  - Default merge mode is "move", so shard data is removed from SHARD_ROOT as it is merged.
  - If INPUT_ROOT/OUTPUT_ROOT/SHARD_ROOT are on the same filesystem, move avoids duplicating data blocks.
  - Generate t5_embeddings.pkl only after the merged dataset is ready.
  - If you rerun into the same OUTPUT_ROOT, this script will refuse to overwrite merged task directories.
EOF
