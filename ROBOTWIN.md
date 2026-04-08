# RoboTwin AgileX in Cosmos Policy

This guide covers both training and evaluation for RoboTwin 2.0 `aloha-agilex` with Cosmos Policy.

The integration is intentionally narrow:

1. Training reuses the ALOHA data layout and `ALOHADataset`.
2. Evaluation reuses the existing Cosmos Policy ALOHA deploy server.
3. RobotWin evaluation runs through the official RoboTwin / RobotWin benchmark harness, using a lightweight HTTP adapter that forwards observations to the Cosmos `/act` server.

## Architecture

RobotWin evaluation is split across two environments:

- **Cosmos server environment**: runs the trained RobotWin policy checkpoint with `cosmos_policy.experiments.robot.aloha.deploy`
- **RobotWin environment**: runs the official RoboTwin benchmark, captures observations, and forwards them to the Cosmos server

This split is useful because RobotWin and Cosmos often need different dependency stacks.

```text
RobotWin benchmark env  --HTTP /act-->  Cosmos Policy server
       |                                      |
       | task/env loop                        | model loading + inference
       | success metrics                      | best-of-N / planning (optional)
```

## Relevant Files

Training

- `cosmos_policy/experiments/robot/aloha/convert_robotwin_agilex_to_aloha.py`
- `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`

Evaluation

- `cosmos_policy/experiments/robot/aloha/deploy.py`
- `cosmos_policy/experiments/robot/robotwin/setup_robotwin_eval.py`
- `cosmos_policy/experiments/robot/robotwin/run_robotwin_eval.py`
- `cosmos_policy/experiments/robot/robotwin/smoke_test_robotwin_server.py`
- `cosmos_policy/experiments/robot/robotwin/template_deploy_policy.py`
- `cosmos_policy/experiments/robot/robotwin/requirements_robotwin_eval.txt`

## 0. Recommended Path Setup

The commands below assume:

```bash
export REPO_ROOT=/Users/oooer/ws/cosmos-policy
export DATA_ROOT=/data

export RAW_ROOT=$DATA_ROOT/robotwin/RoboTwin2.0
export INPUT_ROOT=$RAW_ROOT/dataset
export ROBOTWIN_REPO=$DATA_ROOT/robotwin/RoboTwin

export BASE_DATASETS_DIR=$DATA_ROOT/cosmos_datasets
export OUTPUT_ROOT=$BASE_DATASETS_DIR/RoboTwin-Agilex-Cosmos-Policy

export IMAGINAIRE_OUTPUT_ROOT=$DATA_ROOT/cosmos_runs
```

If your data disk is mounted somewhere else, only change `DATA_ROOT`.

## 1. Convert RoboTwin to ALOHA Format

After extracting the RoboTwin archives, the converter expects a layout like:

```text
$INPUT_ROOT/
  <task_name>/
    aloha-agilex_<variant>/
      data/
        episode0.hdf5
        episode1.hdf5
        ...
```

Run:

```bash
cd $REPO_ROOT

python -m cosmos_policy.experiments.robot.aloha.convert_robotwin_agilex_to_aloha \
  --input_root "$INPUT_ROOT" \
  --output_root "$OUTPUT_ROOT"
```

Useful flags:

```bash
  --task_names open_microwave close_drawer
  --percent_val 0.05
  --img_resize_size 256
  --max_episodes_per_config 200
  --overwrite
```

The converted dataset will be written to:

```text
$OUTPUT_ROOT/
  preprocessed/
    train/
    val/
    conversion_metadata.json
    dataset_statistics.json
    dataset_statistics_post_norm.json
```

When the raw RoboTwin layout includes per-episode language files at
`<task_name>/<task_config>/instructions/episode*.json`, the converter now uses those
instructions for `attrs["task_description"]` instead of falling back to the task name.

Each converted episode contains:

```text
/action
/relative_action
/observations/qpos
/observations/qvel
/observations/effort
/observations/images/cam_high
/observations/images/cam_left_wrist
/observations/images/cam_right_wrist
attrs["task_description"]
attrs["instruction"]
```

The converter follows the same alignment as RoboTwin's ACT preprocessing: current state becomes `qpos`, next state becomes `action`.

## 2. Precompute T5 Embeddings

Run:

```bash
cd $REPO_ROOT

uv run --extra cu128 --group aloha --python 3.10 \
  python -m cosmos_policy.datasets.save_aloha_t5_text_embeddings \
  --data_dir "$OUTPUT_ROOT/preprocessed"
```

This creates:

```text
$OUTPUT_ROOT/preprocessed/t5_embeddings.pkl
```

## 3. Launch Training

Set the training roots:

```bash
export BASE_DATASETS_DIR=$DATA_ROOT/cosmos_datasets
export IMAGINAIRE_OUTPUT_ROOT=$DATA_ROOT/cosmos_runs
```

Then start training:

```bash
cd $REPO_ROOT

uv run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_robotwin_agilex"
```

For quick validation before launching a real run:

```bash
cd $REPO_ROOT

uv run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=1 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py --dryrun -- \
  experiment="cosmos_predict2_2b_480p_robotwin_agilex"
```

## 4. Prepare Evaluation Environments

### 4A. Cosmos server environment

Follow the Docker setup in [SETUP.md](SETUP.md), then install the same dependencies used by the ALOHA deploy server:

```bash
uv sync --extra cu128 --group aloha --python 3.10
```

If Docker is not available on your cluster, the practical fallback is:

1. create a bare-metal Python 3.10 environment on the server
2. clone this repo to `$REPO_ROOT`
3. run the same `uv sync --extra cu128 --group aloha --python 3.10`

The important part is matching the Python package environment, not Docker itself.

### 4B. RobotWin environment

Install the official RoboTwin / RobotWin benchmark environment by following the upstream docs:

- [RobotWin install guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
- [Deploy your policy](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)
- [Control robot / action modes](https://robotwin-platform.github.io/doc/usage/control-robot.html)

If your RobotWin conda environment is missing `requests`, `json_numpy`, or `Pillow`, install:

```bash
pip install -r /PATH/TO/cosmos-policy/cosmos_policy/experiments/robot/robotwin/requirements_robotwin_eval.txt
```

## 5. Start the Cosmos Policy Server with Your Trained RobotWin Model

The RobotWin evaluation client reuses the ALOHA deploy server. Point it at your trained RobotWin checkpoint and the dataset stats from the converted RobotWin dataset.

Example:

```bash
uv run --extra cu128 --group aloha --python 3.10 \
  python -m cosmos_policy.experiments.robot.aloha.deploy \
    --config cosmos_predict2_2b_480p_robotwin_agilex__inference_only \
    --ckpt_path $IMAGINAIRE_OUTPUT_ROOT/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_robotwin_agilex/checkpoints/<YOUR_CHECKPOINT_DIR> \
    --config_file cosmos_policy/config/config.py \
    --use_third_person_image True \
    --use_wrist_image True \
    --num_wrist_images 2 \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path $OUTPUT_ROOT/preprocessed/dataset_statistics.json \
    --t5_text_embeddings_path $OUTPUT_ROOT/preprocessed/t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 50 \
    --ar_future_prediction False \
    --ar_value_prediction False \
    --use_jpeg_compression False \
    --flip_images False \
    --num_denoising_steps_action 10 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --deterministic True \
    --seed 195
```

Notes:

- `--ckpt_path` can be a local checkpoint directory or an HF repo, exactly like other Cosmos deploy flows.
- For RobotWin, `--chunk_size` belongs to the server/model side. The client-side re-query cadence is configured separately through `setup_robotwin_eval --num_open_loop_steps` or `deploy_policy.yml`.
- if you used `IMAGINAIRE_OUTPUT_ROOT=$DATA_ROOT/cosmos_runs`, the checkpoint root is typically:

```text
$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_robotwin_agilex/checkpoints/
```

- `--dataset_stats_path` should point to the RobotWin training dataset stats, typically `$OUTPUT_ROOT/preprocessed/dataset_statistics.json`.
- If you trained a planning model, you can add `--planning_model_config_name` and `--planning_model_ckpt_path` exactly as in ALOHA deploy.

## 6. Optional Smoke Test Before Touching RobotWin

If you want to verify the server without a live RobotWin environment, use one converted episode:

```bash
uv run --extra cu128 --group aloha --python 3.10 \
  python -m cosmos_policy.experiments.robot.robotwin.smoke_test_robotwin_server \
    --episode_path $OUTPUT_ROOT/preprocessed/val/<task>/<config>/episode0.hdf5 \
    --server_endpoint http://127.0.0.1:8777/act
```

This checks that:

- the server is reachable
- your checkpoint loads
- action dimensions and payload formatting are sane

## 7. Install the RobotWin Policy Adapter into the RobotWin Repo

From the `cosmos-policy` repo:

```bash
python -m cosmos_policy.experiments.robot.robotwin.setup_robotwin_eval \
  --robotwin_repo "$ROBOTWIN_REPO" \
  --policy_name CosmosPolicyRemote \
  --server_endpoint http://127.0.0.1:8777/act \
  --input_image_size 224 \
  --num_open_loop_steps 50 \
  --request_timeout_sec 60 \
  --action_type qpos \
  --overwrite
```

`--num_open_loop_steps` is a client-side setting: the adapter will execute only that many actions from each server response before fetching fresh RobotWin observations and querying `/act` again.

This writes:

```text
$ROBOTWIN_REPO/policy/CosmosPolicyRemote/
  deploy_policy.py
  __init__.py
  CosmosPolicyRemote.py
  deploy_policy.yml
  eval.sh
  README.md

$ROBOTWIN_REPO/
  CosmosPolicyRemote.py
```

If RobotWin observation keys differ from the defaults, you can override them while installing:

```bash
python -m cosmos_policy.experiments.robot.robotwin.setup_robotwin_eval \
  --robotwin_repo /PATH/TO/RoboTwin \
  --primary_image_path observation.head_camera.rgb \
  --left_wrist_image_path observation.left_camera.rgb \
  --right_wrist_image_path observation.right_camera.rgb \
  --proprio_path observation.qpos \
  --overwrite
```

## 8. Run RobotWin Evaluation

You can either call the generated `eval.sh` directly from the RobotWin repo:

```bash
cd /PATH/TO/RoboTwin
bash policy/CosmosPolicyRemote/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
```

Or drive the same flow from the Cosmos repo:

```bash
python -m cosmos_policy.experiments.robot.robotwin.run_robotwin_eval \
  --robotwin_repo "$ROBOTWIN_REPO" \
  --task_name <task_name> \
  --task_config <task_config> \
  --ckpt_setting 1 \
  --seed 0 \
  --gpu_id 0 \
  --server_endpoint http://127.0.0.1:8777/act \
  --execute \
  --overwrite
```

Without `--execute`, the wrapper only prints the final `eval.sh` command.

## 9. Troubleshooting

- If you see `module 'CosmosPolicyRemote' has no attribute 'get_model'`, first confirm that `ROBOTWIN_REPO` points to the RoboTwin checkout containing `script/eval_policy.py`, not the `cosmos-policy` repo. The installer now checks this explicitly.
- Re-run `setup_robotwin_eval --overwrite` after updating this repo so the compatibility shims are regenerated.
- The generated `eval.sh` now exports a broader `PYTHONPATH` and performs a small import self-check before launching `script/eval_policy.py`, which makes RobotWin loader differences less brittle.

## 10. Assumptions and Compatibility Notes

- This evaluation path targets **RoboTwin 2.0 `aloha-agilex` only**.
- The adapter assumes the same observation layout used during training:
  - 1 primary camera
  - 2 wrist cameras
  - proprio in the same joint order as the conversion script
- The trained policy is assumed to output the same action semantics used in the converted training data, namely next-step joint targets compatible with RobotWin `qpos` control.
- If your RobotWin task expects a different action mode, adjust `action_type` and, if necessary, add a task-side conversion layer before calling `TASK_ENV.take_action()`.
- This workspace did **not** have a live Cosmos + RobotWin runtime environment during authoring, so the pipeline was implemented statically. Use the smoke test first before running long benchmark jobs.
- For the path setup used in this document, the main concrete paths are:

```text
REPO_ROOT=/Users/oooer/ws/cosmos-policy
RAW_ROOT=/data/robotwin/RoboTwin2.0
INPUT_ROOT=/data/robotwin/RoboTwin2.0/dataset
ROBOTWIN_REPO=/data/robotwin/RoboTwin
BASE_DATASETS_DIR=/data/cosmos_datasets
OUTPUT_ROOT=/data/cosmos_datasets/RoboTwin-Agilex-Cosmos-Policy
IMAGINAIRE_OUTPUT_ROOT=/data/cosmos_runs
```

## Notes

- `ALOHADataset` prefers `attrs["task_description"]`, so converted RoboTwin files load without ALOHA-specific folder-name heuristics.
- The evaluation adapter is intentionally lightweight: it keeps Cosmos model inference on the server side and leaves all RobotWin simulation / benchmarking inside the official RobotWin environment.
