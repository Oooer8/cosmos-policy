# RoboTwin AgileX in Cosmos Policy

This guide covers the minimum path for training Cosmos Policy on the RoboTwin 2.0 `aloha-agilex` embodiment:

1. Convert RoboTwin episodes into ALOHA-style HDF5 episodes.
2. Precompute T5 text embeddings for the converted task descriptions.
3. Launch the native Cosmos Policy training job with the new RoboTwin config.

## Assumed Input Layout

After extracting the RoboTwin archives, the script expects a layout like:

```text
/PATH/TO/ROBOTWIN/
  <task_name>/
    aloha-agilex_<variant>/
      data/
        episode0.hdf5
        episode1.hdf5
        ...
```

The converter scans recursively for `data/*.hdf5` directories whose parent folder starts with `aloha-agilex`.

## 1. Convert RoboTwin to ALOHA Format

Run:

```bash
python -m cosmos_policy.experiments.robot.aloha.convert_robotwin_agilex_to_aloha \
  --input_root /PATH/TO/ROBOTWIN \
  --output_root /PATH/TO/RoboTwin-Agilex-Cosmos-Policy
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
/PATH/TO/RoboTwin-Agilex-Cosmos-Policy/
  preprocessed/
    train/
    val/
    conversion_metadata.json
```

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
```

## 2. Precompute T5 Embeddings

Run:

```bash
uv run --extra cu128 --group aloha --python 3.10 \
  python -m cosmos_policy.datasets.save_aloha_t5_text_embeddings \
  --data_dir /PATH/TO/RoboTwin-Agilex-Cosmos-Policy/preprocessed
```

This creates:

```text
/PATH/TO/RoboTwin-Agilex-Cosmos-Policy/preprocessed/t5_embeddings.pkl
```

## 3. Launch Training

Set `BASE_DATASETS_DIR` to the parent directory that contains `RoboTwin-Agilex-Cosmos-Policy`:

```bash
export BASE_DATASETS_DIR=/PATH/TO
```

Then start training:

```bash
uv run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_robotwin_agilex"
```

For quick validation before launching a real run:

```bash
uv run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=1 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py --dryrun -- \
  experiment="cosmos_predict2_2b_480p_robotwin_agilex"
```

## Notes

- The current integration is intentionally narrow: it only targets the RoboTwin `aloha-agilex` embodiment.
- `ALOHADataset` now prefers `attrs["task_description"]`, so converted RoboTwin files can be loaded without ALOHA-specific folder-name heuristics.
- The converter follows the same joint-state alignment that RoboTwin's official ACT preprocessing script uses: current state becomes `qpos`, next state becomes `action`.
