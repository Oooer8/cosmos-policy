# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert RoboTwin 2.0 ALOHA-AgileX trajectories into ALOHA-style episode HDF5 files.

The converted dataset can be loaded directly by ``cosmos_policy.datasets.aloha_dataset.ALOHADataset``.

Expected RoboTwin layout after extracting the Hugging Face archives:

    INPUT_ROOT/
      <task_name>/
        aloha-agilex_<variant>/
          data/
            episode0.hdf5
            episode1.hdf5
            ...

The script scans recursively for directories that match this pattern, decodes the
bitstream RGB observations, and writes a train/val split in ALOHA format:

    OUTPUT_ROOT/
      preprocessed/
        train/
          <task_name>/
            <task_config>/
              episode0.hdf5
        val/
          <task_name>/
            <task_config>/
              episode1.hdf5

Usage:
    python -m cosmos_policy.experiments.robot.aloha.convert_robotwin_agilex_to_aloha \
      --input_root /path/to/robotwin/dataset \
      --output_root /path/to/RoboTwin-Agilex-Cosmos-Policy
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


CAMERA_OUTPUT_NAMES = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RoboTwin ALOHA-AgileX data into ALOHA-format HDF5 episodes.")
    parser.add_argument("--input_root", required=True, help="Root directory containing extracted RoboTwin tasks.")
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output dataset root. Converted files are written under OUTPUT_ROOT/preprocessed/{train,val}/...",
    )
    parser.add_argument(
        "--instructions_root",
        default="",
        help="Optional path to RoboTwin instruction JSON files. If omitted, the script tries common locations automatically.",
    )
    parser.add_argument(
        "--embodiment_prefix",
        default="aloha-agilex",
        help="Only convert task configs whose folder name starts with this prefix.",
    )
    parser.add_argument(
        "--task_names",
        nargs="*",
        default=None,
        help="Optional allow-list of task names. If omitted, converts all matching tasks under input_root.",
    )
    parser.add_argument(
        "--percent_val",
        type=float,
        default=0.05,
        help="Validation split ratio measured in episodes per task-config directory.",
    )
    parser.add_argument(
        "--img_resize_size",
        type=int,
        default=256,
        help="Resize RGB observations to square images of this size before writing output episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for train/val splitting.",
    )
    parser.add_argument(
        "--max_episodes_per_config",
        type=int,
        default=0,
        help="Optional cap per task-config directory. Use 0 to keep all episodes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted files if they already exist.",
    )
    return parser.parse_args()


def _candidate_instruction_roots(input_root: str, explicit_root: str) -> list[str]:
    candidates = []
    if explicit_root:
        candidates.append(explicit_root)
    candidates.extend(
        [
            os.path.join(input_root, "instructions"),
            os.path.join(os.path.dirname(input_root), "instructions"),
        ]
    )
    deduped = []
    for candidate in candidates:
        if candidate and os.path.isdir(candidate) and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _extract_first_string(obj) -> str:
    if isinstance(obj, str):
        text = obj.strip()
        return text if text else ""
    if isinstance(obj, dict):
        for key in ("task_description", "instruction", "language", "prompt", "text", "description"):
            if key in obj:
                text = _extract_first_string(obj[key])
                if text:
                    return text
        for value in obj.values():
            text = _extract_first_string(value)
            if text:
                return text
    if isinstance(obj, list):
        for item in obj:
            text = _extract_first_string(item)
            if text:
                return text
    return ""


def load_instruction_lookup(input_root: str, explicit_root: str) -> dict[str, str]:
    instruction_lookup = {}
    for root in _candidate_instruction_roots(input_root, explicit_root):
        for filename in sorted(os.listdir(root)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except Exception:
                continue
            text = _extract_first_string(data)
            if text:
                instruction_lookup[os.path.splitext(filename)[0]] = text
    return instruction_lookup


def discover_robotwin_episode_groups(input_root: str, embodiment_prefix: str, task_names: set[str] | None):
    grouped_episodes = defaultdict(list)
    for root, _, files in os.walk(input_root):
        if os.path.basename(root) != "data":
            continue
        task_config = os.path.basename(os.path.dirname(root))
        if not task_config.startswith(embodiment_prefix):
            continue
        task_name = os.path.basename(os.path.dirname(os.path.dirname(root)))
        if task_names is not None and task_name not in task_names:
            continue
        episode_paths = sorted(
            os.path.join(root, filename) for filename in files if filename.lower().endswith((".h5", ".hdf5", ".he5"))
        )
        if episode_paths:
            grouped_episodes[(task_name, task_config)].extend(episode_paths)
    return grouped_episodes


def split_episode_paths(episode_paths: list[str], percent_val: float, rng: random.Random) -> tuple[list[str], list[str]]:
    shuffled = list(episode_paths)
    rng.shuffle(shuffled)
    if percent_val <= 0 or len(shuffled) <= 1:
        return shuffled, []
    num_val = int(len(shuffled) * percent_val)
    if num_val == 0:
        num_val = 1
    if num_val >= len(shuffled):
        num_val = len(shuffled) - 1
    return shuffled[num_val:], shuffled[:num_val]


def _find_dataset(root: h5py.File, candidates: list[tuple[str, ...]]):
    for path_parts in candidates:
        node = root
        found = True
        for part in path_parts:
            if part not in node:
                found = False
                break
            node = node[part]
        if found:
            return node
    return None


def _decode_robotwin_frame(frame_data, resize_size: int) -> np.ndarray:
    if isinstance(frame_data, (bytes, bytearray)):
        encoded = np.frombuffer(frame_data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Failed to decode RoboTwin frame bytes with OpenCV.")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if resize_size > 0 and (frame_rgb.shape[0] != resize_size or frame_rgb.shape[1] != resize_size):
            frame_rgb = np.array(
                Image.fromarray(frame_rgb).resize((resize_size, resize_size), resample=Image.BICUBIC),
                dtype=np.uint8,
            )
        return frame_rgb.astype(np.uint8)

    array = np.asarray(frame_data)
    if array.ndim == 1:
        encoded = np.asarray(array, dtype=np.uint8)
        frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Failed to decode RoboTwin frame bytes with OpenCV.")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    elif array.ndim == 3 and array.shape[-1] == 3:
        frame_rgb = array.astype(np.uint8)
    else:
        raise ValueError(f"Unexpected RoboTwin frame shape: {array.shape}")

    if resize_size > 0 and (frame_rgb.shape[0] != resize_size or frame_rgb.shape[1] != resize_size):
        frame_rgb = np.array(
            Image.fromarray(frame_rgb).resize((resize_size, resize_size), resample=Image.BICUBIC),
            dtype=np.uint8,
        )
    return frame_rgb.astype(np.uint8)


def _load_robotwin_joint_state(root: h5py.File) -> np.ndarray:
    joint_action = root["joint_action"]
    left_gripper = joint_action["left_gripper"][:]
    right_gripper = joint_action["right_gripper"][:]
    if left_gripper.ndim == 1:
        left_gripper = left_gripper[:, None]
    if right_gripper.ndim == 1:
        right_gripper = right_gripper[:, None]
    state = np.concatenate(
        [
            joint_action["left_arm"][:],
            left_gripper,
            joint_action["right_arm"][:],
            right_gripper,
        ],
        axis=1,
    )
    return state.astype(np.float32)


def load_robotwin_episode(episode_path: str, resize_size: int) -> dict:
    with h5py.File(episode_path, "r") as root:
        state = _load_robotwin_joint_state(root)
        camera_datasets = {}
        for camera_name in CAMERA_OUTPUT_NAMES:
            dataset = _find_dataset(
                root,
                [
                    ("observation", camera_name, "rgb"),
                    ("observations", camera_name, "rgb"),
                ],
            )
            if dataset is None:
                raise KeyError(f"Could not find RGB dataset for camera '{camera_name}' in {episode_path}.")
            camera_datasets[camera_name] = dataset

        frame_counts = [len(dataset) for dataset in camera_datasets.values()]
        common_steps = min([state.shape[0], *frame_counts])
        if common_steps < 2:
            raise ValueError(f"Episode {episode_path} has fewer than 2 aligned steps after truncation.")

        qpos = state[: common_steps - 1]
        action = state[1:common_steps]
        qvel = np.zeros_like(qpos, dtype=np.float32)
        effort = np.zeros_like(qpos, dtype=np.float32)
        relative_action = np.zeros_like(action, dtype=np.float32)
        if action.shape[0] > 1:
            relative_action[:-1] = action[1:] - action[:-1]
            relative_action[-1] = relative_action[-2]

        image_dict = {}
        for camera_name, output_name in CAMERA_OUTPUT_NAMES.items():
            dataset = camera_datasets[camera_name]
            frames = [_decode_robotwin_frame(dataset[idx], resize_size) for idx in range(common_steps - 1)]
            image_dict[output_name] = np.stack(frames, axis=0).astype(np.uint8)

    return {
        "qpos": qpos,
        "qvel": qvel,
        "effort": effort,
        "action": action,
        "relative_action": relative_action,
        "images": image_dict,
        "num_steps": qpos.shape[0],
    }


def save_aloha_episode(output_path: str, episode_data: dict, task_description: str, source_episode_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_steps = int(episode_data["num_steps"])

    with h5py.File(output_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = True
        root.attrs["task_description"] = task_description
        root.attrs["source_episode_path"] = source_episode_path

        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=episode_data["qpos"], dtype=np.float32)
        obs.create_dataset("qvel", data=episode_data["qvel"], dtype=np.float32)
        obs.create_dataset("effort", data=episode_data["effort"], dtype=np.float32)

        image_group = obs.create_group("images")
        for camera_name, frames in episode_data["images"].items():
            image_group.create_dataset(
                camera_name,
                data=frames,
                dtype=np.uint8,
                compression="gzip",
                compression_opts=1,
                chunks=(1, frames.shape[1], frames.shape[2], frames.shape[3]),
            )

        root.create_dataset("action", data=episode_data["action"], dtype=np.float32)
        root.create_dataset("relative_action", data=episode_data["relative_action"], dtype=np.float32)

        # Store the aligned sequence length for debugging / sanity checks.
        root.attrs["num_steps"] = num_steps


def get_task_description(task_name: str, instruction_lookup: dict[str, str]) -> str:
    if task_name in instruction_lookup:
        return instruction_lookup[task_name]
    return task_name.replace("_", " ")


def convert_split(
    split_name: str,
    grouped_episode_paths: dict[tuple[str, str], list[str]],
    split_indices: dict[tuple[str, str], tuple[list[str], list[str]]],
    output_root: str,
    instruction_lookup: dict[str, str],
    resize_size: int,
    overwrite: bool,
) -> list[dict]:
    split_metadata = []
    preprocessed_root = os.path.join(output_root, "preprocessed", split_name)

    iterable = []
    for key, (train_paths, val_paths) in split_indices.items():
        split_paths = train_paths if split_name == "train" else val_paths
        for episode_path in split_paths:
            iterable.append((key, episode_path))

    for (task_name, task_config), episode_path in tqdm(iterable, desc=f"Converting {split_name} episodes"):
        task_description = get_task_description(task_name, instruction_lookup)
        relative_dir = os.path.join(task_name, task_config)
        output_dir = os.path.join(preprocessed_root, relative_dir)
        output_path = os.path.join(output_dir, os.path.basename(episode_path))

        if os.path.exists(output_path) and not overwrite:
            split_metadata.append(
                {
                    "task_name": task_name,
                    "task_config": task_config,
                    "source_episode_path": episode_path,
                    "output_episode_path": output_path,
                    "split": split_name,
                    "skipped_existing": True,
                }
            )
            continue

        episode_data = load_robotwin_episode(episode_path, resize_size=resize_size)
        save_aloha_episode(output_path, episode_data, task_description, episode_path)
        split_metadata.append(
            {
                "task_name": task_name,
                "task_config": task_config,
                "task_description": task_description,
                "source_episode_path": episode_path,
                "output_episode_path": output_path,
                "split": split_name,
                "num_steps": int(episode_data["num_steps"]),
                "skipped_existing": False,
            }
        )

    return split_metadata


def main(args):
    task_name_filter = set(args.task_names) if args.task_names else None
    instruction_lookup = load_instruction_lookup(args.input_root, args.instructions_root)
    grouped_episode_paths = discover_robotwin_episode_groups(
        input_root=args.input_root,
        embodiment_prefix=args.embodiment_prefix,
        task_names=task_name_filter,
    )
    if not grouped_episode_paths:
        raise ValueError(
            f"No RoboTwin episode directories matching embodiment prefix '{args.embodiment_prefix}' were found under "
            f"{args.input_root}."
        )

    rng = random.Random(args.seed)
    split_indices = {}
    for key, episode_paths in grouped_episode_paths.items():
        limited_paths = list(episode_paths)
        if args.max_episodes_per_config > 0 and len(limited_paths) > args.max_episodes_per_config:
            limited_paths = limited_paths[: args.max_episodes_per_config]
        split_indices[key] = split_episode_paths(limited_paths, args.percent_val, rng)

    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    train_metadata = convert_split(
        split_name="train",
        grouped_episode_paths=grouped_episode_paths,
        split_indices=split_indices,
        output_root=output_root,
        instruction_lookup=instruction_lookup,
        resize_size=args.img_resize_size,
        overwrite=args.overwrite,
    )
    val_metadata = convert_split(
        split_name="val",
        grouped_episode_paths=grouped_episode_paths,
        split_indices=split_indices,
        output_root=output_root,
        instruction_lookup=instruction_lookup,
        resize_size=args.img_resize_size,
        overwrite=args.overwrite,
    )

    metadata = {
        "dataset_info": {
            "input_root": os.path.abspath(args.input_root),
            "output_root": output_root,
            "instructions_root": os.path.abspath(args.instructions_root) if args.instructions_root else "",
            "embodiment_prefix": args.embodiment_prefix,
            "percent_val": args.percent_val,
            "img_resize_size": args.img_resize_size,
            "seed": args.seed,
            "max_episodes_per_config": args.max_episodes_per_config,
            "num_task_configs": len(split_indices),
            "num_train_episodes": len(train_metadata),
            "num_val_episodes": len(val_metadata),
        },
        "episode_mapping": train_metadata + val_metadata,
    }

    metadata_path = os.path.join(output_root, "preprocessed", "conversion_metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Converted dataset written to: {os.path.join(output_root, 'preprocessed')}")
    print(f"Metadata written to: {metadata_path}")


if __name__ == "__main__":
    main(parse_args())
