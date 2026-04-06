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

"""Helpers for RobotWin evaluation and RobotWin -> Cosmos observation conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


DEFAULT_PRIMARY_IMAGE_PATHS = [
    "primary_image",
    "images.cam_high",
    "observation.images.cam_high",
    "head_camera.rgb",
    "observation.head_camera.rgb",
    "observation.head_camera",
    "head_camera",
]
DEFAULT_LEFT_WRIST_IMAGE_PATHS = [
    "left_wrist_image",
    "images.cam_left_wrist",
    "observation.images.cam_left_wrist",
    "left_camera.rgb",
    "observation.left_camera.rgb",
    "observation.left_camera",
    "left_camera",
]
DEFAULT_RIGHT_WRIST_IMAGE_PATHS = [
    "right_wrist_image",
    "images.cam_right_wrist",
    "observation.images.cam_right_wrist",
    "right_camera.rgb",
    "observation.right_camera.rgb",
    "observation.right_camera",
    "right_camera",
]
DEFAULT_PROPRIO_PATHS = [
    "qpos",
    "observation.qpos",
    "joint_positions",
    "observation.joint_positions",
    "robot_state.qpos",
    "joint_state.qpos",
    "observation.joint_state.qpos",
]
DEFAULT_TASK_DESCRIPTION_PATHS = [
    "task_description",
    "instruction",
    "language_instruction",
    "language",
    "meta.task_description",
    "observation.task_description",
]


def normalize_candidate_paths(paths: Sequence[str] | None, fallback_paths: Sequence[str]) -> list[str]:
    """Normalize a user-provided path list while preserving fallback defaults."""
    if not paths:
        return list(fallback_paths)

    normalized: list[str] = []
    for entry in paths:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized or list(fallback_paths)


def _lookup_child(node: Any, key: str) -> Any:
    if isinstance(node, Mapping) and key in node:
        return node[key]

    if hasattr(node, key):
        return getattr(node, key)

    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        try:
            index = int(key)
        except ValueError:
            return None
        if 0 <= index < len(node):
            return node[index]

    return None


def deep_get(node: Any, path: str) -> Any:
    """Traverse nested dicts / objects using dot-separated paths."""
    current = node
    for part in path.split("."):
        if current is None:
            return None
        current = _lookup_child(current, part)
    return current


def first_present(node: Any, candidate_paths: Sequence[str]) -> Any:
    """Return the first successfully resolved path."""
    for path in candidate_paths:
        value = deep_get(node, path)
        if value is not None:
            return value
    return None


def _resize_square(image: np.ndarray, target_size: int) -> np.ndarray:
    if target_size <= 0 or (image.shape[0] == target_size and image.shape[1] == target_size):
        return image
    return np.asarray(Image.fromarray(image).resize((target_size, target_size), resample=Image.BICUBIC))


def coerce_uint8_rgb_image(image: Any, target_size: int, swap_bgr_to_rgb: bool = False) -> np.ndarray:
    """Convert image-like input into HWC uint8 RGB."""
    array = np.asarray(image)

    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]

    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)

    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected image with shape HxWx3/4 or CxHxW, got {array.shape}.")

    if array.shape[-1] == 4:
        array = array[..., :3]

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.max(array)) if array.size > 0 else 0.0
        if max_value <= 1.5:
            array = np.clip(array, 0.0, 1.0) * 255.0
        else:
            array = np.clip(array, 0.0, 255.0)
        array = np.round(array).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if swap_bgr_to_rgb:
        array = array[..., ::-1]

    array = _resize_square(array, target_size)
    return np.ascontiguousarray(array)


def _coerce_float_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("Expected a non-empty proprio / action vector.")
    return array


def _compose_joint_state_from_prefix(root: Any, prefix: str) -> np.ndarray | None:
    left_arm = deep_get(root, f"{prefix}.left_arm")
    left_gripper = deep_get(root, f"{prefix}.left_gripper")
    right_arm = deep_get(root, f"{prefix}.right_arm")
    right_gripper = deep_get(root, f"{prefix}.right_gripper")

    if left_arm is None or left_gripper is None or right_arm is None or right_gripper is None:
        return None

    left_arm = _coerce_float_vector(left_arm)
    right_arm = _coerce_float_vector(right_arm)
    left_gripper = _coerce_float_vector(left_gripper)
    right_gripper = _coerce_float_vector(right_gripper)
    return np.concatenate([left_arm, left_gripper[:1], right_arm, right_gripper[:1]], axis=0)


def extract_robotwin_proprio(observation: Any, proprio_paths: Sequence[str] | None = None) -> np.ndarray:
    """Extract proprioception in the joint order used by the RobotWin conversion script."""
    for path in normalize_candidate_paths(proprio_paths, DEFAULT_PROPRIO_PATHS):
        value = deep_get(observation, path)
        if value is not None:
            return _coerce_float_vector(value)

    for prefix in (
        "joint_action",
        "observation.joint_action",
        "joint_state",
        "observation.joint_state",
        "joint_position",
        "observation.joint_position",
    ):
        composed = _compose_joint_state_from_prefix(observation, prefix)
        if composed is not None:
            return composed

    raise KeyError(
        "Could not extract proprio from RobotWin observation. "
        "Override `proprio_paths` or expose qpos / joint_action in the environment observation."
    )


def extract_robotwin_policy_observation(
    observation: Any,
    input_image_size: int = 224,
    swap_bgr_to_rgb: bool = False,
    primary_image_paths: Sequence[str] | None = None,
    left_wrist_image_paths: Sequence[str] | None = None,
    right_wrist_image_paths: Sequence[str] | None = None,
    proprio_paths: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Convert a raw RobotWin observation into the ALOHA-style payload expected by the deploy server."""
    primary = first_present(observation, normalize_candidate_paths(primary_image_paths, DEFAULT_PRIMARY_IMAGE_PATHS))
    left = first_present(observation, normalize_candidate_paths(left_wrist_image_paths, DEFAULT_LEFT_WRIST_IMAGE_PATHS))
    right = first_present(
        observation, normalize_candidate_paths(right_wrist_image_paths, DEFAULT_RIGHT_WRIST_IMAGE_PATHS)
    )

    if primary is None:
        raise KeyError("Could not locate RobotWin primary camera image in observation.")
    if left is None:
        raise KeyError("Could not locate RobotWin left wrist camera image in observation.")
    if right is None:
        raise KeyError("Could not locate RobotWin right wrist camera image in observation.")

    return {
        "primary_image": coerce_uint8_rgb_image(primary, input_image_size, swap_bgr_to_rgb=swap_bgr_to_rgb),
        "left_wrist_image": coerce_uint8_rgb_image(left, input_image_size, swap_bgr_to_rgb=swap_bgr_to_rgb),
        "right_wrist_image": coerce_uint8_rgb_image(right, input_image_size, swap_bgr_to_rgb=swap_bgr_to_rgb),
        "proprio": extract_robotwin_proprio(observation, proprio_paths=proprio_paths),
    }


def extract_task_description(observation: Any, task_description_paths: Sequence[str] | None = None) -> str:
    """Try to recover a task description from an observation-like object."""
    for path in normalize_candidate_paths(task_description_paths, DEFAULT_TASK_DESCRIPTION_PATHS):
        value = deep_get(observation, path)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def load_policy_observation_from_episode(
    episode_path: str | Path,
    frame_idx: int = 0,
    input_image_size: int = 224,
) -> tuple[dict[str, np.ndarray], str]:
    """Load a single frame from a converted RobotWin / ALOHA-format HDF5 episode for smoke tests."""
    episode_path = Path(episode_path)
    with h5py.File(episode_path, "r") as handle:
        obs_group = handle["observations"]
        num_steps = int(obs_group["qpos"].shape[0])
        if num_steps <= 0:
            raise ValueError(f"Episode has no frames: {episode_path}")

        index = max(0, min(int(frame_idx), num_steps - 1))
        task_description = str(handle.attrs.get("task_description", "")).strip()
        observation = {
            "primary_image": obs_group["images"]["cam_high"][index],
            "left_wrist_image": obs_group["images"]["cam_left_wrist"][index],
            "right_wrist_image": obs_group["images"]["cam_right_wrist"][index],
            "proprio": obs_group["qpos"][index],
        }

    converted = extract_robotwin_policy_observation(
        observation,
        input_image_size=input_image_size,
        swap_bgr_to_rgb=False,
        primary_image_paths=["primary_image"],
        left_wrist_image_paths=["left_wrist_image"],
        right_wrist_image_paths=["right_wrist_image"],
        proprio_paths=["proprio"],
    )
    return converted, task_description
