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

"""RobotWin policy adapter that forwards RobotWin observations to the Cosmos Policy /act server."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import json_numpy
import numpy as np
import requests
from PIL import Image


json_numpy.patch()

__all__ = [
    "RemoteRobotWinPolicy",
    "encode_obs",
    "eval",
    "get_action",
    "get_model",
    "reset_model",
    "update_obs",
]


_FIXED_TASK_DESCRIPTION_CACHE: dict[str, str] = {}


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


def _normalize_paths(paths: Sequence[str] | None, fallback_paths: Sequence[str]) -> list[str]:
    if not paths:
        return list(fallback_paths)
    normalized = []
    for entry in paths:
        if entry is None:
            continue
        text = str(entry).strip()
        if text:
            normalized.append(text)
    return normalized or list(fallback_paths)


def _merge_config(user_args: Any) -> dict[str, Any]:
    defaults = {
        "server_endpoint": os.environ.get("COSMOS_POLICY_SERVER_ENDPOINT", "http://127.0.0.1:8777/act"),
        "request_timeout_sec": float(os.environ.get("COSMOS_POLICY_REQUEST_TIMEOUT_SEC", "60")),
        "input_image_size": 224,
        # Client-side re-query cadence. The server still returns the full model chunk.
        "num_open_loop_steps": 50,
        "return_all_query_results": False,
        "action_type": "qpos",
        "strict_action_dim": True,
        "swap_bgr_to_rgb": False,
        "use_fixed_task_description": False,
        "sleep_after_action_sec": 0.0,
        "primary_image_paths": list(DEFAULT_PRIMARY_IMAGE_PATHS),
        "left_wrist_image_paths": list(DEFAULT_LEFT_WRIST_IMAGE_PATHS),
        "right_wrist_image_paths": list(DEFAULT_RIGHT_WRIST_IMAGE_PATHS),
        "proprio_paths": list(DEFAULT_PROPRIO_PATHS),
        "default_task_description": "",
        "verbose": True,
    }

    if user_args is None:
        return defaults

    if isinstance(user_args, Mapping):
        merged = dict(defaults)
        merged.update(dict(user_args))
    elif hasattr(user_args, "items"):
        merged = dict(defaults)
        merged.update(dict(user_args.items()))
    else:
        raise TypeError(f"Expected dict-like usr_args, got {type(user_args)}")

    merged["primary_image_paths"] = _normalize_paths(merged.get("primary_image_paths"), DEFAULT_PRIMARY_IMAGE_PATHS)
    merged["left_wrist_image_paths"] = _normalize_paths(
        merged.get("left_wrist_image_paths"), DEFAULT_LEFT_WRIST_IMAGE_PATHS
    )
    merged["right_wrist_image_paths"] = _normalize_paths(
        merged.get("right_wrist_image_paths"), DEFAULT_RIGHT_WRIST_IMAGE_PATHS
    )
    merged["proprio_paths"] = _normalize_paths(merged.get("proprio_paths"), DEFAULT_PROPRIO_PATHS)
    merged["input_image_size"] = int(merged["input_image_size"])
    merged["num_open_loop_steps"] = max(1, int(merged["num_open_loop_steps"]))
    merged["request_timeout_sec"] = float(merged["request_timeout_sec"])
    merged["sleep_after_action_sec"] = float(merged["sleep_after_action_sec"])
    merged["strict_action_dim"] = bool(merged["strict_action_dim"])
    merged["return_all_query_results"] = bool(merged["return_all_query_results"])
    merged["swap_bgr_to_rgb"] = bool(merged["swap_bgr_to_rgb"])
    merged["use_fixed_task_description"] = bool(merged["use_fixed_task_description"])
    merged["verbose"] = bool(merged["verbose"])
    return merged


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


def _deep_get(node: Any, path: str) -> Any:
    current = node
    for part in path.split("."):
        if current is None:
            return None
        current = _lookup_child(current, part)
    return current


def _first_present(node: Any, candidate_paths: Sequence[str]) -> Any:
    for path in candidate_paths:
        value = _deep_get(node, path)
        if value is not None:
            return value
    return None


def _resize_square(image: np.ndarray, target_size: int) -> np.ndarray:
    if target_size <= 0 or (image.shape[0] == target_size and image.shape[1] == target_size):
        return image
    return np.asarray(Image.fromarray(image).resize((target_size, target_size), resample=Image.BICUBIC))


def _coerce_uint8_rgb_image(image: Any, target_size: int, swap_bgr_to_rgb: bool = False) -> np.ndarray:
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
        raise ValueError("Expected a non-empty float vector.")
    return array


def _extract_first_string(obj: Any) -> str:
    if isinstance(obj, str):
        text = obj.strip()
        return text if text else ""
    if isinstance(obj, Mapping):
        for key in ("full_description", "task_description", "instruction", "language", "prompt", "text", "description"):
            if key in obj:
                text = _extract_first_string(obj[key])
                if text:
                    return text
        for value in obj.values():
            text = _extract_first_string(value)
            if text:
                return text
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            text = _extract_first_string(item)
            if text:
                return text
    return ""


def _infer_task_name(task_env: Any, model: "RemoteRobotWinPolicy", observation: Any) -> str:
    for candidate in (
        model.config.get("task_name"),
        getattr(task_env, "task_name", None),
        getattr(getattr(task_env, "__class__", None), "__name__", None),
        _deep_get(observation, "task_name"),
        _deep_get(observation, "meta.task_name"),
    ):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return ""


def _resolve_fixed_task_description(task_name: str, config: Mapping[str, Any]) -> str:
    if config.get("default_task_description"):
        return str(config["default_task_description"]).strip()

    text = _FIXED_TASK_DESCRIPTION_CACHE.get(task_name, "")
    if text:
        return text

    repo_root = Path(__file__).resolve().parents[2]
    task_instruction_path = repo_root / "description" / "task_instruction" / f"{task_name}.json"
    if task_instruction_path.exists():
        with open(task_instruction_path, "r", encoding="utf-8") as file:
            task_data = json.load(file)
        text = _extract_first_string(task_data)

    if not text and task_name:
        text = task_name.replace("_", " ")

    if text:
        _FIXED_TASK_DESCRIPTION_CACHE[task_name] = text
    return text


def _compose_joint_state_from_prefix(root: Any, prefix: str) -> np.ndarray | None:
    left_arm = _deep_get(root, f"{prefix}.left_arm")
    left_gripper = _deep_get(root, f"{prefix}.left_gripper")
    right_arm = _deep_get(root, f"{prefix}.right_arm")
    right_gripper = _deep_get(root, f"{prefix}.right_gripper")
    if left_arm is None or left_gripper is None or right_arm is None or right_gripper is None:
        return None
    left_arm = _coerce_float_vector(left_arm)
    right_arm = _coerce_float_vector(right_arm)
    left_gripper = _coerce_float_vector(left_gripper)
    right_gripper = _coerce_float_vector(right_gripper)
    return np.concatenate([left_arm, left_gripper[:1], right_arm, right_gripper[:1]], axis=0)


def _extract_proprio(observation: Any, proprio_paths: Sequence[str]) -> np.ndarray:
    for path in proprio_paths:
        value = _deep_get(observation, path)
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
        "Could not extract RobotWin proprio from the current observation. "
        "Override `proprio_paths` in deploy_policy.yml."
    )


def encode_obs(observation: Any, usr_args: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Convert a raw RobotWin observation into the ALOHA-style payload expected by the server."""
    config = _merge_config(usr_args)
    primary = _first_present(observation, config["primary_image_paths"])
    left = _first_present(observation, config["left_wrist_image_paths"])
    right = _first_present(observation, config["right_wrist_image_paths"])

    if primary is None:
        raise KeyError("Could not locate RobotWin primary camera image in observation.")
    if left is None:
        raise KeyError("Could not locate RobotWin left wrist image in observation.")
    if right is None:
        raise KeyError("Could not locate RobotWin right wrist image in observation.")

    return {
        "primary_image": _coerce_uint8_rgb_image(
            primary, config["input_image_size"], swap_bgr_to_rgb=config["swap_bgr_to_rgb"]
        ),
        "left_wrist_image": _coerce_uint8_rgb_image(
            left, config["input_image_size"], swap_bgr_to_rgb=config["swap_bgr_to_rgb"]
        ),
        "right_wrist_image": _coerce_uint8_rgb_image(
            right, config["input_image_size"], swap_bgr_to_rgb=config["swap_bgr_to_rgb"]
        ),
        "proprio": _extract_proprio(observation, config["proprio_paths"]),
    }


class RemoteRobotWinPolicy:
    """Thin client that queries the Cosmos /act server from inside RobotWin."""

    def __init__(self, usr_args: dict[str, Any] | None = None):
        self.config = _merge_config(usr_args)
        self.obs_cache: deque[dict[str, np.ndarray]] = deque(maxlen=1)
        self.last_response: dict[str, Any] | list[Any] | None = None
        self.last_instruction = self.config["default_task_description"]
        self.query_count = 0

    def update_obs(self, obs: dict[str, np.ndarray]) -> None:
        self.obs_cache.clear()
        self.obs_cache.append(obs)

    def reset(self) -> None:
        self.obs_cache.clear()
        self.last_response = None
        self.last_instruction = self.config["default_task_description"]

    def _prepare_action(self, action: Any, expected_dim: int) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != expected_dim:
            if not self.config["strict_action_dim"] and action_array.size > expected_dim:
                action_array = action_array[:expected_dim]
            else:
                raise ValueError(
                    f"Server returned action dim {action_array.size}, but RobotWin proprio dim is {expected_dim}."
                )
        action_array = np.nan_to_num(action_array, nan=0.0, posinf=0.0, neginf=0.0)
        return action_array

    def get_action(self, task_instruction: str | None = None) -> list[np.ndarray]:
        if not self.obs_cache:
            raise RuntimeError("No observation cached. Call update_obs() before get_action().")

        instruction = (task_instruction or self.last_instruction or self.config["default_task_description"]).strip()
        if not instruction:
            raise ValueError(
                "Task instruction is empty. Expose TASK_ENV.get_instruction() or set default_task_description."
            )

        payload = dict(self.obs_cache[-1])
        payload["task_description"] = instruction
        payload["return_all_query_results"] = self.config["return_all_query_results"]

        response = requests.post(
            self.config["server_endpoint"],
            json=payload,
            timeout=self.config["request_timeout_sec"],
        )
        response.raise_for_status()
        data = response.json()
        self.last_response = data
        self.last_instruction = instruction
        self.query_count += 1

        if isinstance(data, list):
            actions = data
        elif isinstance(data, Mapping):
            actions = data.get("actions")
        else:
            raise TypeError(f"Unexpected /act response type: {type(data)}")

        if not isinstance(actions, Sequence) or len(actions) == 0:
            raise ValueError("Server returned an empty action chunk.")

        expected_dim = int(payload["proprio"].shape[0])
        max_actions = self.config["num_open_loop_steps"]
        return [self._prepare_action(action, expected_dim) for action in actions[:max_actions]]


def get_model(usr_args: dict[str, Any] | None = None) -> RemoteRobotWinPolicy:
    """RobotWin entrypoint used by script/eval_policy.py."""
    model = RemoteRobotWinPolicy(usr_args)
    if model.config["verbose"]:
        print(
            "[CosmosPolicyRemote] Connected to "
            f"{model.config['server_endpoint']} (open_loop={model.config['num_open_loop_steps']})"
        )
    return model


def update_obs(model: RemoteRobotWinPolicy, obs: Any) -> None:
    if isinstance(obs, Mapping) and {"primary_image", "left_wrist_image", "right_wrist_image", "proprio"} <= set(obs):
        encoded = {
            "primary_image": np.asarray(obs["primary_image"], dtype=np.uint8),
            "left_wrist_image": np.asarray(obs["left_wrist_image"], dtype=np.uint8),
            "right_wrist_image": np.asarray(obs["right_wrist_image"], dtype=np.uint8),
            "proprio": np.asarray(obs["proprio"], dtype=np.float32),
        }
    else:
        encoded = encode_obs(obs, model.config)
    model.update_obs(encoded)


def get_action(model: RemoteRobotWinPolicy, task_instruction: str | None = None) -> list[np.ndarray]:
    return model.get_action(task_instruction)


def reset_model(model: RemoteRobotWinPolicy) -> None:
    model.reset()


def _get_task_instruction(task_env: Any, model: RemoteRobotWinPolicy, observation: Any) -> str:
    if model.config["use_fixed_task_description"]:
        task_name = _infer_task_name(task_env, model, observation)
        text = _resolve_fixed_task_description(task_name, model.config)
        if text:
            return text
        raise ValueError(
            "use_fixed_task_description is enabled, but the adapter could not resolve a task-level description. "
            "Set default_task_description explicitly or ensure task_name is available."
        )

    if hasattr(task_env, "get_instruction"):
        instruction = task_env.get_instruction()
        if instruction is not None:
            text = str(instruction).strip()
            if text:
                return text

    for path in ("task_description", "instruction", "language", "language_instruction"):
        value = _deep_get(observation, path)
        if value is not None:
            text = str(value).strip()
            if text:
                return text

    if model.config["default_task_description"]:
        return str(model.config["default_task_description"]).strip()

    raise ValueError(
        "Could not infer task instruction from TASK_ENV / observation. "
        "Set default_task_description in deploy_policy.yml."
    )


def eval(TASK_ENV: Any, model: RemoteRobotWinPolicy, observation: Any) -> Any:
    """RobotWin evaluation callback that executes one action chunk open-loop."""
    task_instruction = _get_task_instruction(TASK_ENV, model, observation)
    update_obs(model, observation)
    action_chunk = get_action(model, task_instruction)

    for action in action_chunk:
        TASK_ENV.take_action(action, action_type=model.config["action_type"])
        if model.config["sleep_after_action_sec"] > 0:
            time.sleep(model.config["sleep_after_action_sec"])
        if hasattr(TASK_ENV, "get_obs"):
            next_observation = TASK_ENV.get_obs()
            update_obs(model, next_observation)

    return TASK_ENV
