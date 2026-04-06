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

"""Smoke-test a RobotWin checkpoint server using a converted RobotWin / ALOHA-format episode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the /act server with one frame from a converted RobotWin episode.")
    parser.add_argument("--episode_path", required=True, help="Path to a converted RobotWin / ALOHA-format episode HDF5.")
    parser.add_argument("--server_endpoint", default="http://127.0.0.1:8777/act", help="Cosmos Policy /act endpoint.")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to use from the episode.")
    parser.add_argument("--input_image_size", type=int, default=224, help="Image size sent to the server.")
    parser.add_argument(
        "--task_description",
        default="",
        help="Optional override. If omitted, the script reads attrs['task_description'] from the episode.",
    )
    parser.add_argument("--request_timeout_sec", type=float, default=60.0, help="HTTP timeout for the request.")
    parser.add_argument(
        "--return_all_query_results",
        action="store_true",
        help="Request all best-of-N outputs from the server for debugging.",
    )
    return parser.parse_args()


def summarize_response(response: Any) -> None:
    import numpy as np

    if isinstance(response, list):
        actions = np.asarray(response, dtype=np.float32)
        print(f"Received list response with shape: {actions.shape}")
        if actions.size > 0:
            print(f"First action: {actions[0]}")
        return

    if isinstance(response, dict):
        print(f"Received dict response with keys: {sorted(response.keys())}")
        if "actions" in response:
            actions = np.asarray(response["actions"], dtype=np.float32)
            print(f"Action chunk shape: {actions.shape}")
            if actions.size > 0:
                print(f"First action: {actions[0]}")
        if "value_prediction" in response:
            print(f"Value prediction: {response['value_prediction']}")
        if "all_actions" in response:
            print(f"all_actions candidates: {len(response['all_actions'])}")
        return

    print(f"Received unexpected response type: {type(response)}")


def main() -> None:
    args = parse_args()

    try:
        import json_numpy
        import requests
    except ImportError as exc:
        raise ModuleNotFoundError(
            "smoke_test_robotwin_server.py requires `requests` and `json_numpy`. "
            "Install them with: pip install requests json-numpy"
        ) from exc

    from cosmos_policy.experiments.robot.robotwin.robotwin_utils import load_policy_observation_from_episode

    json_numpy.patch()
    observation, task_description = load_policy_observation_from_episode(
        args.episode_path,
        frame_idx=args.frame_idx,
        input_image_size=args.input_image_size,
    )
    observation["task_description"] = args.task_description.strip() or task_description
    observation["return_all_query_results"] = args.return_all_query_results

    if not observation["task_description"]:
        raise ValueError("Task description is empty. Pass --task_description explicitly.")

    response = requests.post(
        args.server_endpoint,
        json=observation,
        timeout=args.request_timeout_sec,
    )
    response.raise_for_status()
    summarize_response(response.json())


if __name__ == "__main__":
    main()
