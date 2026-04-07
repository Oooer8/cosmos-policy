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

"""Prepare and optionally launch RobotWin evaluation through the official RoboTwin harness."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from cosmos_policy.experiments.robot.robotwin.setup_robotwin_eval import install_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the Cosmos->RobotWin adapter and run the official RobotWin eval wrapper."
    )
    parser.add_argument("--robotwin_repo", required=True, help="Absolute path to the local RoboTwin / RobotWin repo.")
    parser.add_argument("--task_name", required=True, help="RobotWin task name passed to eval.sh.")
    parser.add_argument("--task_config", required=True, help="RobotWin task config / embodiment passed to eval.sh.")
    parser.add_argument(
        "--ckpt_setting",
        default="1",
        help="Checkpoint selection argument expected by RobotWin's eval_policy.py harness.",
    )
    parser.add_argument("--seed", default="0", help="Evaluation seed forwarded to eval.sh.")
    parser.add_argument("--gpu_id", default="0", help="GPU id forwarded to eval.sh / CUDA_VISIBLE_DEVICES.")

    parser.add_argument("--policy_name", default="CosmosPolicyRemote", help="Policy directory name to create.")
    parser.add_argument(
        "--server_endpoint",
        default="http://127.0.0.1:8777/act",
        help="Cosmos Policy server endpoint exposed by aloha.deploy.",
    )
    parser.add_argument("--input_image_size", type=int, default=224, help="Square image size sent to the server.")
    parser.add_argument(
        "--num_open_loop_steps",
        type=int,
        default=50,
        help="Client-side open-loop chunk length before re-querying /act.",
    )
    parser.add_argument("--request_timeout_sec", type=float, default=60.0, help="HTTP timeout for /act.")
    parser.add_argument("--action_type", default="qpos", help="RobotWin action mode.")
    parser.add_argument("--default_task_description", default="", help="Fallback task description.")
    parser.add_argument(
        "--use_task_name_as_instruction",
        action="store_true",
        help=(
            "If set, ignore RoboTwin's per-episode generated instruction and always use the task name converted "
            'to plain text, for example "open_microwave" -> "open microwave".'
        ),
    )
    parser.add_argument(
        "--instruction_type",
        default="unseen",
        help="RobotWin instruction split type passed through deploy_policy.yml (default: unseen).",
    )
    parser.add_argument("--return_all_query_results", action="store_true", help="Request full best-of-N metadata.")
    parser.add_argument(
        "--allow_action_dim_mismatch",
        action="store_true",
        help="If set, truncate over-long action vectors instead of failing on mismatch.",
    )
    parser.add_argument("--swap_bgr_to_rgb", action="store_true", help="Swap channel order before sending obs.")
    parser.add_argument("--sleep_after_action_sec", type=float, default=0.0, help="Sleep after each env action.")
    parser.add_argument("--primary_image_path", action="append", default=None, help="Override primary camera path.")
    parser.add_argument("--left_wrist_image_path", action="append", default=None, help="Override left camera path.")
    parser.add_argument("--right_wrist_image_path", action="append", default=None, help="Override right camera path.")
    parser.add_argument("--proprio_path", action="append", default=None, help="Override proprio path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the target policy directory if it exists.")

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run eval.sh after writing the adapter. Without this flag the command is only printed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_dir = install_policy(args)
    eval_script = policy_dir / "eval.sh"
    command = [
        "bash",
        str(eval_script),
        str(args.task_name),
        str(args.task_config),
        str(args.ckpt_setting),
        str(args.seed),
        str(args.gpu_id),
    ]

    print(f"RobotWin adapter is ready at: {policy_dir}")
    print("Evaluation command:")
    print(" ".join(command))

    if not args.execute:
        print("")
        print("`--execute` was not set, so nothing else was run.")
        print("Activate your RobotWin environment first, then re-run with `--execute` or paste the command above.")
        return

    robotwin_repo = Path(args.robotwin_repo).expanduser().resolve()
    subprocess.run(command, cwd=robotwin_repo, check=True)


if __name__ == "__main__":
    main()
