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

"""Install a Cosmos Policy remote-eval adapter into a local RoboTwin / RobotWin checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_POLICY_NAME = "CosmosPolicyRemote"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write deploy_policy.py / deploy_policy.yml / eval.sh into a RobotWin checkout."
    )
    parser.add_argument("--robotwin_repo", required=True, help="Absolute path to the local RoboTwin / RobotWin repo.")
    parser.add_argument(
        "--policy_name",
        default=DEFAULT_POLICY_NAME,
        help=f"Policy directory name to create under <robotwin_repo>/policy (default: {DEFAULT_POLICY_NAME}).",
    )
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
        help="Number of actions from the server chunk to execute before re-querying.",
    )
    parser.add_argument(
        "--request_timeout_sec",
        type=float,
        default=60.0,
        help="HTTP timeout for each /act request.",
    )
    parser.add_argument(
        "--action_type",
        default="qpos",
        help="RobotWin action mode passed to TASK_ENV.take_action(action, action_type=...).",
    )
    parser.add_argument(
        "--default_task_description",
        default="",
        help="Optional fallback instruction if TASK_ENV does not expose get_instruction().",
    )
    parser.add_argument(
        "--instruction_type",
        default="unseen",
        help="RobotWin instruction split type expected by script/eval_policy.py (default: unseen).",
    )
    parser.add_argument(
        "--return_all_query_results",
        action="store_true",
        help="Request all best-of-N query metadata from the server. Usually keep this off for normal eval.",
    )
    parser.add_argument(
        "--allow_action_dim_mismatch",
        action="store_true",
        help="If set, truncate over-long action vectors instead of failing on dimension mismatch.",
    )
    parser.add_argument(
        "--swap_bgr_to_rgb",
        action="store_true",
        help="Swap channel order before sending observations if RobotWin cameras are BGR.",
    )
    parser.add_argument(
        "--sleep_after_action_sec",
        type=float,
        default=0.0,
        help="Optional sleep after each TASK_ENV.take_action() call.",
    )
    parser.add_argument(
        "--primary_image_path",
        action="append",
        default=None,
        help="Override / add a dot-path candidate for the primary camera. Repeat to supply multiple.",
    )
    parser.add_argument(
        "--left_wrist_image_path",
        action="append",
        default=None,
        help="Override / add a dot-path candidate for the left wrist camera. Repeat to supply multiple.",
    )
    parser.add_argument(
        "--right_wrist_image_path",
        action="append",
        default=None,
        help="Override / add a dot-path candidate for the right wrist camera. Repeat to supply multiple.",
    )
    parser.add_argument(
        "--proprio_path",
        action="append",
        default=None,
        help="Override / add a dot-path candidate for proprio / qpos. Repeat to supply multiple.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing policy directory.")
    return parser.parse_args()


def _template_dir() -> Path:
    return Path(__file__).resolve().parent


def build_yaml(args: argparse.Namespace) -> str:
    def _yaml_list(name: str, values: list[str] | None) -> str:
        if not values:
            return ""
        lines = [f"{name}:"]
        for value in values:
            lines.append(f'  - "{value}"')
        return "\n".join(lines)

    parts = [
        "# Basic experiment configuration (required by RoboTwin eval_policy.py)",
        "policy_name: null",
        "task_name: null",
        "task_config: null",
        "ckpt_setting: null",
        "seed: null",
        f'instruction_type: "{args.instruction_type}"',
        "",
        "# Cosmos Policy remote adapter configuration",
        f'server_endpoint: "{args.server_endpoint}"',
        f"request_timeout_sec: {args.request_timeout_sec}",
        f"input_image_size: {args.input_image_size}",
        f"num_open_loop_steps: {args.num_open_loop_steps}",
        f"return_all_query_results: {'true' if args.return_all_query_results else 'false'}",
        f'action_type: "{args.action_type}"',
        f"strict_action_dim: {'false' if args.allow_action_dim_mismatch else 'true'}",
        f"swap_bgr_to_rgb: {'true' if args.swap_bgr_to_rgb else 'false'}",
        f"sleep_after_action_sec: {args.sleep_after_action_sec}",
        f'default_task_description: "{args.default_task_description}"',
        _yaml_list("primary_image_paths", args.primary_image_path),
        _yaml_list("left_wrist_image_paths", args.left_wrist_image_path),
        _yaml_list("right_wrist_image_paths", args.right_wrist_image_path),
        _yaml_list("proprio_paths", args.proprio_path),
    ]
    return "\n".join(part for part in parts if part) + "\n"


def build_eval_sh(policy_name: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail

policy_name="{policy_name}"
task_name="${{1:?task_name is required}}"
task_config="${{2:?task_config is required}}"
ckpt_setting="${{3:?ckpt_setting is required}}"
seed="${{4:?seed is required}}"
gpu_id="${{5:?gpu_id is required}}"

export CUDA_VISIBLE_DEVICES="${{gpu_id}}"
echo -e "\\033[33mgpu id (to use): ${{gpu_id}}\\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "${{SCRIPT_DIR}}/../.."

PYTHONWARNINGS=ignore::UserWarning \\
python script/eval_policy.py \\
  --config "policy/${{policy_name}}/deploy_policy.yml" \\
  --overrides \\
  --task_name "${{task_name}}" \\
  --task_config "${{task_config}}" \\
  --ckpt_setting "${{ckpt_setting}}" \\
  --seed "${{seed}}" \\
  --policy_name "${{policy_name}}"
"""


def build_readme(args: argparse.Namespace) -> str:
    return f"""# {args.policy_name}

This policy directory was generated from the `cosmos-policy` repo to let RoboTwin / RobotWin
query a remote Cosmos Policy server over HTTP.

## Files

- `deploy_policy.py`: RobotWin policy adapter
- `deploy_policy.yml`: adapter configuration
- `eval.sh`: convenience wrapper around `python script/eval_policy.py`

## Expected Server

Start the server from the `cosmos-policy` repo with a RobotWin Agilex checkpoint and point it at:

`{args.server_endpoint}`

## Typical Usage

```bash
cd {Path(args.robotwin_repo).resolve()}
bash policy/{args.policy_name}/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
```

If RobotWin camera names or proprio fields differ from the defaults, edit `deploy_policy.yml`.
"""


def install_policy(args: argparse.Namespace) -> Path:
    robotwin_repo = Path(args.robotwin_repo).expanduser().resolve()
    if not robotwin_repo.exists():
        raise FileNotFoundError(f"RobotWin repo does not exist: {robotwin_repo}")

    policy_root = robotwin_repo / "policy"
    if not policy_root.exists():
        raise FileNotFoundError(
            f"Expected policy/ directory under RobotWin repo, but did not find: {policy_root}"
        )

    policy_dir = policy_root / args.policy_name
    if policy_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Policy directory already exists: {policy_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(policy_dir)

    policy_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(_template_dir() / "template_deploy_policy.py", policy_dir / "deploy_policy.py")
    (policy_dir / "deploy_policy.yml").write_text(build_yaml(args), encoding="utf-8")
    eval_path = policy_dir / "eval.sh"
    eval_path.write_text(build_eval_sh(args.policy_name), encoding="utf-8")
    eval_path.chmod(0o755)
    (policy_dir / "README.md").write_text(build_readme(args), encoding="utf-8")
    return policy_dir


def main() -> None:
    args = parse_args()
    policy_dir = install_policy(args)

    print(f"Installed RobotWin policy adapter at: {policy_dir}")
    print("")
    print("Next steps:")
    print("1. Activate your RobotWin environment in the RobotWin repo.")
    print("2. Start the Cosmos Policy server from the cosmos-policy repo.")
    print(
        "3. Run: "
        f"bash {policy_dir / 'eval.sh'} <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>"
    )


if __name__ == "__main__":
    main()
