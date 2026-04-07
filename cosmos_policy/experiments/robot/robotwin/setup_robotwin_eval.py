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
EXPORTED_POLICY_SYMBOLS = (
    "RemoteRobotWinPolicy",
    "encode_obs",
    "eval",
    "get_action",
    "get_model",
    "reset_model",
    "update_obs",
)


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


def _format_symbol_block(indent: str = "    ") -> str:
    return "\n".join(f"{indent}{symbol}," for symbol in EXPORTED_POLICY_SYMBOLS)


def _format_symbol_list() -> str:
    return ", ".join(f'"{symbol}"' for symbol in EXPORTED_POLICY_SYMBOLS)


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
        "# Client-side open-loop execution length before the adapter re-queries /act",
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


def build_package_init() -> str:
    return (
        '"""Compatibility exports for RobotWin policy loading."""\n\n'
        "from .deploy_policy import (\n"
        f"{_format_symbol_block()}\n"
        ")\n\n"
        f"__all__ = [{_format_symbol_list()}]\n"
    )


def build_policy_local_shim() -> str:
    return (
        '"""Compatibility shim for loaders that add this policy directory to ``PYTHONPATH``."""\n\n'
        "from deploy_policy import (\n"
        f"{_format_symbol_block()}\n"
        ")\n\n"
        f"__all__ = [{_format_symbol_list()}]\n"
    )


def build_repo_root_shim(policy_name: str) -> str:
    return (
        f'"""Compatibility shim for loaders that import ``{policy_name}`` from the RoboTwin repo root."""\n\n'
        "from __future__ import annotations\n\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        f'_POLICY_DIR = Path(__file__).resolve().parent / "policy" / "{policy_name}"\n'
        "if str(_POLICY_DIR) not in sys.path:\n"
        "    sys.path.insert(0, str(_POLICY_DIR))\n\n"
        "from deploy_policy import (\n"
        f"{_format_symbol_block()}\n"
        ")\n\n"
        f"__all__ = [{_format_symbol_list()}]\n"
    )


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
export PYTHONPATH="${{SCRIPT_DIR}}:${{SCRIPT_DIR}}/..:${{SCRIPT_DIR}}/../..${{PYTHONPATH:+:${{PYTHONPATH}}}}"

python - "${{policy_name}}" <<'PY'
import importlib
import sys

policy_name = sys.argv[1]
policy_module = importlib.import_module(policy_name)
required_symbols = ("get_model", "eval", "update_obs")
missing_symbols = [name for name in required_symbols if not hasattr(policy_module, name)]
if missing_symbols:
    module_path = getattr(policy_module, "__file__", "<namespace package>")
    raise SystemExit(
        f"Policy module {{policy_name}} loaded from {{module_path}} is missing required exports: {{missing_symbols}}"
    )
print(f"Loaded policy module {{policy_name}} from {{getattr(policy_module, '__file__', '<namespace package>')}}")
PY

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
- `__init__.py`: package-style exports for RobotWin loaders
- `{args.policy_name}.py`: compatibility shim when the policy directory itself is on `PYTHONPATH`
- `deploy_policy.yml`: adapter configuration
- `eval.sh`: convenience wrapper around `python script/eval_policy.py`
- `{args.policy_name}.py` in the RoboTwin repo root: compatibility shim when the repo root is on `PYTHONPATH`

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

    eval_policy_path = robotwin_repo / "script" / "eval_policy.py"
    if not eval_policy_path.exists():
        raise FileNotFoundError(
            "Expected RobotWin entrypoint at "
            f"{eval_policy_path}, but it was not found. Double-check that --robotwin_repo points to the RoboTwin "
            "checkout rather than the cosmos-policy repo."
        )

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
    (policy_dir / "__init__.py").write_text(build_package_init(), encoding="utf-8")
    (policy_dir / f"{args.policy_name}.py").write_text(build_policy_local_shim(), encoding="utf-8")
    (policy_dir / "deploy_policy.yml").write_text(build_yaml(args), encoding="utf-8")
    eval_path = policy_dir / "eval.sh"
    eval_path.write_text(build_eval_sh(args.policy_name), encoding="utf-8")
    eval_path.chmod(0o755)
    (policy_dir / "README.md").write_text(build_readme(args), encoding="utf-8")
    (robotwin_repo / f"{args.policy_name}.py").write_text(build_repo_root_shim(args.policy_name), encoding="utf-8")
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
