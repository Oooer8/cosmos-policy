"""Batch-run RobotWin evaluation for the RoboTwin Agilex task set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from cosmos_policy.experiments.robot.robotwin.setup_robotwin_eval import install_policy


DEFAULT_TASKS = (
    "adjust_bottle",
    "click_alarmclock",
    "grab_roller",
    "handover_block",
    "lift_pot",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_bread_basket",
    "place_bread_skillet",
    "place_empty_cup",
    "place_phone_stand",
    "press_stapler",
    "put_object_cabinet",
    "scan_object",
    "shake_bottle",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_two",
    "stamp_seal",
)


@dataclass
class TaskEvalResult:
    task_name: str
    returncode: int
    success_rate: float | None
    result_file: str | None
    duration_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the Cosmos->RobotWin adapter once and evaluate a batch of RoboTwin tasks."
    )
    parser.add_argument("--robotwin_repo", required=True, help="Absolute path to the local RoboTwin repo.")
    parser.add_argument(
        "--task_names",
        nargs="+",
        default=None,
        help="Optional subset of task names. Defaults to the 20 RoboTwin Agilex tasks used by cosmos-policy.",
    )
    parser.add_argument(
        "--task_config",
        default="demo_clean",
        help="RobotWin task config to evaluate, for example demo_clean or demo_randomized.",
    )
    parser.add_argument(
        "--rollouts_per_task",
        type=int,
        default=10,
        help="Override RoboTwin episode_num for each task.",
    )
    parser.add_argument(
        "--ckpt_setting",
        default="1",
        help="Checkpoint selection argument expected by RoboTwin's eval_policy.py harness.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed forwarded to eval_policy.py.")
    parser.add_argument(
        "--seed_start",
        type=int,
        default=None,
        help="Optional explicit starting seed forwarded to eval_policy.py.",
    )
    parser.add_argument(
        "--seed_stride",
        type=int,
        default=1,
        help="Seed stride forwarded to eval_policy.py.",
    )
    parser.add_argument("--gpu_id", default="0", help="GPU id forwarded to eval_policy.py.")
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop the batch immediately when any task evaluation fails.",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Print the per-task commands without executing them.",
    )
    parser.add_argument(
        "--collect_data",
        action="store_true",
        help="Keep RoboTwin data collection enabled instead of forcing evaluation-only mode.",
    )
    parser.add_argument(
        "--save_eval_videos",
        action="store_true",
        help="Keep RoboTwin eval video logging enabled instead of disabling it for batch runs.",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Optional save_path override used only when --collect_data is set.",
    )
    parser.add_argument(
        "--summary_dir",
        default=None,
        help="Directory for the JSON batch summary. Defaults to <robotwin_repo>/eval_result/_batch/<policy>/<task_config>.",
    )

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
        "--use_fixed_task_description",
        action="store_true",
        help=(
            "If set, ignore RoboTwin's per-episode generated instruction and always use a fixed task-level "
            "description during evaluation."
        ),
    )
    parser.add_argument(
        "--instruction_type",
        default="unseen",
        help="RobotWin instruction split type passed through deploy_policy.yml.",
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
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite the target policy directory before reinstalling the adapter.",
    )
    parser.add_argument(
        "--no_overwrite",
        dest="overwrite",
        action="store_false",
        help="Reuse the existing policy directory if it already exists.",
    )
    parser.set_defaults(overwrite=True)

    args = parser.parse_args()
    if args.rollouts_per_task <= 0:
        raise SystemExit("--rollouts_per_task must be positive.")
    if args.seed_stride <= 0:
        raise SystemExit("--seed_stride must be positive.")
    return args


def _validate_tasks(robotwin_repo: Path, task_names: tuple[str, ...]) -> None:
    missing_tasks = [task_name for task_name in task_names if not (robotwin_repo / "envs" / f"{task_name}.py").exists()]
    if missing_tasks:
        missing = ", ".join(sorted(missing_tasks))
        raise FileNotFoundError(f"Task env files not found in RoboTwin for: {missing}")


def _build_eval_command(
    args: argparse.Namespace,
    policy_name: str,
    task_name: str,
) -> list[str]:
    command = [
        sys.executable,
        "script/eval_policy.py",
        "--config",
        f"policy/{policy_name}/deploy_policy.yml",
        "--overrides",
        "--task_name",
        task_name,
        "--task_config",
        args.task_config,
        "--ckpt_setting",
        str(args.ckpt_setting),
        "--seed",
        str(args.seed),
        "--policy_name",
        policy_name,
        "--episode_num",
        str(args.rollouts_per_task),
        "--seed_stride",
        str(args.seed_stride),
        "--collect_data",
        str(bool(args.collect_data)),
        "--eval_video_log",
        str(bool(args.save_eval_videos)),
    ]

    if args.seed_start is not None:
        command.extend(["--seed_start", str(args.seed_start)])
    if args.collect_data and args.save_path:
        command.extend(["--save_path", str(args.save_path)])
    return command


def _build_subprocess_env(policy_dir: Path, robotwin_repo: Path, gpu_id: str) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [
        str(policy_dir),
        str(policy_dir.parent),
        str(robotwin_repo),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _find_latest_result_file(
    robotwin_repo: Path,
    task_name: str,
    policy_name: str,
    task_config: str,
    ckpt_setting: str,
) -> Path | None:
    result_root = robotwin_repo / "eval_result" / task_name / policy_name / task_config / str(ckpt_setting)
    if not result_root.exists():
        return None
    candidates = sorted(result_root.glob("*/_result.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _parse_success_rate(result_file: Path | None) -> float | None:
    if result_file is None or not result_file.exists():
        return None
    for line in reversed(result_file.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            return float(stripped)
        except ValueError:
            continue
    return None


def _default_summary_dir(robotwin_repo: Path, policy_name: str, task_config: str) -> Path:
    return robotwin_repo / "eval_result" / "_batch" / policy_name / task_config


def main() -> None:
    args = parse_args()
    robotwin_repo = Path(args.robotwin_repo).expanduser().resolve()
    if not robotwin_repo.exists():
        raise FileNotFoundError(f"RobotWin repo does not exist: {robotwin_repo}")

    task_names = tuple(args.task_names) if args.task_names else DEFAULT_TASKS
    _validate_tasks(robotwin_repo, task_names)

    policy_dir = install_policy(args)
    env = _build_subprocess_env(policy_dir, robotwin_repo, args.gpu_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = Path(args.summary_dir).expanduser().resolve() if args.summary_dir else _default_summary_dir(
        robotwin_repo, args.policy_name, args.task_config
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"batch_eval_{timestamp}.json"

    print(f"RobotWin adapter is ready at: {policy_dir}")
    print(f"Tasks to evaluate ({len(task_names)}): {', '.join(task_names)}")
    print(f"Task config: {args.task_config}")
    print(f"Rollouts per task: {args.rollouts_per_task}")
    print(f"Batch summary will be written to: {summary_path}")

    results: list[TaskEvalResult] = []
    failed_tasks: list[str] = []

    for task_name in task_names:
        command = _build_eval_command(args, args.policy_name, task_name)
        print("")
        print(f"[RobotWin batch] {task_name}")
        print(" ".join(command))

        if args.print_only:
            results.append(
                TaskEvalResult(
                    task_name=task_name,
                    returncode=0,
                    success_rate=None,
                    result_file=None,
                    duration_sec=0.0,
                )
            )
            continue

        start_time = time.time()
        completed = subprocess.run(command, cwd=robotwin_repo, env=env, check=False)
        duration_sec = time.time() - start_time

        result_file = _find_latest_result_file(
            robotwin_repo=robotwin_repo,
            task_name=task_name,
            policy_name=args.policy_name,
            task_config=args.task_config,
            ckpt_setting=str(args.ckpt_setting),
        )
        success_rate = _parse_success_rate(result_file)

        results.append(
            TaskEvalResult(
                task_name=task_name,
                returncode=completed.returncode,
                success_rate=success_rate,
                result_file=str(result_file) if result_file is not None else None,
                duration_sec=round(duration_sec, 3),
            )
        )

        if completed.returncode != 0:
            failed_tasks.append(task_name)
            if args.fail_fast:
                break

    summary = {
        "timestamp": timestamp,
        "robotwin_repo": str(robotwin_repo),
        "policy_name": args.policy_name,
        "task_config": args.task_config,
        "rollouts_per_task": args.rollouts_per_task,
        "seed": args.seed,
        "seed_start": args.seed_start,
        "seed_stride": args.seed_stride,
        "gpu_id": str(args.gpu_id),
        "collect_data": bool(args.collect_data),
        "save_eval_videos": bool(args.save_eval_videos),
        "task_names": list(task_names),
        "failed_tasks": failed_tasks,
        "results": [asdict(result) for result in results],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("")
    print(f"Saved batch summary to: {summary_path}")
    if failed_tasks:
        failed_display = ", ".join(failed_tasks)
        raise SystemExit(f"Batch evaluation finished with failures: {failed_display}")

    print("Batch evaluation finished successfully.")


if __name__ == "__main__":
    main()
