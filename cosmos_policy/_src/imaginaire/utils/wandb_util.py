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

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import attrs
import wandb
import wandb.util
from omegaconf import DictConfig

from cosmos_policy._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_policy._src.imaginaire.utils import distributed, log as imaginaire_log, object_store
from cosmos_policy._src.imaginaire.utils.easy_io import easy_io

if TYPE_CHECKING:
    from cosmos_policy._src.imaginaire.config import CheckpointConfig, Config, JobConfig
    from cosmos_policy._src.imaginaire.model import ImaginaireModel


_ACTIVE_WANDB_ID: str | None = None
_LAST_LOGGED_STEP: int | None = None
_WANDB_DISABLED_REASON: str | None = None


def _get_job_config(config: Config) -> JobConfig:
    if isinstance(config.job, DictConfig):
        from cosmos_policy._src.imaginaire.config import JobConfig

        return JobConfig(**config.job)
    return config.job


def _disable_wandb(reason: str, exc: Exception | None = None) -> None:
    global _WANDB_DISABLED_REASON
    if _WANDB_DISABLED_REASON is not None:
        return
    _WANDB_DISABLED_REASON = reason
    if exc is None:
        imaginaire_log.warning(f"Disabling WandB integration: {reason}")
    else:
        imaginaire_log.warning(f"Disabling WandB integration: {reason}: {exc}")


def is_active() -> bool:
    return _WANDB_DISABLED_REASON is None and wandb.run is not None


@distributed.rank0_only
def init_wandb(config: Config, model: ImaginaireModel | None = None):  # noqa: ANN201
    """Initialize WandB once and reuse the existing run when possible."""

    del model

    global _ACTIVE_WANDB_ID
    global _LAST_LOGGED_STEP

    if _WANDB_DISABLED_REASON is not None:
        return None

    config_job = _get_job_config(config)
    if str(config_job.wandb_mode).lower() == "disabled":
        return None

    config_checkpoint = config.checkpoint
    wandb_id = _read_wandb_id(config_job, config_checkpoint)
    if wandb_id is None:
        wandb_id = wandb.util.generate_id()
        _write_wandb_id(config_job, config_checkpoint, wandb_id=wandb_id)
        imaginaire_log.info(f"Generating new wandb ID: {wandb_id}")
    else:
        imaginaire_log.info(f"Resuming with existing wandb ID: {wandb_id}")

    if wandb.run is not None:
        active_run_id = getattr(wandb.run, "id", None)
        if active_run_id == wandb_id:
            _ACTIVE_WANDB_ID = wandb_id
            return wandb.run
        imaginaire_log.warning(
            f"WandB run {active_run_id} is already active while initializing {wandb_id}; "
            "finishing the previous run before reinitializing."
        )
        try:
            wandb.finish()
        except Exception as exc:  # pragma: no cover - defensive path
            _disable_wandb("failed to finish the previously active WandB run", exc)
            return None

    local_safe_yaml_fp = LazyConfig.save_yaml(config, os.path.join(config_job.path_local, "config.yaml"))
    if os.path.exists(local_safe_yaml_fp):
        config_resolved = easy_io.load(local_safe_yaml_fp)
    else:
        config_resolved = attrs.asdict(config)

    try:
        run = wandb.init(
            force=True,
            id=wandb_id,
            project=config_job.project,
            group=config_job.group,
            name=config_job.name,
            config=config_resolved,
            dir=config_job.path_local,
            resume="allow",
            mode=config_job.wandb_mode,
        )
        wandb.define_metric("iteration")
        _ACTIVE_WANDB_ID = wandb_id
        _LAST_LOGGED_STEP = None
        return run
    except Exception as exc:  # pragma: no cover - defensive path
        _disable_wandb("wandb.init failed", exc)
        return None


@distributed.rank0_only
def update_config(values: dict[str, Any], allow_val_change: bool = True) -> bool:
    if not is_active():
        return False
    try:
        wandb.run.config.update(values, allow_val_change=allow_val_change)
        return True
    except Exception as exc:  # pragma: no cover - defensive path
        _disable_wandb("wandb.run.config.update failed", exc)
        return False


@distributed.rank0_only
def log(data: dict[str, Any], step: int | None = None) -> bool:
    global _LAST_LOGGED_STEP

    if not is_active():
        return False

    if step is not None and _LAST_LOGGED_STEP is not None and step < _LAST_LOGGED_STEP:
        imaginaire_log.warning(
            f"Skipping WandB log for stale step {step} because the latest logged step is {_LAST_LOGGED_STEP}."
        )
        return False

    try:
        wandb.log(data, step=step)
        if step is not None:
            _LAST_LOGGED_STEP = step if _LAST_LOGGED_STEP is None else max(_LAST_LOGGED_STEP, step)
        return True
    except Exception as exc:  # pragma: no cover - defensive path
        _disable_wandb("wandb.log failed", exc)
        return False


@distributed.rank0_only
def alert(title: str, text: str, level: Any) -> bool:
    if not is_active():
        return False
    try:
        if isinstance(level, str):
            alert_level = getattr(wandb.AlertLevel, level.upper(), wandb.AlertLevel.ERROR)
        else:
            alert_level = level
        wandb.alert(title=title, text=text, level=alert_level)
        return True
    except Exception as exc:  # pragma: no cover - defensive path
        _disable_wandb("wandb.alert failed", exc)
        return False


@distributed.rank0_only
def finish() -> bool:
    global _ACTIVE_WANDB_ID
    global _LAST_LOGGED_STEP

    if wandb.run is None:
        _ACTIVE_WANDB_ID = None
        _LAST_LOGGED_STEP = None
        return False

    try:
        wandb.finish()
        _ACTIVE_WANDB_ID = None
        _LAST_LOGGED_STEP = None
        return True
    except Exception as exc:  # pragma: no cover - defensive path
        _disable_wandb("wandb.finish failed", exc)
        return False


def _read_wandb_id(config_job: JobConfig, config_checkpoint: CheckpointConfig) -> str | None:
    """Read the W&B job ID. If it doesn't exist, return None.

    Args:
        config_wandb (JobConfig): The config object for the W&B logger.
        config_checkpoint (CheckpointConfig): The config object for the checkpointer.

    Returns:
        wandb_id (str | None): W&B job ID.
    """
    wandb_id = None
    if config_checkpoint.load_from_object_store.enabled:
        object_store_loader = object_store.ObjectStore(config_checkpoint.load_from_object_store)
        wandb_id_path = f"{config_job.path}/wandb_id.txt"
        if object_store_loader.object_exists(key=wandb_id_path):
            wandb_id = object_store_loader.load_object(key=wandb_id_path, type="text").strip()
    else:
        wandb_id_path = f"{config_job.path_local}/wandb_id.txt"
        if os.path.isfile(wandb_id_path):
            wandb_id = open(wandb_id_path).read().strip()
    return wandb_id


def _write_wandb_id(config_job: JobConfig, config_checkpoint: CheckpointConfig, wandb_id: str) -> None:
    """Write the generated W&B job ID.

    Args:
        config_wandb (JobConfig): The config object for the W&B logger.
        config_checkpoint (CheckpointConfig): The config object for the checkpointer.
        wandb_id (str): The W&B job ID.
    """
    content = f"{wandb_id}\n"
    if config_checkpoint.save_to_object_store.enabled:
        object_store_saver = object_store.ObjectStore(config_checkpoint.save_to_object_store)
        wandb_id_path = f"{config_job.path}/wandb_id.txt"
        object_store_saver.save_object(content, key=wandb_id_path, type="text")
    else:
        wandb_id_path = f"{config_job.path_local}/wandb_id.txt"
        with open(wandb_id_path, "w") as file:
            file.write(content)
