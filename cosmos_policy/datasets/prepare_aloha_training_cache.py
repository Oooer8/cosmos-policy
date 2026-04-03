# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Build cached metadata and statistics for lazy ALOHA training.

This script scans the dataset once, without decoding MP4 videos, and writes:
- `aloha_demo_manifest_train.pkl`
- `aloha_demo_manifest_val.pkl` (if val split exists)
- `dataset_statistics.json`
- `dataset_statistics_post_norm.json`
- `t5_embeddings.pkl`

Usage:
    python -m cosmos_policy.datasets.prepare_aloha_training_cache --data_dir /path/to/preprocessed
"""

import argparse
import json
import os
import pickle

import h5py
import numpy as np

from cosmos_policy.datasets.aloha_dataset import _get_aloha_demo_manifest_cache_path, _get_task_description
from cosmos_policy.datasets.dataset_utils import calculate_dataset_statistics, get_hdf5_files, rescale_data
from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings, save_embeddings


def _read_str_dataset(dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _build_manifest(data_dir: str, is_train: bool) -> list[dict]:
    hdf5_files = get_hdf5_files(data_dir, is_train=is_train)
    manifest = []
    for file_path in hdf5_files:
        with h5py.File(file_path, "r") as f:
            obs_group = f["observations"]
            has_raw_images = "images" in obs_group and all(
                cam_key in obs_group["images"] for cam_key in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
            )
            has_video_paths = "video_paths" in obs_group and all(
                cam_key in obs_group["video_paths"] for cam_key in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
            )
            use_mp4 = has_video_paths and not has_raw_images

            entry = dict(
                file_path=file_path,
                command=_get_task_description(file_path, f),
                num_steps=int(f["action"].shape[0]),
                success=True,
                storage_format="mp4" if use_mp4 else "raw_hdf5",
            )
            if use_mp4:
                file_dir = os.path.dirname(file_path)
                entry["video_paths"] = {
                    cam_key: os.path.join(file_dir, _read_str_dataset(obs_group["video_paths"][cam_key]))
                    for cam_key in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
                }
            manifest.append(entry)
    return manifest


def _write_manifest(data_dir: str, is_train: bool, manifest: list[dict]) -> None:
    cache_path = _get_aloha_demo_manifest_cache_path(data_dir, is_train)
    with open(cache_path, "wb") as file:
        pickle.dump(manifest, file)
    print(f"Saved manifest: {cache_path}")


def _compute_train_stats(data_dir: str) -> None:
    train_files = get_hdf5_files(data_dir, is_train=True)
    if len(train_files) == 0:
        train_files = get_hdf5_files(data_dir, is_train=None)

    data = {}
    for idx, file_path in enumerate(train_files):
        with h5py.File(file_path, "r") as f:
            data[idx] = dict(
                actions=f["action"][:].astype(np.float32),
                proprio=f["observations/qpos"][:].astype(np.float32),
            )

    dataset_stats = calculate_dataset_statistics(data)
    dataset_stats_json = {name: value.tolist() for name, value in dataset_stats.items()}
    dataset_stats_path = os.path.join(data_dir, "dataset_statistics.json")
    with open(dataset_stats_path, "w") as file:
        json.dump(dataset_stats_json, file, indent=4)
    print(f"Saved stats: {dataset_stats_path}")

    normalized_data = data
    normalized_data = rescale_data(normalized_data, dataset_stats, "actions")
    normalized_data = rescale_data(normalized_data, dataset_stats, "proprio")
    dataset_stats_post_norm = calculate_dataset_statistics(normalized_data)
    dataset_stats_post_norm_json = {name: value.tolist() for name, value in dataset_stats_post_norm.items()}
    dataset_stats_post_norm_path = os.path.join(data_dir, "dataset_statistics_post_norm.json")
    with open(dataset_stats_post_norm_path, "w") as file:
        json.dump(dataset_stats_post_norm_json, file, indent=4)
    print(f"Saved post-normalization stats: {dataset_stats_post_norm_path}")


def _compute_t5_embeddings(data_dir: str, manifests: list[list[dict]]) -> None:
    unique_commands = sorted({entry["command"] for manifest in manifests for entry in manifest})
    if len(unique_commands) == 0:
        raise ValueError(f"No commands found while generating T5 embeddings for {data_dir}")
    t5_text_embeddings = generate_t5_embeddings(unique_commands)
    save_embeddings(t5_text_embeddings, data_dir)


def main():
    parser = argparse.ArgumentParser(description="Prepare cached metadata/statistics for lazy ALOHA training")
    parser.add_argument("--data_dir", type=str, required=True, help="Root preprocessed ALOHA dataset directory")
    args = parser.parse_args()

    train_manifest = _build_manifest(args.data_dir, is_train=True)
    if len(train_manifest) == 0:
        train_manifest = _build_manifest(args.data_dir, is_train=False)
        if len(train_manifest) == 0:
            raise ValueError(f"No HDF5 files found under {args.data_dir}")
        print("No explicit train split found; cached val/all files into train manifest path.")
    _write_manifest(args.data_dir, is_train=True, manifest=train_manifest)

    val_manifest = _build_manifest(args.data_dir, is_train=False)
    if len(val_manifest) > 0:
        _write_manifest(args.data_dir, is_train=False, manifest=val_manifest)

    _compute_train_stats(args.data_dir)
    _compute_t5_embeddings(args.data_dir, [train_manifest, val_manifest])


if __name__ == "__main__":
    main()
