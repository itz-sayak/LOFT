import os
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from .utils import *


class NPZFolderTest(Dataset):
    def __init__(self, root: str, features: Optional[str] = None):
        """
        Initialize the NPZFolderTest dataset.

        Args:
            root (str): Path to the root directory.
            features (Optional[str]): Features to include in the dataset.
        """
        super().__init__()
        self.root = root
        self.features = features
        self.files = load_npz_folder(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = self.files[index]
        points = data["points"]
        features = data[self.features] if self.features is not None else None

        # normalize the points
        center = np.mean(points, axis=0)
        points -= center
        scale = np.max(np.linalg.norm(points, axis=1))
        points /= scale

        data = {
            "idx": index,
            "train_points": torch.from_numpy(points).float(),
            "train_points_center": center,
            "train_points_scale": scale,
        }

        if features is not None:
            data["features"] = torch.from_numpy(features).float()

        return data


class ScanNetPP_NPZ(Dataset):
    def __init__(
        self,
        root: str,
        mode: str = "training",
        additional_features: bool = False,
        augment: bool = False,
        transform: Optional[Callable] = None,
        train_split_path: Optional[str] = None,
        val_split_path: Optional[str] = None,
        features_root: Optional[str] = None,
        feature_type: Optional[str] = None,
    ):
        """
        Initialize the ScanNetPP_NPZ dataset.

        Args:
            root (str): Path to the root directory.
            mode (str): Mode of the dataset (training or validation).
            additional_features (bool): Whether to include additional features.
            augment (bool): Whether to apply augmentation.
            transform (Optional[Callable]): Transform to apply to the data.
        """
        super().__init__()
        self.root = root
        self.mode = mode
        self.additional_features = additional_features
        self.augment = augment if mode == "training" else False
        self.transform = transform

        self.features_root = features_root or root
        self.feature_type = feature_type
        self._features_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._max_cached_scenes = 4

        splits_path = "splits"
        train_split_path = train_split_path or os.path.join(splits_path, "snpp_train.txt")
        val_split_path = val_split_path or os.path.join(splits_path, "snpp_val.txt")

        with open(train_split_path, "r") as f:
            train_scans = f.read().splitlines()
        with open(val_split_path, "r") as f:
            val_scans = f.read().splitlines()

        # setup the splits
        if mode == "training":
            scans = train_scans
        elif mode == "validation":
            scans = val_scans
        else:
            raise NotImplementedError(f"Mode {mode} not implemented!")

        # scan paths for ply files
        folders = os.listdir(self.root)
        logger.info(f"Setting up preprocessed {mode} scannet dataset")
        folders = [f for f in folders if os.path.isdir(os.path.join(self.root, f))]
        folders = [f for f in folders if f in scans]

        self.scene_batches = []

        for folder in folders:
            folder_files = os.listdir(os.path.join(self.root, folder))
            points_paths = sorted([f for f in folder_files if f.startswith("points") and f.endswith(".npz")])
            for points in points_paths:
                data = {
                    "scene": folder,
                    "npz": os.path.join(self.root, folder, points),
                }
                self.scene_batches.append(data)

        logger.info(f"Loaded {len(self.scene_batches)} batches")

    def _get_feature_path(self, scene_id: str) -> str:
        if not self.feature_type or str(self.feature_type).lower() in {"none", "null", ""}:
            raise ValueError("feature_type is required to load external features")
        fname = f"{self.feature_type}_iphone.npy"
        return os.path.join(self.features_root, scene_id, "features", fname)

    def _get_scene_features_memmap(self, scene_id: str) -> np.ndarray:
        # Simple LRU cache of memmaps to avoid re-opening huge files every batch.
        if scene_id in self._features_cache:
            self._features_cache.move_to_end(scene_id)
            return self._features_cache[scene_id]

        fpath = self._get_feature_path(scene_id)
        feats = np.load(fpath, mmap_mode="r")
        self._features_cache[scene_id] = feats
        if len(self._features_cache) > self._max_cached_scenes:
            self._features_cache.popitem(last=False)
        return feats

    def __len__(self):
        return len(self.scene_batches)


class ScanNetPP(ScanNetPP_NPZ):
    def __init__(
        self,
        root: str,
        mode: str = "training",
        additional_features: bool = False,
        augment: bool = False,
        transform: Optional[Callable] = None,
        train_split_path: Optional[str] = None,
        val_split_path: Optional[str] = None,
        features_root: Optional[str] = None,
        feature_type: Optional[str] = None,
    ):
        """
        Initialize the ScanNetPP dataset.

        Args:
            root (str): Path to the root directory.
            mode (str): Mode of the dataset (training or validation).
            additional_features (bool): Whether to include additional features.
            augment (bool): Whether to apply augmentation.
            transform (Optional[Callable]): Transform to apply to the data.
        """
        super().__init__(
            root=root,
            mode=mode,
            additional_features=additional_features,
            augment=augment,
            transform=transform,
            train_split_path=train_split_path,
            val_split_path=val_split_path,
            features_root=features_root,
            feature_type=feature_type,
        )

    def __getitem__(self, index):
        batch_data = {}
        while True:
            try:
                data = self.scene_batches[index]
                scene_id = data["scene"]
                with np.load(data["npz"]) as data_dict:
                    clean = data_dict["clean"]
                    noisy = data_dict["noisy"]
                    idxs = data_dict["idxs"] if "idxs" in data_dict.files else None
                    features_npz = data_dict["features"] if "features" in data_dict.files else None
                    has_center = "center" in data_dict.files
                    has_scale = "scale" in data_dict.files
                    center_file = data_dict["center"] if has_center else None
                    scale_file = data_dict["scale"] if has_scale else None
                break
            except Exception as e:
                logger.error(f"Failed to load data {data}")
                logger.exception(e)
                index = np.random.randint(0, self.__len__())

        # extract the points
        points_noisy = noisy[:, :3]
        points_clean = clean[:, :3]

        # extract the colors
        if noisy.shape[1] > 3:
            batch_data["noisy_colors"] = torch.from_numpy(noisy[:, 3:]).float()
        if clean.shape[1] > 3:
            batch_data["clean_colors"] = torch.from_numpy(clean[:, 3:]).float()

        # append the features if they are available
        if self.additional_features:
            if features_npz is not None:
                batch_data["noisy_features"] = torch.from_numpy(features_npz).float()
            else:
                if idxs is None:
                    raise KeyError(
                        f"NPZ is missing both 'features' and 'idxs' keys; can't load external features for scene {scene_id}"
                    )

                # Load per-scene features and gather by idxs.
                feats_all = self._get_scene_features_memmap(scene_id)
                idxs = idxs.astype(np.int64, copy=False)
                if feats_all.ndim != 2:
                    raise ValueError(f"Expected features to be 2D array, got shape {feats_all.shape} for scene {scene_id}")

                # Support both layouts: (C, N) and (N, C)
                if feats_all.shape[0] < feats_all.shape[1]:
                    feats = feats_all[:, idxs].T
                else:
                    feats = feats_all[idxs]

                batch_data["noisy_features"] = torch.from_numpy(np.asarray(feats)).float()

        # normalize the point coordinates
        if center_file is None:
            center = np.mean(points_noisy, axis=0)
            points_noisy -= center
            points_clean -= center
        else:
            center = center_file

        if scale_file is None:
            scale = np.max(np.linalg.norm(points_noisy, axis=1))
            points_noisy /= scale
            points_clean /= scale
        else:
            scale = scale_file

        # random rotation augmentation
        if self.augment and np.random.rand() < 0.5:
            points_noisy, theta = random_rotate_pointcloud_horizontally(points_noisy)
            points_clean, theta = random_rotate_pointcloud_horizontally(points_clean, theta=theta)

        # shuffle the point indexes
        rand_idxs = np.arange(points_noisy.shape[0])
        np.random.shuffle(rand_idxs)

        points_noisy = points_noisy[rand_idxs]
        points_clean = points_clean[rand_idxs]
        if "noisy_colors" in batch_data:
            batch_data["noisy_colors"] = batch_data["noisy_colors"][rand_idxs]
        if "clean_colors" in batch_data:
            batch_data["clean_colors"] = batch_data["clean_colors"][rand_idxs]
        if "noisy_features" in batch_data:
            batch_data["noisy_features"] = batch_data["noisy_features"][rand_idxs]

        if self.transform is not None:
            points_noisy = self.transform(points_noisy)
            points_clean = self.transform(points_clean)

        batch_data["idx"] = index
        # IMPORTANT: Keep naming consistent with training code:
        #   noisy_points = input (iphone)
        #   clean_points = target (faro)
        batch_data["noisy_points"] = torch.from_numpy(points_noisy).float()
        batch_data["clean_points"] = torch.from_numpy(points_clean).float()
        batch_data["center"] = center
        batch_data["scale"] = scale

        return batch_data
