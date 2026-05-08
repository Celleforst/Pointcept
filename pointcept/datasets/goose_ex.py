"""
GOOSE-Ex 3D dataset.

Author: GitHub Copilot
"""

import glob
import os
from collections.abc import Sequence

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class GooseExDataset(DefaultDataset):
    """GOOSE-Ex LiDAR semantic segmentation dataset."""

    def __init__(
        self,
        lidar_folder="lidar",
        label_folder="labels",
        lidar_suffix="_pcl.bin",
        label_suffix="_goose.label",
        **kwargs,
    ):
        """Initialize the dataset with folder and suffix conventions."""
        self.lidar_folder = lidar_folder
        self.label_folder = label_folder
        self.lidar_suffix = lidar_suffix
        self.label_suffix = label_suffix
        super().__init__(**kwargs)

    def get_data_list(self):
        """Collect LiDAR files for the requested split(s)."""
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            lidar_root = os.path.join(self.data_root, self.lidar_folder, split)
            if not os.path.isdir(lidar_root):
                raise FileNotFoundError(f"Split directory not found: {lidar_root}")
            data_list.extend(
                glob.glob(
                    os.path.join(lidar_root, "**", f"*{self.lidar_suffix}"),
                    recursive=True,
                )
            )
        if not data_list:
            raise FileNotFoundError(
                f"No LiDAR files found under: {self.data_root}/{self.lidar_folder}"
            )
        return sorted(data_list)

    def get_data_name(self, idx):
        """Return a stable sample name for logging and caching."""
        data_path = self.data_list[idx % len(self.data_list)]
        base = os.path.basename(data_path)
        if not base.endswith(self.lidar_suffix):
            raise ValueError(f"Unexpected lidar filename: {data_path}")
        return base[: -len(self.lidar_suffix)]

    def get_split_name(self, idx):
        """Infer the split name from the lidar path."""
        data_path = self.data_list[idx % len(self.data_list)]
        return self._split_from_path(data_path)

    def _split_from_path(self, lidar_path):
        """Extract the split name from a lidar file path."""
        parts = os.path.normpath(lidar_path).split(os.sep)
        if self.lidar_folder not in parts:
            raise ValueError(
                f"Lidar path does not contain '{self.lidar_folder}': {lidar_path}"
            )
        idx = parts.index(self.lidar_folder)
        if idx + 1 >= len(parts):
            raise ValueError(f"Cannot infer split from path: {lidar_path}")
        return parts[idx + 1]

    def _label_path_from_lidar(self, lidar_path):
        """Map a lidar file path to its corresponding label path."""
        split = self._split_from_path(lidar_path)
        lidar_root = os.path.join(self.data_root, self.lidar_folder, split)
        rel_path = os.path.relpath(lidar_path, lidar_root)
        if not rel_path.endswith(self.lidar_suffix):
            raise ValueError(f"Unexpected lidar suffix in: {lidar_path}")
        rel_path = rel_path[: -len(self.lidar_suffix)] + self.label_suffix
        return os.path.join(self.data_root, self.label_folder, split, rel_path)

    def _read_lidar(self, lidar_path):
        """Read a LiDAR frame from a .bin file."""
        scan = np.fromfile(lidar_path, dtype=np.float32)
        if scan.size == 0:
            raise ValueError(f"Empty LiDAR file: {lidar_path}")
        if scan.size % 4 == 0:
            scan = scan.reshape(-1, 4)
        elif scan.size % 5 == 0:
            scan = scan.reshape(-1, 5)
        else:
            raise ValueError(
                f"Unexpected point dimension in {lidar_path}: {scan.size} floats"
            )
        coord = scan[:, :3]
        strength = scan[:, 3:4]
        return coord, strength

    def get_data(self, idx):
        """Load a single LiDAR frame and its label mask."""
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)

        coord, strength = self._read_lidar(data_path)
        label_path = self._label_path_from_lidar(data_path)
        if not os.path.isfile(label_path):
            if self.test_mode:
                segment = (
                    np.ones(coord.shape[0], dtype=np.int32) * self.ignore_index
                )
                instance = np.ones_like(segment) * -1
            else:
                raise FileNotFoundError(f"Label file not found: {label_path}")
        else:
            label = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
            if label.shape[0] != coord.shape[0]:
                raise ValueError(
                    f"Label count {label.shape[0]} does not match points {coord.shape[0]}"
                )
            segment = (label & 0xFFFF).astype(np.int32)
            instance = (label >> 16).astype(np.int32)

        data_dict = dict(
            name=name,
            split=split,
            coord=coord.astype(np.float32),
            strength=strength.astype(np.float32),
            color=np.zeros_like(coord, dtype=np.float32),
            normal=np.zeros_like(coord, dtype=np.float32),
            segment=segment,
            instance=instance,
        )
        return data_dict
