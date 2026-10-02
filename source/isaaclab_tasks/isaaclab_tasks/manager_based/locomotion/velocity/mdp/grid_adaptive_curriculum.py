# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grid Adaptive Curriculum (GAC) for sampling velocity commands.

Implementation based on "Rapid Locomotion via Reinforcement Learning" (2022) from
Margolis et. al. https://arxiv.org/abs/2205.02824
"""

from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from isaaclab.utils import configclass


@configclass
class GridAdaptiveConfig:
    """Configuration for the :class:`GridAdaptiveCurriculum`."""

    resolution_x: float = 0.05
    """Bin resolution for the linear-x velocity axis (in m/s)."""

    resolution_w: float = 0.05
    """Bin resolution for the angular-z velocity axis (in rad/s)."""

    init_x_range: tuple[float, float] = (-0.2, 0.2)
    """Initial range of linear-x velocity commands marked as available for sampling."""

    init_w_range: tuple[float, float] = (-0.2, 0.2)
    """Initial range of angular-z velocity commands marked as available for sampling."""

    limit_x_range: tuple[float, float] = (-1.0, 1.0)
    """Absolute limits of the linear-x velocity grid."""

    limit_w_range: tuple[float, float] = (-1.0, 1.0)
    """Absolute limits of the angular-z velocity grid."""

    linvel_thresh: float = 0.8
    """Minimum average linear velocity tracking reward for a command to be marked as successful."""

    angvel_thresh: float = 0.8
    """Minimum average angular velocity tracking reward for a command to be marked as successful."""


class GridAdaptiveCurriculum:
    """Grid adaptive curriculum that grows the sampled command range based on tracking success.

    The curriculum discretizes the (linear-x, angular-z) velocity command space into a grid of bins.
    Only bins marked as "available" can be sampled. Whenever a command is tracked successfully, its
    neighboring bins (one step further out along each axis) are marked as available, gradually
    expanding the range of commands that get sampled over the course of training.
    """

    def __init__(self, config: GridAdaptiveConfig):
        self._config = config
        self._available_bins = np.empty((0, 2), dtype=np.int64)

        self._pdg = self._init_probability_density_grid()
        self._updates_count = 0

    def save_to_disk(self, path: str) -> None:
        with open(path, "wb") as fp:
            print(f"Saving GAC probability density grid to `{path}`.")
            pickle.dump(self._pdg, fp)

    def load_from_disk(self, path: str) -> None:
        with open(path, "rb") as fp:
            print(f"Loading GAC probability density grid from `{path}`.")
            self._pdg = pickle.load(fp)
            self._available_bins = np.array(list(self._pdg.keys()))

    def sample_command(self, n: int) -> np.ndarray:
        """Sample ``n`` (linear-x, angular-z) velocity commands from the available bins."""
        cmds_n = self._available_bins.shape[0]
        rand_inds = np.random.randint(low=0, high=cmds_n, size=(n,))
        sampled_bins = self._available_bins[rand_inds]
        return self._bins_to_cmds(sampled_bins)

    def update_neighbors_from_cmds(self, cmds: np.ndarray) -> None:
        """Mark the neighboring bins of the given successfully tracked commands as available."""
        self._updates_count += len(cmds)

        bins = self._cmds_to_bins(cmds)
        for x, w in bins:
            if x < 0 or x >= self._x_bins or w < 0 or w >= self._w_bins:
                print(f"Warning, invalid bin: `({x}, {w})`")
                continue

            if x + 1 < self._x_bins:
                self._pdg[(x + 1, w)] = 1.0
            if x - 1 >= 0:
                self._pdg[(x - 1, w)] = 1.0
            if w + 1 < self._w_bins:
                self._pdg[(x, w + 1)] = 1.0
            if w - 1 >= 0:
                self._pdg[(x, w - 1)] = 1.0

        self._available_bins = np.array(list(self._pdg.keys()))

    def get_num_updates(self) -> int:
        return self._updates_count

    def get_num_available_bins(self) -> int:
        return len(self._pdg)

    def save_grid_density(self, path: str) -> None:
        """Plot the current probability density grid."""
        sns.set_style("darkgrid")

        complete_grid = np.zeros((self._x_bins, self._w_bins), dtype=np.float32)
        for cell in self._pdg.keys():
            complete_grid[cell[0], cell[1]] = 1.0

        plt.subplots(figsize=(20, 10))
        yticks = [f"[{i}]: {self._x_to_cmd(i):.2f}" for i in range(self._x_bins)]
        xticks = [f"[{i}]: {self._w_to_cmd(i):.2f}" for i in range(self._w_bins)]
        color_palette = sns.color_palette("crest", as_cmap=True)
        sns.heatmap(
            complete_grid,
            cbar=False,
            cmap=color_palette,
            xticklabels=xticks,
            yticklabels=yticks,
            linewidths=0.5,
            linecolor="gray",
            square=True,
            annot=True,
            vmin=0.0,
            vmax=1.0,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved grid curriculum plot to `{path}`.")

    """
    Internal helpers.
    """

    def _init_probability_density_grid(self) -> dict[tuple[int, int], float]:
        self._x_span = self._config.limit_x_range[1] + self._config.resolution_x - self._config.limit_x_range[0]
        self._w_span = self._config.limit_w_range[1] + self._config.resolution_w - self._config.limit_w_range[0]

        self._x_bins = int(self._x_span / self._config.resolution_x)
        self._w_bins = int(self._w_span / self._config.resolution_w)

        pdg_dict = {}
        for init_x in np.arange(
            self._config.init_x_range[0],
            self._config.init_x_range[1] + 1e-4,
            self._config.resolution_x,
        ):
            for init_w in np.arange(
                self._config.init_w_range[0],
                self._config.init_w_range[1] + 1e-4,
                self._config.resolution_w,
            ):
                x_bin, w_bin = self._cmd_to_bin((init_x, init_w))
                pdg_dict[(x_bin, w_bin)] = 1.0

        self._available_bins = np.array(list(pdg_dict.keys()))
        return pdg_dict

    def _cmd_to_bin(self, cmd: tuple[float, float]) -> tuple[int, int]:
        x, w = cmd
        x_bin = ((x - self._config.limit_x_range[0]) / self._x_span) * self._x_bins
        w_bin = ((w - self._config.limit_w_range[0]) / self._w_span) * self._w_bins
        return round(x_bin), round(w_bin)

    def _cmds_to_bins(self, cmds: np.ndarray) -> np.ndarray:
        bins = np.zeros(cmds.shape, dtype=np.int64)
        bins[:, 0] = np.round(((cmds[:, 0] - self._config.limit_x_range[0]) / self._x_span) * self._x_bins)
        bins[:, 1] = np.round(((cmds[:, 1] - self._config.limit_w_range[0]) / self._w_span) * self._w_bins)
        return bins

    def _x_to_cmd(self, x_bin: int) -> float:
        return (x_bin / self._x_bins) * self._x_span + self._config.limit_x_range[0]

    def _w_to_cmd(self, w_bin: int) -> float:
        return (w_bin / self._w_bins) * self._w_span + self._config.limit_w_range[0]

    def _bins_to_cmds(self, bins: np.ndarray) -> np.ndarray:
        cmds = np.zeros(bins.shape, dtype=np.float64)
        cmds[:, 0] = (bins[:, 0] / self._x_bins) * self._x_span + self._config.limit_x_range[0]
        cmds[:, 1] = (bins[:, 1] / self._w_bins) * self._w_span + self._config.limit_w_range[0]
        return cmds
