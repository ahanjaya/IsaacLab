# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for velocity-based locomotion environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def estimated_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Placeholder for estimated linear velocity. Returns zeros of shape (num_envs, 3).

    This observation is populated by the linear velocity estimator during training/inference.
    The function returns a zero tensor as a placeholder that will be replaced by the estimator's output.

    Args:
        env: The environment instance.
        asset_cfg: The scene entity configuration for the robot asset.

    Returns:
        Zero tensor of shape (num_envs, 3) representing [x, y, z] linear velocity estimates.
    """
    # Return zeros as placeholder - will be filled by the lin vel estimator
    return torch.zeros(env.num_envs, 3, device=env.device)
