# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, quat_unique, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def body_pose_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pose of a body in the asset's root frame.

    Returns ``[x, y, z, qw, qx, qy, qz]`` with shape (num_envs, 7), matching the layout of the pose commands. Since
    ``q`` and ``-q`` describe the same rotation, the quaternion is chosen with a non-negative real part.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    pos_b, quat_b = subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_pos_w[:, body_id],
        asset.data.body_quat_w[:, body_id],
    )
    return torch.cat([pos_b, quat_unique(quat_b)], dim=-1)


def body_dual_quaternion_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pose of a body in the asset's root frame as a unit dual quaternion.

    Returns ``[q_r, q_d]`` with shape (num_envs, 8), quaternions in (w, x, y, z) order, where ``q_r`` is the rotation
    and ``q_d = 0.5 * (0, t) * q_r`` encodes the translation ``t``. Since ``(q_r, q_d)`` and ``(-q_r, -q_d)`` describe
    the same pose, the sign is chosen such that ``q_r`` has a non-negative real part.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    pos_b, quat_b = subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_pos_w[:, body_id],
        asset.data.body_quat_w[:, body_id],
    )
    q_r = quat_unique(quat_b)
    t = torch.cat([torch.zeros_like(pos_b[:, :1]), pos_b], dim=-1)
    q_d = 0.5 * quat_mul(t, q_r)
    return torch.cat([q_r, q_d], dim=-1)
