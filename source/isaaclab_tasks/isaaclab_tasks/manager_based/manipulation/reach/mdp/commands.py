# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_unique, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)


class ReachablePoseCommand(UniformPoseCommand):
    """Pose command (position and orientation) that only generates reachable targets.

    Arms with fewer than 6 DOF cannot reach arbitrary poses, so instead of sampling positions and orientations
    independently, a table of reachable poses is built once at initialization: random joint positions are sampled
    uniformly within the soft joint limits of :attr:`ReachablePoseCommandCfg.joint_names`, the body pose is obtained
    through forward kinematics, and only poses whose position lies inside the workspace box given by
    :attr:`ReachablePoseCommandCfg.ranges` are kept. Commands are then drawn uniformly from this table.

    .. note::
        The forward kinematics ignores collisions, so a target may still require (self-)contact if the workspace box
        overlaps with the robot body.
    """

    cfg: ReachablePoseCommandCfg

    def __init__(self, cfg: ReachablePoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.pose_table_b = self._build_pose_table(env)

    def __str__(self) -> str:
        return super().__str__() + f"\n\tReachable pose table size: {self.pose_table_b.shape[0]}"

    def _build_pose_table(self, env: ManagerBasedEnv) -> torch.Tensor:
        robot = self.robot
        joint_ids, _ = robot.find_joints(self.cfg.joint_names, preserve_order=True)
        limits = robot.data.soft_joint_pos_limits[0, joint_ids]
        lower, upper = limits[:, 0], limits[:, 1]
        ranges = self.cfg.ranges
        box_lower = torch.tensor([ranges.pos_x[0], ranges.pos_y[0], ranges.pos_z[0]], device=self.device)
        box_upper = torch.tensor([ranges.pos_x[1], ranges.pos_y[1], ranges.pos_z[1]], device=self.device)

        default_joint_pos = robot.data.default_joint_pos.clone()
        default_joint_vel = robot.data.default_joint_vel.clone()
        poses, num_poses = [], 0
        for _ in range(self.cfg.max_sampling_iterations):
            joint_pos = default_joint_pos.clone()
            joint_pos[:, joint_ids] = lower + (upper - lower) * torch.rand(
                self.num_envs, len(joint_ids), device=self.device
            )
            robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
            # update the link poses from the joint positions without stepping the physics
            env.sim.physics_sim_view.update_articulations_kinematic()
            body_pose_w = robot.data.body_link_pose_w[:, self.body_idx]
            pos_b, quat_b = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, body_pose_w[:, :3], body_pose_w[:, 3:7]
            )
            inside = ((pos_b >= box_lower) & (pos_b <= box_upper)).all(dim=-1)
            poses.append(torch.cat([pos_b, quat_b], dim=-1)[inside])
            num_poses += int(inside.sum())
            if num_poses >= self.cfg.num_samples:
                break
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        env.sim.physics_sim_view.update_articulations_kinematic()

        if num_poses == 0:
            raise RuntimeError(
                f"No reachable pose of '{self.cfg.body_name}' found inside the workspace box {ranges}."
                " Check the workspace ranges and the joint names."
            )
        table = torch.cat(poses)[: self.cfg.num_samples]
        if self.cfg.make_quat_unique:
            table[:, 3:] = quat_unique(table[:, 3:])
        if num_poses < self.cfg.num_samples:
            logger.warning(
                f"Only {num_poses} reachable poses of '{self.cfg.body_name}' were found (requested"
                f" {self.cfg.num_samples}). Consider increasing 'max_sampling_iterations'."
            )
        return table

    def _resample_command(self, env_ids: Sequence[int]):
        indices = torch.randint(self.pose_table_b.shape[0], (len(env_ids),), device=self.device)
        self.pose_command_b[env_ids] = self.pose_table_b[indices]


@configclass
class ReachablePoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for :class:`ReachablePoseCommand`."""

    class_type: type = ReachablePoseCommand

    joint_names: list[str] = MISSING
    """Names of the joints that move :attr:`body_name` (e.g. the arm joints). Other joints stay at their default."""

    num_samples: int = 50000
    """Number of reachable poses to store in the table."""

    max_sampling_iterations: int = 500
    """Maximum number of forward kinematics batches (of ``num_envs`` samples each) used to fill the table."""

    make_quat_unique: bool = True

    @configclass
    class Ranges:
        """Workspace box (in the root frame) the reachable target positions must lie in."""

        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING
        pos_z: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
