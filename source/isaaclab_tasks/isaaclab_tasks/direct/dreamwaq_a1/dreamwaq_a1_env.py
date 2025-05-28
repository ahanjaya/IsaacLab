# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor

from .dreamwaq_a1_env_cfg import DreamWaQA1FlatEnvCfg


class DreamWaQA1Env(DirectRLEnv):
    cfg: DreamWaQA1FlatEnvCfg

    def __init__(self, cfg: DreamWaQA1FlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_two_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._setup_utility_tensors()

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "flat_orientation_l2",
                "dof_acc_l2",
                "dof_power_l2",
                "base_height",
                "feet_air_time",
                # "feet_clearance",
                "action_rate_l2",
                "smoothness_l2",
                "power_distribution",
                "hip_pos_l2",
                "dof_err_l2",
                "dof_pos_limits_l2",
                "dof_torque_limits_l2",
                "foot_ground",
                "termination",
                "dof_torques_l2",
                "undesired_contacts",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("trunk")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh")

        # Get specific joint indices
        self._hip_ids, _ = self._robot.find_joints(".*_hip_joint")

        # Randomize robot friction
        env_ids = self._robot._ALL_INDICES
        mat_props = self._robot.root_physx_view.get_material_properties()
        mat_props[:, :, :2].uniform_(0.6, 0.8)
        self._robot.root_physx_view.set_material_properties(mat_props, env_ids.cpu())

        # Randomize base mass
        base_id, _ = self._robot.find_bodies("trunk")
        masses = self._robot.root_physx_view.get_masses()
        masses[:, base_id] += torch.zeros_like(masses[:, base_id]).uniform_(-1.0, 3.0)
        self._robot.root_physx_view.set_masses(masses, env_ids.cpu())

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_utility_tensors(self):
        # Camera follow actor
        self.follow_cam_pos = np.array(self.cfg.camera_viewer.pos)
        self.follow_cam_target = np.array(self.cfg.camera_viewer.target)
        self.follow_cam_offset = np.array(self.cfg.camera_viewer.offset)

        self.k_smooth = 0.9
        self.i_follow_env = 0

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_two_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        height_data = None
        commands_scale = torch.tensor(
            [
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.ang_vel,
            ],
            device=self.device,
            requires_grad=False,
        )

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b * self.cfg.normalization.obs_scales.lin_vel,
                    self._robot.data.root_ang_vel_b * self.cfg.normalization.obs_scales.ang_vel,
                    self._robot.data.projected_gravity_b,
                    self._commands * commands_scale,
                    (self._robot.data.joint_pos - self._robot.data.default_joint_pos)
                    * self.cfg.normalization.obs_scales.dof_pos,
                    self._robot.data.joint_vel * self.cfg.normalization.obs_scales.dof_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        # TODO: in IsaacGym we add clip obs_buf here

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / self.cfg.rewards.tracking_sigma)

        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)

        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        # joint power
        joint_power = torch.sum(
            torch.abs(self._robot.data.applied_torque) * torch.abs(self._robot.data.joint_vel), dim=1
        )

        # base height
        # TODO: Refactor actual height based on mean of grid scan under robot
        base_height = self._robot.data.root_pos_w[:, 2]  # - torch.mean(self.measured_heights_under_robot, dim=1)
        base_height_error = torch.square(base_height - self.cfg.rewards.base_height_target)

        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )

        # feet clearance
        # TODO: Add this reward

        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # smoothness
        smoothness = torch.sum(
            torch.square(self._actions - (2 * self._previous_actions) + self._previous_two_actions), dim=1
        )

        # power distribution
        power_distribution = torch.var(self._robot.data.applied_torque * self._robot.data.joint_vel, dim=1)

        # hip position
        hip_pos_err = torch.sum(
            torch.square(
                self._robot.data.joint_pos[:, self._hip_ids] - self._robot.data.default_joint_pos[:, self._hip_ids]
            ),
            dim=1,
        )

        # joint error
        joint_err = torch.sum(torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1)

        # joint position limits
        out_of_limits = -(self._robot.data.joint_pos - self._robot.data.soft_joint_pos_limits[0, :, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self._robot.data.joint_pos - self._robot.data.soft_joint_pos_limits[0, :, 1]).clip(
            min=0.0
        )  # upper limit
        joint_pos_limits_err = torch.sum(out_of_limits, dim=1)

        # joint torque limits
        joint_torque_limits_err = torch.sum(
            (
                torch.abs(self._robot.data.applied_torque)
                - self._robot.data.joint_effort_limits * self.cfg.rewards.soft_torque_limit_percentage
            ).clip(min=0.0),
            dim=1,
        )

        # termination
        termination = self.reset_terminated * ~self.reset_time_outs

        # foot on ground
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        foot_contacts = 1 - torch.sum(is_contact, dim=1) * 0.25
        zero_command_env_ids = torch.norm(self._commands[:, :2], dim=1) <= 0.2
        # penalize when feet are not on the ground when the command is zero
        foot_not_on_ground = foot_contacts * zero_command_env_ids

        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.rewards.scales.lin_vel * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.rewards.scales.yaw_rate * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.rewards.scales.z_vel * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.rewards.scales.ang_vel * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.rewards.scales.flat_orientation * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.rewards.scales.joint_accel * self.step_dt,
            "dof_power_l2": joint_power * self.cfg.rewards.scales.joint_power * self.step_dt,
            "base_height": base_height_error * self.cfg.rewards.scales.base_height * self.step_dt,
            "feet_air_time": air_time * self.cfg.rewards.scales.feet_air_time * self.step_dt,
            # "feet_clearance": feet_clearance * self.cfg.feet_clearance_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.rewards.scales.action_rate * self.step_dt,
            "smoothness_l2": smoothness * self.cfg.rewards.scales.smoothness * self.step_dt,
            "power_distribution": power_distribution * self.cfg.rewards.scales.power_distribution * self.step_dt,
            "hip_pos_l2": hip_pos_err * self.cfg.rewards.scales.hip_pos * self.step_dt,
            "dof_err_l2": joint_err * self.cfg.rewards.scales.joint_err * self.step_dt,
            "dof_pos_limits_l2": joint_pos_limits_err * self.cfg.rewards.scales.joint_pos_limits * self.step_dt,
            "dof_torque_limits_l2": (
                joint_torque_limits_err * self.cfg.rewards.scales.joint_torque_limits * self.step_dt
            ),
            "foot_ground": foot_not_on_ground * self.cfg.rewards.scales.foot_ground * self.step_dt,
            "termination": termination * self.cfg.rewards.scales.termination * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.rewards.scales.joint_torque * self.step_dt,
            "undesired_contacts": contacts * self.cfg.rewards.scales.undesired_contact * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_two_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if self.cfg.debug_marker:
            if not hasattr(self, "robot_visualizer"):
                # -- cmd_vel goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)

                # -- cmd_vel current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)

            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "robot_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update camera view
        self._update_camera_follow_env()

        # update markers
        self._update_debug_marker()

    """
    Internal helpers.
    """

    def _update_camera_follow_env(self):
        actor_pos = self._robot.data.root_pos_w[self.i_follow_env].cpu().numpy()

        # Smooth the camera movement with a moving average.
        new_cam_pos = actor_pos + self.follow_cam_offset
        new_cam_target = actor_pos

        self.follow_cam_pos = self.k_smooth * self.follow_cam_pos + (1 - self.k_smooth) * new_cam_pos
        self.follow_cam_target = self.k_smooth * self.follow_cam_target + (1 - self.k_smooth) * new_cam_target

        set_camera_view(self.follow_cam_pos, self.follow_cam_target)

    def _update_debug_marker(self):
        if not self.cfg.debug_marker:
            return
        robot_root_pose_w = self._robot.data.root_state_w
        robot_root_pose_w[:, 2] += 0.1

        # get marker location
        # -- base state
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self._robot.data.root_lin_vel_b[:, :2])

        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self._robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
