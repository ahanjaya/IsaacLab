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
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, quat_rotate_inverse, yaw_quat

from .velocity_nxp_env_cfg import VelocityNXPLowerBodyFlatEnvCfg, VelocityNXPLowerBodyRoughEnvCfg


class VelocityNXPLowerBodyFlatEnv(DirectRLEnv):
    cfg: VelocityNXPLowerBodyFlatEnvCfg | VelocityNXPLowerBodyRoughEnvCfg

    def __init__(
        self,
        cfg: VelocityNXPLowerBodyFlatEnvCfg | VelocityNXPLowerBodyRoughEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
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
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "flat_orientation_l2",
                "dof_pos_limits",
                "termination_penalty",
                "feet_slide",
                "joint_deviation_hip",
                "joint_deviation_torso",
                "orientation_torso",
            ]
        }
        self._metric_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "error_vel_xy",
                "error_vel_yaw",
            ]
        }

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("torso_link")
        self._feet_contact_ids, _ = self._contact_sensor.find_bodies(".*_ankle_roll_link")

        self._feet_body_ids, _ = self._robot.find_bodies(".*_ankle_roll_link")
        self._ankle_joint_ids, _ = self._robot.find_joints([".*_ankle_.*"])
        self._hip_joint_ids, _ = self._robot.find_joints([".*_hip_.*"])
        self._torso_joint_ids, _ = self._robot.find_joints("torso_yaw_joint")
        self._dof_acc_torque_joint_ids, _ = self._robot.find_joints([".*_hip_.*", ".*_knee_joint"])

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, VelocityNXPLowerBodyRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
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

        self.target_body_roll = np.radians(self.cfg.target_body_roll)
        self.target_body_pitch = np.radians(self.cfg.target_body_pitch)
        self.desired_projected_gravity_b = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float,
            device=self.device,
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, VelocityNXPLowerBodyRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        # lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)

        # TODO: Verify this vel_yaw does it equal with the lin_vel_b?
        # track_lin_vel_xy_yaw_frame_exp
        vel_yaw = quat_rotate_inverse(
            yaw_quat(self._robot.data.root_quat_w),
            self._robot.data.root_lin_vel_w[:, :3],
        )
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - vel_yaw[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        # angular velocity tracking
        # ang_vel_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])

        # TODO: Verify this ang_vel_error does it equal with the ang_vel_b?
        # track_ang_vel_z_world_exp
        ang_vel_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_w[:, 2])
        ang_vel_error_mapped = torch.exp(-ang_vel_error / 0.25)

        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)

        # joint torques
        joint_torques = torch.sum(
            torch.square(self._robot.data.applied_torque[:, self._dof_acc_torque_joint_ids]),
            dim=1,
        )

        # joint acceleration
        joint_accel = torch.sum(
            torch.square(self._robot.data.joint_acc[:, self._dof_acc_torque_joint_ids]),
            dim=1,
        )

        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # feet air time positive biped
        air_time = self._contact_sensor.data.current_air_time[:, self._feet_contact_ids]
        contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_contact_ids]
        in_contact = contact_time > 0.0
        in_mode_time = torch.where(in_contact, contact_time, air_time)
        single_stance = torch.sum(in_contact.int(), dim=1) == 1
        air_time_reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
        air_time_reward = torch.clamp(air_time_reward, max=self.cfg.feet_air_time_threshold)
        # no reward for zero command
        air_time_reward *= torch.norm(self._commands[:, :2], dim=1) > 0.1

        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # dof pos limits
        # compute out of limits constraints
        out_of_limits = -(self._robot.data.joint_pos[:, :] - self._robot.data.soft_joint_pos_limits[:, :, 0]).clip(
            max=0.0
        )
        out_of_limits += (self._robot.data.joint_pos[:, :] - self._robot.data.soft_joint_pos_limits[:, :, 1]).clip(
            min=0.0
        )
        out_of_limits = torch.sum(out_of_limits, dim=1)

        # termination penalty
        is_terminated = self.died.float()

        # feet slide
        contacts = (
            self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_ids, :].norm(dim=-1).max(dim=1)[0]
            > 1.0
        )
        body_vel = self._robot.data.body_lin_vel_w[:, self._feet_body_ids, :2]
        feet_slide = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

        # joint deviation hip
        angle = (
            self._robot.data.joint_pos[:, self._hip_joint_ids]
            - self._robot.data.default_joint_pos[:, self._hip_joint_ids]
        )
        joint_deviation_hip = torch.sum(torch.abs(angle), dim=1)

        # joint deviation torso
        angle = (
            self._robot.data.joint_pos[:, self._torso_joint_ids]
            - self._robot.data.default_joint_pos[:, self._torso_joint_ids]
        )
        joint_deviation_torso = torch.sum(torch.abs(angle), dim=1)

        # desired body orientation
        target_roll = torch.tensor([self.target_body_roll], dtype=torch.float, device=self.device).repeat(self.num_envs)
        quat_roll = quat_from_angle_axis(
            target_roll,
            torch.tensor([1, 0, 0], dtype=torch.float, device=self.device),
        )

        target_pitch = torch.tensor([self.target_body_pitch], dtype=torch.float, device=self.device).repeat(
            self.num_envs
        )
        quat_pitch = quat_from_angle_axis(
            target_pitch,
            torch.tensor([0, 1, 0], dtype=torch.float, device=self.device),
        )

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        self.desired_projected_gravity_b = quat_rotate_inverse(desired_base_quat, self._robot.data.GRAVITY_VEC_W)

        orientation_torso = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2] - self.desired_projected_gravity_b[:, :2]),
            dim=1,
        )

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.track_lin_vel_xy_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": ang_vel_error_mapped * self.cfg.track_ang_vel_z_exp_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.lin_vel_z_l2_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_xy_l2_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.dof_torques_l2_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.dof_acc_l2_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_l2_reward_scale * self.step_dt,
            "feet_air_time": air_time_reward * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_l2_reward_scale * self.step_dt,
            "dof_pos_limits": out_of_limits * self.cfg.dof_pos_limits_reward_scale * self.step_dt,
            "termination_penalty": is_terminated * self.cfg.termination_reward_scale * self.step_dt,
            "feet_slide": feet_slide * self.cfg.feet_slide_reward_scale * self.step_dt,
            "joint_deviation_hip": joint_deviation_hip * self.cfg.joint_deviation_hip_reward_scale * self.step_dt,
            "joint_deviation_torso": joint_deviation_torso * self.cfg.joint_deviation_torso_reward_scale * self.step_dt,
            "orientation_torso": orientation_torso * self.cfg.orientation_torso_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        self.died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )

        return self.died, time_out

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

        # Sample new commands
        self._update_command_metrics()
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        self._commands[env_ids, 0] = torch.zeros_like(self._commands[env_ids, 0]).uniform_(-0.5, 1.0)
        self._commands[env_ids, 1] = torch.zeros_like(self._commands[env_ids, 1]).uniform_(-0.5, 0.5)
        self._commands[env_ids, 2] = torch.zeros_like(self._commands[env_ids, 2]).uniform_(-1.0, 1.0)

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

        # Termination logging
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        # Command logging
        extras = dict()
        for key in self._metric_sums.keys():
            extras["Metrics/base_velocity/" + key] = self._metric_sums[key][env_ids]
            self._metric_sums[key][env_ids] = 0.0

        self.extras["log"].update(extras)

    def _update_command_metrics(self):
        # time for which the command was executed
        max_command_time = self.max_episode_length_s
        max_command_step = max_command_time / self.step_dt

        # logs data
        self._metric_sums["error_vel_xy"] += (
            torch.norm(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self._metric_sums["error_vel_yaw"] += (
            torch.abs(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if self.cfg.debug_marker:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

            if not hasattr(self, "robot_visualizer"):
                # -- base state
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Robot/body_cur_pose"
                marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
                self.base_pose_visualizer = VisualizationMarkers(marker_cfg)

                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Robot/body_target_pose"
                marker_cfg.markers["frame"].scale = (0.25, 0.25, 0.25)
                self.target_pose_visualizer = VisualizationMarkers(marker_cfg)

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

                # -- projected gravity
                self.draw_line_interface = omni_debug_draw.acquire_debug_draw_interface()

            # For Debug visualization
            self._yellow_lines_colors = [[1.0, 1.0, 0.0, 1.0]] * self.num_envs
            self._green_lines_colors = [[0.0, 1.0, 0.0, 1.0]] * self.num_envs
            self._line_thicknesses = [5.0] * self.num_envs

            # set their visibility to true
            self.base_pose_visualizer.set_visibility(True)
            self.target_pose_visualizer.set_visibility(True)
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "robot_visualizer"):
                self.base_pose_visualizer.set_visibility(False)
                self.target_pose_visualizer.set_visibility(False)
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
        if not self.cfg.camera_viewer.follow_camera:
            return

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

        self.base_pose_visualizer.visualize(self._robot.data.root_pos_w, self._robot.data.root_quat_w)

        robot_yaw = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)[2]
        roll = torch.zeros_like(robot_yaw)
        pitch = torch.zeros_like(robot_yaw) + self.target_body_pitch
        quat_target = math_utils.quat_from_euler_xyz(roll, pitch, robot_yaw)
        self.target_pose_visualizer.visualize(self._robot.data.root_pos_w[:, :3], quat_target)

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

        # vector projection
        self.draw_line_interface.clear_lines()

        # -- projected gravity
        projected_gravity_vector = self._robot.data.projected_gravity_b + self._robot.data.root_pos_w
        # -- desired projected gravity
        desired_projected_gravity_vector = self.desired_projected_gravity_b + self._robot.data.root_pos_w

        self.draw_line_interface.draw_lines(
            self._robot.data.root_pos_w.tolist(),
            projected_gravity_vector.tolist(),
            self._yellow_lines_colors,
            self._line_thicknesses,
        )
        self.draw_line_interface.draw_lines(
            self._robot.data.root_pos_w.tolist(),
            desired_projected_gravity_vector.tolist(),
            self._green_lines_colors,
            self._line_thicknesses,
        )

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
