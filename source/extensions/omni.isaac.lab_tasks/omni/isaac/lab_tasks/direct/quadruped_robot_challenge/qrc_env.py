# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import torchvision
from collections.abc import Sequence

import cv2
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass, isaacgym_utils

##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG  # isort: skip


@configclass
class QRCEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 40.0
    decimation = 4
    action_scale = 0.25
    num_actions = 12
    num_observations = 105  # Total
    num_proprio_observations = 46
    num_task_objectives = 3

    robot_type = "UNITREE_A1"
    debug_vis = True
    show_depth = True
    follow_env = True
    debug_marker = False

    obs_scales = {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "joint_pos": 1.0,
        "joint_vel": 0.05,
    }
    clip_observations = 100.0
    clip_actions = 100.0
    encoder_history_length = 10
    foot_contact_threshold = 1.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=debug_marker,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=10, env_spacing=2.0, replicate_physics=True
    )

    # robot
    if robot_type == "UNITREE_A1":
        robot: ArticulationCfg = UNITREE_A1_CFG.replace(
            prim_path="/World/envs/env_.*/Robot"
        )
        camera_offset = CameraCfg.OffsetCfg(
            pos=(0.27, 0.0, 0.03), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
        )
    else:
        raise ValueError(
            f"Invalid robot type: {robot_type}, available options: ['UNITREE_A1', 'UNITREE_GO2']"
        )

    # contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=1,
        update_period=0.0,
        debug_vis=debug_marker,
    )

    # camera
    depth_cam = {
        "original_size": (106, 60),  # (width, height)
        "resize_to": (87, 58),
        "far_clip": 2.0,
    }

    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/trunk/front_cam",
        height=depth_cam["original_size"][1],
        width=depth_cam["original_size"][0],
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),
        ),
        offset=camera_offset,
    )

    # reward scales
    dummy_reward_scale = 1.0


class QRCEnv(DirectRLEnv):
    cfg: QRCEnvCfg

    def __init__(self, cfg: QRCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(
            self.num_envs, self.cfg.num_actions, device=self.device
        )

        self._joint_dof_idx, _ = self._robot.find_joints(".*")
        # [4, 8, 12, 16], ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self._foot_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._n_foot = len(self._foot_ids)

        self._setup_utility_tensors()
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._camera = Camera(self.cfg.camera)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["camera"] = self._camera
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

    def _pre_physics_step(self, actions: torch.Tensor):
        clip_actions = self.cfg.clip_actions
        isaaclab_actions = isaacgym_utils.reorder_joints_isaacgym_to_isaaclab(actions)
        clip_actions = torch.clip(isaaclab_actions, -clip_actions, clip_actions)
        self._actions = (
            clip_actions * self.cfg.action_scale + self._robot.data.default_joint_pos
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self._actions)

    def _get_foot_contacts(self) -> torch.Tensor:
        net_contact_forces = self._contact_sensor.data.net_forces_w

        if net_contact_forces is None:
            return torch.zeros(
                self.num_envs, self._n_foot, dtype=torch.bool, device=self.device
            )

        foot_contacts = torch.zeros(
            self.num_envs, self._n_foot, dtype=torch.bool, device=self.device
        )
        for i, foot_id in enumerate(self._foot_ids):
            foot_contacts[:, i] = (
                torch.norm(net_contact_forces[:, foot_id, :], dim=-1)
                >= self.cfg.foot_contact_threshold
            )

        # ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        return foot_contacts

    def _get_proprioceptive_obs(self) -> torch.Tensor:
        isaacgym_joint_pos_proprio = isaacgym_utils.reorder_joints_isaaclab_to_isaacgym(
            (self._robot.data.joint_pos - self._robot.data.default_joint_pos)
            * self.cfg.obs_scales["joint_pos"],
        )
        isaacgym_joint_vel_proprio = isaacgym_utils.reorder_joints_isaaclab_to_isaacgym(
            self._robot.data.joint_vel * self.cfg.obs_scales["joint_vel"]
        )

        obs = torch.cat(
            [
                self._robot.data.root_ang_vel_b * self.cfg.obs_scales["ang_vel"],
                self._robot.data.projected_gravity_b,
                isaacgym_joint_pos_proprio,
                isaacgym_joint_vel_proprio,
                self._actions,
                self._get_foot_contacts(),
            ],
            dim=-1,
        )

        return torch.clip(obs, -self.cfg.clip_observations, self.cfg.clip_observations)

    def _update_depth_map(self) -> None:
        # TODO: Implement the depth map update function based on interval
        depth_frames = self._camera.data.output["distance_to_image_plane"]
        curr_depth_frames = torch.zeros(
            self.num_envs,
            self.cfg.depth_cam["resize_to"][1],
            self.cfg.depth_cam["resize_to"][0],
            device=self.device,
            requires_grad=False,
        )

        # post-process and normalize depth maps
        for i in range(self.num_envs):
            depth_frame = depth_frames[i]
            depth_frame[depth_frame == torch.inf] = 0.0
            depth_frame = depth_frame / self.cfg.depth_cam["far_clip"]
            depth_frame[depth_frame > 0] = 1 - depth_frame[depth_frame > 0]
            curr_depth_frames[i, :, :] = self._depth_resize_transform(
                depth_frame.unsqueeze(0)
            )

        self._depth_map_buf = torch.roll(self._depth_map_buf, 1, dims=1)
        self._depth_map_buf[:, 0, :, :] = curr_depth_frames[:]

    def _get_observations(self) -> dict:
        # TODO: Implement the observation function
        proprio_obs = self._get_proprioceptive_obs()
        task_obs = self._commands[:]

        self._encoder_obs_hist_buf = torch.roll(self._encoder_obs_hist_buf, 1, dims=1)
        self._encoder_obs_hist_buf[:, 0, :] = proprio_obs

        self._update_depth_map()

        observations = {
            "proprio_obs": proprio_obs,
            "task_obs": task_obs,
            "encoder_obs_hist": self._encoder_obs_hist_buf,
            "depth_map_obs": self._depth_map_buf,
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        dummy_rew = torch.zeros(self.num_envs)

        rewards = {
            "track_dummy": dummy_rew * self.cfg.dummy_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out).bool()

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        self._camera.reset(env_ids)
        self._contact_sensor.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Reset buff
        self._actions[env_ids] = 0.0
        self._encoder_obs_hist_buf[env_ids, :, :] = 0.0
        self._depth_map_buf[env_ids, :, :, :] = 0.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if self.cfg.show_depth:
            cv2.namedWindow("Robot Depth Frame", cv2.WINDOW_NORMAL)

    def _show_depth_frame(self):
        if not self.cfg.show_depth:
            return

        depth_frame = (
            self._depth_map_buf[self.i_follow_env, 0, :, :].detach().cpu().numpy()
        )
        depth_frame = cv2.resize(
            depth_frame, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow("Robot Depth Frame", depth_frame)
        cv2.waitKey(1)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._robot.is_initialized:
            return

        # Camera view follow actor
        self._update_camera_follow_env()

        # Robot front depth cam
        self._show_depth_frame()

    ##############################################################################
    def _setup_utility_tensors(self):
        # Camera follow actor
        self.follow_cam_pos = np.array([0.7, 1.5, 0.7])
        self.follow_cam_target = np.array([0.5, 0.0, 0])
        self.follow_cam_offset = np.array([0.0, -3.0, 2.0])
        self.k_smooth = 0.9
        self.i_follow_env = 0

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(
            self.num_envs, self.cfg.num_actions, device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(
            self.num_envs, self.cfg.num_task_objectives, device=self.device
        )
        # TODO: Remove this override commands
        self._commands[:, 1:] = 1.0

        self._encoder_obs_hist_buf = torch.zeros(
            (
                self.num_envs,
                self.cfg.encoder_history_length,
                self.cfg.num_proprio_observations,
            ),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self._depth_map_buf = torch.zeros(
            (
                self.num_envs,
                1,
                self.cfg.depth_cam["resize_to"][1],
                self.cfg.depth_cam["resize_to"][0],
            ),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self._depth_resize_transform = torchvision.transforms.Resize(
            (self.cfg.depth_cam["resize_to"][1], self.cfg.depth_cam["resize_to"][0]),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        )

    def _update_camera_follow_env(self):
        if not self.cfg.follow_env:
            return

        actor_pos = self._robot.data.root_pos_w[self.i_follow_env].cpu().numpy()

        # Smooth the camera movement with a moving average.
        new_cam_pos = actor_pos + self.follow_cam_offset
        new_cam_target = actor_pos

        self.follow_cam_pos = (
            self.k_smooth * self.follow_cam_pos + (1 - self.k_smooth) * new_cam_pos
        )
        self.follow_cam_target = (
            self.k_smooth * self.follow_cam_target
            + (1 - self.k_smooth) * new_cam_target
        )

        set_camera_view(self.follow_cam_pos, self.follow_cam_target)
