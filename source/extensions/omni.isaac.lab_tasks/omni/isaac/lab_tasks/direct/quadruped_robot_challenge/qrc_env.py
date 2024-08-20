# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import os
import cv2
import torchvision
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
)

from omni.isaac.lab.devices import Se2Keyboard
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass, isaacgym_utils

##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG  # isort: skip
from omni.isaac.lab.markers.config import SPHERE_MARKER_CFG


@configclass
class QRCEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 100.0
    decimation = 4
    action_scale = 0.25
    num_actions = 12

    num_observations = 105  # Total
    num_proprio_observations = 46
    num_task_objectives = 3

    # visualization
    robot_type = "UNITREE_A1"
    show_depth = False
    show_waypoint = True

    follow_env = False
    debug_marker = False

    spawn = {
        "cube": False,
        "qrc_map": False,
    }

    # observation
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
    reset_contact_threshold = 1.0
    reach_goal_dist_threshold = 0.2

    start_pos = (6.2, 5.0, -np.pi)  # x, y, yaw

    waypoints = (
        [6.2, 3.0],
        [4.0, 3.0],
        [3.0, 4.2],
        [0.25, 4.2],
        [0.25, 3.0],
        [-2.5, 3.0],
        [-3.3, 4.2],
        [-5.75, 4.2],
        [-5.75, 3.0],
        [-6.1, 1.8],
        [-6.1, 0.6],
        # Half
        [-6.1, -0.6],
        [-6.1, -1.8],
        [-5.75, -3.0],
        [-5.75, -4.2],
        [-3.3, -4.2],
        [-2.5, -3.0],
        [0.25, -3.0],
        [0.25, -4.2],
        [3.0, -4.2],
        [4.0, -3.0],
        [6.2, -3.0],
        [6.2, -4.5],
    )

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
        num_envs=1, env_spacing=2.0, replicate_physics=True
    )

    # robot
    if robot_type == "UNITREE_A1":
        robot: ArticulationCfg = UNITREE_A1_CFG.replace(
            prim_path="/World/envs/env_.*/Robot"
        )
        camera_offset = CameraCfg.OffsetCfg(
            pos=(0.27, 0.0, 0.03), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
        )
        camera_prim_path = "/World/envs/env_.*/Robot/trunk/front_cam"
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
        prim_path=camera_prim_path,
        height=depth_cam["original_size"][1],
        width=depth_cam["original_size"][0],
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),
        ),
        offset=camera_offset,
    )

    # add cube
    cube_height = 0.3
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 10.0, cube_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, 0.0, cube_height)),
    )

    # qrc map
    map_usd = os.path.join(
        os.getcwd(),
        "source/extensions/omni.isaac.lab_assets/data/QRC",
        "map_flat.usd",
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

        # Get specific body indices
        self._body_idx, _ = self._contact_sensor.find_bodies("trunk")
        self._foot_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._hip_ids, _ = self._contact_sensor.find_bodies(".*hip")
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._calf_ids, _ = self._contact_sensor.find_bodies(".*calf")
        self._n_foot = len(self._foot_ids)

        self._setup_utility_tensors()

        # debug vis
        self.set_debug_vis(self.sim.has_gui())

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._camera = Camera(self.cfg.camera)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        if self.cfg.spawn["cube"]:
            self._cube = RigidObject(self.cfg.cube)

        if self.sim.has_gui():
            self._teleop_key = Se2Keyboard()
            self._setup_extra_keyboard_callback()

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["camera"] = self._camera
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        if self.cfg.spawn["cube"]:
            self.scene.rigid_objects["cube"] = self._cube

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
        self._actions = torch.clip(actions, -clip_actions, clip_actions)
        isaaclab_actions = isaacgym_utils.reorder_joints_isaacgym_to_isaaclab(
            self._actions
        )
        self._processed_actions = (
            isaaclab_actions * self.cfg.action_scale
            + self._robot.data.default_joint_pos
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

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

    def _compute_task_objective(self) -> torch.Tensor:
        env_ids = self._robot._ALL_INDICES
        current_goal = self._waypoints_pos[env_ids, self._idx_waypoint]

        target_pos_rel = current_goal[:, :2] - self._robot.data.root_pos_w[:, :2]
        norm_dist_to_target = torch.norm(target_pos_rel, dim=-1, keepdim=True)

        target_vec_norm = target_pos_rel / (norm_dist_to_target + 1e-5)
        target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        theta_diff_to_target = math_utils.wrap_to_pi(
            target_yaw - self._robot.data.heading_w
        )

        self._commands[:, 0] = torch.sin(theta_diff_to_target)
        self._commands[:, 1] = torch.cos(theta_diff_to_target)
        self._commands[:, 2] = torch.clip(norm_dist_to_target, 0.0, 1.0)

        # TODO: Override commands
        # self._commands[:, 1:] = 1.0

        reached_goal = (
            norm_dist_to_target.squeeze(-1) <= self.cfg.reach_goal_dist_threshold
        )
        reached_goal_ids = reached_goal.nonzero(as_tuple=False)

        if len(reached_goal_ids) > 0:
            self._idx_waypoint[reached_goal_ids] += 1
            self._idx_waypoint[reached_goal_ids] %= self._total_waypoint

        return self._commands

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
        proprio_obs = self._get_proprioceptive_obs()
        task_obs = self._compute_task_objective()

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
        is_finished = self._idx_waypoint >= self._total_waypoint

        died = torch.zeros_like(time_out)
        is_contacts = self._check_contacts()

        terminated = died | is_contacts
        done = time_out | is_finished

        return terminated, done

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        self._camera.reset(env_ids)
        self._contact_sensor.reset(env_ids)

        if self.cfg.spawn["cube"]:
            self._cube.reset(env_ids)

        if self.sim.has_gui():
            self._teleop_key.reset()
        super()._reset_idx(env_ids)

        # Reset buff
        self._actions[env_ids] = 0.0
        self._encoder_obs_hist_buf[env_ids, :, :] = 0.0
        self._depth_map_buf[env_ids, :, :, :] = 0.0
        self._idx_waypoint[env_ids] = 0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :7] += self.qrc_start_pose

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not debug_vis:
            return

        if self.cfg.show_depth_frame:
            cv2.namedWindow("Robot Depth Frame", cv2.WINDOW_NORMAL)

        if self.cfg.show_waypoint:
            self._show_all_waypoint()

            # current waypoint
            marker_cfg = SPHERE_MARKER_CFG.replace(
                prim_path="/Visuals/Waypoints",
            )
            marker_cfg.markers["sphere"].radius = 0.15
            marker_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                metallic=0.5,
            )
            self.current_waypoint_visualizer = VisualizationMarkers(marker_cfg)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._robot.is_initialized:
            return

        # Camera view follow actor
        self._update_camera_follow_env()

        # Robot front depth cam
        self._show_depth_frame()

        # Robot current waypoint target
        self._show_current_waypoint()

    ##############################################################################
    def _init_start_pose(self):
        start_pos = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        start_pos[:, 0] = self.cfg.start_pos[0]  # x
        start_pos[:, 1] = self.cfg.start_pos[1]  # y

        ori_euler_xyz = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        ori_euler_xyz[:, -1] = self.cfg.start_pos[2]
        ori_quat = math_utils.quat_from_euler_xyz(
            ori_euler_xyz[:, 0], ori_euler_xyz[:, 1], ori_euler_xyz[:, 2]
        )

        return torch.cat(
            [
                start_pos,
                ori_quat,
            ],
            dim=-1,
        )

    def _setup_utility_tensors(self):
        # Camera follow actor
        self.follow_cam_pos = np.array([0.7, 1.5, 0.7])
        self.follow_cam_target = np.array([0.5, 0.0, 0])
        self.follow_cam_offset = np.array([0.0, -3.0, 2.0])
        self.k_smooth = 0.9
        self.i_follow_env = 0

        self.qrc_start_pose = self._init_start_pose()

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

        self._total_waypoint = len(self.cfg.waypoints)

        self._waypoints_pos = torch.tensor(
            (0.0, 0.0, self._robot.data.default_root_state[:, 2]), device=self.device
        ).repeat(self.num_envs, self._total_waypoint, 1)
        for idx, val in enumerate(self.cfg.waypoints):
            self._waypoints_pos[:, idx, :2] = torch.tensor(val, dtype=torch.float32)

        self._waypoints_pos[:] += self._terrain.env_origins[:]

        self._idx_waypoint = torch.zeros_like(self.episode_length_buf)

    def _check_contacts(self) -> torch.Tensor:
        # check reset contact for all bodies except foot
        net_contact_forces = self._contact_sensor.data.net_forces_w
        if net_contact_forces is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        body_contact = (
            torch.norm(net_contact_forces[:, self._body_idx, :], dim=1)
            >= self.cfg.reset_contact_threshold
        )
        hip_contact = (
            torch.norm(net_contact_forces[:, self._hip_ids, :], dim=1)
            >= self.cfg.reset_contact_threshold
        )
        thigh_contact = (
            torch.norm(net_contact_forces[:, self._thigh_ids, :], dim=1)
            >= self.cfg.reset_contact_threshold
        )
        calf_contact = (
            torch.norm(net_contact_forces[:, self._calf_ids, :], dim=1)
            >= self.cfg.reset_contact_threshold
        )

        all_contacts = torch.cat(
            [body_contact, hip_contact, thigh_contact, calf_contact], dim=-1
        )

        return torch.any(all_contacts, dim=1)

    def _update_camera_follow_env(self):
        if not self.cfg.follow_env:
            return

        robot_pos = self._robot.data.root_pos_w[self.i_follow_env].cpu().numpy()

        # Smooth the camera movement with a moving average.
        new_cam_pos = robot_pos + self.follow_cam_offset
        new_cam_target = robot_pos

        self.follow_cam_pos = (
            self.k_smooth * self.follow_cam_pos + (1 - self.k_smooth) * new_cam_pos
        )
        self.follow_cam_target = (
            self.k_smooth * self.follow_cam_target
            + (1 - self.k_smooth) * new_cam_target
        )

        set_camera_view(self.follow_cam_pos, self.follow_cam_target)

    def _setup_extra_keyboard_callback(self):
        self._teleop_key.add_callback("R", self._keyboard_reset_idx)

    def _keyboard_reset_idx(self):
        print("[R] Key pressed: Resetting environments...")
        env_ids = torch.arange(0, self.num_envs, dtype=torch.int64, device=self.device)
        self.episode_length_buf[env_ids] = torch.ones_like(
            self.episode_length_buf
        ) * int(self.max_episode_length)

    def _show_all_waypoint(self):
        marker_cfg = SPHERE_MARKER_CFG.replace(
            prim_path="/Visuals/Waypoints",
        )
        marker_cfg.markers["sphere"].radius = 0.1
        waypoint_visualizer = VisualizationMarkers(marker_cfg)

        for env_ids in range(self.num_envs):
            waypoint_visualizer.visualize(self._waypoints_pos[env_ids])

    def _show_depth_frame(self):
        if not self.cfg.show_depth_frame:
            return

        depth_frame = (
            self._depth_map_buf[self.i_follow_env, 0, :, :].detach().cpu().numpy()
        )
        depth_frame = cv2.resize(
            depth_frame, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow("Robot Depth Frame", depth_frame)
        cv2.waitKey(1)

    def _show_current_waypoint(self):
        if not self.cfg.show_waypoint:
            return

        env_ids = self._robot._ALL_INDICES
        current_goal = self._waypoints_pos[env_ids, self._idx_waypoint]

        self.current_waypoint_visualizer.visualize(current_goal)
