# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import os
import torch
from collections.abc import Sequence

from omni.isaac.core.utils.viewports import set_camera_view

##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import (
    UNITREE_A1_ANIM_CFG,
    UNITREE_A1_CFG,
)  # isort: skip

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils import motion_imitation_utils as miu


class ImitationPolicyA1EnvWindow(BaseEnvWindow):
    """Window manager for the ImitationPolicyA1 environment."""

    def __init__(self, env: ImitationPolicyA1Env, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("Debug", self.env)


@configclass
class ImitationPolicyA1EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 4
    action_scale = 0.3
    num_actions = 12
    num_observations = 93  # state(25) + action(12) + future_target_joints(12*4) + future_frames_euler_xy(2*4)
    z_offset = 0.0

    # debug vis
    debug_vis = False
    ui_window_class_type = ImitationPolicyA1EnvWindow

    # motions
    motions_root = "source/extensions/omni.isaac.lab_assets/data/Motions"
    motion_fn = "pace_remove_yaw.txt"

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
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=1.0, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = UNITREE_A1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=1,
        update_period=0.0,
    )

    # animation
    animation: ArticulationCfg = UNITREE_A1_ANIM_CFG.replace(
        prim_path="/World/envs/env_.*/Animation"
    )

    # reward scales
    weight_joint_pos = 0.5
    weight_joint_vel = 0.05
    weight_ef = 0.2
    weight_root_pose = 0.15
    weight_root_vel = 0.1

    scale_joint_pos = 5.0
    scale_joint_vel = 0.1
    scale_ef = 40.0
    scale_root_pose = 20.0
    scale_root_vel = 2.0
    scale_err_height = 3.0

    obs_scales = {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
    }
    foot_contact_threshold = 1.0
    reset_contact_threshold = 1.0
    root_reset_dist = 1.0


class ImitationPolicyA1Env(DirectRLEnv):
    cfg: ImitationPolicyA1EnvCfg

    def __init__(
        self, cfg: ImitationPolicyA1EnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_joint_lower_limits = self._robot.data.default_joint_limits[
            0, :, 0
        ].to(device=self.device)
        self.robot_joint_upper_limits = self._robot.data.default_joint_limits[
            0, :, 1
        ].to(device=self.device)

        # Get specific body indices
        self._body_idx, _ = self._contact_sensor.find_bodies("trunk")
        self._foot_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._hip_ids, _ = self._contact_sensor.find_bodies(".*hip")
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._calf_ids, _ = self._contact_sensor.find_bodies(".*calf")
        self._n_foot = len(self._foot_ids)

        self._load_motion(
            motion_path=os.path.join(
                os.getcwd(), self.cfg.motions_root, self.cfg.motion_fn
            ),
        )
        self._setup_utility_tensors()

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "joint_pos",
                "joint_vel",
                "end_effector_pos",
                "root_pose",
                "root_vel",
            ]
        }

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._animation = Articulation(self.cfg.animation)

        # TODO: Change visual material of animation environment after make_uninstanceable
        # for prim_path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Animation/.*/visuals"):
        #     print(f"Prim path: {prim_path}")
        #     sim_utils.make_uninstanceable(prim_path)
        #     sim_utils.bind_visual_material(prim_path, material_path=?)

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.articulations["animation"] = self._animation

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
        processed_actions = (
            actions * self.cfg.action_scale * 3.1415
            + self._robot.data.default_joint_pos
        )
        self._actions = torch.clamp(
            processed_actions,
            self.robot_joint_lower_limits,
            self.robot_joint_upper_limits,
        )

        # Update animation
        self._update_animation()

    def _apply_action(self):
        self._robot.set_joint_position_target(self._actions)

        self._animation.set_joint_position_target(self._anim_joint_pos)
        self._animation.set_joint_velocity_target(self._anim_joint_vel)
        self._animation.write_root_pose_to_sim(self._anim_root_pos)
        self._animation.write_root_velocity_to_sim(self._anim_root_vel)

    def _get_observations(self) -> dict:
        curr_state = compute_agent_state(
            robot_euler_xyz=self._get_robot_euler_xyz(),
            robot_root_lin_vel=self._robot.data.root_lin_vel_b,
            robot_root_ang_vel=self._robot.data.root_ang_vel_b,
            robot_joint_pos=self._robot.data.joint_pos,
            robot_foot_contacts=self._get_foot_contacts(),
            scale_lin_vel=self.cfg.obs_scales["lin_vel"],
            scale_ang_vel=self.cfg.obs_scales["ang_vel"],
        )

        future_frames = compute_future_frames(
            ref_motion_frames=self.tensor_ref_pose,
            ref_motion_index=self.ref_motion_index,
            target_pose_inc_indices=self.target_pose_inc_indices,
            robot_heading=self._robot.data.heading_w,
            z_basis_vec=self.z_basis_vec,
            len_target_pose_inc=self.len_target_pose_inc,
            num_envs=self.num_envs,
            num_actions=self.cfg.num_actions,
            device=self.device,
        )

        obs = torch.cat(
            [
                curr_state,
                self._actions,
                future_frames,
            ],
            dim=1,
        )
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        rewards = compute_rewards(
            animation_joint_pos=self._animation.data.joint_pos,
            robot_joint_pos=self._robot.data.joint_pos,
            animation_joint_vel=self._animation.data.joint_vel,
            robot_joint_vel=self._robot.data.joint_vel,
            animation_end_effector_pos=self._animation.data.body_state_w[
                :, self._foot_ids, :3
            ],
            robot_end_effector_pos=self._robot.data.body_state_w[:, self._foot_ids, :3],
            animation_root_pos=self._animation.data.root_pos_w,
            robot_root_pos=self._robot.data.root_pos_w,
            animation_root_ori=self._animation.data.root_quat_w,
            robot_root_ori=self._robot.data.root_quat_w,
            animation_heading=self._animation.data.heading_w,
            robot_heading=self._robot.data.heading_w,
            z_basis_vec=self.z_basis_vec,
            n_foot=self._n_foot,
            animation_root_lin_vel=self._animation.data.root_lin_vel_b,
            robot_root_lin_vel=self._robot.data.root_lin_vel_b,
            animation_root_ang_vel=self._animation.data.root_ang_vel_b,
            robot_root_ang_vel=self._robot.data.root_ang_vel_b,
            scale_joint_pos=self.cfg.scale_joint_pos,
            weight_joint_pos=self.cfg.weight_joint_pos,
            scale_joint_vel=self.cfg.scale_joint_vel,
            weight_joint_vel=self.cfg.weight_joint_vel,
            scale_ef_err_height=self.cfg.scale_err_height,
            scale_ef=self.cfg.scale_ef,
            weight_ef=self.cfg.weight_ef,
            scale_root_pose=self.cfg.scale_root_pose,
            weight_root_pose=self.cfg.weight_root_pose,
            scale_root_vel=self.cfg.scale_root_vel,
            weight_root_vel=self.cfg.weight_root_vel,
        )

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # if reset condition is met, set all rewards to 0
        rewards = torch.sum(torch.stack(list(rewards.values())), dim=0)
        zeros = torch.zeros_like(rewards)
        rewards = torch.where(self.reset_terminated, zeros, rewards)

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.ref_motion_index += 1
        # update camera view
        self._update_camera_follow_env()

        time_outs = self.episode_length_buf >= self.max_episode_length - 1

        # Check position error
        root_pos_diff = torch.square(
            self._animation.data.root_pos_w - self._robot.data.root_pos_w
        )
        root_pos_err = root_pos_diff.sum(dim=1)
        is_root_pos_err = root_pos_err > self.cfg.root_reset_dist

        # Check orientation error
        # TODO: understand this part of code
        root_rot_diff = math_utils.quat_mul(
            self._animation.data.root_quat_w,
            math_utils.quat_conjugate(self._robot.data.root_quat_w),
        )
        root_rot_diff = math_utils.normalize(root_rot_diff)
        root_rot_diff_angle = math_utils.normalize_angle(
            2 * torch.acos(root_rot_diff[:, 0])
        )
        is_root_rot_err = torch.square(root_rot_diff_angle) > self.cfg.root_reset_dist

        is_contacts = self._check_contacts()

        terminated = is_root_pos_err | is_root_rot_err | is_contacts

        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        self._animation.reset(env_ids)
        self._contact_sensor.reset(env_ids)
        super()._reset_idx(env_ids)

        # if len(env_ids) == self.num_envs:
        #     # Spread out the resets to avoid spikes in training when many environments reset at a similar time
        #     self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self.ref_motion_index[env_ids] = 0
        self.tensor_ref_offset_pos[env_ids, :] = 0.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._animation.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._animation.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._animation.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        extras["Episode Termination/base_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "robot_pose_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.15)
                marker_cfg.prim_path = "/Visuals/Robot/body_pose"
                self.robot_pose_visualizer = VisualizationMarkers(marker_cfg)

                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.15)
                marker_cfg.prim_path = "/Visuals/Animation/body_pose"
                self.animation_pose_visualizer = VisualizationMarkers(marker_cfg)

            self.robot_pose_visualizer.set_visibility(True)
            self.animation_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "robot_pose_visualizer"):
                self.robot_pose_visualizer.set_visibility(False)
                self.animation_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update markers
        self._update_debug_marker()

    ##############################################################################
    def _setup_utility_tensors(self):
        # Camera follow actor
        self.follow_cam_pos = np.array([0.7, 1.5, 0.7])
        self.follow_cam_target = np.array([0.5, 0.0, 0])
        self.follow_cam_offset = np.array([0.0, -1.0, 0.20])
        self.k_smooth = 0.9
        self.i_follow_env = 0

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(
            self.num_envs, self.cfg.num_actions, device=self.device
        )

        # The current reference motion index, per actor.
        self.ref_motion_index = torch.zeros_like(self.episode_length_buf)

        self.z_basis_vec = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.z_basis_vec[:, 2] = 1.0

    def _update_camera_follow_env(self):
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

    def _update_debug_marker(self):
        robot_root_pose_w = self._robot.data.root_state_w
        robot_root_pose_w[:, 2] += 0.1
        self.robot_pose_visualizer.visualize(
            robot_root_pose_w[:, :3], robot_root_pose_w[:, 3:7]
        )

        animation_root_pose_w = self._animation.data.root_state_w
        animation_root_pose_w[:, 2] += 0.1
        self.animation_pose_visualizer.visualize(
            animation_root_pose_w[:, :3], animation_root_pose_w[:, 3:7]
        )

    def _load_motion(self, motion_path):
        """Loads a reference motion from disk. Pre-generates all frames and push
        them to GPU for later use.

        """
        print("Loading motion data...")
        self.motion = miu.MotionData(motion_path)
        print(f"\tFrames: {self.motion.get_num_frames()}")
        print(f"\tFrame duration: {self.motion.get_frame_duration()}")

        step_size = self.sim.get_physics_dt() * self.cfg.decimation
        self.motion_length = self.motion.get_num_frames()

        # Pre-generate all frames for the whole episode + some extra cycles.
        # The extra cycles are needed because the robot is reset to a random
        # reference index between 0 and 2 cycles.
        time_axis = np.arange(
            0, self.cfg.episode_length_s + 5 * step_size * self.motion_length, step_size
        )
        print(f"\tTime_axis: {time_axis.shape}")

        self.np_pose_frames = []
        self.np_vel_frames = []
        for t in time_axis:
            pose = self.motion.calc_frame(t)
            vels = self.motion.calc_frame_vel(t)
            # NOTE: Order of joints in IsaacLab differs from PyBullet.
            # PyBullet:
            # FR Hip, FR Thigh, FR Calf,
            # FL Hip, FL Thigh, FL Calf,
            # RR Hip, RR Thigh, RR Calf,
            # RL Hip, RL Thigh, RL Calf,

            reordered_pose = np.array(
                [
                    pose[0],  # X Pose
                    pose[1],  # Y Pose
                    pose[2],  # Z Pose
                    pose[6],  # W Quat
                    pose[3],  # X Quat
                    pose[4],  # Y Quat
                    pose[5],  # Z Quat
                    pose[10],  # FL Hip
                    pose[7],  # FR Hip
                    pose[16],  # RL Hip
                    pose[13],  # RR Hip
                    pose[11],  # FL Thigh
                    pose[8],  # FR Thigh
                    pose[17],  # RL Thigh
                    pose[14],  # RR Thigh
                    pose[12],  # FL Calf
                    pose[9],  # FR Calf
                    pose[18],  # RL Calf
                    pose[15],  # RR Calf
                ]
            )

            reordered_vels = np.array(
                [
                    vels[0],  # X Lin Vel
                    vels[1],  # Y Lin Vel
                    vels[2],  # Z Lin Vel
                    vels[3],  # X Ang Vel
                    vels[4],  # Y Ang Vel
                    vels[5],  # Z Ang Vel
                    pose[9],  # FL Hip Vel
                    pose[6],  # FR Hip Vel
                    pose[15],  # RL Hip Vel
                    pose[12],  # RR Hip Vel
                    pose[10],  # FL Thigh Vel
                    pose[7],  # FR Thigh Vel
                    pose[16],  # RL Thigh Vel
                    pose[13],  # RR Thigh Vel
                    pose[11],  # FL Calf Vel
                    pose[8],  # FR Calf Vel
                    pose[17],  # RL Calf Vel
                    pose[14],  # RR Calf Vel
                ]
            )

            self.np_pose_frames.append(reordered_pose)
            self.np_vel_frames.append(reordered_vels)

        self.np_pose_frames = np.array(self.np_pose_frames)
        self.np_vel_frames = np.array(self.np_vel_frames)

        # Offset reference motion Z axis. Used to unstuck the reference motion
        # from the ground.
        self.np_pose_frames[:, 2] += self.cfg.z_offset
        assert self.np_pose_frames.shape[0] == self.np_vel_frames.shape[0]

        # Convert to PyTorch GPU tensors.
        self.tensor_ref_pose = torch.tensor(
            self.np_pose_frames, dtype=torch.float32, device=self.device
        )
        self.tensor_ref_vels = torch.tensor(
            self.np_vel_frames, dtype=torch.float32, device=self.device
        )

        # Create other useful views.
        self.tensor_ref_root_pose = self.tensor_ref_pose[:, :7]  # XYZ + Quat WXYZ
        self.tensor_ref_joint_targets = self.tensor_ref_pose[:, 7:]  # 12 joints
        self.tensor_ref_root_vels = self.tensor_ref_vels[
            :, :6
        ]  # Linear XYZ + Angular XYZ
        self.tensor_ref_joint_vels = self.tensor_ref_vels[:, 6:]

        # Used to sync the postion of kin character to sim character by offseting
        # its position.
        self.tensor_ref_offset_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )

        lookahead_secs = [0.0333, 0.0666, 0.3333, 1.0]  # Lookahead time in seconds.
        lookahead_inds = [int(s * (1 / step_size) + 0.5) for s in lookahead_secs]
        # Used to increment from current index to get future target poses from
        # the reference motion.
        self.target_pose_inc_indices = torch.tensor(
            lookahead_inds, dtype=torch.long, device=self.device
        )
        self.len_target_pose_inc = len(self.target_pose_inc_indices)

    def _update_animation(self):
        self._anim_root_pos = self.tensor_ref_root_pose[self.ref_motion_index]
        self._anim_root_pos[:, :3] += self._terrain.env_origins

        # Reset root position (x, y) offset every cycle of the motion length.
        curr_phase = self.episode_length_buf // self.motion_length > 0
        reset_phase = (self.episode_length_buf % self.motion_length) == 0
        resync_env_ids = (
            curr_phase.logical_and(reset_phase).nonzero(as_tuple=False).flatten()
        )

        self.tensor_ref_offset_pos[resync_env_ids, :2] = (
            self._anim_root_pos[resync_env_ids, :2]
            - self._robot.data.root_pos_w[resync_env_ids, :2]
        )
        self._anim_root_pos[:, :3] -= self.tensor_ref_offset_pos

        self._anim_root_vel = self.tensor_ref_root_vels[self.ref_motion_index]
        self._anim_joint_pos = self.tensor_ref_joint_targets[self.ref_motion_index]
        self._anim_joint_vel = self.tensor_ref_joint_vels[self.ref_motion_index]

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
                torch.norm(net_contact_forces[:, foot_id, :], dim=1)
                >= self.cfg.foot_contact_threshold
            )
        return foot_contacts

    def _get_robot_euler_xyz(self) -> torch.Tensor:
        robot_euler_xyz = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
        robot_roll, robot_pitch, robot_yaw = robot_euler_xyz

        robot_roll = math_utils.wrap_to_pi(robot_roll)
        robot_pitch = math_utils.wrap_to_pi(robot_pitch)
        robot_yaw = math_utils.wrap_to_pi(robot_yaw)

        return torch.cat(
            [
                robot_roll.unsqueeze(-1),
                robot_pitch.unsqueeze(-1),
                robot_yaw.unsqueeze(-1),
            ],
            dim=-1,
        )


@torch.jit.script
def compute_agent_state(
    robot_euler_xyz: torch.Tensor,
    robot_root_lin_vel: torch.Tensor,
    robot_root_ang_vel: torch.Tensor,
    robot_joint_pos: torch.Tensor,
    robot_foot_contacts: torch.Tensor,
    scale_lin_vel: float,
    scale_ang_vel: float,
):

    return torch.cat(
        [
            robot_euler_xyz,  # num_envs x 3
            robot_root_lin_vel * scale_lin_vel,  # num_envs x 3
            robot_root_ang_vel * scale_ang_vel,  # num_envs x 3
            robot_joint_pos,  # num_envs x 12
            robot_foot_contacts,  # num_envs x 4
        ],
        dim=1,
    )


@torch.jit.script
def compute_future_frames(
    ref_motion_frames: torch.Tensor,
    ref_motion_index: torch.Tensor,
    target_pose_inc_indices: torch.Tensor,
    robot_heading: torch.Tensor,
    z_basis_vec: torch.Tensor,
    len_target_pose_inc: int,
    num_envs: int,
    num_actions: int,
    device: torch.device,
):
    inv_heading_rot = math_utils.quat_from_angle_axis(
        robot_heading,
        z_basis_vec,
    )
    future_indices = ref_motion_index.unsqueeze(1) + target_pose_inc_indices
    future_target_frames = ref_motion_frames[future_indices, 3:]

    # normalize orientation
    future_frames_euler_xy = torch.zeros(
        (num_envs, len_target_pose_inc, 2), dtype=torch.float32, device=device
    )

    for idx_frame in range(len_target_pose_inc):
        current_frame_quat = future_target_frames[:, idx_frame, :4]
        current_frame_quat = math_utils.quat_mul(inv_heading_rot, current_frame_quat)
        current_frame_quat = math_utils.normalize(current_frame_quat)

        current_frame_euler_xyz = math_utils.euler_xyz_from_quat(current_frame_quat)
        frame_roll, frame_pitch, _ = current_frame_euler_xyz

        future_frames_euler_xy[:, idx_frame, 0] = math_utils.wrap_to_pi(frame_roll)
        future_frames_euler_xy[:, idx_frame, 1] = math_utils.wrap_to_pi(frame_pitch)

    # flatten future frames euler xy
    future_frames_euler_xy = future_frames_euler_xy.reshape(-1, 2 * len_target_pose_inc)

    # flatten future target joints
    future_target_joints = future_target_frames[:, :, 4:].reshape(
        -1, num_actions * len_target_pose_inc
    )

    return torch.cat(
        [
            future_target_joints,
            future_frames_euler_xy,
        ],
        dim=1,
    )


@torch.jit.script
def compute_rewards(
    animation_joint_pos: torch.Tensor,
    robot_joint_pos: torch.Tensor,
    animation_joint_vel: torch.Tensor,
    robot_joint_vel: torch.Tensor,
    animation_end_effector_pos: torch.Tensor,
    robot_end_effector_pos: torch.Tensor,
    animation_root_pos: torch.Tensor,
    robot_root_pos: torch.Tensor,
    animation_root_ori: torch.Tensor,
    robot_root_ori: torch.Tensor,
    animation_heading: torch.Tensor,
    robot_heading: torch.Tensor,
    z_basis_vec: torch.Tensor,
    n_foot: int,
    animation_root_lin_vel: torch.Tensor,
    robot_root_lin_vel: torch.Tensor,
    animation_root_ang_vel: torch.Tensor,
    robot_root_ang_vel: torch.Tensor,
    scale_joint_pos: float,
    weight_joint_pos: float,
    scale_joint_vel: float,
    weight_joint_vel: float,
    scale_ef_err_height: float,
    scale_ef: float,
    weight_ef: float,
    scale_root_pose: float,
    weight_root_pose: float,
    scale_root_vel: float,
    weight_root_vel: float,
):
    # Joint pos reward.
    joint_pos_diff = torch.square(animation_joint_pos - robot_joint_pos)
    joint_pos_rew = torch.exp(-scale_joint_pos * joint_pos_diff.sum(dim=1))

    # Joint velocity reward.
    joint_vel_diff = torch.square(animation_joint_vel - robot_joint_vel)
    joint_vel_rew = torch.exp(-scale_joint_vel * joint_vel_diff.sum(dim=1))

    # End-effector position reward.
    # Normalized foot end-effector position relative to root body.
    animation_end_effector_pos -= animation_root_pos.unsqueeze(1)
    robot_end_effector_pos -= robot_root_pos.unsqueeze(1)

    # Angle around Z axis.
    animation_inv_heading_rot = math_utils.quat_from_angle_axis(
        animation_heading,
        z_basis_vec,
    )
    robot_inv_heading_rot = math_utils.quat_from_angle_axis(
        robot_heading,
        z_basis_vec,
    )

    for idx in range(n_foot):
        animation_end_effector_pos[:, idx] = math_utils.quat_rotate(
            animation_inv_heading_rot, animation_end_effector_pos[:, idx]
        )
        robot_end_effector_pos[:, idx] = math_utils.quat_rotate(
            robot_inv_heading_rot, robot_end_effector_pos[:, idx]
        )

    ef_diff_xy = torch.square(
        animation_end_effector_pos[:, :, :2] - robot_end_effector_pos[:, :, :2]
    )
    ef_diff_xy = ef_diff_xy.sum(dim=2)
    ef_diff_z = scale_ef_err_height * torch.square(
        animation_end_effector_pos[:, :, 2] - robot_end_effector_pos[:, :, 2]
    )
    ef_diff = (ef_diff_xy + ef_diff_z).sum(dim=1)
    ef_rew = torch.exp(-scale_ef * ef_diff)

    # Root pose reward. Position + Orientation.
    root_pos_diff = torch.square(animation_root_pos - robot_root_pos)
    root_pos_err = root_pos_diff.sum(dim=1)

    root_rot_diff = math_utils.quat_mul(
        animation_root_ori, math_utils.quat_conjugate(robot_root_ori)
    )
    root_rot_diff = math_utils.normalize(root_rot_diff)

    # axis-angle representation but we only care about the angle
    root_rot_diff_angle = math_utils.normalize_angle(
        2 * torch.acos(root_rot_diff[:, 0])
    )
    root_rot_err = torch.square(root_rot_diff_angle)

    # Compound position and orientation error for root as in motion_imitation codebase.
    root_pose_err = root_pos_err + 0.5 * root_rot_err
    root_pose_rew = torch.exp(-scale_root_pose * root_pose_err)

    # Root velocity reward.
    root_linvel_diff = torch.square(animation_root_lin_vel - robot_root_lin_vel)
    root_linvel_err = root_linvel_diff.sum(dim=1)
    root_angvel_diff = torch.square(animation_root_ang_vel - robot_root_ang_vel)
    root_angvel_err = root_angvel_diff.sum(dim=1)

    root_vel_diff = root_linvel_err + 0.1 * root_angvel_err
    root_vel_rew = torch.exp(-scale_root_vel * root_vel_diff)

    # TODO: all rewards multiply with step_dt
    rewards = {
        "joint_pos": joint_pos_rew * weight_joint_pos,
        "joint_vel": joint_vel_rew * weight_joint_vel,
        "end_effector_pos": ef_rew * weight_ef,
        "root_pose": root_pose_rew * weight_root_pose,
        "root_vel": root_vel_rew * weight_root_vel,
    }

    return rewards
