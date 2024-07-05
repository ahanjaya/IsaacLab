# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import random
import torch
import numpy as np

import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils import motion_imitation_utils as miu
from omni.isaac.core.utils.viewports import set_camera_view


##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG, UNITREE_A1_ANIM_CFG, UNITREE_A1_UNINSTANCEABLE_CFG  # isort: skip


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
                    self._create_debug_vis_ui_element("Follow Cam", self.env)


@configclass
class ImitationPolicyA1EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 4
    action_scale = 0.5
    num_actions = 12
    num_observations = 24

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8, env_spacing=1.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = UNITREE_A1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # animation
    animation: ArticulationCfg = UNITREE_A1_ANIM_CFG.replace(prim_path="/World/envs/env_.*/Animation")

    # reward scales
    dummy_reward_scale = 1.0


class ImitationPolicyA1Env(DirectRLEnv):
    cfg: ImitationPolicyA1EnvCfg

    def __init__(self, cfg: ImitationPolicyA1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self._load_motion(
            motion_path=os.path.join(os.getcwd(), self.cfg.motions_root, self.cfg.motion_fn),
        )
        self._setup_utility_tensors()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._animation = Articulation(self.cfg.animation)

        # TODO: Change visual material of animation environment after make_uninstanceable
        # for prim_path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Animation/.*/visuals"):
        #     print(f"Prim path: {prim_path}")
        #     sim_utils.make_uninstanceable(prim_path)
        #     sim_utils.bind_visual_material(prim_path, material_path=?)

        self.scene.articulations["robot"] = self._robot
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
        self._actions: torch.Tensor = torch.zeros_like(actions)

        joint_pos = self._robot.data.default_joint_pos + (self.cfg.action_scale * self._actions * 3.1415)
        self._processed_actions = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        self._root_pose_anim = self.tensor_ref_root_pose[self.ref_motion_index].clone()
        self._root_pose_anim[:, :3] += self._terrain.env_origins
        self._root_vel_anim = self.tensor_ref_root_vels[self.ref_motion_index].clone()
        self._joint_pos_anim = self.tensor_ref_pd_targets[self.ref_motion_index].clone()
        self._joint_vel_anim = self.tensor_ref_pd_vels[self.ref_motion_index].clone()

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

        self._animation.write_root_pose_to_sim(self._root_pose_anim)
        self._animation.write_root_velocity_to_sim(self._root_vel_anim)
        self._animation.set_joint_position_target(self._joint_pos_anim)
        self._animation.set_joint_velocity_target(self._joint_vel_anim)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        obs = torch.zeros((self.num_envs, self.num_observations))
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        dummy_rew = torch.zeros(self.num_envs)

        rewards = {
            "track_dummy": dummy_rew * self.cfg.dummy_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.ref_motion_index += 1

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out).bool()

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        self._animation.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self.ref_motion_index[env_ids] = 0

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

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            self._init_debug()
            # set_camera_view([0.2, 4.0, 1.0], [0.2, 0.0, 0.0])

    def _debug_vis_callback(self, event):
        # update camera view
        self._update_debug_camera()

    ##############################################################################
    def _setup_utility_tensors(self):
        # The current reference motion index, per actor.
        self.ref_motion_index = torch.zeros_like(self.episode_length_buf)

    def _init_debug(self):
        self.debug_cam_pos = np.array([0.7, 1.5, 0.7])
        self.debug_cam_target = np.array([0.5, 0.0, 0])
        self.debug_cam_offset = np.array([0.0, -1.0, 0.20])
        self.k_smooth = 0.9
        self.i_follow_env = 0

    def _update_debug_camera(self):
        actor_pos = self._robot.data.root_pos_w[self.i_follow_env].cpu().numpy()

        # Smooth the camera movement with a moving average.
        new_cam_pos = actor_pos + self.debug_cam_offset
        new_cam_target = actor_pos

        self.debug_cam_pos = self.k_smooth * self.debug_cam_pos + (1 - self.k_smooth) * new_cam_pos
        self.debug_cam_target = self.k_smooth * self.debug_cam_target + (1 - self.k_smooth) * new_cam_target

        set_camera_view(self.debug_cam_pos, self.debug_cam_target)

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
        time_axis = np.arange(0, self.cfg.episode_length_s + 5 * step_size * self.motion_length, step_size)
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

            reordered_pose = np.array([
                pose[0], pose[1], pose[2],  # X, Y, Z Pos
                pose[6], pose[3], pose[4], pose[5],  # IsaacLab WXYZ, IsaacGym XYZW
                pose[10], pose[7], pose[16], pose[13],  # HIP -> FL, FR, RL, RR
                pose[11], pose[8], pose[17], pose[14],  # Thigh -> FL, FR, RL, RR
                pose[12], pose[9], pose[18], pose[15],  # Calf -> FL, FR, RL, RR
            ])

            reordered_vels = np.array([
                vels[0], vels[1], vels[2],  # Lin vel (No change)
                vels[3], vels[4], vels[5],  # Ang vel (No change)
                pose[9], pose[6], pose[15], pose[12],  # HIP -> FL, FR, RL, RR
                pose[10], pose[7], pose[16], pose[13],  # Thigh -> FL, FR, RL, RR
                pose[11], pose[8], pose[17], pose[14],  # Calf -> FL, FR, RL, RR
            ])

            self.np_pose_frames.append(reordered_pose)
            self.np_vel_frames.append(reordered_vels)

        self.np_pose_frames = np.array(self.np_pose_frames)
        self.np_vel_frames = np.array(self.np_vel_frames)

        # Offset reference motion Z axis. Used to unstuck the reference motion
        # from the ground.
        self.np_pose_frames[:, 2] += 0.0
        assert self.np_pose_frames.shape[0] == self.np_vel_frames.shape[0]

        # Animation length also defines the maximum episode length
        # Makes sure episode finished before we run out of future frames to index
        # in the observations.
        # self.max_episode_length = self.np_pose_frames.shape[0] - 4 * self.motion_length - 1
        # print(f"Max episode length is {self.max_episode_length}.")

        # Convert to PyTorch GPU tensors.
        self.tensor_ref_pose = torch.tensor(self.np_pose_frames, dtype=torch.float32, device=self.device)
        self.tensor_ref_vels = torch.tensor(self.np_vel_frames, dtype=torch.float32, device=self.device)

        # Create other useful views.
        self.tensor_ref_root_pose = self.tensor_ref_pose[:, :7]  # XYZ + Quat
        self.tensor_ref_pd_targets = self.tensor_ref_pose[:, 7:]  # 12 joints
        self.tensor_ref_root_vels = self.tensor_ref_vels[:, :6]  # Linear XYZ + Angular XYZ
        self.tensor_ref_pd_vels = self.tensor_ref_vels[:, 6:]

        # Used to sync the postion of kin character to sim character by offseting
        # its position.
        self.tensor_ref_offset_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        lookahead_secs = [0.0333, 0.0666, 0.3333, 1.0]  # Lookahead time in seconds.
        lookahead_inds = [int(s * (1 / step_size) + 0.5) for s in lookahead_secs]
        # Used to increment from current index to get future target poses from
        # the reference motion.
        self.target_pose_inc_indices = torch.tensor(lookahead_inds, dtype=torch.long, device=self.device)
