# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import cv2
import torch
import numpy as np
from collections.abc import Sequence
from torchvision.utils import make_grid

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import Camera, CameraCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG, UNITREE_GO2_CFG  # isort: skip


@configclass
class TemplateEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5.0
    decimation = 4
    action_scale = 0.5
    num_actions = 12
    num_observations = 24
    robot_type = "UNITREE_A1"
    debug_vis = True
    debug_marker = False

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8, env_spacing=2.0, replicate_physics=True)

    # robot
    if robot_type == "UNITREE_A1":
        robot: ArticulationCfg = UNITREE_A1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        camera_offset = CameraCfg.OffsetCfg(pos=(0.27, 0.0, 0.03), rot=(0.5, -0.5, 0.5, -0.5), convention="ros")
    elif robot_type == "UNITREE_GO2":
        robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        camera_offset = CameraCfg.OffsetCfg(pos=(0.32487, -0.00095, 0.05362), rot=(0.5, -0.5, 0.5, -0.5), convention="ros")
    elif robot_type == "ANYMAL_C":
        robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        camera_offset = CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros")
    else:
        raise ValueError(f"Invalid robot type: {robot_type}, available options: ['UNITREE_A1', 'UNITREE_GO2', 'ANYMAL_C']")

    # camera
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/Camera",
        height=120,
        width=160,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
        ),
        offset=camera_offset,
    )

    # reward scales
    dummy_reward_scale = 1.0


class TemplateEnv(DirectRLEnv):
    cfg: TemplateEnvCfg

    def __init__(self, cfg: TemplateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._camera = Camera(self.cfg.camera)

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["camera"] = self._camera

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
        self._actions = torch.zeros_like(actions)
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
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
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out).bool()

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        # RGB Images
        # Permute the tensor from (num_envs, height, wdith, channel) to shape (num_envs, channel, height, width)
        rgb_imgs = self._camera.data.output["rgb"].clone().permute(0, 3, 1, 2)

        # Create a grid of images
        grid_rgb_img = make_grid(rgb_imgs, nrow=round(rgb_imgs.shape[0] ** 0.5))
        grid_rgb_img = grid_rgb_img.permute(1, 2, 0).cpu().numpy()
        grid_rgb_img = cv2.cvtColor(grid_rgb_img, cv2.COLOR_RGB2BGR)

        # Depth Images
        # Add a channel dimension to the tensor from (num_envs, height, width) to (num_envs, 1, height, width)
        depth_imgs = self._camera.data.output["distance_to_image_plane"].clone().unsqueeze(1)

        # Create a grid of images
        grid_depth_img = make_grid(depth_imgs, nrow=round(depth_imgs.shape[0] ** 0.5))
        grid_depth_img = grid_depth_img.permute(1, 2, 0).cpu().numpy()

        # Normalize the image to the range [0, 255]
        grid_depth_img = (grid_depth_img * 255).astype(np.uint8)

        cv2.imshow("Robot RGB Frame", grid_rgb_img)
        cv2.imshow("Robot Depth Frame", grid_depth_img)
        cv2.waitKey(1)
