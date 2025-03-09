# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.envs.ui import BaseEnvWindow

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import (
    UNITREE_A1_ANIM_CFG,
    UNITREE_A1_CFG,
)


class ImitationA1EnvWindow(BaseEnvWindow):
    """Window manager for the ImitationPolicyA1 environment."""

    def __init__(self, env, window_name="IsaacLab"):
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
class ImitationA1EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 4
    action_scale = 0.3
    action_space = 12
    observation_space = 93  # state(25) + action(12) + future_target_joints(12*4) + future_frames_euler_xy(2*4)
    state_space = 0
    z_offset = 0.0

    # debug visualization
    debug_vis = True
    debug_marker = False
    ui_window_class_type = ImitationA1EnvWindow

    # motions
    motions_root = "source/isaaclab_assets/data/Motions"
    motion_fn = "pace_remove_yaw.txt"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        # disable_contact_processing=True,
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
            friction_combine_mode="multiply",  # average
            restitution_combine_mode="multiply",  # average
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=debug_marker,
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
        debug_vis=debug_marker,
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
