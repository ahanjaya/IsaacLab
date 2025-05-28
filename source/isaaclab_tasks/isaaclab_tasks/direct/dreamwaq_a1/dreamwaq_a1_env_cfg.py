# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class DreamWaQA1FlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 48
    state_space = 0

    # debug visualization
    debug_vis = True
    debug_marker = True

    class camera_viewer:
        # # Closer Side View
        # pos = [0.7, 1.5, 0.7]
        # target = [0.5, 0.0, 0.0]
        # offset = [0.0, -1.0, 0.2]

        # Far View
        pos = [10.0, 0.0, 6.0]
        target = [11.0, 5.0, 3.0]
        offset = [0.0, -3.0, 2.0]

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = UNITREE_A1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05

    class rewards:
        class scales:
            lin_vel = 1.0
            yaw_rate = 0.5
            z_vel = -2.0
            ang_vel = -0.05
            flat_orientation = -5.0
            joint_accel = -2.5e-7
            joint_power = -2.0e-5
            base_height = -1.0
            feet_air_time = 1.0
            feet_clearance = -0.01
            action_rate = -0.001
            smoothness = -0.001
            power_distribution = -1.0e-5
            hip_pos = -0.5
            joint_err = -0.05
            joint_pos_limits = -10.0
            joint_torque_limits = -1.0
            foot_ground = -1.0
            termination = -1.0
            joint_torque = -2.5e-5
            undesired_contact = -1.0

        base_height_target = 0.28
        tracking_sigma = 0.25
        soft_torque_limit_percentage = 0.85
