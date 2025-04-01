# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

##
# Pre-defined configs
##
from isaaclab_assets.robots.erc import NXP_LOWER_BODY_WITH_TORSO_MINIMAL_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class EventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    add_base_mass = None

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    push_robot = None


@configclass
class VelocityNXPLowerBodyFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 13
    observation_space = 48
    state_space = 0

    # debug visualization
    debug_vis = True
    debug_marker = True

    class camera_viewer:
        follow_camera = False

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
        dt=1 / 200,  # 0.005
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=10 * 2**15,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # events
    # events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = NXP_LOWER_BODY_WITH_TORSO_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )

    # no height scan
    height_scanner = None
    # self.observations.policy.height_scan = None

    # no terrain curriculum
    # self.curriculum.terrain_levels = None

    # Rewards
    track_lin_vel_xy_reward_scale = 1.0
    track_ang_vel_z_exp_reward_scale = 1.0
    lin_vel_z_l2_reward_scale = -0.2
    ang_vel_xy_l2_reward_scale = -0.05
    dof_torques_l2_reward_scale = -2.0e-6
    dof_acc_l2_reward_scale = -1.0e-7
    action_rate_l2_reward_scale = -0.005
    feet_air_time_reward_scale = 1.0
    feet_air_time_threshold = 0.6
    flat_orientation_l2_reward_scale = -1.0
    dof_pos_limits_reward_scale = -1.0
    termination_reward_scale = -200.0
    feet_slide_reward_scale = -0.1
    joint_deviation_hip_reward_scale = -0.1
    joint_deviation_torso_reward_scale = -0.1
