# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import copy
import math
from typing import Literal

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg as SceneEntity
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets import NXP_V1_HUMANOID_CFG  # isort: skip

# Joint names for NXP humanoid
NXP_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
]

# GAC (Grid Adaptive Config)
NXP_GAC_CFG = mdp.GridAdaptiveConfig()


@configclass
class NXPActions(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=NXP_JOINT_NAMES,
        scale=0.05,
        preserve_order=True,
        use_zero_offset=True,
    )


@configclass
class NXPRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_berkeley,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntity("contact_forces", body_names=".*ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold_min": 0.2,
            "threshold_max": 0.5,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntity("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntity("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize all joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntity("robot", joint_names=[".*_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntity("robot", joint_names=[".*_hip_.*"])},
    )
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntity("robot", joint_names=[".*_knee_.*"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntity("robot", joint_names=[".*_ankle_.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntity("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"])},
    )

    # Upright posture: L1 keeps a gradient near zero tilt, so the policy cannot lean to accelerate
    # roll and pitch are penalized separately so the pitch term alone sets the lean target
    base_roll_l1 = RewTerm(func=mdp.base_roll_l1, weight=-2.0)
    base_pitch_l1 = RewTerm(
        func=mdp.base_pitch_cmd_l1,
        weight=-2.0,
        # lean forward on backward commands to counter the backward lean seen on hardware
        params={"command_name": "base_velocity", "backward_lean": math.radians(3.0), "vel_ref": 0.5},
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.85},
    )

    smoothness_rate = RewTerm(func=mdp.smoothness_rate_l2, weight=-0.001)


@configclass
class NXPObervations:
    """Observation specifications for the MDP."""

    @configclass
    class ProprioceptiveCfg(ObsGroup):
        """Observations for proprioceptive group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class LinVelCfg(ObsGroup):
        """Observations for lin_vel group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class EstimatedLinVelCfg(ObsGroup):
        """Observations for estimated_lin_vel group."""

        # observation terms (order preserved)
        estimated_lin_vel = ObsTerm(func=mdp.estimated_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HeightScanCfg(ObsGroup):
        """Observations for height_scan group."""

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntity("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    proprioceptive: ProprioceptiveCfg = ProprioceptiveCfg()
    lin_vel: LinVelCfg = LinVelCfg()
    estimated_lin_vel: EstimatedLinVelCfg = EstimatedLinVelCfg()
    height_scan: HeightScanCfg = HeightScanCfg()


@configclass
class NXPEvent(EventCfg):
    """Randomization configurations for the MDP."""

    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntity("robot", body_names=".*"),
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
            "asset_cfg": SceneEntity("robot", body_names="base"),
            "mass_distribution_params": (-10.0, 10.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntity("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntity("robot", body_names="base"),
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
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
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


UNIFORM_VEL_CMD_CFG = mdp.UniformVelocityCommandCfg(
    asset_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.02,
    rel_heading_envs=1.0,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    ),
)

GAC_VEL_CMD_CFG = mdp.GridAdaptiveVelocityCommandCfg(
    asset_name="robot",
    resampling_time_range=(10.0, 10.0),
    debug_vis=True,
    gac_config=NXP_GAC_CFG,
    lin_vel_xy_std=0.5,
    ang_vel_z_std=0.5,
    ranges=mdp.GridAdaptiveVelocityCommandCfg.Ranges(
        lin_vel_x=NXP_GAC_CFG.limit_x_range, lin_vel_y=(-0.5, 0.5), ang_vel_z=NXP_GAC_CFG.limit_w_range
    ),
)

NXP_VEL_CMD_CFGS = {
    "uniform": UNIFORM_VEL_CMD_CFG,
    "gac": GAC_VEL_CMD_CFG,
}


@configclass
class NXPCommands:
    """Command specifications for the MDP."""

    base_velocity: mdp.UniformVelocityCommandCfg = NXP_VEL_CMD_CFGS


@configclass
class NXPRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: NXPActions = NXPActions()
    rewards: NXPRewards = NXPRewards()
    events: NXPEvent = NXPEvent()
    commands: NXPCommands = NXPCommands()
    observations: NXPObervations = NXPObervations()

    """Velocity command generator: ``"uniform"`` or ``"gac"`` (grid adaptive curriculum)."""
    command_type: Literal["uniform", "gac"] = "uniform"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Command generator
        self.commands.base_velocity = copy.deepcopy(NXP_VEL_CMD_CFGS[self.command_type])
        # Scene
        self.scene.robot = NXP_V1_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis_link"

        track_lin_vel_xy_std = 0.5
        track_ang_vel_z_std = 0.5

        # Commands
        if self.command_type == "gac":
            self.commands.base_velocity.lin_vel_xy_std = track_lin_vel_xy_std
            self.commands.base_velocity.ang_vel_z_std = track_ang_vel_z_std
            
            # lin_vel_x / ang_vel_z are sampled from the GAC grid, not from `ranges`
            self.commands.base_velocity.gac_config.linvel_thresh = 0.75
            self.commands.base_velocity.gac_config.angvel_thresh = 0.75

        # Observation
        self.observations.proprioceptive.joint_pos.params["asset_cfg"] = SceneEntity(
            "robot", joint_names=NXP_JOINT_NAMES, preserve_order=True
        )
        self.observations.proprioceptive.joint_pos.noise = Unoise(n_min=-0.05, n_max=0.05)
        self.observations.proprioceptive.joint_vel.params["asset_cfg"] = SceneEntity(
            "robot", joint_names=NXP_JOINT_NAMES, preserve_order=True
        )

        # Randomization
        # startup
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.events.base_com.params["asset_cfg"].body_names = ["torso_link"]
        # reset
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # interval
        self.events.push_robot.interval_range_s = (5.0, 10.0)

        # Rewards inspired by isaac berkeley
        self.rewards.track_lin_vel_xy_exp.params["std"] = track_lin_vel_xy_std
        self.rewards.track_ang_vel_z_exp.params["std"] = track_ang_vel_z_std

        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        # damps the pitch/roll transient when the gait starts
        self.rewards.ang_vel_xy_l2.weight = -0.2
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.dof_acc_l2 = None
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntity("robot", joint_names=[".*_joint"])
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_slide.weight = -0.25
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntity(
            "contact_forces",
            body_names=[".*_upper_arm_link", ".*_hip_yaw_link", "torso_link"],
        )
        self.rewards.joint_deviation_hip.weight = -0.2
        self.rewards.joint_deviation_knee.weight = -0.01
        self.rewards.joint_deviation_arms.weight = -0.1
        # base height is measured against the terrain on rough ground
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntity("height_scanner")

        # Commands: lin_vel_y is sampled from the ranges
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class NXPRoughEnvCfg_PLAY(NXPRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.proprioceptive.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
