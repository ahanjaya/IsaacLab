# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import SceneEntityCfg as SceneEntity
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import NXP_V1_UPPER_BODY_CFG  # isort: skip


##
# Environment configuration
##

# Joint names for NXP humanoid
NXP_JOINT_NAMES = [
    # "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    # "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    # "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    # "left_elbow_joint",
    "right_elbow_joint",
]


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=NXP_JOINT_NAMES,
        scale=0.5,
        preserve_order=True,
        use_default_offset=True,
        clip={
            "right_shoulder_pitch_joint": (-1.570796, 1.570796),
            "right_shoulder_roll_joint": (-0.087266, 3.141592),
            "right_shoulder_yaw_joint": (-1.570796, 1.570796),
            "right_elbow_joint": (-1.570796, -0.087266),
        },
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPositionCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPositionCommandCfg.Ranges(
            pos_x=(0.2, 0.4),
            pos_y=(-0.5, -0.1),
            pos_z=(0.05, 0.5),
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    smoothness_rate = RewTerm(func=mdp.smoothness_rate_l2, weight=-0.001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=NXP_JOINT_NAMES)},
    )


@configclass
class NXPReachEnvCfg(ReachEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation

        # simulation settings
        self.sim.dt = 1.0 / 200.0

        # self.episode_length_s = 20.0

        # switch robot to nxp
        self.scene.robot = NXP_V1_UPPER_BODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)

        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["right_ee_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["right_ee_link"]
        # self.rewards.end_effector_position_tracking_fine_grained.weight = 0.2
        self.rewards.end_effector_orientation_tracking = None

        # observation
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntity(
            "robot", joint_names=NXP_JOINT_NAMES, preserve_order=True
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntity(
            "robot", joint_names=NXP_JOINT_NAMES, preserve_order=True
        )

        # override command generator body
        self.commands.ee_pose.body_name = "right_ee_link"

        # # tighten smoothness curriculum targets to reduce joint shaking
        # self.curriculum.action_rate.params["weight"] = -0.01
        # self.curriculum.joint_vel.params["weight"] = -0.005


@configclass
class NXPReachEnvCfg_PLAY(NXPReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 3000.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
