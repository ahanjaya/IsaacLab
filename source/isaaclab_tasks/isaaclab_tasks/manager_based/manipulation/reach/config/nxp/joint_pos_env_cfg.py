# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import NXP_V1_UPPER_BODY_CFG  # isort: skip


##
# Environment configuration
##

NXP_LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]

NXP_RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=NXP_LEFT_ARM_JOINT_NAMES,
        scale=0.5,
        preserve_order=True,
        use_default_offset=True,
        clip={
            "left_shoulder_pitch_joint": (-1.570796, 1.570796),
            "left_shoulder_roll_joint": (-3.141592, 0.087266),
            "left_shoulder_yaw_joint": (-1.570796, 1.570796),
            "left_elbow_joint": (-0.087266, 1.570796),
        },
    )
    right_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=NXP_RIGHT_ARM_JOINT_NAMES,
        scale=0.5,
        preserve_order=True,
        use_default_offset=True,
        clip={
            "right_shoulder_pitch_joint": (-1.570796, 1.570796),
            "right_shoulder_roll_joint": (-0.087266, 3.141592),
            "right_shoulder_yaw_joint": (-1.570796, 1.570796),
            "right_elbow_joint": (-1.570796, 0.087266),
        },
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    left_ee_pose = mdp.UniformPositionCommandCfg(
        asset_name="robot",
        body_name="left_ee_link",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPositionCommandCfg.Ranges(
            pos_x=(0.2, 0.4),
            pos_y=(0.1, 0.5),
            pos_z=(0.05, 0.5),
        ),
    )
    right_ee_pose = mdp.UniformPositionCommandCfg(
        asset_name="robot",
        body_name="right_ee_link",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPositionCommandCfg.Ranges(
            pos_x=(0.2, 0.4),
            pos_y=(-0.5, -0.1),
            pos_z=(0.05, 0.5),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=NXP_LEFT_ARM_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=NXP_RIGHT_ARM_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=NXP_LEFT_ARM_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=NXP_RIGHT_ARM_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_pose"})
        right_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "right_ee_pose"})
        left_actions = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        right_actions = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    left_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="left_ee_link"), "command_name": "left_ee_pose"},
    )
    right_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_ee_link"), "command_name": "right_ee_pose"},
    )
    left_end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_ee_link"),
            "std": 0.1,
            "command_name": "left_ee_pose",
        },
    )
    right_end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_ee_link"),
            "std": 0.1,
            "command_name": "right_ee_pose",
        },
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    smoothness_rate = RewTerm(func=mdp.smoothness_rate_l2, weight=-0.001)

    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=NXP_LEFT_ARM_JOINT_NAMES, preserve_order=True)},
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=NXP_RIGHT_ARM_JOINT_NAMES, preserve_order=True)},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )

    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "left_joint_vel", "weight": -0.001, "num_steps": 4500}
    )

    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "right_joint_vel", "weight": -0.001, "num_steps": 4500}
    )


@configclass
class NXPReachEnvCfg(ReachEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation

        # simulation settings
        self.sim.dt = 1.0 / 200.0

        # switch robot to nxp
        self.scene.robot = NXP_V1_UPPER_BODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)


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
