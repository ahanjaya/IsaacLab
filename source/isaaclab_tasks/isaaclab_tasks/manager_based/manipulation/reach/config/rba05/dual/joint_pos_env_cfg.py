# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.rba05.dual.reach_rba05_dual_env_cfg import ReachEnvCfg

from isaaclab_assets import RBA05_DUAL_CFG  # isort: skip

##
# Environment configuration
##


@configclass
class RBA05DualReachEnvCfg(ReachEnvCfg):
    """Configuration for the Dual RBA05 Reach Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to rba05
        self.scene.robot = RBA05_DUAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override rewards
        self.rewards.left_end_effector_position_tracking.params["asset_cfg"].body_names = ["left_ee_link"]
        self.rewards.left_end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["left_ee_link"]
        self.rewards.left_end_effector_orientation_tracking.params["asset_cfg"].body_names = ["left_ee_link"]

        self.rewards.right_end_effector_position_tracking.params["asset_cfg"].body_names = ["right_ee_link"]
        self.rewards.right_end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [
            "right_ee_link"
        ]
        self.rewards.right_end_effector_orientation_tracking.params["asset_cfg"].body_names = ["right_ee_link"]

        # override actions
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "left_joint[1-6]",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "right_joint[1-6]",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # override command generator body
        # end-effector is along z-direction
        self.commands.left_ee_pose.body_name = "left_ee_link"
        self.commands.right_ee_pose.body_name = "right_ee_link"


@configclass
class RBA05DualReachEnvCfg_PLAY(RBA05DualReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 3000.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # disable command resampling for play
        self.commands.left_ee_pose.resampling_time_range = (1e10, 1e10)
        self.commands.right_ee_pose.resampling_time_range = (1e10, 1e10)
