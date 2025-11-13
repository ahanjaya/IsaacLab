# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import NXPRoughEnvCfg


@configclass
class NXPFlatEnvCfg(NXPRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Randomization
        self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        }
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }

        # Rewards
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.flat_orientation_l2.weight = -5.0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class NXPFlatEnvCfg_PLAY(NXPFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # commands
        self.use_teleop = False
        if not self.use_teleop:
            return

        self.scene.num_envs = 1
        self.episode_length_s = 2000.0
        self.commands.base_velocity = mdp.Se2GamepadVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(0.0, 0.0),  # No automatic resampling for teleop
            rel_standing_envs=0.0,  # No standing environments for teleop
            rel_heading_envs=1.0,
            heading_command=False,  # Use direct angular velocity from gamepad
            debug_vis=True,
            gamepad_sensitivity=(0.5, 0.5, 0.5),  # Sensitivity for vx, vy, omega_z
            dead_zone=0.01,
            invert_axes=(
                False,
                True,
                True,
            ),  # Invert y and z axes for correct direction
            ranges=mdp.Se2GamepadVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
            ),
        )
