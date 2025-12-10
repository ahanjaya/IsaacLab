# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for ERC robots."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##
NXP_DIR = "source/isaaclab_assets/data/Robots/ERC/Humanoids"
# Full Size Configuration [URDF]
NXP_V1_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.getcwd(),
            NXP_DIR,
            "nxp_v1_humanoid/nxp_v1_humanoid.usd",
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        # joint_pos={
        #     "left_hip_pitch_joint": 0.52,  # 30
        #     "right_hip_pitch_joint": -0.52,  # -30
        #     "left_hip_roll_joint": 0.05,  # 3
        #     "right_hip_roll_joint": -0.05,  # -3
        #     "left_hip_yaw_joint": -0.35,  # -20
        #     "right_hip_yaw_joint": 0.35,  # 20
        #     "left_knee_joint": -0.785,  # -45
        #     "right_knee_joint": 0.785,  # 45
        #     "left_ankle_pitch_joint": 0.436,  # 25
        #     "right_ankle_pitch_joint": -0.436,  # -25
        #     "left_ankle_roll_joint": 0.0,
        #     "right_ankle_roll_joint": 0.0,
        #     "torso_yaw_joint": 0.0,
        #     "left_shoulder_pitch_joint": 0.0,
        #     "right_shoulder_pitch_joint": 0.0,
        #     "left_shoulder_roll_joint": -0.13,  # -7.5
        #     "right_shoulder_roll_joint": 0.13,  # 7.5
        #     "left_shoulder_yaw_joint": -0.13,  # -7.5
        #     "right_shoulder_yaw_joint": 0.13,  # 7.5
        #     "left_elbow_joint": 0.52,  # 30
        #     "right_elbow_joint": -0.52,  # -30
        #     "head_pan_joint": 0.0,
        #     "head_tilt_joint": 0.0,
        # },
        joint_pos={
            "left_hip_pitch_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0,
            "right_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "torso_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "right_elbow_joint": 0.0,
            "head_pan_joint": 0.0,
            "head_tilt_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "torso": DCMotorCfg(
            joint_names_expr=["torso_yaw_joint"],
            effort_limit=17.0,
            saturation_effort=17.0,
            velocity_limit=33.93,
            stiffness=40.0,
            damping=2.0,
            armature=0.1,
        ),
        "hip_pitch": DCMotorCfg(
            joint_names_expr=[".*_hip_pitch_.*"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=15.74,
            stiffness=200.0,
            damping=6.0,
            armature=0.1,
        ),
        "hip_roll": DCMotorCfg(
            joint_names_expr=[".*_hip_roll_.*"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=15.74,
            stiffness=200.0,
            damping=6.0,
            armature=0.1,
        ),
        "hip_yaw": DCMotorCfg(
            joint_names_expr=[".*_hip_yaw_.*"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=15.74,
            stiffness=200.0,
            damping=6.0,
            armature=0.1,
        ),
        "knee": DCMotorCfg(
            joint_names_expr=[".*_knee_.*"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=15.74,
            stiffness=200.0,
            damping=6.0,
            armature=0.1,
        ),
        "ankle_pitch": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_.*"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=15.74,
            stiffness=200.0,
            damping=6.0,
            armature=0.1,
        ),
        "ankle_roll": DCMotorCfg(
            joint_names_expr=[".*_ankle_roll_.*"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=15.74,
            stiffness=200.0,
            damping=6.0,
            armature=0.1,
        ),
        "arm": DCMotorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*"],
            effort_limit=17.0,
            saturation_effort=17.0,
            velocity_limit=33.93,
            stiffness=30.0,
            damping=1.5,
            armature=0.1,
        ),
        "head": DCMotorCfg(
            joint_names_expr=["head_.*"],
            effort_limit=14.0,
            saturation_effort=14.0,
            velocity_limit=24.50,
            stiffness=30.0,
            damping=1.5,
            armature=0.1,
        ),
    },
)
