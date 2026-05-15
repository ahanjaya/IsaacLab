# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for ERC robots."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            # fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            "left_hip_pitch_joint": 0.1309,
            "right_hip_pitch_joint": -0.1309,
            "left_hip_roll_joint": -0.0174,
            "right_hip_roll_joint": 0.0174,
            "left_hip_yaw_joint": -0.0872,
            "right_hip_yaw_joint": 0.0872,
            "left_knee_joint": -0.1745,
            "right_knee_joint": 0.1745,
            "left_ankle_pitch_joint": 0.0872,
            "right_ankle_pitch_joint": -0.0872,
            "left_ankle_roll_joint": -0.0174,
            "right_ankle_roll_joint": 0.0174,
            "left_shoulder_pitch_joint": 0.0000,
            "right_shoulder_pitch_joint": 0.0000,
            "left_shoulder_roll_joint": -0.0436,
            "right_shoulder_roll_joint": 0.0436,
            "left_shoulder_yaw_joint": -0.1309,
            "right_shoulder_yaw_joint": 0.1309,
            "left_elbow_joint": 0.1745,
            "right_elbow_joint": -0.1745,
            "head_pan_joint": 0.0000,
            "head_tilt_joint": 0.0000,
        },
        # joint_pos={
        #     "left_hip_pitch_joint": 0.0,
        #     "right_hip_pitch_joint": 0.0,
        #     "left_hip_roll_joint": 0.0,
        #     "right_hip_roll_joint": 0.0,
        #     "left_hip_yaw_joint": 0.0,
        #     "right_hip_yaw_joint": 0.0,
        #     "left_knee_joint": 0.0,
        #     "right_knee_joint": 0.0,
        #     "left_ankle_pitch_joint": 0.0,
        #     "right_ankle_pitch_joint": 0.0,
        #     "left_ankle_roll_joint": 0.0,
        #     "right_ankle_roll_joint": 0.0,
        #     "left_shoulder_pitch_joint": 0.0,
        #     "right_shoulder_pitch_joint": 0.0,
        #     "left_shoulder_roll_joint": 0.0,
        #     "right_shoulder_roll_joint": 0.0,
        #     "left_shoulder_yaw_joint": 0.0,
        #     "right_shoulder_yaw_joint": 0.0,
        #     "left_elbow_joint": 0.0,
        #     "right_elbow_joint": 0.0,
        #     "head_pan_joint": 0.0,
        #     "head_tilt_joint": 0.0,
        # },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "leg": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_.*",
                ".*_hip_roll_.*",
                ".*_hip_yaw_.*",
                ".*_knee_.*",
                ".*_ankle_pitch_.*",
                ".*_ankle_roll_.*",
            ],
            effort_limit_sim={
                ".*_hip_pitch_.*": 120.0,
                ".*_hip_roll_.*": 120.0,
                ".*_hip_yaw_.*": 60.0,
                ".*_knee_.*": 120.0,
                ".*_ankle_pitch_.*": 60.0,
                ".*_ankle_roll_.*": 17.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_.*": 17.48,
                ".*_hip_roll_.*": 17.48,
                ".*_hip_yaw_.*": 18.85,
                ".*_knee_.*": 17.48,
                ".*_ankle_pitch_.*": 18.85,
                ".*_ankle_roll_.*": 37.68,
            },
            stiffness={
                ".*_hip_pitch_.*": 400.0,
                ".*_hip_roll_.*": 300.0,
                ".*_hip_yaw_.*": 150.0,
                ".*_knee_.*": 300.0,
                ".*_ankle_pitch_.*": 150.0,
                ".*_ankle_roll_.*": 75.0,
            },
            damping={
                ".*_hip_pitch_.*": 2.5,
                ".*_hip_roll_.*": 3.0,
                ".*_hip_yaw_.*": 1.0,
                ".*_knee_.*": 1.5,
                ".*_ankle_pitch_.*": 1.0,
                ".*_ankle_roll_.*": 1.0,
            },
        ),
        "arm": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_.*",
                ".*_shoulder_roll_.*",
                ".*_shoulder_yaw_.*",
                ".*_elbow_.*",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_.*": 17.0,
                ".*_shoulder_roll_.*": 17.0,
                ".*_shoulder_yaw_.*": 17.0,
                ".*_elbow_.*": 17.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_.*": 37.68,
                ".*_shoulder_roll_.*": 37.68,
                ".*_shoulder_yaw_.*": 37.68,
                ".*_elbow_.*": 37.68,
            },
            stiffness={
                ".*_shoulder_pitch_.*": 10.0,
                ".*_shoulder_roll_.*": 10.0,
                ".*_shoulder_yaw_.*": 5.0,
                ".*_elbow_.*": 5.0,
            },
            damping={
                ".*_shoulder_pitch_.*": 1.0,
                ".*_shoulder_roll_.*": 1.0,
                ".*_shoulder_yaw_.*": 0.1,
                ".*_elbow_.*": 0.1,
            },
        ),
        "head": IdealPDActuatorCfg(
            joint_names_expr=["head_.*"],
            effort_limit_sim=5.0,
            velocity_limit_sim=27.22,
            stiffness=5.0,
            damping=0.1,
        ),
        # "hip_pitch": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_hip_pitch_.*"],
        #     effort_limit_sim=120.0,
        #     stiffness=400.0,
        #     damping=30.0,
        # ),
        # "hip_roll": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_hip_roll_.*"],
        #     effort_limit_sim=120.0,
        #     stiffness=400.0,
        #     damping=50.0,
        # ),
        # "hip_yaw": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_hip_yaw_.*"],
        #     effort_limit_sim=60.0,
        #     stiffness=300.0,
        #     damping=30.0,
        # ),
        # "knee": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_knee_.*"],
        #     effort_limit_sim=120.0,
        #     stiffness=300.0,
        #     damping=30.0,
        # ),
        # "ankle_pitch": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_ankle_pitch_.*"],
        #     effort_limit_sim=60.0,
        #     stiffness=300.0,
        #     damping=30.0,
        # ),
        # "ankle_roll": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_ankle_roll_.*"],
        #     effort_limit_sim=17.0,
        #     stiffness=300.0,
        #     damping=20.0,
        # ),
        # "shoulder_pitch": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_shoulder_pitch_.*"],
        #     effort_limit_sim=17.0,
        #     stiffness=20.0,
        #     damping=5.0,
        # ),
        # "shoulder_roll": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_shoulder_roll_.*"],
        #     effort_limit_sim=17.0,
        #     stiffness=25.0,
        #     damping=5.0,
        # ),
        # "shoulder_yaw": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_shoulder_yaw_.*"],
        #     effort_limit_sim=17.0,
        #     stiffness=20.0,
        #     damping=5.0,
        # ),
        # "elbow": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_elbow_.*"],
        #     effort_limit_sim=17.0,
        #     stiffness=25.0,
        #     damping=5.0,
        # ),
        # "head_pan": ImplicitActuatorCfg(
        #     joint_names_expr=["head_pan.*"],
        #     effort_limit_sim=5.0,
        #     stiffness=20.0,
        #     damping=5.0,
        # ),
        # "head_tilt": ImplicitActuatorCfg(
        #     joint_names_expr=["head_tilt.*"],
        #     effort_limit_sim=5.0,
        #     stiffness=20.0,
        #     damping=5.0,
        # ),
    },
)
