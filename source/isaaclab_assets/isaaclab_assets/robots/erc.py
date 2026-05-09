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
        "hip_pitch": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*"],
            effort_limit=120.0,
            velocity_limit=17.48,
            stiffness=400.0,
            damping=2.5,
        ),
        "hip_roll": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*"],
            effort_limit=120.0,
            velocity_limit=17.48,
            stiffness=300.0,
            damping=3.0,
        ),
        "hip_yaw": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_yaw_.*"],
            effort_limit=60.0,
            velocity_limit=18.85,
            stiffness=150.0,
            damping=1.0,
        ),
        "knee": IdealPDActuatorCfg(
            joint_names_expr=[".*_knee_.*"],
            effort_limit=200.0,
            velocity_limit=17.48,
            stiffness=300.0,
            damping=1.5,
        ),
        "ankle_pitch": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_.*"],
            effort_limit=60.0,
            velocity_limit=18.85,
            stiffness=150.0,
            damping=1.0,
        ),
        "ankle_roll": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_roll_.*"],
            effort_limit=17.0,
            velocity_limit=37.68,
            stiffness=75.0,
            damping=1.0,
        ),
        "shoulder_pitch": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_.*"],
            effort_limit=17.0,
            velocity_limit=37.68,
            stiffness=10.0,
            damping=1.0,
        ),
        "shoulder_roll": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_roll_.*"],
            effort_limit=17.0,
            velocity_limit=37.68,
            stiffness=10.0,
            damping=1.0,
        ),
        "shoulder_yaw": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_yaw_.*"],
            effort_limit=17.0,
            velocity_limit=37.68,
            stiffness=5.0,
            damping=0.1,
        ),
        "elbow": IdealPDActuatorCfg(
            joint_names_expr=[".*_elbow_.*"],
            effort_limit=17.0,
            velocity_limit=37.68,
            stiffness=5.0,
            damping=0.1,
        ),
        "head": IdealPDActuatorCfg(
            joint_names_expr=["head_.*"],
            effort_limit=5.0,
            velocity_limit=27.22,
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
