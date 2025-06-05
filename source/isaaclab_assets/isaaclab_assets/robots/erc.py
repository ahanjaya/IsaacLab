# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for ERC robots."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##
NXP_DIR = "source/isaaclab_assets/data/Robots/ERC/NXP"
NXP_LOWER_BODY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.getcwd(), NXP_DIR, "nxp_lower_body/nxp_lower_body.usd"),
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
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            "left_hip_pitch_joint": 0.52,  # 30 degrees
            "right_hip_pitch_joint": -0.52,  # -30 degrees
            "left_hip_roll_joint": -0.05,  # -3 degrees
            "right_hip_roll_joint": 0.05,  # 3 degrees
            "left_hip_yaw_joint": -0.35,  # -20 degrees
            "right_hip_yaw_joint": 0.35,  # 20 degrees
            ".*_knee_joint": 0.79,  # 45 degrees
            ".*_ankle_pitch_joint": -0.44,  # -25 degrees
            ".*_ankle_roll_joint": 0.0,  # 0 degrees
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)

NXP_LOWER_BODY_MINIMAL_CFG = NXP_LOWER_BODY_CFG.copy()
NXP_LOWER_BODY_MINIMAL_CFG.spawn.usd_path = os.path.join(
    os.getcwd(),
    NXP_DIR,
    "nxp_lower_body_minimal/nxp_lower_body_minimal.usd",
)


NXP_LOWER_BODY_WITH_TORSO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.getcwd(),
            NXP_DIR,
            "nxp_lower_body_with_torso/nxp_lower_body_with_torso.usd",
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
        joint_pos={
            "left_hip_pitch_joint": 0.52,  # 30 degrees
            "right_hip_pitch_joint": -0.52,  # -30 degrees
            "left_hip_roll_joint": -0.05,  # -3 degrees
            "right_hip_roll_joint": 0.05,  # 3 degrees
            "left_hip_yaw_joint": -0.35,  # -20 degrees
            "right_hip_yaw_joint": 0.35,  # 20 degrees
            ".*_knee_joint": 0.79,  # 45 degrees
            ".*_ankle_pitch_joint": -0.44,  # -25 degrees
            ".*_ankle_roll_joint": 0.0,  # 0 degrees
            "torso_yaw_joint": 0.0,  # 0 degrees
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "torso": DCMotorCfg(
            joint_names_expr=["torso_yaw_joint"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            armature=0.01,
            friction=0.0,
        ),
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_.*"],
            effort_limit=33.5,  # 88.0
            saturation_effort=33.5,  # 88.0
            velocity_limit=21.0,
            stiffness=25.0,  # 88.0
            damping=0.5,  # 5.0
            armature=0.01,
            friction=0.0,
        ),
        "knee": DCMotorCfg(
            joint_names_expr=[".*_knee_.*"],
            effort_limit=33.5 * 2,  # 139.0
            saturation_effort=33.5 * 2,  # 139.0
            velocity_limit=21.0,
            stiffness=25.0,  # 139.0
            damping=0.5,  # 5.0
            armature=0.01,
            friction=0.0,
        ),
        "ankle": DCMotorCfg(
            joint_names_expr=[".*_ankle_.*"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            armature=0.01,
            friction=0.0,
        ),
    },
)

NXP_LOWER_BODY_WITH_TORSO_MINIMAL_CFG = NXP_LOWER_BODY_WITH_TORSO_CFG.copy()
NXP_LOWER_BODY_WITH_TORSO_MINIMAL_CFG.spawn.usd_path = os.path.join(
    os.getcwd(),
    NXP_DIR,
    "nxp_lower_body_with_torso_minimal/nxp_lower_body_with_torso_minimal.usd",
)

# NXP Lower Body with Torso - OnShape
NXP_LOWER_BODY_WITH_TORSO_MINIMAL_ONSHAPE_CFG = NXP_LOWER_BODY_WITH_TORSO_CFG.copy()
NXP_LOWER_BODY_WITH_TORSO_MINIMAL_ONSHAPE_CFG.spawn.usd_path = os.path.join(
    os.getcwd(),
    NXP_DIR,
    "nxp_lower_body_w_torso_onshape/nxp_lower_body_w_torso_edit.usd",
)
NXP_LOWER_BODY_WITH_TORSO_MINIMAL_ONSHAPE_CFG.init_state.joint_pos = {
    "left_hip_pitch_joint": -0.52,  # 30 degrees
    "right_hip_pitch_joint": 0.52,  # -30 degrees
    "left_hip_roll_joint": -0.05,  # -3 degrees
    "right_hip_roll_joint": 0.05,  # 3 degrees
    "left_hip_yaw_joint": 0.35,  # 20 degrees
    "right_hip_yaw_joint": -0.35,  # 20 degrees
    "left_knee_joint": 0.79,  # 45 degrees
    "right_knee_joint": -0.79,  # -45 degrees
    "left_ankle_pitch_joint": 0.44,  # 25 degrees
    "right_ankle_pitch_joint": -0.44,  # -25 degrees
    "left_ankle_roll_joint": 0.0,  # 0 degrees
    "right_ankle_roll_joint": 0.0,  # 0 degrees
    "torso_yaw_joint": 0.0,  # 0 degrees
}


# NXP Humanoid Robot Full Size Configuration
NXP_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.getcwd(),
            NXP_DIR,
            "nxp_humanoid/nxp_humanoid.usd",
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
        joint_pos={
            "left_hip_pitch_joint": 0.52,  # 30 degrees
            "right_hip_pitch_joint": -0.52,  # -30 degrees
            "left_hip_roll_joint": -0.05,  # -3 degrees
            "right_hip_roll_joint": 0.05,  # 3 degrees
            "left_hip_yaw_joint": -0.35,  # -20 degrees
            "right_hip_yaw_joint": 0.35,  # 20 degrees
            ".*_knee_joint": 0.79,  # 45 degrees
            ".*_ankle_pitch_joint": -0.44,  # -25 degrees
            ".*_ankle_roll_joint": 0.0,  # 0 degrees
            "torso_yaw_joint": 0.0,  # 0 degrees
            ".*_shoulder_pitch_joint": 0.0,  # 0 degrees
            "left_shoulder_roll_joint": 0.13,  # 7.5 degrees
            "right_shoulder_roll_joint": -0.13,  # -7.5 degrees
            "left_shoulder_yaw_joint": -0.13,  # -7.5 degrees
            "right_shoulder_yaw_joint": 0.13,  # 7.5 degrees
            ".*_elbow_joint": -0.52,  # -30 degrees
            "head_.*": 0.0,  # 0 degrees
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "torso": DCMotorCfg(
            joint_names_expr=["torso_yaw_joint"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            armature=0.01,
            friction=0.0,
        ),
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_.*"],
            effort_limit=33.5,  # 88.0
            saturation_effort=33.5,  # 88.0
            velocity_limit=21.0,
            stiffness=25.0,  # 88.0
            damping=0.5,  # 5.0
            armature=0.01,
            friction=0.0,
        ),
        "knee": DCMotorCfg(
            joint_names_expr=[".*_knee_.*"],
            effort_limit=33.5 * 2,  # 139.0
            saturation_effort=33.5 * 2,  # 139.0
            velocity_limit=21.0,
            stiffness=25.0,  # 139.0
            damping=0.5,  # 5.0
            armature=0.01,
            friction=0.0,
        ),
        "ankle": DCMotorCfg(
            joint_names_expr=[".*_ankle_.*"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            armature=0.01,
            friction=0.0,
        ),
        "arm": DCMotorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            armature=0.01,
            friction=0.0,
        ),
        "head": DCMotorCfg(
            joint_names_expr=["head_.*"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            armature=0.01,
            friction=0.0,
        ),
    },
)

NXP_HUMANOID_MINIMAL_CFG = NXP_HUMANOID_CFG.copy()
NXP_HUMANOID_MINIMAL_CFG.spawn.usd_path = os.path.join(
    os.getcwd(),
    NXP_DIR,
    "nxp_humanoid_minimal/nxp_humanoid_minimal.usd",
)
