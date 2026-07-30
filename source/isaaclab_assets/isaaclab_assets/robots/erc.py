# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
            "left_ankle_pitch_joint": -0.0872,
            "right_ankle_pitch_joint": 0.0872,
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
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "rs04": DCMotorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_roll_.*", ".*_knee_.*"],
            saturation_effort=120.0,
            effort_limit=120.0,
            velocity_limit=17.48,
            stiffness={
                "left_hip_pitch_.*": 2999.710938,
                "left_hip_roll_.*": 2999.903809,
                "left_knee_.*": 1649.400635,
                "right_hip_pitch_.*": 2039.258179,
                "right_hip_roll_.*": 2999.894043,
                "right_knee_.*": 1413.421509,
            },  # P gain in Nm/rad
            damping={
                "left_hip_pitch_.*": 26.821861,
                "left_hip_roll_.*": 26.393042,
                "left_knee_.*": 13.200838,
                "right_hip_pitch_.*": 19.454243,
                "right_hip_roll_.*": 27.310148,
                "right_knee_.*": 11.965180,
            },  # D gain in Nm s/rad
            armature={
                "left_hip_pitch_.*": 0.018970,
                "left_hip_roll_.*": 0.000062,
                "left_knee_.*": 0.010761,
                "right_hip_pitch_.*": 0.086239,
                "right_hip_roll_.*": 0.000066,
                "right_knee_.*": 0.000132,
            },  # rotor inertia (kg m^2)
            friction={
                "left_hip_pitch_.*": 0.305404,
                "left_hip_roll_.*": 0.010399,
                "left_knee_.*": 0.386539,
                "right_hip_pitch_.*": 0.010378,
                "right_hip_roll_.*": 0.003964,
                "right_knee_.*": 0.475646,
            },  # static (Coulomb) friction coefficient (Nm)
            dynamic_friction={
                "left_hip_pitch_.*": 0.305404,
                "left_hip_roll_.*": 0.010399,
                "left_knee_.*": 0.386539,
                "right_hip_pitch_.*": 0.010378,
                "right_hip_roll_.*": 0.003964,
                "right_knee_.*": 0.475646,
            },  # dynamic friction coefficient (Nm); equal to static for Coulomb model
            viscous_friction={
                "left_hip_pitch_.*": 0.498768,
                "left_hip_roll_.*": 0.010890,
                "left_knee_.*": 0.250857,
                "right_hip_pitch_.*": 0.937754,
                "right_hip_roll_.*": 0.011357,
                "right_knee_.*": 0.025867,
            },  # viscous friction coefficient (Nm s/rad)
        ),
        "rs03": DCMotorCfg(
            joint_names_expr=[".*_hip_yaw_.*", ".*_ankle_pitch_.*"],
            saturation_effort=60.0,
            effort_limit=60.0,
            velocity_limit=18.85,
            stiffness={
                "left_hip_yaw_.*": 2997.733154,
                "left_ankle_pitch_.*": 251.658020,
                "right_hip_yaw_.*": 842.287354,
                "right_ankle_pitch_.*": 357.985382,
            },  # P gain in Nm/rad
            damping={
                "left_hip_yaw_.*": 22.630350,
                "left_ankle_pitch_.*": 1.386608,
                "right_hip_yaw_.*": 9.492960,
                "right_ankle_pitch_.*": 1.833874,
            },  # D gain in Nm s/rad
            armature={
                "left_hip_yaw_.*": 0.177752,
                "left_ankle_pitch_.*": 0.050932,
                "right_hip_yaw_.*": 0.008274,
                "right_ankle_pitch_.*": 0.052342,
            },  # rotor inertia (kg m^2)
            friction={
                "left_hip_yaw_.*": 0.435820,
                "left_ankle_pitch_.*": 0.230562,
                "right_hip_yaw_.*": 0.001204,
                "right_ankle_pitch_.*": 0.297085,
            },  # static (Coulomb) friction coefficient (Nm)
            dynamic_friction={
                "left_hip_yaw_.*": 0.435820,
                "left_ankle_pitch_.*": 0.230562,
                "right_hip_yaw_.*": 0.001204,
                "right_ankle_pitch_.*": 0.297085,
            },  # dynamic friction coefficient (Nm); equal to static for Coulomb model
            viscous_friction={
                "left_hip_yaw_.*": 3.635791,
                "left_ankle_pitch_.*": 0.612513,
                "right_hip_yaw_.*": 0.052139,
                "right_ankle_pitch_.*": 0.971763,
            },  # viscous friction coefficient (Nm s/rad)
        ),
        "rs02": DCMotorCfg(
            joint_names_expr=[".*_ankle_roll_.*", ".*_shoulder_.*", ".*_elbow_.*"],
            saturation_effort=17.0,
            effort_limit=17.0,
            velocity_limit=37.68,
            stiffness={
                "left_ankle_roll_.*": 495.149200,
                "right_ankle_roll_.*": 909.981262,
                "left_shoulder_pitch_joint": 222.121582,
                "right_shoulder_pitch_joint": 285.139221,
                "left_shoulder_roll_joint": 490.476776,
                "right_shoulder_roll_joint": 491.228821,
                "left_shoulder_yaw_joint": 498.726837,
                "right_shoulder_yaw_joint": 497.425354,
                "left_elbow_joint": 498.454926,
                "right_elbow_joint": 499.244324,
            },  # P gain in Nm/rad
            damping={
                "left_ankle_roll_.*": 2.183070,
                "right_ankle_roll_.*": 6.398871,
                "left_shoulder_pitch_joint": 1.196230,
                "right_shoulder_pitch_joint": 2.321509,
                "left_shoulder_roll_joint": 3.956753,
                "right_shoulder_roll_joint": 4.069930,
                "left_shoulder_yaw_joint": 1.769438,
                "right_shoulder_yaw_joint": 1.635402,
                "left_elbow_joint": 3.328801,
                "right_elbow_joint": 2.782490,
            },  # D gain in Nm s/rad
            armature={
                "left_ankle_roll_.*": 0.078677,
                "right_ankle_roll_.*": 0.029191,
                "left_shoulder_pitch_joint": 0.010347,
                "right_shoulder_pitch_joint": 0.016394,
                "left_shoulder_roll_joint": 0.009989,
                "right_shoulder_roll_joint": 0.017516,
                "left_shoulder_yaw_joint": 0.002867,
                "right_shoulder_yaw_joint": 0.020833,
                "left_elbow_joint": 0.108637,
                "right_elbow_joint": 0.084163,
            },  # rotor inertia (kg m^2)
            friction={
                "left_ankle_roll_.*": 0.190270,
                "right_ankle_roll_.*": 0.374299,
                "left_shoulder_pitch_joint": 0.286563,
                "right_shoulder_pitch_joint": 0.492618,
                "left_shoulder_roll_joint": 0.025404,
                "right_shoulder_roll_joint": 0.026012,
                "left_shoulder_yaw_joint": 0.488045,
                "right_shoulder_yaw_joint": 0.495313,
                "left_elbow_joint": 0.484832,
                "right_elbow_joint": 0.488344,
            },  # static (Coulomb) friction coefficient (Nm)
            dynamic_friction={
                "left_ankle_roll_.*": 0.190270,
                "right_ankle_roll_.*": 0.374299,
                "left_shoulder_pitch_joint": 0.286563,
                "right_shoulder_pitch_joint": 0.492618,
                "left_shoulder_roll_joint": 0.025404,
                "right_shoulder_roll_joint": 0.026012,
                "left_shoulder_yaw_joint": 0.488045,
                "right_shoulder_yaw_joint": 0.495313,
                "left_elbow_joint": 0.484832,
                "right_elbow_joint": 0.488344,
            },  # dynamic friction coefficient (Nm); equal to static for Coulomb model
            viscous_friction={
                "left_ankle_roll_.*": 1.496426,
                "right_ankle_roll_.*": 0.205528,
                "left_shoulder_pitch_joint": 0.919618,
                "right_shoulder_pitch_joint": 0.403605,
                "left_shoulder_roll_joint": 0.149416,
                "right_shoulder_roll_joint": 0.161509,
                "left_shoulder_yaw_joint": 2.833519,
                "right_shoulder_yaw_joint": 3.108745,
                "left_elbow_joint": 0.326940,
                "right_elbow_joint": 1.044234,
            },  # viscous friction coefficient (Nm s/rad)
        ),
        "rs00": DCMotorCfg(
            joint_names_expr=["head_.*"],
            saturation_effort=17.0,
            effort_limit=5.0,
            velocity_limit=27.22,
            stiffness=5.0,
            damping=0.1,
            armature=0.001,
        ),
    },
)

NXP_V1_UPPER_BODY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.getcwd(),
            NXP_DIR,
            "nxp_v1_upper_body/nxp_v1_upper_body.usd",
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # disable_gravity=False,
            # retain_accelerations=False,
            # linear_damping=0.0,
            # angular_damping=0.0,
            # max_linear_velocity=1000.0,
            # max_angular_velocity=1000.0,
            # max_depenetration_velocity=1.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=8,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": -0.0436,
            "right_shoulder_roll_joint": 0.0436,
            "left_shoulder_yaw_joint": -0.1309,
            "right_shoulder_yaw_joint": 0.1309,
            "left_elbow_joint": 0.1745,
            "right_elbow_joint": -0.1745,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arm": DCMotorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*"],
            saturation_effort=17.0,
            effort_limit=17.0,
            velocity_limit=37.68,
            stiffness={
                "left_shoulder_pitch_joint": 222.121582,
                "right_shoulder_pitch_joint": 285.139221,
                "left_shoulder_roll_joint": 490.476776,
                "right_shoulder_roll_joint": 491.228821,
                "left_shoulder_yaw_joint": 498.726837,
                "right_shoulder_yaw_joint": 497.425354,
                "left_elbow_joint": 498.454926,
                "right_elbow_joint": 499.244324,
            },  # P gain in Nm/rad
            damping={
                "left_shoulder_pitch_joint": 1.196230,
                "right_shoulder_pitch_joint": 2.321509,
                "left_shoulder_roll_joint": 3.956753,
                "right_shoulder_roll_joint": 4.069930,
                "left_shoulder_yaw_joint": 1.769438,
                "right_shoulder_yaw_joint": 1.635402,
                "left_elbow_joint": 3.328801,
                "right_elbow_joint": 2.782490,
            },  # D gain in Nm s/rad
            armature={
                "left_shoulder_pitch_joint": 0.010347,
                "right_shoulder_pitch_joint": 0.016394,
                "left_shoulder_roll_joint": 0.009989,
                "right_shoulder_roll_joint": 0.017516,
                "left_shoulder_yaw_joint": 0.002867,
                "right_shoulder_yaw_joint": 0.020833,
                "left_elbow_joint": 0.108637,
                "right_elbow_joint": 0.084163,
            },  # rotor inertia (kg m^2)
            friction={
                "left_shoulder_pitch_joint": 0.286563,
                "right_shoulder_pitch_joint": 0.492618,
                "left_shoulder_roll_joint": 0.025404,
                "right_shoulder_roll_joint": 0.026012,
                "left_shoulder_yaw_joint": 0.488045,
                "right_shoulder_yaw_joint": 0.495313,
                "left_elbow_joint": 0.484832,
                "right_elbow_joint": 0.488344,
            },  # static (Coulomb) friction coefficient (Nm)
            dynamic_friction={
                "left_shoulder_pitch_joint": 0.286563,
                "right_shoulder_pitch_joint": 0.492618,
                "left_shoulder_roll_joint": 0.025404,
                "right_shoulder_roll_joint": 0.026012,
                "left_shoulder_yaw_joint": 0.488045,
                "right_shoulder_yaw_joint": 0.495313,
                "left_elbow_joint": 0.484832,
                "right_elbow_joint": 0.488344,
            },  # dynamic friction coefficient (Nm); equal to static for Coulomb model
            viscous_friction={
                "left_shoulder_pitch_joint": 0.919618,
                "right_shoulder_pitch_joint": 0.403605,
                "left_shoulder_roll_joint": 0.149416,
                "right_shoulder_roll_joint": 0.161509,
                "left_shoulder_yaw_joint": 2.833519,
                "right_shoulder_yaw_joint": 3.108745,
                "left_elbow_joint": 0.326940,
                "right_elbow_joint": 1.044234,
            },  # viscous friction coefficient (Nm s/rad)
        ),
        "head": DCMotorCfg(
            joint_names_expr=["head_.*"],
            saturation_effort=17.0,
            effort_limit=5.0,
            velocity_limit=27.22,
            stiffness=5.0,
            damping=0.1,
            armature=0.001,
        ),
    },
)
