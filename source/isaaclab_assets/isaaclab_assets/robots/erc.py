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
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
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
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "rs04": DCMotorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_roll_.*", ".*_knee_.*", ".*_ankle_pitch_.*"],
            saturation_effort=120.0,
            effort_limit=120.0,
            velocity_limit=17.48,
            stiffness={
                "left_hip_pitch_joint": 2999.7544,
                "right_hip_pitch_joint": 2998.6531,
                "left_hip_roll_joint": 2999.8499,
                "right_hip_roll_joint": 2999.8149,
                "left_knee_joint": 2998.0867,
                "right_knee_joint": 2992.7515,
                "left_ankle_pitch_joint": 2989.7522,
                "right_ankle_pitch_joint": 2988.4546,
            },  # P gain in Nm/rad
            damping={
                "left_hip_pitch_joint": 26.3391,
                "right_hip_pitch_joint": 27.7255,
                "left_hip_roll_joint": 27.7248,
                "right_hip_roll_joint": 28.2316,
                "left_knee_joint": 23.1373,
                "right_knee_joint": 23.8175,
                "left_ankle_pitch_joint": 20.1623,
                "right_ankle_pitch_joint": 22.0971,
            },
            armature={
                "left_hip_pitch_joint": 0.0155,
                "right_hip_pitch_joint": 0.0307,
                "left_hip_roll_joint": 0.0001,
                "right_hip_roll_joint": 0.0002,
                "left_knee_joint": 0.0600,
                "right_knee_joint": 0.0382,
                "left_ankle_pitch_joint": 0.0746,
                "right_ankle_pitch_joint": 0.0744,
            },  # rotor inertia (kg m^2)
            friction={
                "left_hip_pitch_joint": 0.3054,
                "right_hip_pitch_joint": 0.0782,
                "left_hip_roll_joint": 0.0194,
                "right_hip_roll_joint": 0.0157,
                "left_knee_joint": 0.3334,
                "right_knee_joint": 0.4033,
                "left_ankle_pitch_joint": 0.4694,
                "right_ankle_pitch_joint": 0.4522,
            },  # static (Coulomb) friction coefficient (Nm)
            dynamic_friction={
                "left_hip_pitch_joint": 0.3054,
                "right_hip_pitch_joint": 0.0782,
                "left_hip_roll_joint": 0.0194,
                "right_hip_roll_joint": 0.0157,
                "left_knee_joint": 0.3334,
                "right_knee_joint": 0.4033,
                "left_ankle_pitch_joint": 0.4694,
                "right_ankle_pitch_joint": 0.4522,
            },  # dynamic friction coefficient (Nm); equal to static for Coulomb model
            viscous_friction={
                "left_hip_pitch_joint": 0.9959,
                "right_hip_pitch_joint": 2.2262,
                "left_hip_roll_joint": 0.0246,
                "right_hip_roll_joint": 0.0335,
                "left_knee_joint": 1.5067,
                "right_knee_joint": 1.8661,
                "left_ankle_pitch_joint": 4.8783,
                "right_ankle_pitch_joint": 2.6610,
            },  # viscous friction coefficient (Nm s/rad)
        ),
        "rs03": DCMotorCfg(
            joint_names_expr=[".*_hip_yaw_.*", ".*_ankle_roll_.*"],
            saturation_effort=60.0,
            effort_limit=60.0,
            velocity_limit=18.85,
            stiffness={
                "left_hip_yaw_joint": 2998.8665,
                "right_hip_yaw_joint": 1977.4078,
                "left_ankle_roll_joint": 847.1395,
                "right_ankle_roll_joint": 1393.1254,
            },  # P gain in Nm/rad
            damping={
                "left_hip_yaw_joint": 25.1778,
                "right_hip_yaw_joint": 21.6429,
                "left_ankle_roll_joint": 3.7022,
                "right_ankle_roll_joint": 8.1858,
            },  # D gain in Nm s/rad
            armature={
                "left_hip_yaw_joint": 0.1008,
                "right_hip_yaw_joint": 0.0887,
                "left_ankle_roll_joint": 0.1333,
                "right_ankle_roll_joint": 0.0489,
            },  # rotor inertia (kg m^2)
            friction={
                "left_hip_yaw_joint": 0.3409,
                "right_hip_yaw_joint": 0.0263,
                "left_ankle_roll_joint": 0.2694,
                "right_ankle_roll_joint": 0.3892,
            },  # static (Coulomb) friction coefficient (Nm)
            dynamic_friction={
                "left_hip_yaw_joint": 0.3409,
                "right_hip_yaw_joint": 0.0263,
                "left_ankle_roll_joint": 0.2694,
                "right_ankle_roll_joint": 0.3892,
            },  # dynamic friction coefficient (Nm); equal to static for Coulomb model
            viscous_friction={
                "left_hip_yaw_joint": 1.1551,
                "right_hip_yaw_joint": 0.7342,
                "left_ankle_roll_joint": 2.6901,
                "right_ankle_roll_joint": 2.0963,
            },  # viscous friction coefficient (Nm s/rad)
        ),
        "rs02": DCMotorCfg(
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
