# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for ERC robots."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg
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
            # "head_pan_joint": 0.0,
            # "head_tilt_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # "arm": IdealPDActuatorCfg(
        #     joint_names_expr=[".*"],
        #     effort_limit_sim=17.0,
        #     velocity_limit_sim=37.68,
        #     stiffness=17.0,
        #     damping=1.0,
        #     armature=0.0042,
        # ),
        # "arm": DelayedPDActuatorCfg(
        #     joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*"],
        #     effort_limit_sim=17.0,
        #     velocity_limit_sim=37.68,
        #     stiffness=17.0,
        #     damping=1.0,
        #     armature=0.0042,
        #     min_delay=0,
        #     max_delay=3,
        # ),
        # --- DelayedPD actuator configuration (identified gains) ---
        "arm": DelayedPDActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*"],
            effort_limit_sim=17.0,
            velocity_limit_sim=37.68,
            stiffness={
                "left_shoulder_pitch_joint": 222.659470,
                "right_shoulder_pitch_joint": 286.036621,
                "left_shoulder_roll_joint": 497.008209,
                "right_shoulder_roll_joint": 498.704407,
                "left_shoulder_yaw_joint": 499.942169,
                "right_shoulder_yaw_joint": 499.883453,
                "left_elbow_joint": 499.921722,
                "right_elbow_joint": 499.969849,
            },  # P gain in Nm/rad
            damping={
                "left_shoulder_pitch_joint": 1.215397,
                "right_shoulder_pitch_joint": 2.400836,
                "left_shoulder_roll_joint": 3.795808,
                "right_shoulder_roll_joint": 3.913681,
                "left_shoulder_yaw_joint": 1.732408,
                "right_shoulder_yaw_joint": 1.659626,
                "left_elbow_joint": 3.657888,
                "right_elbow_joint": 1.687085,
            },  # D gain in Nm s/rad
            armature={
                "left_shoulder_pitch_joint": 0.010149,
                "right_shoulder_pitch_joint": 0.016689,
                "left_shoulder_roll_joint": 0.014932,
                "right_shoulder_roll_joint": 0.021913,
                "left_shoulder_yaw_joint": 0.002554,
                "right_shoulder_yaw_joint": 0.020064,
                "left_elbow_joint": 0.111841,
                "right_elbow_joint": 0.079700,
            },  # rotor inertia (kg m^2)
            friction={
                "left_shoulder_pitch_joint": 0.284765,
                "right_shoulder_pitch_joint": 0.499680,
                "left_shoulder_roll_joint": 0.001165,
                "right_shoulder_roll_joint": 0.001890,
                "left_shoulder_yaw_joint": 0.499420,
                "right_shoulder_yaw_joint": 0.499784,
                "left_elbow_joint": 0.499287,
                "right_elbow_joint": 0.499473,
            },  # static (Coulomb) friction coefficient (Nm)
            dynamic_friction={
                "left_shoulder_pitch_joint": 0.284765,
                "right_shoulder_pitch_joint": 0.499680,
                "left_shoulder_roll_joint": 0.001165,
                "right_shoulder_roll_joint": 0.001890,
                "left_shoulder_yaw_joint": 0.499420,
                "right_shoulder_yaw_joint": 0.499784,
                "left_elbow_joint": 0.499287,
                "right_elbow_joint": 0.499473,
            },  # dynamic friction coefficient (Nm); equal to static for Coulomb model
            viscous_friction={
                "left_shoulder_pitch_joint": 0.911175,
                "right_shoulder_pitch_joint": 0.332654,
                "left_shoulder_roll_joint": 0.388731,
                "right_shoulder_roll_joint": 0.387890,
                "left_shoulder_yaw_joint": 2.870754,
                "right_shoulder_yaw_joint": 3.098242,
                "left_elbow_joint": 0.016841,
                "right_elbow_joint": 2.136223,
            },  # viscous friction coefficient (Nm s/rad)
            min_delay=0,
            max_delay=0,
        ),
        # "left_arm": PaceDCMotorCfg(
        #     joint_names_expr=["left_shoulder_.*", "left_elbow_.*"],
        #     saturation_effort=17.0,
        #     effort_limit=17.0,
        #     velocity_limit=37.68,
        #     stiffness={
        #         ".*_shoulder_pitch_.*": 197.056122,
        #         ".*_shoulder_roll_.*": 490.979828,
        #         ".*_shoulder_yaw_.*": 499.144501,
        #         ".*_elbow_.*": 497.635437,
        #     },  # P gain in Nm/rad
        #     damping={
        #         ".*_shoulder_pitch_.*": 0.742269,
        #         ".*_shoulder_roll_.*": 4.030707,
        #         ".*_shoulder_yaw_.*": 1.776445,
        #         ".*_elbow_.*": 3.101271,
        #     },  # D gain in Nm s/rad
        #     # --- Identified parameters ---
        #     encoder_bias={
        #         ".*_shoulder_pitch_.*": 0.097939,
        #         ".*_shoulder_roll_.*": 0.099312,
        #         ".*_shoulder_yaw_.*": -0.042336,
        #         ".*_elbow_.*": 0.099604,
        #     },  # encoder bias in radians
        #     armature={
        #         ".*_shoulder_pitch_.*": 0.015928,
        #         ".*_shoulder_roll_.*": 0.035970,
        #         ".*_shoulder_yaw_.*": 0.000436,
        #         ".*_elbow_.*": 0.075686,
        #     },  # rotor inertia (kg m^2)
        #     friction={
        #         ".*_shoulder_pitch_.*": 0.223557,
        #         ".*_shoulder_roll_.*": 0.017051,
        #         ".*_shoulder_yaw_.*": 0.493055,
        #         ".*_elbow_.*": 0.489961,
        #     },  # static (Coulomb) friction coefficient (Nm)
        #     dynamic_friction={
        #         ".*_shoulder_pitch_.*": 0.223557,
        #         ".*_shoulder_roll_.*": 0.017051,
        #         ".*_shoulder_yaw_.*": 0.493055,
        #         ".*_elbow_.*": 0.489961,
        #     },  # dynamic friction coefficient (Nm); equal to static for Coulomb model
        #     viscous_friction={
        #         ".*_shoulder_pitch_.*": 1.070898,
        #         ".*_shoulder_roll_.*": 0.092758,
        #         ".*_shoulder_yaw_.*": 2.680460,
        #         ".*_elbow_.*": 0.516676,
        #     },  # viscous friction coefficient (Nm s/rad)
        #     max_delay=0,  #
        # ),
        # "right_arm": PaceDCMotorCfg(
        #     joint_names_expr=["right_shoulder_.*", "right_elbow_.*"],
        #     saturation_effort=17.0,
        #     effort_limit=17.0,
        #     velocity_limit=37.68,
        #     stiffness={
        #         ".*_shoulder_pitch_joint": 241.678772,
        #         ".*_shoulder_roll_joint": 492.960785,
        #         ".*_shoulder_yaw_joint": 498.990387,
        #         ".*_elbow_joint": 499.138458,
        #     },  # P gain in Nm/rad
        #     damping={
        #         ".*_shoulder_pitch_joint": 0.166352,
        #         ".*_shoulder_roll_joint": 4.097325,
        #         ".*_shoulder_yaw_joint": 1.964294,
        #         ".*_elbow_joint": 2.457608,
        #     },  # D gain in Nm s/rad
        #     # --- Identified parameters ---
        #     encoder_bias={
        #         ".*_shoulder_pitch_joint": -0.094001,
        #         ".*_shoulder_roll_joint": -0.099424,
        #         ".*_shoulder_yaw_joint": 0.049218,
        #         ".*_elbow_joint": -0.099565,
        #     },  # encoder bias in radians
        #     armature={
        #         ".*_shoulder_pitch_joint": 0.015554,
        #         ".*_shoulder_roll_joint": 0.042742,
        #         ".*_shoulder_yaw_joint": 0.010356,
        #         ".*_elbow_joint": 0.051234,
        #     },  # rotor inertia (kg m^2)
        #     friction={
        #         ".*_shoulder_pitch_joint": 0.493800,
        #         ".*_shoulder_roll_joint": 0.019700,
        #         ".*_shoulder_yaw_joint": 0.496976,
        #         ".*_elbow_joint": 0.492035,
        #     },  # static (Coulomb) friction coefficient (Nm)
        #     dynamic_friction={
        #         ".*_shoulder_pitch_joint": 0.493800,
        #         ".*_shoulder_roll_joint": 0.019700,
        #         ".*_shoulder_yaw_joint": 0.496976,
        #         ".*_elbow_joint": 0.492035,
        #     },  # dynamic friction coefficient (Nm); equal to static for Coulomb model
        #     viscous_friction={
        #         ".*_shoulder_pitch_joint": 2.015668,
        #         ".*_shoulder_roll_joint": 0.126374,
        #         ".*_shoulder_yaw_joint": 2.831339,
        #         ".*_elbow_joint": 1.337506,
        #     },  # viscous friction coefficient (Nm s/rad)
        #     max_delay=0,  # identified delay: round(0.49767881631851196 steps) = 0 sim steps
        # ),
        "head": DelayedPDActuatorCfg(
            joint_names_expr=["head_.*"],
            effort_limit_sim=5.0,
            velocity_limit_sim=27.22,
            stiffness=5.0,
            damping=0.1,
            armature=0.001,
            min_delay=0,
            max_delay=3,
        ),
        # "arm": ImplicitActuatorCfg(
        #     joint_names_expr=[".*"],
        #     effort_limit_sim=17.0,
        #     stiffness=17.0,
        #     damping=1.0,
        # ),
        # "shoulder_pitch": IdealPDActuatorCfg(
        #     joint_names_expr=[".*_shoulder_pitch_.*"],
        #     effort_limit=17.0,
        #     velocity_limit=37.68,
        #     stiffness=10.0,
        #     damping=1.0,
        # ),
        # "shoulder_roll": IdealPDActuatorCfg(
        #     joint_names_expr=[".*_shoulder_roll_.*"],
        #     effort_limit=17.0,
        #     velocity_limit=37.68,
        #     stiffness=10.0,
        #     damping=1.0,
        # ),
        # "shoulder_yaw": IdealPDActuatorCfg(
        #     joint_names_expr=[".*_shoulder_yaw_.*"],
        #     effort_limit=17.0,
        #     velocity_limit=37.68,
        #     stiffness=5.0,
        #     damping=0.1,
        # ),
        # "elbow": IdealPDActuatorCfg(
        #     joint_names_expr=[".*_elbow_.*"],
        #     effort_limit=17.0,
        #     velocity_limit=37.68,
        #     stiffness=5.0,
        #     damping=0.1,
        # ),
    },
)
