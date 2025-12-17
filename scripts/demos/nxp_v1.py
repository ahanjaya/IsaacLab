# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates nxp humanoid robot.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/nxp_humanoid.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import time
import torch
from collections import deque
from enum import Enum

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates nxp v1 humanoid robot.")
parser.add_argument("--plot", action="store_true", default=False, help="Plot robot joint actions and positions.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

##
# Pre-defined configs
##
from isaaclab_assets.robots.erc import NXP_V1_HUMANOID_CFG  # isort:skip


##
# State Machine Enum
##
class RobotState(Enum):
    """Enum for robot state machine states."""

    IDLE = 0
    RESET = 1
    SQUAT = 2
    STRAIGHT = 3
    INFERENCE = 4


def design_scene() -> Articulation:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    prim_utils.create_prim("/World/Origin1", "Xform", translation=[0.0, 0.0, 0.0])
    # -- Robot
    robot = Articulation(NXP_V1_HUMANOID_CFG.replace(prim_path="/World/Origin/Robot"))

    return robot


def joint_linear_interpolation(init_pos, target_pos, rate):
    """Interpolate between current and target by alpha."""
    rate = min(max(rate, 0.0), 1.0)
    return init_pos * (1 - rate) + target_pos * rate


def _plot_joint_visualization(queue) -> None:
    plt.ion()  # Turn on interactive mode

    # Configuration
    joint_names = ["Hip Pitch", "Hip Roll", "Hip Yaw", "Knee", "Ankle Pitch", "Ankle Roll"]
    num_joints = len(joint_names)
    window_size = 750
    ylim = (-0.75, 0.75)

    # Create figure and axes
    fig, ax = plt.subplots(num_joints, 2, figsize=(20, 15))

    # Initialize data structures
    joint_data = {}
    lines = {}

    for i, joint_name in enumerate(joint_names):
        for side in ["left", "right"]:
            col = 0 if side == "left" else 1
            key = f"{side}_{joint_name.lower().replace(' ', '_')}"

            # Create queues
            joint_data[f"{key}_target"] = deque(maxlen=window_size)
            joint_data[f"{key}_current"] = deque(maxlen=window_size)

            # Create lines
            (lines[f"{key}_target"],) = ax[i, col].plot(
                joint_data[f"{key}_target"], label=f"{side.title()} {joint_name} Target"
            )
            (lines[f"{key}_current"],) = ax[i, col].plot(
                joint_data[f"{key}_current"],
                label=f"{side.title()} {joint_name} Current",
            )

            # Configure subplot
            _configure_subplot(ax[i, col], f"{side.title()} {joint_name}", ylim)

    def _update_joint_data(data_tuple):
        """Update joint data from queue data."""
        data_keys = [
            "left_hip_pitch_target",
            "left_hip_pitch_current",
            "right_hip_pitch_target",
            "right_hip_pitch_current",
            "left_hip_roll_target",
            "left_hip_roll_current",
            "right_hip_roll_target",
            "right_hip_roll_current",
            "left_hip_yaw_target",
            "left_hip_yaw_current",
            "right_hip_yaw_target",
            "right_hip_yaw_current",
            "left_knee_target",
            "left_knee_current",
            "right_knee_target",
            "right_knee_current",
            "left_ankle_pitch_target",
            "left_ankle_pitch_current",
            "right_ankle_pitch_target",
            "right_ankle_pitch_current",
            "left_ankle_roll_target",
            "left_ankle_roll_current",
            "right_ankle_roll_target",
            "right_ankle_roll_current",
        ]

        for i, key in enumerate(data_keys):
            value = data_tuple[i]
            if value is None:
                joint_data[key].clear()
            else:
                joint_data[key].append(value)

    def _update_plots():
        """Update all plot lines."""
        for i, joint_name in enumerate(joint_names):
            for side in ["left", "right"]:
                col = 0 if side == "left" else 1
                key = f"{side}_{joint_name.lower().replace(' ', '_')}"

                target_queue = joint_data[f"{key}_target"]
                current_queue = joint_data[f"{key}_current"]

                if target_queue:
                    # Update action line
                    lines[f"{key}_target"].set_xdata(range(len(target_queue)))
                    lines[f"{key}_target"].set_ydata(target_queue)

                    # Update position line
                    lines[f"{key}_current"].set_xdata(range(len(current_queue)))
                    lines[f"{key}_current"].set_ydata(current_queue)

                    # Update axes
                    ax[i, col].relim()
                    ax[i, col].autoscale_view()

    # Main loop
    while True:
        # Process queue data
        while not queue.empty():
            data_tuple = queue.get()
            _update_joint_data(data_tuple)

        # Update plots
        _update_plots()

        # Refresh display
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.tight_layout()
        time.sleep(0.01)


def _configure_subplot(subplot, title, ylim):
    """Configure a subplot with common settings."""
    subplot.set_title(title)
    subplot.set_xlabel("Time Step")
    subplot.set_ylabel("Angle (rad)")
    subplot.legend()
    subplot.set_ylim(ylim)
    subplot.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)


def run_simulator(sim: sim_utils.SimulationContext, entity: Articulation):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    # print(f"[INFO]: Robot Joint Names: {entity.data.joint_names}")
    # print(f"[INFO]: Robot Default Joint Pos: {entity.data.default_joint_pos}")

    default_pose = entity.data.default_joint_pos.clone() * 0.0
    squat_pose = torch.tensor(
        [[
            0.0000,
            0.1309,
            0.0000,
            -0.1309,
            0.0000,
            0.0000,
            -0.0174,
            -0.0436,
            0.0174,
            0.0436,
            -0.0872,
            -0.1309,
            0.0872,
            0.1309,
            -0.1745,
            0.1745,
            0.1745,
            -0.1745,
            0.0872,
            -0.0872,
            -0.0174,
            0.0174,
        ]],
        device="cuda:0",
    )
    joint_pos_target = default_pose.clone()

    ticks_per_second = 1.0 / sim_dt
    list_ticks = np.arange(0.2, 1.0, 0.2)
    num_cycles = len(list_ticks)

    # State machine initialization
    cycle_count = 0
    current_state = RobotState.RESET
    idle_duration = 200
    idle_count = 0
    squat_count = 0
    straight_count = 0

    if args_cli.plot:
        # Joint names for NXP humanoid
        lower_body_joint_names = [
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
        ]

        # upper_body_joint_names = [
        #     "left_shoulder_pitch_joint",
        #     "right_shoulder_pitch_joint",
        #     "left_shoulder_roll_joint",
        #     "right_shoulder_roll_joint",
        #     "left_shoulder_yaw_joint",
        #     "right_shoulder_yaw_joint",
        #     "left_elbow_joint",
        #     "right_elbow_joint",
        # ]

        # get real_indices of the nxp joints
        lower_body_indices = [entity.data.joint_names.index(name) for name in lower_body_joint_names]
        # upper_body_indices = [entity.data.joint_names.index(name) for name in upper_body_joint_names]

        # Start the plotting function in a separate process
        queue = mp.Queue()
        mp_plot = mp.Process(target=_plot_joint_visualization, args=(queue,))
        mp_plot.start()

    # Simulate physics
    while simulation_app.is_running():
        robot = entity

        # Execute state actions
        if current_state == RobotState.RESET:
            # reset counters
            sim_time = 0.0
            cycle_count = 0

            # root state
            root_state = robot.data.default_root_state.clone()
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            # joint state
            joint_pos, joint_vel = (
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # reset the internal state
            robot.reset()
            print(f"[INFO]: State = {current_state.name} - Resetting robot state...")

            idle_count = 0
            current_state = RobotState.IDLE

        elif current_state == RobotState.IDLE:
            joint_pos_target = robot.data.default_joint_pos

            if idle_count >= idle_duration:
                current_state = RobotState.SQUAT
                idle_count = 0

            idle_count += 1

        elif current_state == RobotState.SQUAT:
            squat_rate = squat_count / (list_ticks[cycle_count % num_cycles] * ticks_per_second)
            joint_pos_target = joint_linear_interpolation(default_pose, squat_pose, squat_rate)

            if squat_rate >= 1.5:
                current_state = RobotState.STRAIGHT
                squat_count = 0

            squat_count += 1

        elif current_state == RobotState.STRAIGHT:
            straight_rate = straight_count / (list_ticks[cycle_count % num_cycles] * ticks_per_second)
            joint_pos_target = joint_linear_interpolation(squat_pose, default_pose, straight_rate)

            if straight_rate >= 1.5:
                current_state = RobotState.SQUAT
                straight_count = 0
                cycle_count += 1

            straight_count += 1

        elif current_state == RobotState.INFERENCE:
            pass

        if args_cli.plot:
            lower_body_target_pose = joint_pos_target[:, lower_body_indices].detach().cpu().numpy()[0]
            # upper_body_target_pose = joint_pos_target[:, upper_body_indices].detach().cpu().numpy()[0]
            lower_body_current_pose = robot.data.joint_pos[:, lower_body_indices].detach().cpu().numpy()[0]
            # upper_body_current_pose = robot.data.joint_pos[:, upper_body_indices].detach().cpu().numpy()[0]

            queue.put((
                lower_body_target_pose[0],  # left_hip_pitch_action
                lower_body_current_pose[0],  # left_hip_pitch_pos
                lower_body_target_pose[1],  # right_hip_pitch_action
                lower_body_current_pose[1],  # right_hip_pitch_pos
                lower_body_target_pose[2],  # left_hip_roll_action
                lower_body_current_pose[2],  # left_hip_roll_pos
                lower_body_target_pose[3],  # right_hip_roll_action
                lower_body_current_pose[3],  # right_hip_roll_pos
                lower_body_target_pose[4],  # left_hip_yaw_action
                lower_body_current_pose[4],  # left_hip_yaw_pos
                lower_body_target_pose[5],  # right_hip_yaw_action
                lower_body_current_pose[5],  # right_hip_yaw_pos
                lower_body_target_pose[6],  # left_knee_action
                lower_body_current_pose[6],  # left_knee_pos
                lower_body_target_pose[7],  # right_knee_action
                lower_body_current_pose[7],  # right_knee_pos
                lower_body_target_pose[8],  # left_ankle_pitch_action
                lower_body_current_pose[8],  # left_ankle_pitch_pos
                lower_body_target_pose[9],  # right_ankle_pitch_action
                lower_body_current_pose[9],  # right_ankle_pitch_pos
                lower_body_target_pose[10],  # left_ankle_roll_action
                lower_body_current_pose[10],  # left_ankle_roll_pos
                lower_body_target_pose[11],  # right_ankle_roll_action
                lower_body_current_pose[11],  # right_ankle_roll_pos
            ))

        # apply action to the robot
        robot.set_joint_position_target(joint_pos_target)
        # write data to sim
        robot.write_data_to_sim()

        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        # update buffers
        robot.update(sim_dt)


def main():
    """Main function."""

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.02))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene
    entity = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, entity)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
