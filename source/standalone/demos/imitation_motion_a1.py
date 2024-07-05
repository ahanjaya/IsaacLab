# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different legged robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/a1_imitation_motion.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates a1 imitation motion from animation.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import os

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation

##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG  # isort:skip
from omni.isaac.lab.utils import motion_imitation_utils as miu


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)

    # create a grid of origins
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")

    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0

    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=1, spacing=1.5)

    # Origin with Unitree A1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins)

    # -- Robot
    unitree_a1 = Articulation(UNITREE_A1_CFG.replace(prim_path="/World/Origin1/Robot"))

    # return the scene information
    scene_entities = {
        "unitree_a1": unitree_a1
    }

    return scene_entities, origins


def _reset(entities, origins, sim_time, count, max_episode_length):
    if count % max_episode_length == 0:
        # reset counters
        sim_time = 0.0
        count = 0

        # reset robots
        for index, robot in enumerate(entities.values()):
            # root state
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins[index]
            robot.write_root_state_to_sim(root_state)

            # joint state
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # reset the internal state
            robot.reset()

        print("[INFO]: Resetting robots state...")

    return sim_time, count


def _load_motion(motion_path, max_time, sim):
    """Loads a reference motion from disk. Pre-generates all frames and push
    them to GPU for later use.

    """
    dt = sim.get_physics_dt() / 2
    decimation = 4

    print("Loading motion data...")
    motion = miu.MotionData(motion_path)
    print(f"\tFrames: {motion.get_num_frames()}")
    print(f"\tFrame duration: {motion.get_frame_duration()}")

    step_size = dt * decimation
    motion_length = motion.get_num_frames()

    # Pre-generate all frames for the whole episode + some extra cycles.
    # The extra cycles are needed because the robot is reset to a random
    # reference index between 0 and 2 cycles.
    time_axis = np.arange(0, max_time + 5 * step_size * motion_length, step_size)
    print(f"\tTime_axis: {time_axis.shape}")

    np_pose_frames = []
    np_vel_frames = []
    for t in time_axis:
        pose = motion.calc_frame(t)
        vels = motion.calc_frame_vel(t)

        # NOTE: Order of joints in IsaacLab differs from PyBullet.
        # PyBullet:
        # FR Hip, FR Thigh, FR Calf,
        # FL Hip, FL Thigh, FL Calf,
        # RR Hip, RR Thigh, RR Calf,
        # RL Hip, RL Thigh, RL Calf,

        # IsaacLab
        # FL Hip,   FR Hip,     RL Hip,     RR Hip
        # FL Thigh, FR Thigh,   RL Thigh,   RR Thigh
        # FL Calf,  FF Calf,    RL Calf,    RR calf

        reordered_pose = np.array([
            pose[0], pose[1], pose[2],  # X, Y, Z Pos
            pose[6], pose[3], pose[4], pose[5],  # W, X, Y, Z Quat
            pose[10], pose[7], pose[16], pose[13],  # HIP -> FL, FR, RL, RR
            pose[11], pose[8], pose[17], pose[14],  # Thigh -> FL, FR, RL, RR
            pose[12], pose[9], pose[18], pose[15],  # Calf -> FL, FR, RL, RR
        ])

        reordered_vels = np.array([
            vels[0], vels[1], vels[2],  # Lin vel (No change).
            vels[3], vels[4], vels[5],  # Ang vel
            pose[9],  pose[6], pose[15], pose[12],  # HIP -> FL, FR, RL, RR
            pose[10], pose[7], pose[16], pose[13],  # Thigh -> FL, FR, RL, RR
            pose[11], pose[8], pose[17], pose[14],  # Calf -> FL, FR, RL, RR
        ])

        np_pose_frames.append(reordered_pose)
        np_vel_frames.append(reordered_vels)

    np_pose_frames = np.array(np_pose_frames)
    np_vel_frames = np.array(np_vel_frames)

    assert np_pose_frames.shape[0] == np_vel_frames.shape[0]

    # Animation length also defines the maximum episode length
    # Makes sure episode finished before we run out of future frames to index
    # in the observations.
    max_episode_length = np_pose_frames.shape[0] - 4 * motion_length - 1
    print(f"Max episode length is {max_episode_length}.")

    # Convert to PyTorch GPU tensors.
    tensor_ref_pose = torch.tensor(np_pose_frames, dtype=torch.float32, device=sim.device)
    tensor_ref_vels = torch.tensor(np_vel_frames, dtype=torch.float32, device=sim.device)

    # Create other useful views.
    tensor_ref_root_pose = tensor_ref_pose[:, :7]  # XYZ + Quat
    tensor_ref_pd_targets = tensor_ref_pose[:, 7:]  # 12 joints
    tensor_ref_root_vels = tensor_ref_vels[:, :6]  # Linear XYZ + Angular XYZ
    tensor_ref_pd_vels = tensor_ref_vels[:, 6:]

    return max_episode_length, tensor_ref_root_pose, tensor_ref_pd_targets, tensor_ref_root_vels, tensor_ref_pd_vels


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    motions_root = "source/extensions/omni.isaac.lab_assets/data/Motions"
    motion_fn = "pace_remove_yaw.txt"
    # motion_fn = "trot_remove_yaw.txt"
    # motion_fn = "pace_backward_remove_yaw.txt"
    # motion_fn = "trot_backward_remove_yaw.txt"
    # motion_fn = "hop-inplace-3_unity.txt"
    max_time = 15.0

    (
        max_episode_length,
        tensor_ref_root_pose,
        tensor_ref_pd_targets,
        tensor_ref_root_vels,
        tensor_ref_pd_vels
    ) = _load_motion(
        motion_path=os.path.join(os.getcwd(), motions_root, motion_fn),
        max_time=max_time,
        sim=sim,
    )

    cam_pos = np.array([0.7, 1.5, 0.7])
    cam_target = np.array([0.5, 0.0, 0])
    cam_offset = np.array([0.0, -1.0, 0.20])
    k_smooth = 0.9

    # Simulate physics
    while simulation_app.is_running():
        # reset
        sim_time, count = _reset(entities, origins, sim_time, count, max_episode_length)

        # apply default actions to the quadrupedal robots
        for robot in entities.values():
            # apply action to the robot
            robot.set_joint_position_target(tensor_ref_pd_targets[count])
            robot.set_joint_velocity_target(tensor_ref_pd_vels[count])

            # write data to sim
            robot.write_root_pose_to_sim(tensor_ref_root_pose[count])
            robot.write_root_velocity_to_sim(tensor_ref_root_vels[count])

            robot.write_data_to_sim()

        # perform step
        sim.step()

        actor_pos = robot.data.root_pos_w.squeeze().cpu().numpy()
        new_cam_pos = actor_pos + cam_offset
        cam_pos = k_smooth * cam_pos + (1 - k_smooth) * new_cam_pos
        cam_target = k_smooth * cam_target + (1 - k_smooth) * actor_pos
        sim.set_camera_view(eye=cam_pos, target=cam_target)

        # update sim-time
        sim_time += sim_dt
        count += 1

        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.005,  # slower in visual 200Hz
            # dt=0.01, # faster in visual 100Hz
            gravity=(0.0, 0.0, -9.81),
            render_interval=1, # decimation
        )
    )

    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()
