# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different legged robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/qrc.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script for benchmark QRC.")
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

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils import play_utils


##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG  # isort:skip


def design_scene() -> tuple[Articulation, list[float]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Origin Unitree A1
    origin = [0.0, 0.0, 0.0]
    prim_utils.create_prim("/World/Envs", "Xform", translation=origin)
    # -- Robot
    unitree_a1 = Articulation(UNITREE_A1_CFG.replace(prim_path="/World/Envs/Robot"))

    # return the scene information
    return unitree_a1, origin


def reset(robot: Articulation, origin: torch.Tensor):
    # root state
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin
    robot.write_root_state_to_sim(root_state)
    # joint state
    joint_pos, joint_vel = (
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
    )
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    # reset the internal state
    robot.reset()
    print("[INFO]: Resetting robots state...")

    #
    proprio_obs = compute_proprio_obs(robot)


def compute_proprio_obs(robot: Articulation):
    obs_scales = {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "yaw": 1.0,
    }

    obs = torch.cat(
        (
            robot.data.root_ang_vel_b * obs_scales["ang_vel"],
            robot.data.projected_gravity_b,
            (robot.data.joint_pos - robot.data.default_joint_pos) * obs_scales["dof_pos"],
            robot.data.joint_vel * obs_scales["dof_vel"],
            # self.actions,
        ),
        dim=-1,
    )


def step(robot: Articulation, sim: sim_utils.SimulationContext, action: torch.Tensor):
    # apply default actions to the quadrupedal robots
    # generate random joint positions
    joint_pos_target = robot.data.default_joint_pos
    # apply action to the robot
    robot.set_joint_position_target(joint_pos_target)
    # write data to sim
    robot.write_data_to_sim()
    # perform step
    sim.step()


def run_simulator(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    origin: torch.Tensor,
    modules: play_utils.ModulesContainer,
):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Camera setup
    cam_pos = np.array([0.7, 1.5, 0.7])
    cam_target = np.array([0.5, 0.0, 0])
    cam_offset = np.array([0.0, -3.0, 2.0])
    k_smooth = 0.9

    # Reset the scene
    reset(robot, origin)

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 20000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset robots
            reset(robot, origin)

        # step
        step(robot, sim, action=torch.zeros(12))

        # update camera
        actor_pos = robot.data.root_pos_w.squeeze().cpu().numpy()
        new_cam_pos = actor_pos + cam_offset
        cam_pos = k_smooth * cam_pos + (1 - k_smooth) * new_cam_pos
        cam_target = k_smooth * cam_target + (1 - k_smooth) * actor_pos
        sim.set_camera_view(eye=tuple(cam_pos), target=tuple(cam_target))

        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)


def main():
    """Main function."""

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # Set main camera
    sim.set_camera_view(eye=(2.5, 2.5, 2.5), target=(0.0, 0.0, 0.0))
    # design scene
    scene_robot, scene_origin = design_scene()
    scene_origin = torch.tensor(scene_origin, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Load the checkpoint
    # modules = play_utils.load_modules(args_cli, sim)
    # print(f"[INFO]: Modules: {modules}")
    modules = None

    # Run the simulator
    run_simulator(sim, scene_robot, scene_origin, modules)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
