# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Torch Jit."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Torch Jit.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Torch Jit checkpoint file to load from.",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

from omni.isaac.lab.utils import play_utils
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.torch_jit import JitVecEnvWrapper

np.set_printoptions(precision=2, suppress=True)

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap around environment for rsl-rl
    env = JitVecEnvWrapper(env)

    # load previously trained model
    modules_runner = play_utils.load_modules(args_cli, env.device)
    print(f"[INFO]: Loading model checkpoint from: {args_cli.checkpoint_path}")
    print(modules_runner)

    # reset environment
    obs = env.get_observations()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # process observations
            process_obs = play_utils.process_observations(
                proprio_obs=obs["proprio_obs"],
                task_obs=obs["task_obs"],
                encoder_obs_hist=obs["encoder_obs_hist"],
                depth_map_obs=obs["depth_map_obs"],
                modules=modules_runner,
            )

            # agent stepping
            actions = modules_runner.policy(process_obs.detach())

            # TODO: replace the following line with the above line
            # actions = torch.zeros(
            #     (env.num_envs, env.num_actions), dtype=torch.float32, device=env.device
            # )

            # env stepping
            obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
