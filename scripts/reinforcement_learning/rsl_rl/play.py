# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gymnasium as gym
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import time

import gymnasium as gym
import torch
from collections import deque

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _plot_joint_visualization(queue) -> None:
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(4, 2, figsize=(20, 15))

    window_size = 300
    left_hip_pitch_action_queue = deque(maxlen=window_size)
    left_hip_pitch_pos_queue = deque(maxlen=window_size)
    right_hip_pitch_action_queue = deque(maxlen=window_size)
    right_hip_pitch_pos_queue = deque(maxlen=window_size)

    left_hip_roll_action_queue = deque(maxlen=window_size)
    left_hip_roll_pos_queue = deque(maxlen=window_size)
    right_hip_roll_action_queue = deque(maxlen=window_size)
    right_hip_roll_pos_queue = deque(maxlen=window_size)

    left_hip_yaw_action_queue = deque(maxlen=window_size)
    left_hip_yaw_pos_queue = deque(maxlen=window_size)
    right_hip_yaw_action_queue = deque(maxlen=window_size)
    right_hip_yaw_pos_queue = deque(maxlen=window_size)

    left_knee_action_queue = deque(maxlen=window_size)
    left_knee_pos_queue = deque(maxlen=window_size)
    right_knee_action_queue = deque(maxlen=window_size)
    right_knee_pos_queue = deque(maxlen=window_size)

    (line_left_hip_pitch,) = ax[0, 0].plot(left_hip_pitch_action_queue, label="Left Hip Pitch Action")
    (line_left_hip_pitch_pos,) = ax[0, 0].plot(left_hip_pitch_pos_queue, label="Left Hip Pitch Pos")
    (line_right_hip_pitch,) = ax[0, 1].plot(right_hip_pitch_action_queue, label="Right Hip Pitch Action")
    (line_right_hip_pitch_pos,) = ax[0, 1].plot(right_hip_pitch_pos_queue, label="Right Hip Pitch Pos")

    (line_left_hip_roll,) = ax[1, 0].plot(left_hip_roll_action_queue, label="Left Hip Roll Action")
    (line_left_hip_roll_pos,) = ax[1, 0].plot(left_hip_roll_pos_queue, label="Left Hip Roll Pos")
    (line_right_hip_roll,) = ax[1, 1].plot(right_hip_roll_action_queue, label="Right Hip Roll Action")
    (line_right_hip_roll_pos,) = ax[1, 1].plot(right_hip_roll_pos_queue, label="Right Hip Roll Pos")

    (line_left_hip_yaw,) = ax[2, 0].plot(left_hip_yaw_action_queue, label="Left Hip Yaw Action")
    (line_left_hip_yaw_pos,) = ax[2, 0].plot(left_hip_yaw_pos_queue, label="Left Hip Yaw Pos")
    (line_right_hip_yaw,) = ax[2, 1].plot(right_hip_yaw_action_queue, label="Right Hip Yaw Action")
    (line_right_hip_yaw_pos,) = ax[2, 1].plot(right_hip_yaw_pos_queue, label="Right Hip Yaw Pos")

    (line_left_knee,) = ax[3, 0].plot(left_knee_action_queue, label="Left Knee Action")
    (line_left_knee_pos,) = ax[3, 0].plot(left_knee_pos_queue, label="Left Knee Pos")
    (line_right_knee,) = ax[3, 1].plot(right_knee_action_queue, label="Right Knee Action")
    (line_right_knee_pos,) = ax[3, 1].plot(right_knee_pos_queue, label="Right Knee Pos")

    ax[0, 0].set_title("Left Hip Pitch")
    ax[0, 0].set_xlabel("Time Step")
    ax[0, 0].set_ylabel("Angle (rad)")
    ax[0, 0].legend()
    ax[0, 0].set_ylim(-1.5, 1.5)
    ax[0, 0].grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    ax[0, 1].set_title("Right Hip Pitch")
    ax[0, 1].set_xlabel("Time Step")
    ax[0, 1].set_ylabel("Angle (rad)")
    ax[0, 1].legend()
    ax[0, 1].set_ylim(-1.5, 1.5)
    ax[0, 1].grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    ax[1, 0].set_title("Left Hip Roll")
    ax[1, 0].set_xlabel("Time Step")
    ax[1, 0].set_ylabel("Angle (rad)")
    ax[1, 0].legend()
    ax[1, 0].set_ylim(-1.5, 1.5)
    ax[1, 0].grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    ax[1, 1].set_title("Right Hip Roll")
    ax[1, 1].set_xlabel("Time Step")
    ax[1, 1].set_ylabel("Angle (rad)")
    ax[1, 1].legend()
    ax[1, 1].set_ylim(-1.5, 1.5)
    ax[1, 1].grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    ax[2, 0].set_title("Left Hip Yaw")
    ax[2, 0].set_xlabel("Time Step")
    ax[2, 0].set_ylabel("Angle (rad)")
    ax[2, 0].legend()
    ax[2, 0].set_ylim(-1.5, 1.5)
    ax[2, 0].grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    ax[2, 1].set_title("Right Hip Yaw")
    ax[2, 1].set_xlabel("Time Step")
    ax[2, 1].set_ylabel("Angle (rad)")
    ax[2, 1].legend()
    ax[2, 1].set_ylim(-1.5, 1.5)
    ax[2, 1].grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    ax[3, 0].set_title("Left Knee")
    ax[3, 0].set_xlabel("Time Step")
    ax[3, 0].set_ylabel("Angle (rad)")
    ax[3, 0].legend()
    ax[3, 0].set_ylim(-1.5, 1.5)
    ax[3, 0].grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    ax[3, 1].set_title("Right Knee")
    ax[3, 1].set_xlabel("Time Step")
    ax[3, 1].set_ylabel("Angle (rad)")
    ax[3, 1].legend()
    ax[3, 1].set_ylim(-1.5, 1.5)
    ax[3, 1].grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    while True:
        # clear the plot if the queue is empty
        while not queue.empty():
            (
                left_hip_pitch_action,
                left_hip_pitch_pos,
                right_hip_pitch_action,
                right_hip_pitch_pos,
                left_hip_roll_action,
                left_hip_roll_pos,
                right_hip_roll_action,
                right_hip_roll_pos,
                left_hip_yaw_action,
                left_hip_yaw_pos,
                right_hip_yaw_action,
                right_hip_yaw_pos,
                left_knee_action,
                left_knee_pos,
                right_knee_action,
                right_knee_pos,
            ) = queue.get()

            if left_hip_pitch_action is None:
                left_hip_pitch_action_queue.clear()
                left_hip_pitch_pos_queue.clear()
            else:
                left_hip_pitch_action_queue.append(left_hip_pitch_action)
                left_hip_pitch_pos_queue.append(left_hip_pitch_pos)

            if right_hip_pitch_action is None:
                right_hip_pitch_action_queue.clear()
                right_hip_pitch_pos_queue.clear()
            else:
                right_hip_pitch_action_queue.append(right_hip_pitch_action)
                right_hip_pitch_pos_queue.append(right_hip_pitch_pos)

            if left_hip_roll_action is None:
                left_hip_roll_action_queue.clear()
                left_hip_roll_pos_queue.clear()
            else:
                left_hip_roll_action_queue.append(left_hip_roll_action)
                left_hip_roll_pos_queue.append(left_hip_roll_pos)

            if right_hip_roll_action is None:
                right_hip_roll_action_queue.clear()
                right_hip_roll_pos_queue.clear()
            else:
                right_hip_roll_action_queue.append(right_hip_roll_action)
                right_hip_roll_pos_queue.append(right_hip_roll_pos)

            if left_hip_yaw_action is None:
                left_hip_yaw_action_queue.clear()
                left_hip_yaw_pos_queue.clear()
            else:
                left_hip_yaw_action_queue.append(left_hip_yaw_action)
                left_hip_yaw_pos_queue.append(left_hip_yaw_pos)

            if right_hip_yaw_action is None:
                right_hip_yaw_action_queue.clear()
                right_hip_yaw_pos_queue.clear()
            else:
                right_hip_yaw_action_queue.append(right_hip_yaw_action)
                right_hip_yaw_pos_queue.append(right_hip_yaw_pos)

            if left_knee_action is None:
                left_knee_action_queue.clear()
                left_knee_pos_queue.clear()
            else:
                left_knee_action_queue.append(left_knee_action)
                left_knee_pos_queue.append(left_knee_pos)

            if right_knee_action is None:
                right_knee_action_queue.clear()
                right_knee_pos_queue.clear()
            else:
                right_knee_action_queue.append(right_knee_action)
                right_knee_pos_queue.append(right_knee_pos)

        if left_hip_pitch_action_queue:
            line_left_hip_pitch.set_xdata(range(len(left_hip_pitch_action_queue)))
            line_left_hip_pitch.set_ydata(left_hip_pitch_action_queue)
            line_left_hip_pitch_pos.set_xdata(range(len(left_hip_pitch_pos_queue)))
            line_left_hip_pitch_pos.set_ydata(left_hip_pitch_pos_queue)

            ax[0, 0].relim()
            ax[0, 0].autoscale_view()

        if right_hip_pitch_action_queue:
            line_right_hip_pitch.set_xdata(range(len(right_hip_pitch_action_queue)))
            line_right_hip_pitch.set_ydata(right_hip_pitch_action_queue)
            line_right_hip_pitch_pos.set_xdata(range(len(right_hip_pitch_pos_queue)))
            line_right_hip_pitch_pos.set_ydata(right_hip_pitch_pos_queue)

            ax[0, 1].relim()
            ax[0, 1].autoscale_view()

        if left_hip_roll_action_queue:
            line_left_hip_roll.set_xdata(range(len(left_hip_roll_action_queue)))
            line_left_hip_roll.set_ydata(left_hip_roll_action_queue)
            line_left_hip_roll_pos.set_xdata(range(len(left_hip_roll_pos_queue)))
            line_left_hip_roll_pos.set_ydata(left_hip_roll_pos_queue)

            ax[1, 0].relim()
            ax[1, 0].autoscale_view()

        if right_hip_roll_action_queue:
            line_right_hip_roll.set_xdata(range(len(right_hip_roll_action_queue)))
            line_right_hip_roll.set_ydata(right_hip_roll_action_queue)
            line_right_hip_roll_pos.set_xdata(range(len(right_hip_roll_pos_queue)))
            line_right_hip_roll_pos.set_ydata(right_hip_roll_pos_queue)

            ax[1, 1].relim()
            ax[1, 1].autoscale_view()

        if left_hip_yaw_action_queue:
            line_left_hip_yaw.set_xdata(range(len(left_hip_yaw_action_queue)))
            line_left_hip_yaw.set_ydata(left_hip_yaw_action_queue)
            line_left_hip_yaw_pos.set_xdata(range(len(left_hip_yaw_pos_queue)))
            line_left_hip_yaw_pos.set_ydata(left_hip_yaw_pos_queue)

            ax[2, 0].relim()
            ax[2, 0].autoscale_view()

        if right_hip_yaw_action_queue:
            line_right_hip_yaw.set_xdata(range(len(right_hip_yaw_action_queue)))
            line_right_hip_yaw.set_ydata(right_hip_yaw_action_queue)
            line_right_hip_yaw_pos.set_xdata(range(len(right_hip_yaw_pos_queue)))
            line_right_hip_yaw_pos.set_ydata(right_hip_yaw_pos_queue)

            ax[2, 1].relim()
            ax[2, 1].autoscale_view()

        if left_knee_action_queue:
            line_left_knee.set_xdata(range(len(left_knee_action_queue)))
            line_left_knee.set_ydata(left_knee_action_queue)
            line_left_knee_pos.set_xdata(range(len(left_knee_pos_queue)))
            line_left_knee_pos.set_ydata(left_knee_pos_queue)

            ax[3, 0].relim()
            ax[3, 0].autoscale_view()

        if right_knee_action_queue:
            line_right_knee.set_xdata(range(len(right_knee_action_queue)))
            line_right_knee.set_ydata(right_knee_action_queue)
            line_right_knee_pos.set_xdata(range(len(right_knee_pos_queue)))
            line_right_knee_pos.set_ydata(right_knee_pos_queue)

            ax[3, 1].relim()
            ax[3, 1].autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.tight_layout()

        time.sleep(0.01)  # Add a small sleep to prevent high CPU usage


# Start the plotting function in a separate process
queue = mp.Queue()
mp_plot = mp.Process(target=_plot_joint_visualization, args=(queue,))
mp_plot.start()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    obs_pos_idx = 12

    default_joint_pos = np.array(
        [
            0.52,
            -0.52,
            -0.05,
            0.05,
            -0.35,
            0.35,
            0.785,
            0.785,
            -0.436,
            -0.436,
            0.0,
            0.0,
            0.0,
            0.0,
            0.13,
            -0.13,
            -0.13,
            0.13,
            -0.52,
            -0.52,
        ],
        dtype=np.float32,
    )

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # from IPython import embed; embed()
            obs_numpy = obs["policy"].detach().cpu().numpy()[0]
            obs_pos_numpy = obs_numpy[obs_pos_idx : obs_pos_idx + 20] + default_joint_pos
            actions_numpy = actions.detach().cpu().numpy()[0] * 0.25 + default_joint_pos

            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        queue.put((
            actions_numpy[0],  # left_hip_pitch_action
            obs_pos_numpy[0],  # left_hip_pitch_pos
            actions_numpy[1],  # right_hip_pitch_action
            obs_pos_numpy[1],  # right_hip_pitch_pos
            actions_numpy[2],  # left_hip_roll_action
            obs_pos_numpy[2],  # left_hip_roll_pos
            actions_numpy[3],  # right_hip_roll_action
            obs_pos_numpy[3],  # right_hip_roll_pos
            actions_numpy[4],  # left_hip_yaw_action
            obs_pos_numpy[4],  # left_hip_yaw_pos
            actions_numpy[5],  # right_hip_yaw_action
            obs_pos_numpy[5],  # right_hip_yaw_pos
            actions_numpy[6],  # left_knee_action
            obs_pos_numpy[6],  # left_knee_pos
            actions_numpy[7],  # right_knee_action
            obs_pos_numpy[7],  # right_knee_pos
        ))

        done_env_ids = dones.nonzero(as_tuple=False).flatten()
        if done_env_ids.shape[0] > 0:
            queue.put((
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ))

        timestep += 1

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
