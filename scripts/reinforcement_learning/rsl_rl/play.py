# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
parser.add_argument("--plot", action="store_true", default=False, help="Plot robot joint actions and positions.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time.")
parser.add_argument("--follow_robot", action="store_true", default=False, help="Follow the robot with the camera.")
parser.add_argument("--udp_host", type=str, default="localhost", help="UDP host for publishing actions.")
parser.add_argument("--udp_port", type=int, default=8888, help="UDP port for publishing actions.")
parser.add_argument("--enable_udp", action="store_true", default=False, help="Enable UDP publishing of actions.")
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

"""Rest everything follows."""

import gymnasium as gym
import json
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import socket
import time
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
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

np.set_printoptions(precision=4, suppress=True)

# PLACEHOLDER: Extension template (do not remove this comment)


def _plot_joint_visualization(queue) -> None:
    plt.ion()  # Turn on interactive mode

    # Configuration
    joint_names = [
        "Hip Pitch",
        "Hip Roll",
        "Hip Yaw",
        "Knee",
        "Ankle Pitch",
        "Ankle Roll",
        "Shoulder Pitch",
        "Shoulder Roll",
        "Shoulder Yaw",
        "Elbow",
    ]
    num_joints = len(joint_names)
    window_size = 50
    ylim = (-1.0, 1.0)

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
            "left_shoulder_pitch_target",
            "left_shoulder_pitch_current",
            "right_shoulder_pitch_target",
            "right_shoulder_pitch_current",
            "left_shoulder_roll_target",
            "left_shoulder_roll_current",
            "right_shoulder_roll_target",
            "right_shoulder_roll_current",
            "left_shoulder_yaw_target",
            "left_shoulder_yaw_current",
            "right_shoulder_yaw_target",
            "right_shoulder_yaw_current",
            "left_elbow_target",
            "left_elbow_current",
            "right_elbow_target",
            "right_elbow_current",
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
        time.sleep(0.02)


def _configure_subplot(subplot, title, ylim):
    """Configure a subplot with common settings."""
    subplot.set_title(title)
    subplot.set_xlabel("Time Step")
    subplot.set_ylabel("Angle (rad)")
    subplot.legend()
    subplot.set_ylim(ylim)
    subplot.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)



@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

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

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    print(f"[INFO] Environment step dt: {dt:.4f} seconds.")

    if args_cli.plot:
        # Start the plotting function in a separate process
        queue = mp.Queue()
        mp_plot = mp.Process(target=_plot_joint_visualization, args=(queue,))
        mp_plot.start()

    # initialize UDP socket for publishing actions
    udp_socket = None
    if args_cli.enable_udp:
        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"[INFO] UDP socket initialized. Publishing to {args_cli.udp_host}:{args_cli.udp_port}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize UDP socket: {e}")
            udp_socket = None

    # reset environment
    obs = env.get_observations()
    timestep = 0
    obs_pos_idx = 12

    # Set up viewport camera to track the robot
    if args_cli.follow_robot:
        vcc = env.unwrapped.viewport_camera_controller
        vcc.update_view_to_asset_root("robot")
        # vcc.set_view_env_index(0)  # Track environment 0
        # vcc.update_view_to_env()
        vcc.update_view_location(eye=[2.0, 2.0, 0.5], lookat=[0.0, 0.0, 0.0])

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            obs_numpy = obs["policy"].detach().cpu().numpy()[0]
            obs_pos_numpy = obs_numpy[obs_pos_idx : obs_pos_idx + 20]

            actions_numpy = actions.detach().cpu().numpy()[0] * 0.5
            actions_publish = actions.detach().cpu().numpy()[:12]

            # actions_publish = obs["policy"].detach().cpu().numpy()[:, obs_pos_idx : obs_pos_idx + 12]

            # publish actions via UDP
            if udp_socket is not None and args_cli.enable_udp:
                try:
                    # create data payload
                    data_payload = {"timestamp": time.time(), "timestep": timestep, "actions": actions_publish.tolist()}

                    # convert to JSON and send via UDP
                    json_data = json.dumps(data_payload)
                    udp_socket.sendto(json_data.encode("utf-8"), (args_cli.udp_host, args_cli.udp_port))
                except Exception as e:
                    print(f"[WARNING] Failed to send UDP data: {e}")

            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

            if args_cli.plot:
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
                    actions_numpy[8],  # left_ankle_pitch_action
                    obs_pos_numpy[8],  # left_ankle_pitch_pos
                    actions_numpy[9],  # right_ankle_pitch_action
                    obs_pos_numpy[9],  # right_ankle_pitch_pos
                    actions_numpy[10],  # left_ankle_roll_action
                    obs_pos_numpy[10],  # left_ankle_roll_pos
                    actions_numpy[11],  # right_ankle_roll_action
                    obs_pos_numpy[11],  # right_ankle_roll_pos
                    actions_numpy[12],  # left_shoulder_pitch_action
                    obs_pos_numpy[12],  # left_shoulder_pitch_pos
                    actions_numpy[13],  # right_shoulder_pitch_action
                    obs_pos_numpy[13],  # right_shoulder_pitch_pos
                    actions_numpy[14],  # left_shoulder_roll_action
                    obs_pos_numpy[14],  # left_shoulder_roll_pos
                    actions_numpy[15],  # right_shoulder_roll_action
                    obs_pos_numpy[15],  # right_shoulder_roll_pos
                    actions_numpy[16],  # left_shoulder_yaw_action
                    obs_pos_numpy[16],  # left_shoulder_yaw_pos
                    actions_numpy[17],  # right_shoulder_yaw_action
                    obs_pos_numpy[17],  # right_shoulder_yaw_pos
                    actions_numpy[18],  # left_elbow_action
                    obs_pos_numpy[18],  # left_elbow_pos
                    actions_numpy[19],  # right_elbow_action
                    obs_pos_numpy[19],  # right_elbow_pos
                ))

        done_env_ids = dones.nonzero(as_tuple=False).flatten()

        if done_env_ids.shape[0] > 0:
            if args_cli.plot:
                none_tuple = (None,) * 40
                queue.put(none_tuple)

        timestep += 1

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        # print the frequency in this loop
        end_time = time.time()
        loop_dt = end_time - start_time
        loop_freq = 1.0 / loop_dt if loop_dt > 0 else float("inf")
        print(f"[INFO] Timestep: {timestep}, Loop Frequency: {loop_freq:.2f} Hz")

    # close UDP socket if initialized
    if udp_socket is not None:
        udp_socket.close()
        print("[INFO] UDP socket closed.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
