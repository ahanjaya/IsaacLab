# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates nxp humanoid robot with runtime PD-gain tuning.

Features:
  * State machine: RESET / IDLE / SQUAT / STRAIGHT / CHIRP
  * Chirp excitation (linear or log sweep) on a single selected lower-body joint
  * Hot-reloaded PD gains from `gains.yaml` (edit file -> next loop picks it up)
  * Live plot of target / current joint position + applied torque per joint
  * Torque saturation detector (compares computed vs applied torque)
  * Keyboard hotkeys (type letter + Enter in the launching terminal):
        r  -> reset state
        c  -> start chirp on currently selected joint
        n  -> advance chirp joint index
        s  -> toggle squat/straight cycling
        i  -> go to IDLE
        q  -> quit

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/nxp_humanoid.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import multiprocessing as mp
import os
import queue
import select
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates nxp v1 humanoid robot.")
parser.add_argument(
    "--gains-file",
    type=str,
    default="gains.yaml",
    help="Path to YAML file with hot-reloadable PD gains.",
)
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
    CHIRP = 4


@dataclass
class ChirpConfig:
    """Configuration for chirp signal excitation."""

    duration: float = 20.0  # sweep duration in seconds
    f_start: float = 0.2  # start frequency in Hz
    f_end: float = 2.0  # end frequency in Hz
    amplitude: float = 0.2  # amplitude in radians
    offset: float = 0.0  # steady-state position offset (rad)
    log_scale: bool = False  # True = logarithmic, False = linear frequency sweep
    active_joint_idx: int = 10  # index into lower_body_joint_names to excite


##
# Runtime gain tuner (YAML hot-reload)
##
class GainTuner:
    """Hot-reload joint stiffness/damping from a YAML file.

    YAML format (regex pattern -> kp/kd):

        ".*_hip_pitch_.*": {kp: 400.0, kd: 30.0}
        ".*_knee_.*":      {kp: 300.0, kd: 30.0}
    """

    def __init__(self, robot: Articulation, gain_file: str = "gains.yaml"):
        self.robot = robot
        self.gain_file = gain_file
        self.last_mtime = 0.0
        self._pattern_cache: dict[str, tuple[list[int], list[str]]] = {}

    def _resolve(self, pattern: str):
        if pattern not in self._pattern_cache:
            ids, names = self.robot.find_joints(pattern)
            self._pattern_cache[pattern] = (ids, names)
        return self._pattern_cache[pattern]

    def maybe_reload(self) -> bool:
        """Reload gains if YAML mtime changed. Returns True if updated."""
        if not os.path.exists(self.gain_file):
            return False

        mtime = os.path.getmtime(self.gain_file)
        if mtime <= self.last_mtime:
            return False

        self.last_mtime = mtime

        try:
            with open(self.gain_file) as f:
                gains = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[GainTuner] Failed to parse {self.gain_file}: {e}")
            return False

        device = self.robot.device
        print(f"[GainTuner] Reloading gains from {self.gain_file}")
        for pattern, vals in gains.items():
            ids, names = self._resolve(pattern)
            if not ids:
                print(f"[GainTuner]   {pattern}: no matching joints, skipping")
                continue
            try:
                kp_val = float(vals["kp"])
                kd_val = float(vals["kd"])
            except (KeyError, TypeError, ValueError) as e:
                print(f"[GainTuner]   {pattern}: invalid kp/kd ({e}), skipping")
                continue

            # For explicit actuators (IdealPDActuator, etc.) the PD gains live on
            # the actuator object's .stiffness / .damping tensors, NOT in PhysX.
            # write_joint_stiffness_to_sim only touches PhysX (used by implicit
            # actuators), so we must update the actuator tensors directly.
            matched_any = False
            for act_name, actuator in self.robot.actuators.items():
                act_joint_ids = list(actuator.joint_indices)
                for global_idx, joint_name in zip(ids, names):
                    if global_idx not in act_joint_ids:
                        continue
                    local_idx = act_joint_ids.index(global_idx)
                    kp_before = actuator.stiffness[0, local_idx].item()
                    kd_before = actuator.damping[0, local_idx].item()
                    actuator.stiffness[:, local_idx] = kp_val
                    actuator.damping[:, local_idx] = kd_val
                    print(
                        f"[GainTuner]   {joint_name:30s}  "
                        f"kp: {kp_before:7.2f} -> {kp_val:7.2f}  | "
                        f"kd: {kd_before:6.2f} -> {kd_val:6.2f}"
                    )
                    matched_any = True
            if not matched_any:
                print(f"[GainTuner]   {pattern}: joints found but not in any actuator group")

        return True


##
# Keyboard listener (background thread reading stdin lines)
##
class KeyboardListener:
    """Non-blocking stdin reader. Type a single letter + Enter to issue a command."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_key: str | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    line = sys.stdin.readline()
                    if not line:
                        continue
                    key = line.strip().lower()
                    if key:
                        with self._lock:
                            self._last_key = key[0]  # only first char
            except Exception:
                # stdin may not be available in all launch contexts (e.g. detached)
                time.sleep(0.5)

    def poll(self) -> str | None:
        with self._lock:
            k = self._last_key
            self._last_key = None
        return k

    def stop(self):
        self._stop = True


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


def chirp_reference(t: float, cfg: ChirpConfig) -> tuple:
    """Compute chirp desired position offset and velocity at elapsed time t.

    Supports linear and logarithmic frequency sweeps. The sweep repeats every
    ``cfg.duration`` seconds.
    """
    T = cfg.duration
    t_m = t % T  # time within current sweep (0 .. T)
    tau = t_m / T  # normalized time (0 .. 1)

    if cfg.log_scale and cfg.f_start > 0.0 and cfg.f_end > cfg.f_start:
        k = np.log(cfg.f_end / cfg.f_start)
        phase = 2.0 * np.pi * cfg.f_start * T / k * (np.exp(k * tau) - 1.0)
        f_inst = cfg.f_start * np.exp(k * tau)
    else:
        c = (cfg.f_end - cfg.f_start) / (2.0 * T)
        phase = 2.0 * np.pi * (cfg.f_start * t_m + c * t_m * t_m)
        f_inst = cfg.f_start + 2.0 * c * t_m

    q_des = cfg.offset + cfg.amplitude * np.sin(phase)
    qd_des = cfg.amplitude * (2.0 * np.pi * f_inst) * np.cos(phase)
    return q_des, qd_des


def _plot_joint_visualization(plot_queue) -> None:
    """Live plot: per joint, target vs current position (left axis) and torque (right axis)."""
    plt.ion()

    joint_names = ["Hip Pitch", "Hip Roll", "Hip Yaw", "Knee", "Ankle Pitch", "Ankle Roll"]
    num_joints = len(joint_names)
    window_size = 1000
    pos_ylim = (-0.4, 0.4)

    fig, ax = plt.subplots(num_joints, 2, figsize=(20, 15))

    joint_data: dict[str, deque] = {}
    lines_pos: dict[str, any] = {}
    lines_tau: dict[str, any] = {}
    ax_tau: dict[str, any] = {}

    for i, joint_name in enumerate(joint_names):
        for side in ["left", "right"]:
            col = 0 if side == "left" else 1
            key = f"{side}_{joint_name.lower().replace(' ', '_')}"

            joint_data[f"{key}_target"] = deque(maxlen=window_size)
            joint_data[f"{key}_current"] = deque(maxlen=window_size)
            joint_data[f"{key}_torque"] = deque(maxlen=window_size)

            (lines_pos[f"{key}_target"],) = ax[i, col].plot([], [], label="Target", color="blue", linewidth=1.2)
            (lines_pos[f"{key}_current"],) = ax[i, col].plot(
                [], [], label="Current", color="green", linewidth=1.0, linestyle="--"
            )

            # # Twin axis for torque
            # ax_t = ax[i, col].twinx()
            # (lines_tau[f"{key}_torque"],) = ax_t.plot(
            #     [], [], label=f"{side.title()} {joint_name} Torque (Nm)",
            #     color="red", alpha=0.5, linewidth=0.8
            # )
            # ax_t.set_ylabel("Torque (Nm)", color="red")
            # ax_t.tick_params(axis="y", labelcolor="red")
            # ax_tau[key] = ax_t

            _configure_subplot(ax[i, col], f"{side.title()} {joint_name}", pos_ylim)

    # Mapping order: must match the producer-side packing order.
    # 6 joint groups x 2 sides x 3 channels (target, current, torque) = 36 items
    data_keys: list[str] = []
    for jn in joint_names:
        for side in ["left", "right"]:
            key = f"{side}_{jn.lower().replace(' ', '_')}"
            data_keys.extend([f"{key}_target", f"{key}_current", f"{key}_torque"])

    total_steps = 0  # global step counter for sliding x-axis

    def _update_joint_data(data_tuple):
        nonlocal total_steps
        for i, key in enumerate(data_keys):
            value = data_tuple[i]
            if value is None:
                joint_data[key].clear()
            else:
                joint_data[key].append(value)
        total_steps += 1

    def _update_plots():
        for i, jn in enumerate(joint_names):
            for side in ["left", "right"]:
                col = 0 if side == "left" else 1
                key = f"{side}_{jn.lower().replace(' ', '_')}"
                tq = joint_data[f"{key}_target"]
                cq = joint_data[f"{key}_current"]
                taq = joint_data[f"{key}_torque"]
                if tq:
                    n = len(tq)
                    x = range(total_steps - n, total_steps)
                    lines_pos[f"{key}_target"].set_data(x, tq)
                    lines_pos[f"{key}_current"].set_data(x, cq)
                    # lines_tau[f"{key}_torque"].set_data(x, taq)
                    ax[i, col].relim()
                    ax[i, col].autoscale_view(scalex=True, scaley=False)
                    ax[i, col].set_xlim(total_steps - window_size, total_steps)

    while True:
        try:
            while not plot_queue.empty():
                _update_joint_data(plot_queue.get_nowait())
        except queue.Empty:
            pass

        _update_plots()
        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            pass
        plt.tight_layout()
        time.sleep(0.02)


def _configure_subplot(subplot, title, ylim):
    subplot.set_title(title)
    subplot.set_xlabel("Times (s)")
    subplot.set_ylabel("Angle (rad)")
    subplot.legend(loc="upper left", fontsize=14)
    subplot.set_ylim(ylim)
    subplot.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)


def _print_help():
    print()
    print("=" * 60)
    print(" Keyboard hotkeys (type letter + Enter in this terminal):")
    print("   r  reset robot state")
    print("   c  start chirp on currently selected joint")
    print("   n  advance chirp joint index (wraps around)")
    print("   s  start squat/straight cycling")
    print("   i  go to IDLE")
    print("   h  show this help")
    print("   q  quit simulation")
    print()
    print(" Edit gains.yaml to retune PD gains live. Saves are picked up")
    print(" automatically on the next control step.")
    print("=" * 60)
    print()


def run_simulator(sim: sim_utils.SimulationContext, entity: Articulation, gains_file: str):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    print(f"[INFO]: Robot Default Joint Pos: {entity.data.default_joint_pos}")

    default_pose = entity.data.default_joint_pos.clone()
    squad_pose = torch.tensor(
        [
            [
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
            ]
        ],
        device=entity.device,
    )
    joint_pos_target = default_pose.clone()

    decimation = 1
    control_dt = sim_dt * decimation
    ticks_per_second = 1.0 / control_dt
    list_ticks = np.arange(0.2, 1.0, 0.2)
    num_cycles = len(list_ticks)

    # State machine init
    cycle_count = 0
    current_state = RobotState.RESET
    idle_duration = 200
    idle_count = 0
    squat_count = 0
    straight_count = 0
    auto_chirp_after_idle = True  # chirp automatically once idle finishes (original behavior)

    # Chirp state
    chirp_cfg = ChirpConfig()
    chirp_t0 = 0.0
    chirp_hold_positions = None

    # Joint name lists
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
    lower_body_indices = [entity.data.joint_names.index(n) for n in lower_body_joint_names]

    # Plot order is per-joint-group {left,right} -> matches producer-side packing.
    # Map from lower_body_joint_names ordering (l/r interleaved per group) to the
    # plot's expected order (left first within each joint group, then right).
    plot_pair_order = [
        ("left_hip_pitch_joint", "right_hip_pitch_joint"),
        ("left_hip_roll_joint", "right_hip_roll_joint"),
        ("left_hip_yaw_joint", "right_hip_yaw_joint"),
        ("left_knee_joint", "right_knee_joint"),
        ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
        ("left_ankle_roll_joint", "right_ankle_roll_joint"),
    ]
    plot_indices: list[int] = []
    for left_name, right_name in plot_pair_order:
        plot_indices.append(entity.data.joint_names.index(left_name))
        plot_indices.append(entity.data.joint_names.index(right_name))

    # Runtime helpers
    gain_tuner = GainTuner(entity, gain_file=gains_file)
    gain_tuner.maybe_reload()

    keyboard = KeyboardListener()
    _print_help()

    # Plotting process
    plot_queue = mp.Queue()
    mp_plot = mp.Process(target=_plot_joint_visualization, args=(plot_queue,))
    mp_plot.start()

    # Saturation print throttle
    last_sat_print_time = 0.0
    sat_print_interval = 0.5  # seconds

    try:
        while simulation_app.is_running():
            robot = entity

            # ---- Hot-reload gains ----
            gain_tuner.maybe_reload()

            # ---- Keyboard handling ----
            key = keyboard.poll()
            if key:
                if key == "r":
                    print("[KB] -> RESET")
                    current_state = RobotState.RESET
                elif key == "c":
                    print(f"[KB] -> CHIRP on '{lower_body_joint_names[chirp_cfg.active_joint_idx]}'")
                    current_state = RobotState.CHIRP
                    chirp_t0 = sim_time
                    chirp_hold_positions = robot.data.joint_pos.clone()
                elif key == "n":
                    chirp_cfg.active_joint_idx = (chirp_cfg.active_joint_idx + 1) % len(lower_body_joint_names)
                    print(f"[KB] active chirp joint: '{lower_body_joint_names[chirp_cfg.active_joint_idx]}'")
                elif key == "s":
                    print("[KB] -> SQUAT cycling")
                    current_state = RobotState.SQUAT
                    squat_count = 0
                    straight_count = 0
                    cycle_count = 0
                elif key == "i":
                    print("[KB] -> IDLE")
                    current_state = RobotState.IDLE
                    idle_count = 0
                    auto_chirp_after_idle = False
                elif key == "h":
                    _print_help()
                elif key == "q":
                    print("[KB] -> QUIT")
                    break

            # ---- State machine ----
            if current_state == RobotState.RESET:
                sim_time = 0.0
                cycle_count = 0

                root_state = robot.data.default_root_state.clone()
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])

                joint_pos, joint_vel = (
                    robot.data.default_joint_pos.clone(),
                    robot.data.default_joint_vel.clone(),
                )
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()
                print(f"[INFO]: State = {current_state.name} - Resetting robot state...")

                idle_count = 0
                auto_chirp_after_idle = True
                current_state = RobotState.IDLE

            elif current_state == RobotState.IDLE:
                joint_pos_target = robot.data.default_joint_pos

                if auto_chirp_after_idle and idle_count >= idle_duration:
                    current_state = RobotState.CHIRP
                    chirp_t0 = sim_time
                    chirp_hold_positions = robot.data.joint_pos.clone()
                    print(
                        f"[INFO]: State = CHIRP - Starting chirp on joint "
                        f"'{lower_body_joint_names[chirp_cfg.active_joint_idx]}'"
                    )
                    idle_count = 0

                idle_count += 1

            elif current_state == RobotState.SQUAT:
                squat_rate = squat_count / (list_ticks[cycle_count % num_cycles] * ticks_per_second)
                joint_pos_target = joint_linear_interpolation(default_pose, squad_pose, squat_rate)

                if squat_rate >= 1.5:
                    current_state = RobotState.STRAIGHT
                    squat_count = 0

                squat_count += 1

            elif current_state == RobotState.STRAIGHT:
                straight_rate = straight_count / (list_ticks[cycle_count % num_cycles] * ticks_per_second)
                joint_pos_target = joint_linear_interpolation(squad_pose, default_pose, straight_rate)

                if straight_rate >= 1.5:
                    current_state = RobotState.SQUAT
                    straight_count = 0
                    cycle_count += 1

                straight_count += 1

            elif current_state == RobotState.CHIRP:
                t_elapsed = sim_time - chirp_t0
                q_ref, _qd_ref = chirp_reference(t_elapsed, chirp_cfg)

                joint_pos_target = chirp_hold_positions.clone()
                active_idx = lower_body_indices[chirp_cfg.active_joint_idx]
                hold_val = chirp_hold_positions[0, active_idx].item()
                joint_pos_target[0, active_idx] = hold_val + q_ref

                if t_elapsed >= chirp_cfg.duration:
                    print(
                        f"[INFO]: Chirp sweep complete on joint "
                        f"'{lower_body_joint_names[chirp_cfg.active_joint_idx]}' - returning to IDLE."
                    )
                    current_state = RobotState.IDLE
                    idle_count = 0
                    auto_chirp_after_idle = False  # don't auto-restart

            # ---- Pull state for plotting ----
            target_np = joint_pos_target[:, plot_indices].detach().cpu().numpy()[0]
            current_np = robot.data.joint_pos[:, plot_indices].detach().cpu().numpy()[0]
            torque_np = robot.data.applied_torque[:, plot_indices].detach().cpu().numpy()[0]

            # Pack as (target, current, torque) per plot slot in the order the consumer expects.
            packed = []
            for k in range(len(plot_indices)):
                packed.extend([target_np[k], current_np[k], torque_np[k]])
            plot_queue.put(tuple(packed))

            # ---- Saturation detection ----
            try:
                computed = robot.data.computed_torque[:, lower_body_indices]
                applied = robot.data.applied_torque[:, lower_body_indices]
                # Saturated iff |applied| < |computed| - eps
                eps = 1e-2
                sat_mask = (computed.abs() - applied.abs()) > eps
                if sat_mask.any() and (sim_time - last_sat_print_time) > sat_print_interval:
                    sat_joint_idx = sat_mask.any(dim=0).nonzero(as_tuple=False).flatten().tolist()
                    msgs = []
                    for j in sat_joint_idx:
                        msgs.append(
                            f"{lower_body_joint_names[j]}: cmd={computed[0, j].item():+7.1f} "
                            f"applied={applied[0, j].item():+7.1f}"
                        )
                    # print(f"[SAT] t={sim_time:6.2f}  " + " | ".join(msgs))
                    last_sat_print_time = sim_time
            except AttributeError:
                # `computed_torque` may not be exposed for ImplicitActuator -- ignore silently.
                pass

            # ---- Apply target & step ----
            robot.set_joint_position_target(joint_pos_target)
            robot.write_data_to_sim()

            for _ in range(decimation):
                sim.step()
                robot.update(sim_dt)

            sim_time += control_dt
    finally:
        keyboard.stop()
        try:
            mp_plot.terminate()
            mp_plot.join(timeout=1.0)
        except Exception:
            pass


def main():
    """Main function."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    entity = design_scene()
    sim.reset()
    print("[INFO]: Setup complete...")

    run_simulator(sim, entity, gains_file=args_cli.gains_file)


if __name__ == "__main__":
    main()
    simulation_app.close()
