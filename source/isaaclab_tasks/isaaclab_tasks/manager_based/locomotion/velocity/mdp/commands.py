# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task with teleop support."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.devices import Se2Gamepad, Se2GamepadCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from .grid_adaptive_curriculum import GridAdaptiveConfig, GridAdaptiveCurriculum

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class Se2GamepadVelocityCommand(UniformVelocityCommand):
    """Command generator that generates velocity commands from SE2 gamepad input.

    This command generator receives input from an SE2 gamepad device and converts it to
    velocity commands for the robot. The gamepad provides linear velocity in x and y
    direction and angular velocity around the z-axis in the robot's base frame.

    The command is updated in real-time based on gamepad input rather than being
    randomly sampled like the base UniformVelocityCommand.
    """

    cfg: Se2GamepadVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: Se2GamepadVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the SE2 gamepad command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Initialize gamepad device
        self._gamepad_config = Se2GamepadCfg(
            v_x_sensitivity=self.cfg.gamepad_sensitivity[0],
            v_y_sensitivity=self.cfg.gamepad_sensitivity[1],
            omega_z_sensitivity=self.cfg.gamepad_sensitivity[2],
            dead_zone=self.cfg.dead_zone,
            swap_stick_dpad=self.cfg.swap_stick_dpad,
            sim_device=self.device,
        )

        # Create gamepad device
        self._initialize_gamepad()

        # Initialize command to zero
        self.vel_command_b.zero_()

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "Se2GamepadVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tGamepad sensitivity: {self.cfg.gamepad_sensitivity}\n"
        msg += f"\tDead zone: {self.cfg.dead_zone}\n"
        msg += f"\tInvert axes: {self.cfg.invert_axes}"
        return msg

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command generator."""
        # For teleop, we don't track traditional error metrics
        # Instead, we could track gamepad input responsiveness
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the given environment IDs.

        For gamepad teleop, we don't resample commands automatically.
        Commands are updated based on gamepad input in _update_command.
        """
        # For teleop, commands are updated continuously, not resampled
        pass

    def _update_command(self):
        """Update the velocity command based on gamepad input."""
        # Get gamepad input
        gamepad_command = self._gamepad_device.advance()

        if gamepad_command is not None:
            # Process gamepad input
            processed_cmd = self._process_gamepad_input(gamepad_command)

            # Update velocity command for all environments
            self.vel_command_b[:] = processed_cmd.unsqueeze(0).expand(self.num_envs, -1)

            # Apply range limits
            self._apply_command_limits()

    def _apply_command_limits(self):
        """Apply velocity limits to the command."""
        # Clamp linear velocity x
        self.vel_command_b[:, 0] = torch.clamp(
            self.vel_command_b[:, 0], min=self.cfg.ranges.lin_vel_x[0], max=self.cfg.ranges.lin_vel_x[1]
        )

        # Clamp linear velocity y
        self.vel_command_b[:, 1] = torch.clamp(
            self.vel_command_b[:, 1], min=self.cfg.ranges.lin_vel_y[0], max=self.cfg.ranges.lin_vel_y[1]
        )

        # Clamp angular velocity z
        self.vel_command_b[:, 2] = torch.clamp(
            self.vel_command_b[:, 2], min=self.cfg.ranges.ang_vel_z[0], max=self.cfg.ranges.ang_vel_z[1]
        )

    def _process_gamepad_input(self, gamepad_cmd: torch.Tensor) -> torch.Tensor:
        """Process raw gamepad input and apply necessary transformations.

        Args:
            gamepad_cmd: Raw gamepad command tensor [vx, vy, omega_z]

        Returns:
            Processed command tensor [vx, vy, omega_z]
        """
        processed_cmd = gamepad_cmd.clone()

        # Apply axis inversions
        for i, invert in enumerate(self.cfg.invert_axes):
            if invert:
                processed_cmd[i] *= -1

        return processed_cmd

    def _initialize_gamepad(self):
        """Initialize the gamepad device."""
        try:
            self._gamepad_device = Se2Gamepad(self._gamepad_config)
            omni.log.info(f"SE2 Gamepad teleoperation initialized: {self._gamepad_device}")
        except AttributeError as e:
            if "omni" in str(e) and "appwindow" in str(e):
                raise RuntimeError(
                    "Gamepad teleop is not supported in headless mode. "
                    "Please run with GUI mode enabled to use gamepad functionality."
                ) from e
            raise RuntimeError(f"Failed to initialize gamepad device: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize gamepad device: {e}") from e

    def get_gamepad_device(self) -> Se2Gamepad:
        """Get the gamepad device instance.

        Returns:
            The gamepad device instance.
        """
        return self._gamepad_device

    def is_gamepad_connected(self) -> bool:
        """Check if gamepad is connected and functional.

        Returns:
            True if gamepad is connected, False otherwise.
        """
        return self._gamepad_device is not None


@configclass
class Se2GamepadVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the SE2 gamepad velocity command generator."""

    class_type: type = Se2GamepadVelocityCommand

    # Note: class_type will be set after class definition to avoid circular reference

    # Gamepad-specific parameters
    gamepad_sensitivity: tuple[float, float, float] = (0.5, 0.5, 0.5)
    """Sensitivity for linear-x, linear-y, and angular-z velocity commands from gamepad. Defaults to (0.5, 0.5, 0.5)."""

    dead_zone: float = 0.01
    """Dead zone for gamepad input to avoid drift. Defaults to 0.01."""

    swap_stick_dpad: bool = False
    """Set True if the controller reports the D-pad and left stick swapped. Defaults to False."""

    invert_axes: tuple[bool, bool, bool] = (False, True, True)
    """Whether to invert x, y, z axes from gamepad input. Defaults to (False, True, True)."""

    # Override some defaults for teleop use
    resampling_time_range: tuple[float, float] = (0.0, 0.0)  # No automatic resampling for teleop
    rel_standing_envs: float = 0.0  # No standing environments for teleop
    debug_vis: bool = True  # Enable visualization by default for teleop


class GridAdaptiveVelocityCommand(UniformVelocityCommand):
    """Command generator that samples linear-x and angular-z velocity commands from a grid adaptive curriculum.

    Implementation based on "Rapid Locomotion via Reinforcement Learning" (2022) from Margolis et. al.
    https://arxiv.org/abs/2205.02824

    Instead of sampling commands uniformly from a fixed range, commands are drawn from a
    :class:`~isaaclab_tasks.manager_based.locomotion.velocity.mdp.grid_adaptive_curriculum.GridAdaptiveCurriculum`.
    Every time a command is about to be resampled, the tracking performance achieved while it was active is
    evaluated. If both the linear and angular velocity tracking rewards (computed with the same exponential
    kernel as the ``track_lin_vel_xy_yaw_frame_exp``/``track_ang_vel_z_world_exp`` reward terms) stayed above
    their configured thresholds on average, the curriculum grid is expanded around that command, gradually
    widening the range of commands that can be sampled. The linear-y velocity command is still sampled
    uniformly from :attr:`~UniformVelocityCommandCfg.Ranges.lin_vel_y`.
    """

    cfg: GridAdaptiveVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: GridAdaptiveVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the grid adaptive velocity command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)

        self._gac = GridAdaptiveCurriculum(self.cfg.gac_config)

        # running sums used to score the tracking success of the command since its last resampling
        self._lin_vel_xy_reward_sum = torch.zeros(self.num_envs, device=self.device)
        self._ang_vel_z_reward_sum = torch.zeros(self.num_envs, device=self.device)
        self._command_steps = torch.zeros(self.num_envs, device=self.device)

        # same sums but over the whole episode (read by the terrain curriculum before the reset)
        self._episode_lin_vel_xy_reward_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_ang_vel_z_reward_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, device=self.device)

        self.metrics["gac_lin_score"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gac_ang_score"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gac_num_bins"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gac_success_rate"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GridAdaptiveVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tGAC updates so far: {self._gac.get_num_updates()}"
        return msg

    def save_grid_density(self, path: str) -> None:
        """Save a plot of the curriculum grid to ``path``."""
        self._gac.save_grid_density(path)

    def get_episode_tracking_scores(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Average linear and angular velocity tracking rewards over the current episode."""
        steps = self._episode_steps[env_ids].clamp(min=1.0)
        return (
            self._episode_lin_vel_xy_reward_sum[env_ids] / steps,
            self._episode_ang_vel_z_reward_sum[env_ids] / steps,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._episode_lin_vel_xy_reward_sum[ids] = 0.0
        self._episode_ang_vel_z_reward_sum[ids] = 0.0
        self._episode_steps[ids] = 0.0
        return extras

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        super()._update_metrics()

        # tracking rewards computed with the same exponential kernel used by the locomotion reward terms
        vel_yaw = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w), self.robot.data.root_lin_vel_w[:, :3]
        )
        lin_vel_xy_error = torch.sum(torch.square(self.vel_command_b[:, :2] - vel_yaw[:, :2]), dim=1)
        ang_vel_z_error = torch.square(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_w[:, 2])

        lin_vel_xy_reward = torch.exp(-lin_vel_xy_error / self.cfg.lin_vel_xy_std**2)
        ang_vel_z_reward = torch.exp(-ang_vel_z_error / self.cfg.ang_vel_z_std**2)
        self._lin_vel_xy_reward_sum += lin_vel_xy_reward
        self._ang_vel_z_reward_sum += ang_vel_z_reward
        self._command_steps += 1
        self._episode_lin_vel_xy_reward_sum += lin_vel_xy_reward
        self._episode_ang_vel_z_reward_sum += ang_vel_z_reward
        self._episode_steps += 1

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # score the command that is about to be replaced and expand the curriculum around it if successful
        tracked = self._command_steps[ids] > 0
        if tracked.any():
            tracked_ids = ids[tracked]
            lin_vel_xy_avg = self._lin_vel_xy_reward_sum[tracked_ids] / self._command_steps[tracked_ids]
            ang_vel_z_avg = self._ang_vel_z_reward_sum[tracked_ids] / self._command_steps[tracked_ids]
            self.metrics["gac_lin_score"][tracked_ids] = lin_vel_xy_avg
            self.metrics["gac_ang_score"][tracked_ids] = ang_vel_z_avg
            successful = (lin_vel_xy_avg >= self.cfg.gac_config.linvel_thresh) & (
                ang_vel_z_avg >= self.cfg.gac_config.angvel_thresh
            )
            self.metrics["gac_success_rate"][tracked_ids] = successful.float()
            if successful.any():
                success_ids = tracked_ids[successful]
                successful_cmds = torch.stack(
                    (self.vel_command_b[success_ids, 0], self.vel_command_b[success_ids, 2]), dim=1
                )
                self._gac.update_neighbors_from_cmds(successful_cmds.cpu().numpy())
        self.metrics["gac_num_bins"][:] = self._gac.get_num_available_bins()

        self._lin_vel_xy_reward_sum[ids] = 0.0
        self._ang_vel_z_reward_sum[ids] = 0.0
        self._command_steps[ids] = 0.0

        # sample new linear-x and angular-z commands from the curriculum grid
        new_cmds = self._gac.sample_command(len(ids))
        new_cmds = torch.as_tensor(new_cmds, dtype=torch.float32, device=self.device)
        self.vel_command_b[ids, 0] = new_cmds[:, 0]
        self.vel_command_b[ids, 2] = new_cmds[:, 1]

        # linear-y velocity is still sampled uniformly
        r = torch.empty(len(ids), device=self.device)
        self.vel_command_b[ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)

        # update standing envs
        self.is_standing_env[ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs


@configclass
class GridAdaptiveVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the grid adaptive velocity command generator.

    Note:
        :attr:`~UniformVelocityCommandCfg.Ranges.lin_vel_x` and :attr:`~UniformVelocityCommandCfg.Ranges.ang_vel_z`
        are ignored since those commands are sampled from :attr:`gac_config` instead. They must still be set to
        satisfy the base class, e.g. to :attr:`gac_config`'s ``limit_x_range``/``limit_w_range``.
        :attr:`~UniformVelocityCommandCfg.Ranges.lin_vel_y` is still used as-is.
    """

    class_type: type = GridAdaptiveVelocityCommand

    gac_config: GridAdaptiveConfig = GridAdaptiveConfig()
    """Configuration of the underlying grid adaptive curriculum (bin resolution, limits, thresholds)."""

    lin_vel_xy_std: float = 0.5
    """Standard deviation of the exponential kernel used to score linear velocity tracking success."""

    ang_vel_z_std: float = 0.5
    """Standard deviation of the exponential kernel used to score angular velocity tracking success."""

    heading_command: bool = False  # --> the curriculum directly controls the angular-z velocity command.
    rel_standing_envs: float = 0.0  # --> commands are always moving, matching the GAC reference implementation.
