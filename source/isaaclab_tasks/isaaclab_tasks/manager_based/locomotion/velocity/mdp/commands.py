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

from isaaclab.devices import Se2Gamepad, Se2GamepadCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

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
