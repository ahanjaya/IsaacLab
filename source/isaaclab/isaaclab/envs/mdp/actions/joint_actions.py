# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

# import logger
logger = logging.getLogger(__name__)


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        logger.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        print(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the joint action.
        It adds the following information to the base descriptor:
        - joint_names: The names of the joints.
        - scale: The scale of the action term.
        - offset: The offset of the action term.
        - clip: The clip of the action term.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "JointAction"
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.scale = self._scale
        # This seems to be always [4xNum_joints] IDK why. Need to check.
        if isinstance(self._offset, torch.Tensor):
            self._IO_descriptor.offset = self._offset[0].detach().cpu().numpy().tolist()
        else:
            self._IO_descriptor.offset = self._offset
        # FIXME: This is not correct. Add list support.
        if self.cfg.clip is not None:
            if isinstance(self._clip, torch.Tensor):
                self._IO_descriptor.clip = self._clip[0].detach().cpu().numpy().tolist()
            else:
                self._IO_descriptor.clip = self._clip
        else:
            self._IO_descriptor.clip = None
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


class RelativeJointPositionAction(JointAction):
    r"""Joint action term that applies the processed actions to the articulation's joints as relative position commands.

    Unlike :class:`JointPositionAction`, this action term applies the processed actions as relative position commands.
    This means that the processed actions are added to the current joint positions of the articulation's joints
    before being sent as position commands.

    This means that the action applied at every step is:

    .. math::

         \text{applied action} = \text{current joint positions} + \text{processed actions}

    where :math:`\text{current joint positions}` are the current joint positions of the articulation's joints.
    """

    cfg: actions_cfg.RelativeJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.RelativeJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use zero offset for relative position
        if cfg.use_zero_offset:
            self._offset = 0.0

    def apply_actions(self):
        # add current joint positions to the processed actions
        current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        # set position targets
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)


class PolynomialRelativeJointPositionAction(RelativeJointPositionAction):
    r"""Relative joint position action with a polynomial mapping applied to a subset of the joints.

    Intended for joints whose real response is nonlinear (e.g. closed-loop linkages), described by the absolute
    mapping :math:`q = p(x) = \sum_i c_i x^{n-i}` from the linear command space :math:`x` to the joint position
    :math:`q`. For the joints matching :attr:`polynomial_joint_names`, the processed action
    (:math:`\Delta x = \text{scale} \times \text{input action}`) is applied in the command space:

    .. math::

        q_{target} = p(p^{-1}(q_{current}) + \Delta x)

    so a zero action holds the current position. Each joint may use its own polynomial (e.g. mirrored left/right
    joints, where the mirrored mapping is :math:`-p(-x)`). The polynomial must be strictly monotonic so that
    :math:`p^{-1}` exists; it is evaluated with Newton's method. Other joints behave exactly like
    :class:`RelativeJointPositionAction`.
    """

    cfg: actions_cfg.PolynomialRelativeJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.PolynomialRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # indices (within this action term) of the joints that use the polynomial
        self._poly_ids = [
            i
            for i, name in enumerate(self._joint_names)
            if any(re.fullmatch(expr, name) for expr in cfg.polynomial_joint_names)
        ]
        if len(self._poly_ids) == 0:
            raise ValueError(
                f"None of the joints {self._joint_names} match the polynomial joint names {cfg.polynomial_joint_names}."
            )
        # per-joint coefficients ordered from the highest to the lowest degree, shape (num_poly_joints, degree + 1)
        poly_coeffs = [
            self._resolve_coefficients(self._joint_names[i], cfg.polynomial_coefficients) for i in self._poly_ids
        ]
        max_len = max(len(c) for c in poly_coeffs)
        poly_coeffs = [[0.0] * (max_len - len(c)) + list(c) for c in poly_coeffs]
        self._poly_coeffs = torch.tensor(poly_coeffs, dtype=torch.float32, device=self.device)
        degree = max_len - 1
        self._poly_deriv_coeffs = self._poly_coeffs[:, :-1] * torch.arange(degree, 0, -1, device=self.device)

    @staticmethod
    def _resolve_coefficients(joint_name: str, coefficients: list[float] | dict[str, list[float]]) -> list[float]:
        if isinstance(coefficients, dict):
            matches = [coeffs for expr, coeffs in coefficients.items() if re.fullmatch(expr, joint_name)]
            if len(matches) != 1:
                raise ValueError(
                    f"Joint '{joint_name}' must match exactly one key of the polynomial coefficients"
                    f" {list(coefficients.keys())}, but matched {len(matches)}."
                )
            coefficients = matches[0]
        if len(coefficients) < 2:
            raise ValueError(f"The polynomial coefficients for joint '{joint_name}' must be at least of degree 1.")
        return coefficients

    @staticmethod
    def _polyval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Horner's method; coeffs has shape (num_poly_joints, degree + 1) and x has shape (num_envs, num_poly_joints)
        y = torch.zeros_like(x)
        for k in range(coeffs.shape[1]):
            y = y * x + coeffs[:, k]
        return y

    def _polynomial_inverse(self, q: torch.Tensor, num_iters: int = 6) -> torch.Tensor:
        # Newton's method; the identity is a good initial guess for near-linear joint mappings
        x = q.clone()
        for _ in range(num_iters):
            x = x - (self._polyval(self._poly_coeffs, x) - q) / self._polyval(self._poly_deriv_coeffs, x)
        return x

    def apply_actions(self):
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        target_pos = current_pos + self.processed_actions
        poly_current = current_pos[:, self._poly_ids]
        poly_delta = self.processed_actions[:, self._poly_ids]
        target_pos[:, self._poly_ids] = self._polyval(
            self._poly_coeffs, self._polynomial_inverse(poly_current) + poly_delta
        )
        self._asset.set_joint_position_target(target_pos, joint_ids=self._joint_ids)


class JointVelocityAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    cfg: actions_cfg.JointVelocityActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint velocity as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()

    def apply_actions(self):
        # set joint velocity targets
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)


class JointEffortAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

    cfg: actions_cfg.JointEffortActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint effort targets
        self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)
