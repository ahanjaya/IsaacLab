# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class LinVelEstimator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.learning_rate = kwargs["learning_rate"]
        self.input_dim = kwargs["input_dim"]
        self.output_dim = kwargs["output_dim"]
        self.hidden_dims = kwargs["hidden_dims"]
        activation = resolve_nn_activation(kwargs["activation"])

        mlp_layer = []
        input_linear = self.input_dim

        for i in range(len(self.hidden_dims)):
            output_linear = self.hidden_dims[i]
            mlp_layer.append(nn.Linear(input_linear, output_linear))
            mlp_layer.append(activation)
            input_linear = output_linear
        mlp_layer.append(nn.Linear(input_linear, self.output_dim))

        self.linvel_estimator = nn.Sequential(*mlp_layer)
        print(f"LinVelEstimator MLP: {self.linvel_estimator}")

    def forward(self, obs):
        return self.linvel_estimator(obs)

    def act_inference(self, obs):
        with torch.no_grad():
            estimated_lin_vel = self.linvel_estimator(obs)
        return estimated_lin_vel

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
