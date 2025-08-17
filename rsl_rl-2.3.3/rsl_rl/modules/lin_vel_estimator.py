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

    def forward(self, observations):
        return self.linvel_estimator(observations)

    def act_inference(self, observations):
        with torch.no_grad():
            estimated_lin_vel = self.linvel_estimator(observations)
        return estimated_lin_vel
