# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import MLP


class LinVelEstimatorModel(nn.Module):
    """MLP model for estimating linear velocity from proprioceptive observations.

    Unlike MLPModel, this model operates directly on a raw tensor input rather than
    a TensorDict, since it is always fed a single observation group (proprioceptive).
    """

    is_recurrent: bool = False

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        activation: str = "elu",
        learning_rate: float = 1e-4,
    ) -> None:
        """Initialize the linear velocity estimator model.

        Args:
            input_dim: Dimensionality of the input tensor.
            output_dim: Dimensionality of the output tensor.
            hidden_dims: Sizes of the hidden layers in the MLP.
            activation: Activation function name to use in the MLP.
            learning_rate: Learning rate for the optimizer.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.mlp = MLP(input_dim, output_dim, hidden_dims, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass through the MLP.

        Args:
            x: Input tensor of proprioceptive observations.

        Returns:
            Estimated linear velocity tensor.
        """
        return self.mlp(x)
