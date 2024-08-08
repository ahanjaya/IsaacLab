import os
import argparse

import torch
from dataclasses import dataclass
from typing import Optional, Union

ModuleType = Optional[Union[torch.nn.Module, torch.jit.ScriptModule]]


@dataclass
class ModulesContainer:
    policy: ModuleType = None
    estimator: ModuleType = None
    explicit_privileged_encoder: ModuleType = None
    history_privileged_encoder: ModuleType = None
    perception_encoder: ModuleType = None
    depth_encoder: ModuleType = None
    privileged_from: str = "history"  # `history` or `explicit`.
    jit: bool = False


def load_modules(args: argparse.Namespace, device: torch.device) -> ModulesContainer:
    container = ModulesContainer()

    container.jit = True
    policy_file = os.path.join(args.checkpoint_path, "actor_mlp_jit.pt")
    container.policy = torch.jit.load(policy_file).to(device).eval()

    estimator_file = os.path.join(args.checkpoint_path, "estimator_jit.pt")
    container.estimator = torch.jit.load(estimator_file).to(device).eval()

    history_enc_file = os.path.join(
        args.checkpoint_path, "history_state_encoder_jit.pt"
    )
    container.history_privileged_encoder = (
        torch.jit.load(history_enc_file).to(device).eval()
    )

    depth_enc_file = os.path.join(args.checkpoint_path, "depth_encoder_jit.pt")
    container.depth_encoder = torch.jit.load(depth_enc_file).to(device).eval()

    return container


def process_observations(
    proprio_obs: torch.Tensor,
    privileged_obs: Optional[torch.Tensor],
    estimated_gt_obs: Optional[torch.Tensor],
    encoder_obs_hist: Optional[torch.Tensor],
    depth_map_obs: Optional[torch.Tensor],
    modules: ModulesContainer,
) -> torch.Tensor:
    num_task_objectives = 3
    obs = proprio_obs
    proprio_without_task = proprio_obs.detach()[:, :-num_task_objectives]

    if modules.jit and depth_map_obs is not None:
        depth_map_obs = depth_map_obs[0]

    if estimated_gt_obs is not None:
        assert modules.estimator is not None
        estimator_input = obs.detach()[:, :-num_task_objectives]
        estimated_obs = modules.estimator(estimator_input)
        obs = torch.cat([obs, estimated_obs], dim=1).detach()

    if privileged_obs is not None:
        if modules.privileged_from == "history":
            assert modules.history_privileged_encoder is not None
            encoded_priv = modules.history_privileged_encoder(encoder_obs_hist)
        elif modules.privileged_from == "explicit":
            assert modules.explicit_privileged_encoder is not None
            encoded_priv = modules.explicit_privileged_encoder(privileged_obs)
        else:
            raise ValueError()

        obs = torch.cat([obs, encoded_priv], dim=1).detach()

    assert modules.depth_encoder is not None
    encoded_depth = modules.depth_encoder(depth_map_obs, proprio_without_task)
    obs = torch.cat([obs, encoded_depth], dim=1).detach()

    return obs
