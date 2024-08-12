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


def load_modules(args: argparse.Namespace, device: torch.device) -> ModulesContainer:
    container = ModulesContainer()

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
    task_obs: torch.Tensor,
    encoder_obs_hist: torch.Tensor,
    depth_map_obs: torch.Tensor,
    modules: ModulesContainer,
) -> torch.Tensor:
    # proprioceptive and task observations
    obs = torch.cat([proprio_obs, task_obs], dim=1).detach()

    # estimated observations
    assert modules.estimator is not None
    estimated_obs = modules.estimator(proprio_obs.detach())
    obs = torch.cat([obs, estimated_obs], dim=1).detach()

    # privileged observations
    assert modules.history_privileged_encoder is not None
    encoded_priv = modules.history_privileged_encoder(encoder_obs_hist)
    obs = torch.cat([obs, encoded_priv], dim=1).detach()

    # depth encoding for student policy
    assert modules.depth_encoder is not None
    encoded_depth = modules.depth_encoder(depth_map_obs[0], proprio_obs)
    obs = torch.cat([obs, encoded_depth], dim=1).detach()

    # TODO: check if requires to add clip observations

    return obs
