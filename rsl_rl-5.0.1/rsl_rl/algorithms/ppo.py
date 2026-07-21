# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from isaaclab_rl.rsl_rl.rl_cfg import L2C2Cfg

from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        lin_vel_estimator: MLPModel | None = None,
        l2c2_cfg: L2C2Cfg | None = None,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # Resolve the data augmentation function (supports string names or direct callables)
            symmetry_cfg["data_augmentation_func"] = resolve_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Check if the policy is compatible with symmetry
            if actor.is_recurrent or critic.is_recurrent:
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Create the optimizer
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
        )  # type: ignore

        # Lin Vel Estimator components
        self.lin_vel_estimator = lin_vel_estimator

        if self.lin_vel_estimator is not None:
            self.lin_vel_estimator = self.lin_vel_estimator.to(self.device)
            self.lin_vel_estimator_optimizer = optim.Adam(
                self.lin_vel_estimator.parameters(), lr=self.lin_vel_estimator.learning_rate
            )

        # L2C2 components
        self.l2c2_cfg = l2c2_cfg
        if self.l2c2_cfg is not None:
            print("Using L2C2 regularization.")
            self.l2c2_epsilon = (l2c2_cfg["sigma"] * l2c2_cfg["lambda_under"]) / (
                l2c2_cfg["lambda_upper"] - l2c2_cfg["sigma"] * l2c2_cfg["lambda_under"]
            )
            self.l2c2_lambda_actor = l2c2_cfg["lambda_upper"] * self.l2c2_epsilon
        else:
            self.l2c2_epsilon = None
            self.l2c2_lambda_actor = None

        self._prev_actor_input: torch.Tensor | None = None
        self._prev_critic_input: torch.Tensor | None = None
        self._prev_input_valid: torch.Tensor | None = None

        # Losses
        self.l2_loss = nn.MSELoss()

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def _get_model_input(self, model: MLPModel, obs: TensorDict) -> torch.Tensor:
        """Concatenate the observation groups consumed by a model without applying model-specific transforms."""
        return torch.cat([obs[obs_group] for obs_group in model.obs_groups], dim=-1)

    def _masked_mse(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        masks: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute an MSE reduction that ignores invalid recurrent samples when masks are provided."""
        sample_loss = (left - right).pow(2).mean(dim=-1)
        combined_mask = None
        if masks is not None:
            combined_mask = masks.to(sample_loss.dtype)
        if valid_mask is not None:
            valid_mask = valid_mask.squeeze(-1).to(sample_loss.dtype)
            combined_mask = valid_mask if combined_mask is None else combined_mask * valid_mask
        if combined_mask is None:
            return sample_loss.mean()
        return (sample_loss * combined_mask).sum() / combined_mask.sum().clamp_min(1.0)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())

        if self.lin_vel_estimator:
            obs["estimated_lin_vel"] = self.lin_vel_estimator(obs["proprioceptive"]).detach()

        # Record the previous actor & critic input for L2C2 loss
        if self.l2c2_cfg is not None:
            actor_input = self._get_model_input(self.actor, obs).detach()
            critic_input = self._get_model_input(self.critic, obs).detach()
            self.transition.prev_actor_input = (
                self._prev_actor_input if self._prev_actor_input is not None else torch.zeros_like(actor_input)
            )
            self.transition.prev_critic_input = (
                self._prev_critic_input if self._prev_critic_input is not None else torch.zeros_like(critic_input)
            )
            self.transition.prev_input_valid = (
                self._prev_input_valid
                if self._prev_input_valid is not None
                else torch.zeros(actor_input.shape[0], 1, device=self.device, dtype=torch.bool)
            )
            self._prev_actor_input = actor_input
            self._prev_critic_input = critic_input
            self._prev_input_valid = torch.ones(actor_input.shape[0], 1, device=self.device, dtype=torch.bool)
        else:
            self.transition.prev_actor_input = None
            self.transition.prev_critic_input = None
            self.transition.prev_input_valid = None

        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute value for the last step
        last_values = self.critic(obs).detach()
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # Linear velocity estimator loss
        mean_lin_vel_estimator_loss = 0 if self.lin_vel_estimator else None
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # L2C2 loss
        mean_l2c2_loss = 0 if self.l2c2_cfg else None

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # -- Linear velocity estimator loss
            if self.lin_vel_estimator:
                estimated_lin_vel = self.lin_vel_estimator(batch.observations["proprioceptive"])
                true_lin_vel = batch.observations["lin_vel"]
                lin_vel_estimator_loss = self.l2_loss(estimated_lin_vel, true_lin_vel)
                batch.observations["estimated_lin_vel"] = estimated_lin_vel.detach()

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_mean = self.actor.output_mean  # type: ignore
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            # L2C2 loss on the actor/critic inputs stored during rollout collection.
            if self.l2c2_cfg and batch.prev_input_valid is not None:
                actor_input = self._get_model_input(self.actor, batch.observations)
                critic_input = self._get_model_input(self.critic, batch.observations)

                near_actor_input = (
                    batch.prev_actor_input
                    + (actor_input - batch.prev_actor_input) * torch.rand_like(actor_input) * self.l2c2_cfg["sigma"]
                )
                near_critic_input = (
                    batch.prev_critic_input
                    + (critic_input - batch.prev_critic_input) * torch.rand_like(critic_input) * self.l2c2_cfg["sigma"]
                )
                near_actor_output = self.actor.mlp(near_actor_input)
                near_critic_output = self.critic.mlp(near_critic_input)

                actor_input_diff = self._masked_mse(
                    actor_input, batch.prev_actor_input, batch.masks, batch.prev_input_valid
                )
                critic_input_diff = self._masked_mse(
                    critic_input, batch.prev_critic_input, batch.masks, batch.prev_input_valid
                )
                near_actor_output_diff = self._masked_mse(
                    near_actor_output, actions_mean, batch.masks, batch.prev_input_valid
                )
                near_critic_output_diff = self._masked_mse(
                    near_critic_output, values, batch.masks, batch.prev_input_valid
                )

                l2c2_actor_loss = self.l2c2_lambda_actor * (near_actor_output_diff / actor_input_diff)
                l2c2_lambda_value = self.l2c2_lambda_actor * self.l2c2_cfg["beta"]
                l2c2_value_loss = l2c2_lambda_value * (near_critic_output_diff / critic_input_diff)
                l2c2_loss = l2c2_actor_loss + l2c2_value_loss
            else:
                l2c2_loss = None

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # L2C2 loss
            if l2c2_loss is not None:
                loss += l2c2_loss

            # RND loss
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()

            if self.lin_vel_estimator:
                self.lin_vel_estimator.zero_grad()
                lin_vel_estimator_loss.backward()

            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Apply the gradients for LinVelEstimator
            if self.lin_vel_estimator:
                nn.utils.clip_grad_norm_(self.lin_vel_estimator.parameters(), 1.0)
                self.lin_vel_estimator_optimizer.step()

            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # Linear velocity estimator loss
            if mean_lin_vel_estimator_loss is not None:
                mean_lin_vel_estimator_loss += lin_vel_estimator_loss.item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # L2C2 loss
            if mean_l2c2_loss is not None and l2c2_loss is not None:
                mean_l2c2_loss += l2c2_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_lin_vel_estimator_loss is not None:
            mean_lin_vel_estimator_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_l2c2_loss is not None:
            mean_l2c2_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.lin_vel_estimator:
            loss_dict["lin_vel_estimator"] = mean_lin_vel_estimator_loss
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.l2c2_cfg:
            loss_dict["l2c2"] = mean_l2c2_loss

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()
        if self.lin_vel_estimator:
            self.lin_vel_estimator.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()
        if self.lin_vel_estimator:
            self.lin_vel_estimator.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd_optimizer.state_dict()
        if self.lin_vel_estimator:
            saved_dict["lin_vel_estimator_state_dict"] = self.lin_vel_estimator.state_dict()
            saved_dict["lin_vel_estimator_optimizer_state_dict"] = self.lin_vel_estimator_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
                "lin_vel_estimator": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if load_cfg.get("lin_vel_estimator") and self.lin_vel_estimator:
            self.lin_vel_estimator.load_state_dict(loaded_dict["lin_vel_estimator_state_dict"], strict=strict)
            self.lin_vel_estimator_optimizer.load_state_dict(loaded_dict["lin_vel_estimator_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self.actor

    def get_lin_vel_estimator(self) -> MLPModel | None:
        """Get the linear velocity estimator model, or None if not used."""
        return self.lin_vel_estimator

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # If linear velocity estimator is used, initialize it
        lin_vel_estimator = None
        if "lin_vel_estimator" in cfg and cfg["lin_vel_estimator"] is not None:
            lin_vel_estimator_class: type[MLPModel] = resolve_callable(cfg["lin_vel_estimator"].pop("class_name"))  # type: ignore
            output_dim = cfg["lin_vel_estimator"].pop("output_dim")
            input_dim = obs["proprioceptive"].shape[-1]
            lin_vel_estimator = lin_vel_estimator_class(input_dim, output_dim, **cfg["lin_vel_estimator"]).to(device)
            print(f"Linear Velocity Estimator Model: {lin_vel_estimator}")

        # L2C2 configuration
        l2c2_cfg = cfg.get("l2c2")

        # Initialize the storage
        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], actor.obs_dim, critic.obs_dim, device
        )

        # Initialize the algorithm
        alg: PPO = alg_class(
            actor,
            critic,
            storage,
            lin_vel_estimator=lin_vel_estimator,
            l2c2_cfg=l2c2_cfg,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        if self.lin_vel_estimator:
            model_params.append(self.lin_vel_estimator.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])
        if self.lin_vel_estimator:
            self.lin_vel_estimator.load_state_dict(model_params[-1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        if self.lin_vel_estimator:
            all_params = chain(all_params, self.lin_vel_estimator.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
