import gymnasium as gym

from . import agents
from .imitation_policy_a1_env import ImitationPolicyA1Env, ImitationPolicyA1EnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-ImitationA1-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.imitation_policy_a1:ImitationPolicyA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ImitationPolicyA1EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.ImitationPolicyA1PPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
