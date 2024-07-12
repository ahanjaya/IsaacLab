# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .unitree_a1_env import UnitreeA1Env, UnitreeA1FlatEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-A1-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.unitree_a1:UnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnitreeA1FlatEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.UnitreeA1FlatPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
