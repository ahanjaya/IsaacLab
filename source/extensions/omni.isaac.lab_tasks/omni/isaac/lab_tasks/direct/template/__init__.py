# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .template_env import TemplateEnv, TemplateEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Template-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.template:TemplateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TemplateEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
