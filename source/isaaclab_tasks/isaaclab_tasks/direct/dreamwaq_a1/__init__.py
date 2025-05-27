# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-DreamWaQ-A1-Direct-Flat-v0",
    entry_point=f"{__name__}.dreamwaq_a1_env:DreamWaQA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dreamwaq_a1_env:DreamWaQA1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DreamWaQA1FlatPPORunnerCfg",
    },
)
