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
    id="Isaac-Velocity-Flat-NXP-Lower-Body-Direct-v0",
    entry_point=f"{__name__}.velocity_nxp_env:VelocityNXPLowerBodyFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_nxp_env:VelocityNXPLowerBodyFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VelocityNXPLowerBodyFlatPPORunnerCfg",
    },
)
