from gymnasium.envs.registration import register

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

"""
Installs irl_gym envs

"""
register(
    id='gym_coop_search/GridWorld-v0',
    entry_point='gym_coop_search.envs:GridWorldEnv',
    max_episode_steps=100,
    reward_threshold = None,
    disable_env_checker=False,
    nondeterministic = True,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "params":
        {
            "dimensions": [40,40],
            "goal": [10,10],
            "state": 
            {
                "pose": [20,20],
            },
            "p_false_pos":0.1,
            "p_false_neg":0.1,
            "render": "none",
            "cell_size": 50,
            "prefix": current + "/plot/",
            "save_frames": False,
            "log_level": "WARNING"
        }
    }
)

register(
    id='irl_gym/GridTunnel-v0',
    entry_point='irl_gym.envs:GridTunnelEnv',
    max_episode_steps=100,
    reward_threshold = None,
    disable_env_checker=True,
    nondeterministic = True,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "params":
        {
            "dimensions": [40,40],
            "goal": [10,10],
            "state": 
            {
                "pose": [20,20]
            },
            "r_radius": 5,
            "r_range": (-0.01, 1),
            "p": 0.1,
            "render": "none",
            "cell_size": 50,
            "prefix": current + "/plot/",
            "save_frames": False,
            "log_level": "WARNING"
        }
    }
)

register(
    id='irl_gym/Sailing-v0',
    entry_point='irl_gym.envs:SailingEnv',
    max_episode_steps=100 ,
    reward_threshold = None,
    disable_env_checker=True,
    nondeterministic = True,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "params":
        {
            "dimensions": [40,40],
            "goal": [10,10],
            "state_offset": 15,
            "trap_offset": 17,
            "r_radius": 5,
            "r_range": (-400,1100),
            "p": 0.1,
            "render": "none",
            "cell_size": 50,
            "prefix": current + "/plot/",
            "save_frames": False,
            "log_level": "WARNING"
        }
    }
)