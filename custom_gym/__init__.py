#__all__ = ["gridworld", "gridtrap", "sailing"]

# import sys
# import os
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)

from gym.envs.registration import register


register(
    id='custom_gym/GridWorld-v0',
    entry_point='custom_gym.envs:GridWorldEnv',
    max_episode_steps=100,
    reward_threshold = None,
    nondeterministic = False,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "agent": [20,20],
        "dimensions": [40,40],
        "goal": [10,10],
        "p": 0.1,
        "prefix": "/home/jared/ambiguity_ws/data/gridworld/",
        "reward_bounds": [0,1]
    }
)

register(
    id='custom_gym/GridTunnel-v0',
    entry_point='custom_gym.envs:GridTunnelEnv',
    max_episode_steps=100,
    reward_threshold = None,
    nondeterministic = False,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "agent": [20,20],
        "dimensions": [35,10],
        "goal": [10,10],
        "p": 0.1,
        "prefix": "/home/jared/ambiguity_ws/data/gridworld/",
        "reward_bounds": [0,1]
    }
)

register(
    id='custom_gym/Sailing-v0',
    entry_point='custom_gym.envs:SailingEnv',
    max_episode_steps=100,
    reward_threshold = None,
    nondeterministic = False,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "agent": [20,20],
        "dimensions": [40,40],
        "goal": [10,10],
        "p": 0.1,
        "prefix": "/home/jared/ambiguity_ws/data/gridworld/",
        "reward_bounds": [0,1]
    }
)