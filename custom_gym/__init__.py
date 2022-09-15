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
)

register(
    id='custom_gym/GridTunnel-v0',
    entry_point='custom_gym.envs:GridTunnelEnv',
    max_episode_steps=100,
)

register(
    id='custom_gym/Sailing-v0',
    entry_point='custom_gym.envs:SailingEnv',
    max_episode_steps=100,
)