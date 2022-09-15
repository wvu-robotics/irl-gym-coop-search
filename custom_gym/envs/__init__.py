#__all__ = ["gridworld", "gridtrap", "sailing"]

from gym.envs.registration import register
from grid_world import GridWorldEnv

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