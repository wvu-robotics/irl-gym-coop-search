#__all__ = ["gridworld", "gridtrap", "sailing"]

from gym.envs.registration import register
from grid_world import GridWorldEnv

register(
    id='decision_making/GridWorld-v0',
    entry_point='decision_making.envs:GridWorldEnv',
    max_episode_steps=100,
)

register(
    id='decision_making/GridTunnel-v0',
    entry_point='decision_making.envs:GridTunnelEnv',
    max_episode_steps=100,
)

register(
    id='decision_making/Sailing-v0',
    entry_point='decision_making.envs:SailingEnv',
    max_episode_steps=100,
)