import gym, irl_gym

env = gym.make("irl_gym/GridWorld-v0", max_episode_steps=5)

# env.reset()

import inspect
print(inspect.signature(env.reset))
print(env.reset)
env.reset()
env.reset(seed=4, options={})