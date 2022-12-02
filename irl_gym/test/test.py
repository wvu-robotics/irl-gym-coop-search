import gym, irl_gym
import numpy as np
env = gym.make("irl_gym/GridWorld-v0", max_episode_steps=5)

# env.reset()
# import inspect
# print(inspect.signature(env.reset))
# print(env.reset)
env.reset()
print(env.step(3))
env.reset(seed=4, options={"state":{"pose":np.array([5,0])}})
print(env.step(3))
env.reset(seed=4, options={"state":{"pose":np.array([3,3])}})
print(env.step(3))
env.reset(seed=4, options={"state":{"pose":np.array([2,0])}})