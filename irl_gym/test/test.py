import gym, irl_gym

env = gym.make("irl_gym/GridWorld-v0", max_episode_steps=5)

env.reset()
env.reset(4)