import gym, irl_gym
import numpy as np

import matplotlib.pyplot as plt

#"log_level": "DEBUG",
param = {"render": "plot", "dimensions": [50,50], "cell_size": 20, "goal": [15,15]}
env = gym.make("irl_gym/Sailing-v0", max_episode_steps=5, params=param)
env.reset()
done = False
while not done:
    s, r, done, _, _ = env.step(0)
    print(s, r)
    plt.pause(1)
    env.render()
