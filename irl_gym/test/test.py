import gymnasium as gym
import, irl_gym
import numpy as np

import matplotlib.pyplot as plt

#"log_level": "DEBUG",
param = {"render": "plot", "dimensions": [40,11], "cell_size": 20, "goal": [10,5]}
env = gym.make("irl_gym/GridTunnel-v0", max_episode_steps=5, params=param)
env.reset()
done = False
while not done:
    s, r, done, is_trunc, _ = env.step(0)
    env.reset()
    print(s, r)
    plt.pause(1)
    env.render()
    done = done or is_trunc
