import gym, irl_gym
import numpy as np

import matplotlib.pyplot as plt

#"log_level": "DEBUG",
param = {"render": "plot", "dimensions": [50,30], "cell_size": 20, "goal": [15,15], "save_frames": True}
env = gym.make("irl_gym/GridWorld-v0", max_episode_steps=5, params=param)
env.reset()
done = False
while not done:
    s, r, done, _, _ = env.step(1)
    print(s, r)
    plt.pause(1)
    env.render()
