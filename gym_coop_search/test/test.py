import gymnasium as gym
# import irl_gym
import gym_coop_search
import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt
from probability_distribution import gaussian_filter

rng = r.default_rng()

# Probability Distributions
# size of environment
size_x = 35
size_y = 25
# define the ranges for random values
num_peaks_range = [2, 6]  # range for number of peaks
peak_height_range = [0.15, 0.9]  # range for peak heights
peak_width_range_x = [0.1, 0.4]  # range for peak width in x
peak_width_range_y = [0.15, 0.3]  # range for peak width in y
peak_rot_range = [0, np.pi]  # range for peak orientation

gaussian, peaks, centers = gaussian_filter(size_x, size_y, num_peaks_range=num_peaks_range, peak_height_range=peak_height_range, peak_width_range_x=peak_width_range_x, peak_width_range_y=peak_width_range_y, peak_rot_range=peak_rot_range)
# print(centers)

# set the goal at the center of a generated distribution
goalChoice = rng.choice(list(range(len(centers))))
goal = centers[goalChoice]
# print(goal)

#"log_level": "DEBUG",
param = {"render": "plot", "render_fps": 15, "dimensions": [size_x, size_y], "cell_size": 22, "goal": [goal[1], goal[0]]}
env = gym.make("gym_coop_search/GridWorld-v0", max_episode_steps=150, params=param)
env.reset()
done = False

while not done:
    s, r, done, is_trunc, _ = env.step(rng.choice(list(range(4)))) # step with random movement
    print(s, r)
    env.custom_render(gaussian)
    done = done or is_trunc


