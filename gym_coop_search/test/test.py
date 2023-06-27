import gymnasium as gym
# import irl_gym
import gym_coop_search
import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt
from decision_maker import search_decision_maker
from probability_distribution import gaussian_filter
from obstacles import obstacles0, obstacles1, obstacles2
import time

seed = 26 # set the random seed for the gaussian distribution generation
rng = r.default_rng(seed=seed)

# size of environment
size_x = 28
size_y = 20

# start and goal positions
start = [26, 18]
goal = [5, 3]

# Probability Distributions
# define the ranges for random values
goal_dist_offset_range = [-4, 4] # range to offset the distribution around the goal so that the goal is not in the exact center. values in squares
num_peaks_range = [3, 6]  # range for number of peaks
peak_height_range = [0.18, 0.88]  # range for peak heights. values in probability within a square
peak_width_range_x = [0.08, 0.5]  # range for peak width in x. values in % of total environment size
peak_width_range_y = [0.08, 0.5]  # range for peak width in y. values in % of total environment size
peak_rot_range = [0, np.deg2rad(180)]  # range for peak orientation

# create the search distribution
search_distribution, peaks, centers = gaussian_filter(seed, size_x, size_y, goal, goal_dist_offset_range, num_peaks_range, peak_height_range, peak_width_range_x, peak_width_range_y, peak_rot_range)

# search_distribution = np.full((size_x, size_y), 0.5) # creates a uniform search distribution

# Obstacles
obs = obstacles2(size_x, size_y, start, goal)

#"log_level": "DEBUG",
param = {"render": "plot", "render_fps": 150, "dimensions": [size_x, size_y], "cell_size": 30, "goal": [goal[0], goal[1]], "state": {"pose":[start[0], start[1]]}, "p_false_pos":0.1, "p_false_neg":0.1}
env = gym.make("gym_coop_search/GridWorld-v0", max_episode_steps=1200, params=param, obstacles=obs)
env.reset()
done = False
current_position = start
waypoint = start
waypoint_reached = True
search_observation = False # initial search observation (the agent shouldn't be at the goal intially)
# print(obs)
visited = np.zeros((size_x, size_y), dtype=bool)
action=None
step = 0


# time.sleep(3)

while not done:
    action, search_distribution, waypoint_reached, waypoint = search_decision_maker(size_x, size_y, current_position, search_observation, action, search_distribution, waypoint_reached, waypoint) # step with decision maker for search
    s, r, done, is_trunc, information = env.step(action)

    # print(updated_search_distribution.argmax())
    # s, r, done, is_trunc, _ = env.step(rng.choice(list(range(4)))) # step with random movement
    # print(s)
    current_position = s["pose"] # update current position
    # print(current_position) # print current position
    search_observation = s["obs"]
    print(step)
    # print(search_observation)
    env.custom_render(search_distribution, obs) # render the environment with search probabilities
    step += 1
    done = done or is_trunc

if is_trunc == False:
    print('Goal Reached')
    print(step)
else:
    print('Ran out of steps')
