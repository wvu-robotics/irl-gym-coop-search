import gymnasium as gym
# import irl_gym
import gym_coop_search
import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt
import pickle
from view_results import view_results
from decision_maker import search_decision_maker
from probability_distribution import gaussian_filter
from obstacles import obstacles0, obstacles1, obstacles2
from gym_coop_search.test.update_distribution import update_distribution
import datetime
import time

num_trials = 3  # define the number of trials

distribution_seed = 26 # set the random seed for the gaussian distribution generation
rng = r.default_rng(seed=distribution_seed)

# size of environment
size_x = 28
size_y = 20

# start and goal positions [x, y]
start = [26, 18]
goal = [4, 4]

# Probability Distributions
# define the ranges for random values
goal_dist_offset_range = [-4, 4] # range to offset the distribution around the goal so that the goal is not in the exact center. values in squares
num_peaks_range = [3, 6]  # range for number of peaks (could be overlapping peaks)
peak_height_range = [0.18, 0.88]  # range for peak heights. values in probability within a square
peak_width_range_x = [0.08, 0.5]  # range for peak width in x. values in % of total environment size
peak_width_range_y = [0.08, 0.5]  # range for peak width in y. values in % of total environment size
peak_rot_range = [0, np.deg2rad(180)]  # range for peak orientation

# create the search distribution
search_distribution, peaks, centers = gaussian_filter(distribution_seed, size_x, size_y, goal, goal_dist_offset_range, num_peaks_range, peak_height_range, peak_width_range_x, peak_width_range_y, peak_rot_range)

# search_distribution = np.full((size_x, size_y), 0.5) # creates a uniform search distribution

# create obstacles within the environment (call 'obstacles0' for no obstacles)
obstacles, obstacles_name = obstacles2(size_x, size_y, start, goal)
obstacles = obstacles.astype(int)

search_distribution = search_distribution * ~obstacles.astype(bool) # set the search distribution cells to zero where there are obstacles

param = {"render": "plot", "render_fps": 150, "dimensions": [size_x, size_y], "cell_size": 30, "goal": [goal[0], goal[1]], "state": {"pose":[start[0], start[1]]}, "p_false_pos":0.1, "p_false_neg":0.1}
env = gym.make("gym_coop_search/GridWorld-v0", max_episode_steps=1200, params=param, obstacles=obstacles)

original_search_distribution = np.copy(search_distribution)


# save setup
setup = {"distribution_seed": distribution_seed, "size_x": size_x, "size_y": size_y, "start": start, "goal": goal,
         "goal_dist_offset_range": goal_dist_offset_range, "num_peaks_range": num_peaks_range,
         "peak_height_range": peak_height_range, "peak_width_range_x": peak_width_range_x,
         "peak_width_range_y": peak_width_range_y, "peak_rot_range": np.round(peak_rot_range,4),
         "obstacles": obstacles_name, "p_false_pos": param["p_false_pos"], "p_false_neg": param["p_false_neg"],
         "max_episode_steps": 1200, "num_trials": num_trials}

results = []  # store results from each trial

for i in range(num_trials):
    print(f'Starting trial {i+1}')

    search_distribution = np.copy(original_search_distribution) # reset search distribution
    env.reset()
    done = False
    current_position = start
    waypoint = start
    waypoint_reached = True
    search_observation = False # initial search observation (the agent shouldn't be at the goal intially)
    visited = np.zeros((size_x, size_y), dtype=bool)
    action=None
    step = 0
    
    while not done:
        action, search_distribution, waypoint_reached, waypoint = search_decision_maker(size_x, size_y, current_position, search_observation, obstacles, action, search_distribution, waypoint_reached, waypoint) # step with decision maker for search
        s, r, done, is_trunc, information = env.step(action)
        current_position = s["pose"] # update current position
        search_observation = s["obs"]
        search_distribution = update_distribution(size_x, size_y, current_position, search_observation, obstacles, search_distribution) # update the search distribution
        print(f'{i+1} {step}')
        env.custom_render(search_distribution, obstacles) # render the environment with search probabilities
        step += 1
        done = done or is_trunc

    results.append((step, not is_trunc))  # save the number of steps and is_trunc status for each trial
    if is_trunc == False:
        print(f'Trial {i+1}: Found goal after {step} steps')
    else:
        print(f'Trial {i+1}: Ran out of steps after {step} steps')

print(f'\nRan {num_trials} trials \n')


# get current time to use in file name
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_name = f'experiment data/experiment_data_{current_time}.pickle'

# save setup and results
with open(file_name, 'wb') as f:
    pickle.dump((setup, results), f)

# print results
for i, result in enumerate(results):
    steps, is_trunc = result
    if is_trunc == False:
        print(f'Trial {i+1}: Goal Reached in {steps} steps')
    else:
        print(f'Trial {i+1}: Ran out of steps after {steps} steps')

print(f'Setup and results saved within: {file_name}')

view_results(file_name)
