
# Generates obstacles within the environment

# Can add different environments here

import numpy as np

def obstacles0(size_x, size_y, start, goal): # Use this to make an environment with no obstacles
    obs = np.zeros((size_x, size_y))
    return obs

def obstacles1(size_x, size_y, start, goal): # Env notes: x = 42, y = 25, start = [7, 10], goal = [8, 3]
    obs = np.zeros((size_x, size_y))
    obs[3, 16:22] = 1.0
    return obs

def obstacles2(size_x, size_y, start, goal): # Env notes: x = 42, y = 25, start = [7, 10], goal = [8, 3]
    obs = np.zeros((size_x, size_y))
    obs[4:9, 9] = 1.0
    obs[4, 9:18] = 1.0
    obs[9, 9:20] = 1.0
    return obs


