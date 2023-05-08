
# import sys
import numpy as np
from search_algorithms.greedy_search_cells import greedy_search_cells



# Takes in the probability density at each location using the search distribution

def search_decision_maker(cur_pos, obs, distribution, waypoint_reached, waypoint):

    # Adjust the search distribution using the sensor measurements
    if not obs: # if the agent does not see the object (false)
        distribution[cur_pos[0], cur_pos[1]] *= 0.4 # adjust the probability for the current cell
        distribution += 0.0001 # adjust the probability for all cells
        # print(np.amax(distribution))
        if np.amax(distribution) > 0.999:
            # Normalize the Gaussian filter to 1.0 if any value is over 1.0
            distribution /= np.max(distribution)

    i = waypoint[0]
    j = waypoint[1]

    relative_pos = greedy_search_cells(cur_pos, distribution) # do greedy search for individual cells



    # Perform random search (go to a random cell)
    # if waypoint_reached is True:
    #     # Compute random indices
    #     i = np.random.randint(0, distribution.shape[0])
    #     j = np.random.randint(0, distribution.shape[1])

    #     waypoint[0] = i
    #     waypoint[1] = j
    #     # Compute the relative position from the current position to the maximum probability density
    
    # relative_pos = np.array([i, j]) - cur_pos

    # Choose an action - go to the desired cell
    # Choose the action that minimizes the relative position in both x and y axes
    if np.abs(relative_pos[0]) >= np.abs(relative_pos[1]):
        if relative_pos[0] < 0:
            action = 1  # Move up
        else:
            action = 3  # Move down
    else:
        if relative_pos[1] < 0:
            action = 0  # Move left
        else:
            action = 2  # Move right
    if relative_pos[0] == 0 and relative_pos[1] == 0:
        action = 4

    if np.abs(relative_pos[0]) < 1 and np.abs(relative_pos[1]) < 1:
        waypoint_reached = True
    else:
        waypoint_reached = False

    return action, waypoint_reached, waypoint

    # From gym env:
    # 0: np.array([0, -1]), # move up
    # 1: np.array([-1, 0]), # move left
    # 2: np.array([0, 1]),  # move down
    # 3: np.array([1, 0]),  # move right