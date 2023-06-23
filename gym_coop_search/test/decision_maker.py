
# import sys
import numpy as np
from search_algorithms.greedy_search_cells import greedy_search_cells
from search_algorithms.event_horizon_search import event_horizon
from search_algorithms.simulated_search import SimBFS
from search_algorithms.ra_ucb_search import RA_UCB
from search_algorithms.beam_search import beam_search
from search_algorithms.depth_limited_search import DLS
from search_algorithms.receding_horizon_search import receding_horizon_search
from search_algorithms.mcts_search import MCTS
from make_observation import make_observation


# Takes in the probability density at each location using the search distribution

def search_decision_maker(size_x, size_y, cur_pos, obs, last_action, distribution, waypoint_reached, waypoint):

    temp_distribution = distribution.copy()

    distribution = make_observation(size_x, size_y, cur_pos, obs, distribution)

    ########################################
    # Search methods

    # relative_pos = greedy_search_cells(cur_pos, distribution) # do greedy search for individual cells

    # Event horizon search
    # relative_pos = event_horizon(cur_pos, distribution)

    # Receding horizon search
    # best_action, visited, reward = receding_horizon_search(size_x, size_y, cur_pos, temp_distribution, obs, last_action, horizon=40, rollout=5, epsilon=0.1)
    # relative_pos = np.array(cur_pos) - np.array(cur_pos)
    # print(reward)

    # BFS event horizon search
    # next_pos = SimBFS(size_x, size_y, cur_pos, distribution, steps=6)
    # relative_pos = np.array(next_pos) - np.array(cur_pos)

    # Rollout Allocation using Upper Confidence Bounds (RA-UCB)
    # next_pos = RA_UCB(size_x, size_y, cur_pos, distribution, steps=15, num_samples=100)
    # relative_pos = np.array(next_pos) - np.array(cur_pos)

    # Beam search
    # next_pos = beam_search(size_x, size_y, cur_pos, distribution, steps=25, beam_width=20)
    # relative_pos = np.array(next_pos) - np.array(cur_pos)

    # Depth limited search
    # next_pos = DLS(size_x, size_y, cur_pos, distribution, limit=4)
    # relative_pos = np.array(next_pos) - np.array(cur_pos)

    mcts = MCTS(size_x, size_y, cur_pos, distribution, num_iterations=200, c_param=0.6) # higher c values prioritize exploration (0 to 1)
    # Run the MCTS to get the optimal action
    best_action = mcts.simulate()
    relative_pos = np.array(cur_pos) - np.array(cur_pos)

    # Perform random search (go to a random cell)
    # if waypoint_reached is True:
    #     # Compute random indices
    #     i = np.random.randint(0, distribution.shape[0])
    #     j = np.random.randint(0, distribution.shape[1])

    #     waypoint[0] = i
    #     waypoint[1] = j
    #     # Compute the relative position from the current position to the maximum probability density
    
    # relative_pos = np.array([i, j]) - cur_pos

    ########################################

    # Choose an action - go to the desired cell
    # Choose the action that minimizes the relative position in both x and y axes
    # From gym env:
    # 0: np.array([0, -1]), # up
    # 1: np.array([-1, 0]), # left
    # 2: np.array([0, 1]),  # down
    # 3: np.array([1, 0]),  # right

    if np.abs(relative_pos[0]) >= np.abs(relative_pos[1]):
        if relative_pos[0] < 0:
            action = 1  # Move left
        else:
            action = 3  # Move right
    else:
        if relative_pos[1] < 0:
            action = 0  # Move up
        else:
            action = 2  # Move down
    if relative_pos[0] == 0 and relative_pos[1] == 0:
        action = 4

    if np.abs(relative_pos[0]) < 1 and np.abs(relative_pos[1]) < 1:
        waypoint_reached = True
    else:
        waypoint_reached = False

    action = best_action
    return action, distribution, waypoint_reached, waypoint
