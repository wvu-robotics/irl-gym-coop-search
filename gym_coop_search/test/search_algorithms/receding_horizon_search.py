

import numpy as np
from make_observation import make_observation
import time

# Define possible actions
actions = [
    np.array([0, -1]), # up
    np.array([-1, 0]), # left
    np.array([0, 1]),  # down
    np.array([1, 0]),  # right
]

# Simulates a step
def simulate_step(size_x, size_y, position, action, obs, temp_distribution):
    new_position = position + actions[action]
    off_grid = np.any(new_position < 0) or new_position[0] >= size_x or new_position[1] >= size_y
    new_position = np.clip(new_position, [0, 0], [size_x - 1, size_y - 1])
    temp_distribution = make_observation(size_x, size_y, new_position, obs, temp_distribution.copy())
    return tuple(new_position), temp_distribution

# Receding horizon search function
def receding_horizon_search(size_x, size_y, cur_pos, temp_distribution, obs, last_action=None, horizon=5, rollout=5, epsilon=0.1):
    # Initialize best action and maximum reward
    best_action = None
    max_reward = -np.inf

    # Define opposite actions
    opposite_actions = [2, 3, 0, 1]

    # Loop over all possible actions
    for initial_action in range(4):
        # Skip if the initial action is the opposite of the last action
        if last_action is not None and initial_action == opposite_actions[last_action]:
            continue

        # Initialize position and total reward
        position = cur_pos
        total_reward = 0  # Initialize total reward as 0

        # Temporary distribution for each path simulation
        temp_distribution_path = temp_distribution.copy()

        for i in range(rollout):
            # Simulate 'horizon' steps into the future
            for h in range(horizon):
                # Apply decaying epsilon-greedy strategy
                epsilon_decay = epsilon / (i+1) # Increase the exploitation ratio as the rollout goes on
                if np.random.rand() < epsilon_decay:
                    action = np.random.randint(4) # Random action
                else:
                    action = initial_action if h == 0 else np.argmax([temp_distribution_path[new_pos] for new_pos in [simulate_step(size_x, size_y, position, a, obs, temp_distribution_path)[0] for a in range(4)]])
                # Simulate one step
                position, temp_distribution_path = simulate_step(size_x, size_y, position, action, obs, temp_distribution_path)
                # Update total reward
                total_reward += temp_distribution_path[position] * 0.95 ** h

        # Calculate average reward
        avg_reward = total_reward / (rollout * horizon)
        # Update best action if this action has higher average reward
        if avg_reward > max_reward:
            max_reward = avg_reward
            best_action = initial_action

    # print(round(temp_distribution[cur_pos[0], cur_pos[1]],3))
    print(round(max_reward,3))
    # Simulate the step with the best action
    new_position, temp_distribution = simulate_step(size_x, size_y, cur_pos, best_action, obs, temp_distribution)
    # Return the best action, the updated distribution, and the new action
    return best_action, temp_distribution, max_reward











# import numpy as np
# from make_observation import make_observation

# actions = [
#     np.array([0, -1]), # up
#     np.array([-1, 0]), # left
#     np.array([0, 1]),  # down
#     np.array([1, 0]),  # right
# ]

# def is_valid_position(size_x, size_y, position):
#     return 0 <= position[0] < size_x and 0 <= position[1] < size_y

# def simulate_step(size_x, size_y, position, action, obs, temp_distribution, penalty, visited_cells):
#     new_position = position + actions[action]
#     off_grid = not is_valid_position(size_x, size_y, new_position)
#     new_position = tuple(np.clip(new_position, [0, 0], [size_x - 1, size_y - 1]))
    
#     if new_position in visited_cells:
#         return position, temp_distribution, penalty

#     temp_distribution = make_observation(size_x, size_y, new_position, obs, temp_distribution)
#     return new_position, temp_distribution, 2*penalty if off_grid else 0

# def receding_horizon_search(size_x, size_y, cur_pos, temp_distribution, obs, visited, horizon=5, rollout=5, epsilon=0.1, penalty=-6):
#     best_action = None
#     max_reward = -np.inf
#     visited_cells = {tuple(cur_pos)}
    
#     for initial_action in range(4):
#         position = tuple(cur_pos)
#         total_reward = np.zeros(rollout)

#         for i in range(rollout):
#             for h in range(horizon):
#                 if np.random.rand() < epsilon:
#                     action = np.random.randint(4) # Random action
#                 else:
#                     action = initial_action if h == 0 else np.argmax([temp_distribution[new_pos] if new_pos not in visited_cells else penalty for new_pos in [simulate_step(size_x, size_y, position, a, obs, temp_distribution, penalty, visited_cells)[0] for a in range(4)]])

#                 position, temp_distribution, off_grid_penalty = simulate_step(size_x, size_y, position, action, obs, temp_distribution, penalty, visited_cells)
#                 reward = temp_distribution[position] if position not in visited_cells else penalty
#                 total_reward[i] += reward + off_grid_penalty

#                 if position not in visited_cells:
#                     visited_cells.add(position)

#         avg_reward = total_reward.mean()
#         if avg_reward > max_reward:
#             max_reward = avg_reward
#             best_action = initial_action

#     print(max_reward)

#     new_position, temp_distribution, off_grid_penalty = simulate_step(size_x, size_y, tuple(cur_pos), best_action, obs, temp_distribution, penalty, visited_cells)
#     visited[new_position[0], new_position[1]] = True
    
#     return best_action, visited, max_reward



