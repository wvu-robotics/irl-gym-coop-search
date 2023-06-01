

import numpy as np
from make_observation import make_observation

# Define possible actions
actions = [
    np.array([0, -1]), # up
    np.array([-1, 0]), # left
    np.array([0, 1]),  # down
    np.array([1, 0]),  # right
]

# Simulates a step
def simulate_step(size_x, size_y, position, action, obs, temp_distribution, penalty):
    new_position = position + actions[action]
    off_grid = np.any(new_position < 0) or new_position[0] >= size_x or new_position[1] >= size_y
    new_position = np.clip(new_position, [0, 0], [size_x - 1, size_y - 1])
    temp_distribution = make_observation(size_x, size_y, new_position, obs, temp_distribution)
    return tuple(new_position), temp_distribution, penalty if off_grid else 0


# Receding horizon search function
def receding_horizon_search(size_x, size_y, cur_pos, temp_distribution, obs, visited, horizon=4, epsilon=0.1, penalty=-6):
    # Initialize best action and maximum reward
    best_action = None
    max_reward = -np.inf
    
    # Loop over all possible actions
    for initial_action in range(4):
        # Initialize position and total reward
        position = cur_pos
        total_reward = 0
        
        # Copy visited cells map for this look-ahead sequence
        visited_copy = visited.copy()

        # Simulate 'horizon' steps into the future
        for h in range(horizon):
            # Apply epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action = np.random.randint(4) # Random action
            else:
                action = initial_action if h == 0 else np.argmax([temp_distribution[new_pos] if not visited_copy[new_pos[0], new_pos[1]] else penalty for new_pos in [simulate_step(size_x, size_y, position, a, obs, temp_distribution, penalty)[0] for a in range(4)]])
            
            # Simulate one step
            position, temp_distribution, off_grid_penalty = simulate_step(size_x, size_y, position, action, obs, temp_distribution, penalty)
            
            # Update total reward and mark cell as visited
            if visited_copy[position[0], position[1]]:
                total_reward += penalty
            else:
                total_reward += temp_distribution[position[0], position[1]] + off_grid_penalty
                visited_copy[position[0], position[1]] = True

        # Update best action if this action has higher total reward
        if total_reward > max_reward:
            max_reward = total_reward
            best_action = initial_action

    # Mark the new position as visited
    new_position, temp_distribution, off_grid_penalty = simulate_step(size_x, size_y, cur_pos, best_action, obs, temp_distribution, penalty)
    visited[new_position[0], new_position[1]] = True # the current cell has now been visited
    
    # print(max_reward)

    # Return the new position that results from taking the best action
    return best_action, visited, max_reward


