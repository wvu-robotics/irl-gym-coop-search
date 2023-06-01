
import numpy as np

def simulate_step(size_x, size_y, position, action):
    action_effects = [
        np.array([0, -1]),  # Move up
        np.array([-1, 0]),  # Move left
        np.array([0, 1]),   # Move down
        np.array([1, 0]),   # Move right
    ]
    new_position = position + action_effects[action]
    new_position = np.clip(new_position, [0, 0], [size_x - 1, size_y - 1])
    return tuple(new_position)

def DLS(size_x, size_y, cur_pos, distribution, limit=5):
    best_path = None
    best_score = -np.inf

    def recursive_DLS(position, path, score, depth):
        nonlocal best_path
        nonlocal best_score
        if depth == 0:
            if score > best_score:
                best_score = score
                best_path = path
        else:
            for action in range(4):
                new_position = simulate_step(size_x, size_y, position, action)
                new_score = score + distribution[new_position[0], new_position[1]]
                recursive_DLS(new_position, path + [new_position], new_score, depth - 1)

    recursive_DLS(cur_pos, [cur_pos], 0, limit)

    return best_path[1]  # return the next position in the best path
