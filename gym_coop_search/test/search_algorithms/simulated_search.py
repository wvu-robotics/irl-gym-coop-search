
import numpy as np

def simulate_step(size_x, size_y, position, action):
    # Define the effect of each action.
    action_effects = [
        np.array([0, -1]), # left
        np.array([-1, 0]), # up
        np.array([0, 1]),  # right
        np.array([1, 0]),  # down
    ]
    # Apply the action.
    new_position = position + action_effects[action]
    # Check for bounds and clip if necessary.
    new_position = np.clip(new_position, [0, 0], [size_x - 1, size_y - 1])
    return tuple(new_position)

def SimBFS(size_x, size_y, cur_pos, distribution, steps=5):
    # The queue for the BFS, each element is a path (a list of positions).
    queue = [[cur_pos]]
    best_path = []
    best_score = -np.inf

    while queue:
        path = queue.pop(0)
        position = path[-1]
        score = sum(distribution[pos[0], pos[1]] for pos in path)
        if len(path) < steps:
            # Continue expanding this path.
            for action in range(4):
                new_position = tuple(simulate_step(size_x, size_y, position, action))
                # if new_position not in path:  # avoid cycles
                queue.append(path + [new_position])
        elif score > best_score:
            best_path = path
            best_score = score

    return best_path[1] if best_path else cur_pos  # Return the next step in the best path.
