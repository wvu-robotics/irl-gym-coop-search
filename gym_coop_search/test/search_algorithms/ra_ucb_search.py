
import numpy as np

def simulate_step(size_x, size_y, position, action):
    action_effects = [
        np.array([0, -1]), # left
        np.array([-1, 0]), # up
        np.array([0, 1]),  # right
        np.array([1, 0]),  # down
    ]
    new_position = position + action_effects[action]
    new_position = np.clip(new_position, [0, 0], [size_x - 1, size_y - 1])
    return tuple(new_position)

def RA_UCB(size_x, size_y, cur_pos, distribution, steps=7, num_samples=40):
    best_action = None
    best_score = -np.inf

    for action in range(4):
        total_score = 0
        for _ in range(num_samples):
            position = simulate_step(size_x, size_y, cur_pos, action)
            score = distribution[position[0], position[1]]
            for _ in range(steps - 1):
                position = simulate_step(size_x, size_y, position, np.argmax(distribution[position[0], position[1]]))
                score += distribution[position[0], position[1]]
            total_score += score
        avg_score = total_score / num_samples
        if avg_score > best_score:
            best_score = avg_score
            best_action = action

    return simulate_step(size_x, size_y, cur_pos, best_action)
