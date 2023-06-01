
import numpy as np
import heapq

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

def beam_search(size_x, size_y, cur_pos, distribution, steps=5, beam_width=10):
    beam = [(0, [cur_pos])]

    for _ in range(steps):
        new_beam = []
        for score, path in beam:
            position = path[-1]
            for action in range(4):
                new_position = simulate_step(size_x, size_y, position, action)
                new_score = score + distribution[new_position[0], new_position[1]]
                new_path = path + [new_position]
                new_beam.append((new_score, new_path))
        beam = heapq.nlargest(beam_width, new_beam)

    best_score, best_path = beam[0]
    return best_path[1]  # return the next position in the best path
