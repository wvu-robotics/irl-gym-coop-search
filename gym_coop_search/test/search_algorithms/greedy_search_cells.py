
import numpy as np

# Perform greedy search (go to the individual cell with the highest value)

def greedy_search_cells(cur_pos, distribution):
    
    # Compute the indices of the maximum probability density
    i, j = np.unravel_index(distribution.argmax(), distribution.shape)
    # Compute the relative position from the current position to the maximum probability density
    relative_pos = np.array([i, j]) - cur_pos

    return relative_pos
