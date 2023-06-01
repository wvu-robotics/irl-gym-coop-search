
import numpy as np

def event_horizon(cur_pos, distribution, horizon=5):
    """
    This function performs an event horizon search within the given horizon.
    It returns the position with the highest probability density within the horizon.
    
    cur_pos: Current position of the agent.
    distribution: The search distribution.
    horizon: The event horizon distance.
    """
    # Calculate the bounds of the event horizon.
    min_x = max(0, cur_pos[0] - horizon)
    max_x = min(distribution.shape[0], cur_pos[0] + horizon)
    min_y = max(0, cur_pos[1] - horizon)
    max_y = min(distribution.shape[1], cur_pos[1] + horizon)
    
    # Extract the part of the distribution within the event horizon.
    horizon_distribution = distribution[min_x:max_x, min_y:max_y]
    
    # Find the highest probability density within the event horizon.
    max_prob_pos = np.unravel_index(horizon_distribution.argmax(), horizon_distribution.shape)
    
    # Convert the position back to the coordinates in the original distribution.
    max_prob_pos = (max_prob_pos[0] + min_x, max_prob_pos[1] + min_y)
    
    return max_prob_pos
