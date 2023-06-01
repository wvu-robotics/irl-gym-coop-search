
import numpy as np

def make_observation(size_x, size_y, cur_pos, obs, new_distribution):
    # Adjust the search distribution using the sensor measurements
    if not obs: # if the agent does not see the object (false)
        new_distribution[cur_pos[0], cur_pos[1]] *= 0.22 # adjust the probability for the current cell
        new_distribution += 0.00005 # adjust the probability for all cells
        # print(np.amax(new_distribution))
        if np.amax(new_distribution) > 0.999:
            # Normalize the Gaussian filter to 1.0 if any value is over 1.0
            new_distribution /= np.max(new_distribution)

    return new_distribution
