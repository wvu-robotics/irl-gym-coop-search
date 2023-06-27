
import numpy as np

def update_distribution(size_x, size_y, cur_pos, observation, obstacles, new_distribution):
    # Adjust the search distribution using the sensor measurements
    if not observation: # if the agent does not see the object (false)
        # Keep track of original probability at current position
        original_prob = new_distribution[cur_pos[0], cur_pos[1]]

        # Reduce the probability at current position
        new_distribution[cur_pos[0], cur_pos[1]] = 0.0
        reduced_prob = new_distribution[cur_pos[0], cur_pos[1]]

        # Calculate the probability to be redistributed to the other cells
        redistributed_prob = original_prob - reduced_prob
        prob_per_cell = redistributed_prob / ((size_x * size_y) - np.sum(obstacles))

        # Increase the probability for all non-obstacle cells
        new_distribution += prob_per_cell

        # Set the search distribution cells to zero where there are obstacles
        new_distribution = new_distribution * ~obstacles.astype(bool)

        # Normalize the distribution if necessary
        if np.amax(new_distribution) > 0.999:
            # Normalize the Gaussian filter to 1.0 if any value is over 1.0
            new_distribution /= np.max(new_distribution)

    return new_distribution