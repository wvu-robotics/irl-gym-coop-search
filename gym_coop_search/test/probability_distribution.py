
# Generates multiple 2D Gaussian distributions as a 2D array

import numpy as np

def gaussian_filter(size_x, size_y, goal, goal_dist_offset_range, num_peaks_range, peak_height_range, peak_width_range_x, peak_width_range_y, peak_rot_range):
    # Initializing value of x, y as grid of defined size
    x, y = np.meshgrid(np.linspace(-1, 1, size_x),
                       np.linspace(-1, 1, size_y), indexing='ij')
    # Initialize Gaussian filter
    gauss = np.zeros((size_x, size_y))
    # Initialize list to store center coordinates of peaks
    peak_centers = []
    # Generate random values for peak heights, widths, and rotations
    num_peaks = np.random.randint(num_peaks_range[0], num_peaks_range[1])
    peak_heights = np.random.uniform(peak_height_range[0], peak_height_range[1], size=num_peaks)
    peak_widths_x = np.random.uniform(peak_width_range_x[0], peak_width_range_x[1], size=num_peaks)
    peak_widths_y = np.random.uniform(peak_width_range_y[0], peak_width_range_y[1], size=num_peaks)
    peak_rots = np.random.uniform(peak_rot_range[0], peak_rot_range[1], size=num_peaks)
    # Add peaks to Gaussian filter
    firstPeak = True
    withinRange = False
    for i in range(num_peaks): # loop through each peak
        if firstPeak == True: # set the first peak at the goal location
            while withinRange == False: # loop until satisfied
                x_offset = np.random.randint(goal_dist_offset_range[0], goal_dist_offset_range[1]) # generate offset in x
                y_offset = np.random.randint(goal_dist_offset_range[0], goal_dist_offset_range[1]) # generate offset in y
                peak_x = goal[0] + x_offset # offset goal distribution in x
                peak_y = goal[1] + y_offset # offset goal distribution in y
                if 0 <= peak_x <= size_x and 0 <= peak_y <= size_y: # make sure the new center of the peak is within the sim's size
                    withinRange = True
                    firstPeak = False # make sure only the first peak deals with the goal location
            withinRange = False
        else: # generate each peak except for the first one
            # Generate random positions for peaks
            peak_x = np.random.randint(0, size_x)
            peak_y = np.random.randint(0, size_y)
        # Append center coordinates to list
        peak_centers.append([peak_x, peak_y])
        # Calculate the maximum value of the peak
        max_peak_value = peak_heights[i]
        # Calculate the rotated x and y coordinates based on peak rotation
        x_rot = (x - x[peak_x, peak_y]) * np.cos(peak_rots[i]) + (y - y[peak_x, peak_y]) * np.sin(peak_rots[i])
        y_rot = -(x - x[peak_x, peak_y]) * np.sin(peak_rots[i]) + (y - y[peak_x, peak_y]) * np.cos(peak_rots[i])
        # Add Gaussian function centered at the randomly generated positions with specified width, height, and orientation (x_rot and y_rot)
        gauss += max_peak_value * np.exp(-(((x_rot / peak_widths_x[i])**2) + ((y_rot / peak_widths_y[i])**2)) / 2.0)
    
    if np.amax(gauss) > 1.0:
        # Normalize the Gaussian filter to 1.0 if any value is over 1.0
        gauss /= np.max(gauss)
    
    return gauss, num_peaks, peak_centers



