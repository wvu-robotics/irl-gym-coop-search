
# Generates multiple 2D Gaussian distributions as a 2D array

import numpy as np

def gaussian_filter(seed, size_x, size_y, goal, goal_dist_offset_range, num_peaks_range, peak_height_range, peak_width_range_x, peak_width_range_y, peak_rot_range):
    rng = np.random.default_rng(seed=seed)  # Create a random number generator with the specified seed

    x, y = np.meshgrid(np.linspace(-1, 1, size_x),
                       np.linspace(-1, 1, size_y), indexing='ij')
    
    gauss = np.zeros((size_x, size_y))
    peak_centers = []

    num_peaks = rng.integers(num_peaks_range[0], num_peaks_range[1])
    peak_heights = rng.uniform(peak_height_range[0], peak_height_range[1], size=num_peaks)
    peak_widths_x = rng.uniform(peak_width_range_x[0], peak_width_range_x[1], size=num_peaks)
    peak_widths_y = rng.uniform(peak_width_range_y[0], peak_width_range_y[1], size=num_peaks)
    peak_rots = rng.uniform(peak_rot_range[0], peak_rot_range[1], size=num_peaks)

    firstPeak = True
    withinRange = False
    for i in range(num_peaks):
        if firstPeak:
            while not withinRange:
                x_offset = rng.integers(goal_dist_offset_range[0], goal_dist_offset_range[1])
                y_offset = rng.integers(goal_dist_offset_range[0], goal_dist_offset_range[1])
                peak_x = goal[0] + x_offset
                peak_y = goal[1] + y_offset
                if 0 <= peak_x < size_x and 0 <= peak_y < size_y:
                    withinRange = True
                    firstPeak = False
            withinRange = False
        else:
            peak_x = rng.integers(0, size_x)
            peak_y = rng.integers(0, size_y)

        peak_centers.append([peak_x, peak_y])
        max_peak_value = peak_heights[i]
        x_rot = (x - x[peak_x, peak_y]) * np.cos(peak_rots[i]) + (y - y[peak_x, peak_y]) * np.sin(peak_rots[i])
        y_rot = -(x - x[peak_x, peak_y]) * np.sin(peak_rots[i]) + (y - y[peak_x, peak_y]) * np.cos(peak_rots[i])
        gauss += max_peak_value * np.exp(-(((x_rot / peak_widths_x[i])**2) + ((y_rot / peak_widths_y[i])**2)) / 2.0)
    
    if np.amax(gauss) > 1.0:
        gauss /= np.max(gauss)
    
    return gauss, num_peaks, peak_centers




