import numpy as np

def get_distance(s1, s2):
    """
    Computes the Euclidean distance between two 2D points
    
    :param s1: ([x,y]) point 1
    :param s2: ([x,y]) point 2
    :return: (float) distance
    """
    return np.sqrt( (s1[0]-s2[0])**2 + (s1[1]-s2[1])**2 )