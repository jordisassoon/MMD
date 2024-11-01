import numpy as np


def rbf_kernel(x, y, sigma=1.0):
    """
    Computes the Radial Basis Function (RBF) kernel between two vectors x and y.

    Parameters:
    - x (numpy array): First input vector.
    - y (numpy array): Second input vector.
    - sigma (float): Bandwidth parameter. Default is 1.0.

    Returns:
    - float: The RBF kernel value.
    """
    distance_squared = np.linalg.norm(x - y) ** 2
    return np.exp(-distance_squared / (2 * sigma ** 2))
