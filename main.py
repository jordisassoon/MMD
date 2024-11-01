import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from kernels import rbf_kernel

# Step 1: Load the speed of light data (as deviations from 24,800 nanoseconds)
data = np.genfromtxt("data/newcomb.txt", delimiter=",", filling_values=np.nan)

# Step 2: Fit a normal distribution to the data
mu, std = norm.fit(data)

# Step 3: Sample from the fitted normal distribution
fitted_samples = np.random.normal(mu, std, size=1000)


def mmd(X, Y, sigma=1.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two samples X and Y using an RBF kernel.

    Parameters:
    - X (numpy array): Sample set from distribution P (shape: [m, d]).
    - Y (numpy array): Sample set from distribution Q (shape: [n, d]).
    - sigma (float): Bandwidth parameter for the RBF kernel.

    Returns:
    - float: The MMD statistic.
    """
    m, n = len(X), len(Y)

    # Compute all kernel values for X vs X, Y vs Y, and X vs Y
    xx = np.sum([rbf_kernel(X[i], X[j], sigma) for i in range(m) for j in range(m)]) / (m * m)
    yy = np.sum([rbf_kernel(Y[i], Y[j], sigma) for i in range(n) for j in range(n)]) / (n * n)
    xy = np.sum([rbf_kernel(X[i], Y[j], sigma) for i in range(m) for j in range(n)]) / (m * n)

    # Calculate MMD^2
    mmd_stat = xx + yy - 2 * xy
    return np.sqrt(mmd_stat)  # Optional: return sqrt(MMD^2) for interpretability


mmd_stat = mmd(data, fitted_samples, sigma=1.0)

# Step 5: Plot histogram, density estimate, and witness function
plt.hist(data, bins=15, alpha=0.5, label="Data", density=True)
plt.plot(np.linspace(-60, 60, 100), norm.pdf(np.linspace(-60, 60, 100), mu, std), 'r-', lw=2, label="Fitted normal")

# Optional: Calculate and plot witness function to highlight discrepancies (e.g., using KDE)
plt.legend()
plt.show()

print('MMD: ', mmd_stat)
