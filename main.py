import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from kernels import rbf_kernel

# Step 1: Load the speed of light data (as deviations from 24,800 nanoseconds)
data = np.genfromtxt("data/newcomb.txt", delimiter=",", filling_values=np.nan)
data = data[~np.isnan(data)]  # Remove NaN values if any

# Step 2: Fit a normal distribution to the data
mu, std = norm.fit(data)

# Step 3: Sample from the fitted normal distribution
fitted_samples = np.random.normal(mu, std, size=100)

print(fitted_samples)

# Function to calculate MMD using an RBF kernel
def mmd(X, Y, sigma=1.0):
    m, n = len(X), len(Y)
    xx = np.sum([rbf_kernel(X[i], X[j], sigma) for i in range(m) for j in range(m)]) / (m * m)
    yy = np.sum([rbf_kernel(Y[i], Y[j], sigma) for i in range(n) for j in range(n)]) / (n * n)
    xy = np.sum([rbf_kernel(X[i], Y[j], sigma) for i in range(m) for j in range(n)]) / (m * n)
    mmd_stat = xx + yy - 2 * xy
    return mmd_stat


# Bootstrap test to compute p-value
def mmd_bootstrap_test(X, Y, num_bootstraps=100, sigma=1.0):
    observed_mmd = mmd(X, Y, sigma=sigma)

    np.random.seed(0)

    # Generate bootstrap samples
    mmd_values = []
    for i in range(num_bootstraps):
        print(i)
        X_bootstrap = np.random.choice(X, size=len(X), replace=True)
        Y_bootstrap = np.random.choice(Y, size=len(Y), replace=True)
        mmd_values.append(mmd(X_bootstrap, Y_bootstrap, sigma=sigma))

    # Calculate p-value as the proportion of bootstrap MMDs >= observed MMD
    p_value = np.mean(np.array(mmd_values) >= observed_mmd)

    return observed_mmd, p_value

# Calculate the observed MMD and p-value using the bootstrap method
mmd_stat, p_value = mmd_bootstrap_test(data, fitted_samples, num_bootstraps=100, sigma=1.0)

print('Observed MMD:', mmd_stat)
print('P-value:', p_value)

# Step 5: Plot histogram, density estimate, and witness function
plt.hist(data, bins=15, alpha=0.5, label="Data", density=True)
plt.plot(np.linspace(-60, 60, 100), norm.pdf(np.linspace(-60, 60, 100), mu, std), 'r-', lw=2, label="Fitted normal")

# Optional: Calculate and plot witness function to highlight discrepancies (e.g., using KDE)
plt.legend()
plt.title(f"MMD: {mmd_stat:.4f}, p-value: {p_value:.4f}")
plt.show()
