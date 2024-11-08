import torch
import random
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
from pprint import pprint
import argparse

import plotter
from kernels import rbf_kernel, laplacian_kernel, exponential_kernel


def set_seed():
    """Set the random seed for reproducibility."""
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def load_newcomb(ignore_outliers=False):
    """Load Newcomb's speed of light data and optionally ignore outliers.

    Args:
        ignore_outliers (bool): Whether to ignore outliers.

    Returns:
        tuple: Data, mean, and standard deviation.
    """
    # Load and clean data
    data = np.genfromtxt("data/newcomb.txt", delimiter=",", filling_values=np.nan)
    data = data[~np.isnan(data)]  # Remove NaNs

    # Fit a normal distribution and optionally ignore outliers
    if ignore_outliers:
        mu, std = norm.fit(data[data > 0])  # Fit only positive values if ignoring outliers
    else:
        mu, std = norm.fit(data)
    return data, mu, std


def mmd(x, y, sigma, kernel):
    """Compute the Maximum Mean Discrepancy (MMD) between two distributions.

    Args:
        x, y (Tensor): Data tensors.
        sigma (float): Bandwidth for the kernel.
        kernel (function): Kernel function to use.

    Returns:
        float: MMD statistic.
    """
    n, d = x.shape
    m, d2 = y.shape
    assert d == d2, "Dimension mismatch between x and y."

    # Compute kernel matrices
    k_x = kernel(x, x, sigma)
    k_y = kernel(y, y, sigma)
    k_xy = kernel(x, y, sigma)

    return k_x.sum() / (n * n) + k_y.sum() / (m * m) - 2 * k_xy.sum() / (n * m)


def compute_p(data, mu, std, sample_size, kernel, fitted_samples=None, bootstrap_size=1000, sigma=None,
              pretty_print=True, n_outliers=0):
    """Compute the p-value for the MMD test.

    Args:
        data: Original data.
        mu, std (float): Mean and standard deviation for sample generation.
        sample_size (int): Number of samples to generate.
        kernel (function): Kernel function to use.
        fitted_samples (array, optional): Precomputed samples for comparison.
        bootstrap_size (int): Number of bootstrap iterations.
        sigma (float, optional): Bandwidth for the kernel.
        pretty_print (bool): If True, pretty-print parameters.
        n_outliers (int): Number of outliers to add.

    Returns:
        float: p-value from MMD test.
    """
    # Print parameters if pretty print is enabled
    if pretty_print:
        locket = {
            'mu': mu,
            'std': std,
            'sample_size': sample_size,
            'kernel': kernel,
            'bootstrap_size': bootstrap_size
        }
        print("Test Parameters:")
        pprint(locket)

    # Generate normal samples if none are provided
    if fitted_samples is None:
        fitted_samples = np.random.normal(mu, std, size=sample_size)

    # Convert data to torch tensors
    x = torch.from_numpy(data)
    y = torch.from_numpy(fitted_samples)

    # Add outliers if specified
    if n_outliers > 0:
        outlier = torch.tensor(np.random.normal(mu + 4 * std, std, n_outliers))
        y = torch.cat([y, outlier])

    # Compute median distance as sigma if not provided
    dists = torch.pdist(torch.cat([x, y], dim=0)[:, None])
    if sigma is None:
        sigma = dists.median() / 2

    # Calculate MMD between original and generated samples
    our_mmd = mmd(x[:, None], y[:, None], sigma, kernel)

    # Concatenate and initialize bootstrap MMD list
    N_X = len(x)
    xy = torch.cat([x, y], dim=0)[:, None].double()
    mmds = []

    # Bootstrap MMD calculations with progress bar
    for _ in tqdm(range(bootstrap_size), desc="Bootstrap iteration", disable=not pretty_print):
        xy = xy[torch.randperm(len(xy))]  # Shuffle data
        mmds.append(mmd(xy[:N_X], xy[N_X:], sigma, kernel))

    mmds = torch.tensor(mmds)
    p = (our_mmd < mmds).float().mean()  # Calculate p-value
    return x, y, sigma, p.item()


def witness_function(z_values, x, y, sigma, kernel):
    """Compute the witness function over a range of points.

    Args:
        z_values (Tensor): Range of points for witness function.
        x, y (array): Data tensors.
        sigma (float): Bandwidth for kernel.
        kernel (function): Kernel function.

    Returns:
        array: Computed witness values.
    """
    witness_values = []
    for z in tqdm(z_values, desc="Computing witness function"):
        # Calculate witness value at point z
        k_x_X = torch.mean(torch.tensor([kernel(z, x_i, sigma, linear=True) for x_i in x]))
        k_x_Y = torch.mean(torch.tensor([kernel(z, y_j, sigma, linear=True) for y_j in y]))
        witness_values.append((k_x_X - k_x_Y).item())
    return np.array(witness_values)


def main(args):
    set_seed()  # Set random seed for reproducibility

    # Select the kernel function based on user input
    kernel = {'rbf': rbf_kernel, 'lap': laplacian_kernel, 'exp': exponential_kernel}.get(args.kernel, laplacian_kernel)

    if args.test == 'newcomb':
        print("Running Newcomb's speed of light experiment...")
        data, mu, std = load_newcomb(ignore_outliers=args.ignore_outliers)
        print("Data loaded and parameters fitted.")

        # Run the MMD test and display p-value
        x, y, sigma, p_value = compute_p(data, mu, std, sample_size=1000, kernel=kernel)
        print(f"The p-value for Newcomb's test is: {p_value}")

        # Compute witness function and plot
        z_values = torch.linspace(-60, 60, 100)
        w = witness_function(z_values, y, x, sigma, kernel)
        plotter.hist(x, mu, std, w)

    elif args.test == 'sigma':
        print("Exploring different sigma values for kernel...")
        data, mu, std = load_newcomb(ignore_outliers=args.ignore_outliers)
        sigmas = np.linspace(0.1, 10, 25)
        p_values = []
        x, y, sigma = None, None, None
        for sigma in tqdm(sigmas):
            x, y, z, p_value = compute_p(data, mu, std, sample_size=1000, bootstrap_size=100, kernel=kernel,
                                         sigma=sigma,
                                         pretty_print=False)
            p_values.append(p_value)

        # Compute witness function and plot p-values for sigma
        z_values = torch.linspace(-60, 60, 100)
        w = witness_function(z_values, y, x, sigma, kernel)
        plotter.hist(x, mu, std, w)
        plotter.simple_plot(sigmas, p_values, "sigma", "p-value")

    elif args.test == 'sample_size':
        print("Running experiment on varying sample sizes...")
        mu, std = 2, 1
        x, y, sigma = None, None, None
        p_values = []
        sizes = list(range(4, 100))
        for size in tqdm(sizes, desc="Sample size iteration"):
            data = np.random.normal(mu, std, size=size)
            x, y, sigma, p_value = compute_p(data, mu + 1, std, sample_size=size, bootstrap_size=100, kernel=kernel,
                                             pretty_print=False)
            p_values.append(p_value)

        # Compute witness function and plot for varying sample sizes
        z_values = torch.linspace(-2, 8, 100)
        w = witness_function(z_values, y, x, sigma, kernel)
        plotter.double_gauss(mu, std, mu + 1, std, w)
        plotter.simple_plot(sizes, p_values, "sample size", "p-value")

    elif args.test == 'outliers':
        print("Testing effect of outliers...")
        mu, std = 2, 1
        x, y, sigma = None, None, None
        n = None
        p_values = []
        n_values = list(range(0, 26))
        for n in tqdm(n_values, desc="Outlier iteration"):
            data = np.random.normal(mu, std, size=100)
            x, y, sigma, p_value = compute_p(data, mu, std, fitted_samples=data, sample_size=100 - n,
                                             bootstrap_size=100,
                                             kernel=kernel, pretty_print=False, n_outliers=n)
            p_values.append(p_value)

        # Compute witness function and plot with outliers
        z_values = torch.linspace(-2, 8, 100)
        w = witness_function(z_values, y, x, sigma, kernel)
        outliers = y[-n:]
        plotter.outliers(mu, std, mu, std, w, outliers)
        plotter.simple_plot(n_values, p_values, "n of outliers", "p-value")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run statistical experiments.")
    parser.add_argument('--test', type=str, required=True,
                        help="Specify the test to run. Options: 'newcomb', 'sigma', 'sample_size', 'outliers'")
    parser.add_argument('--ignore_outliers', type=bool, help="Ignore outliers in the data", default=True)
    parser.add_argument('--kernel', type=str, default='lap', help="Kernel type: 'rbf', 'lap', 'exp'")
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    main(_args)
