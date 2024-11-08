from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_density_line(x_range, mu, std, color, label, linestyle='-'):
    """Plot a Gaussian density line over a specified range."""
    plt.plot(x_range, norm.pdf(x_range, mu, std), color=color, lw=2, linestyle=linestyle, label=label)

def plot_witness_function(x_range, w):
    """Plot the witness function over a specified range."""
    plt.plot(x_range, w, color='g', linestyle='--', lw=2, label="Witness function")

def plot_setup(x_label, y_label):
    """Set up plot labels and legend."""
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

def hist(x, mu, std, w):
    """Plot histogram of data, fitted normal distribution, and witness function."""
    x_range = np.linspace(-60, 60, 100)
    plt.hist(x, bins=15, alpha=0.5, label="Data", density=True)
    plot_density_line(x_range, mu, std, color='r', label="Fitted normal")
    plot_witness_function(x_range, w)
    plot_setup("Deviations from 24,800 nanoseconds", "Density / Witness function")
    plt.show()

def simple_plot(xs, ys, x_lab, y_lab):
    """Plot simple line graph with custom labels."""
    plt.plot(xs, ys, color='b', lw=2, label="p-values")
    plot_setup(x_lab, y_lab)
    plt.show()

def double_gauss(fmu, fstd, smu, sstd, w):
    """Plot two Gaussian distributions and witness function."""
    x_range = np.linspace(-2, 8, 100)
    plot_density_line(x_range, fmu, fstd, color='b', label="Data")
    plot_density_line(x_range, smu, sstd, color='r', label="Fitted normal")
    plot_witness_function(x_range, w)
    plot_setup("", "Density / Witness function")
    plt.show()

def outliers(fmu, fstd, smu, sstd, w, os):
    """Plot Gaussian distributions, witness function, and highlight outliers."""
    x_range = np.linspace(-2, 8, 100)
    plot_density_line(x_range, fmu, fstd, color='b', label="Data")
    plot_density_line(x_range, smu, sstd, color='r', label="Fitted normal")
    plot_witness_function(x_range, w)
    plt.scatter(os, [1 for _ in range(len(os))], s=20, color='purple', label="Outliers")
    plot_setup("", "Density / Witness function")
    plt.show()
