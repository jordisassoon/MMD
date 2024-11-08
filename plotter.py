from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm

def hist(x, mu, std, w):
    plt.hist(x, bins=15, alpha=0.5, label="Data", density=True)
    plt.plot(np.linspace(-60, 60, 100), norm.pdf(np.linspace(-60, 60, 100), mu, std), 'r-', lw=2, label="Fitted normal")
    plt.plot(np.linspace(-60, 60, 100), w, 'g--', lw=2, label="Witness function")

    plt.xlabel("Deviations from 24,800 nanoseconds")
    plt.ylabel("Density / Witness function")
    plt.legend()
    plt.show()

def sigma_plot(sigmas, ps):
    plt.plot(sigmas, ps, lw=2, label="p-values")
    plt.xlabel("sigma")
    plt.ylabel("p-value")
    plt.legend()
    plt.show()

def double_gauss(fmu, fstd, smu, sstd, w):
    plt.plot(np.linspace(-2, 8, 100), norm.pdf(np.linspace(-2, 8, 100), fmu, fstd), 'b-', lw=2, label="Data")
    plt.plot(np.linspace(-2, 8, 100), norm.pdf(np.linspace(-2, 8, 100), smu, sstd), 'r-', lw=2, label="Fitted normal")
    plt.plot(np.linspace(-2, 8, 100), w, 'g--', lw=2, label="Witness function")

def outliers(fmu, fstd, smu, sstd, w, os):
    plt.plot(np.linspace(-2, 8, 100), norm.pdf(np.linspace(-2, 8, 100), fmu, fstd), 'b-', lw=2, label="Data")
    plt.plot(np.linspace(-2, 8, 100), norm.pdf(np.linspace(-2, 8, 100), smu, sstd), 'r-', lw=2, label="Fitted normal")
    plt.plot(np.linspace(-2, 8, 100), w, 'g--', lw=2, label="Witness function")
    plt.scatter(os, [1 for _ in range(len(os))], s=20, color='purple', label="Outliers")
