import torch
from matplotlib import pyplot as plt
import tqdm
import numpy as np
from scipy.stats import norm

# 1. Radial Basis Function (RBF) Kernel
def rbf_kernel(x, y, sigma=1.0, linear=False):
    if linear:
        return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))
    return torch.exp(-torch.cdist(x, y, p=2) ** 2 / (2 * sigma ** 2))


# 2. Polynomial Kernel
def polynomial_kernel(x, y, degree=3, c=1):
    return (torch.dot(x, y) + c) ** degree


# 3. Linear Kernel
def linear_kernel(x, y):
    return (x @ y.T).sum()


# 4. Sigmoid Kernel
def sigmoid_kernel(x, y, alpha=0.01, c=1):
    return torch.tanh(alpha * torch.dot(x, y) + c)


# 5. Laplacian Kernel
def laplacian_kernel(x, y, sigma=1.0, linear=False):
    if linear:
        return torch.exp(-torch.norm(x - y, p=1) / sigma)
    return torch.exp(-torch.cdist(x, y, p=1) / sigma)


# 6. Exponential Kernel
def exponential_kernel(x, y, sigma=1.0, linear=False):
    if linear:
        return torch.exp(-torch.norm(x - y, p=1) ** 2 / (2 * sigma ** 2))
    return torch.exp(-torch.cdist(x, y, p=1) / (2 * sigma ** 2))


# 7. Cosine Similarity Kernel
def cosine_similarity_kernel(x, y):
    return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))


def mmd(x, y, sigma):
    # compare kernel MMD paper and code:
    # A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    # x shape [n, d] y shape [m, d]
    # n_perm number of bootstrap permutations to get p-value, pass none to not get p-value
    n, d = x.shape
    m, d2 = y.shape
    assert d == d2
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    # note that sigma should be squared in the RBF to match the Gretton et al heuristic
    k_x = kernel(x, x, sigma)
    k_y = kernel(y, y, sigma)
    k_xy = kernel(x, y, sigma)
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    mmd = k_x.sum() / (n * n) + k_y.sum() / (m * m) - 2 * k_xy.sum() / (n * m)
    return mmd

np.random.seed(0)
torch.manual_seed(0)

# data = np.genfromtxt("data/newcomb.txt", delimiter=",", filling_values=np.nan)
# data = data[~np.isnan(data)]  # Remove NaN values if any
#
# # Step 2: Fit a normal distribution to the data
# mu, std = norm.fit(data[data > 0])

ps = []

mu, std = 2, 1
for n in tqdm.tqdm(range(4, 100)):
    data = np.random.normal(mu, std, size=n)
    fitted_samples = np.random.normal(mu + 1, std, size=n)

    dist_x = torch.from_numpy(data)
    dist_y = torch.from_numpy(fitted_samples)
    x = dist_x
    y = dist_y


    # z = torch.linspace(-0.5, 1.5, 100)
    # pyplot.scatter(x, torch.zeros_like(x), marker='+')
    # raw_density_x = dist_x.log_prob(z).exp()
    # raw_density_y = dist_y.log_prob(z).exp()
    # density_x = torch.where(raw_density_x.isnan(), torch.tensor(0.0), raw_density_x)
    # density_y = torch.where(raw_density_y.isnan(), torch.tensor(0.0), raw_density_y)
    # pyplot.plot(z, density_x)
    # pyplot.plot(z, density_y)

    # sigma = 0.1
    # raw_hat_p_x = torch.exp(-((x[None] - z[:, None])**2)/(2*sigma**2)).sum(1)
    # hat_p_x = (raw_hat_p_x / raw_hat_p_x.sum() / (z[1]-z[0]))
    # pyplot.plot(z, hat_p_x)
    # pyplot.plot(z, density_x)

    kernel = laplacian_kernel

    dists = torch.pdist(torch.cat([x, y], dim=0)[:, None])
    sigma = dists.median() / 2

    # print(sigma) # 2.5

    our_mmd = mmd(x[:, None], y[:, None], sigma)

    N_X = len(x)
    N_Y = len(y)
    xy = torch.cat([x, y], dim=0)[:, None].double()

    mmds = []
    for i in range(100):
        xy = xy[torch.randperm(len(xy))]
        mmds.append(mmd(xy[:N_X], xy[N_X:], sigma).item())
    mmds = torch.tensor(mmds)

    p = (our_mmd < mmds).float().mean()

    # print(p)
    ps.append(p)

# # Define the witness function
# def witness_function(z_values, x, y, sigma):
#     # Compute witness function over a range of points
#     witness_values = []
#     for z in tqdm.tqdm(z_values):
#         k_x_X = torch.mean(torch.tensor([rbf_kernel(z, x_i, sigma, linear=True) for x_i in x]))
#         k_x_Y = torch.mean(torch.tensor([rbf_kernel(z, y_j, sigma, linear=True) for y_j in y]))
#         witness_values.append((k_x_X - k_x_Y).item())
#     return np.array(witness_values)
#
#
# # Generate points on which to evaluate the witness function
# z_values = torch.linspace(-60, 60, 100)
#
# # Calculate witness function values
# witness_values = torch.tensor(witness_function(z_values, y, x, sigma))
#
# # Plot the witness function along with the data histogram and fitted distribution
# plt.hist(data, bins=15, alpha=0.5, label="Data", density=True)
# plt.plot(np.linspace(-60, 60, 100), norm.pdf(np.linspace(-60, 60, 100), mu, std), 'r-', lw=2, label="Fitted normal")
# plt.plot(z_values.numpy(), witness_values.numpy(), 'g--', lw=2, label="Witness function")
#
# plt.xlabel("Deviations from 24,800 nanoseconds")
# plt.ylabel("Density / Witness function")
# plt.legend()
# plt.show()

plt.plot(list(range(4, 100)), ps, lw=2, label="p-values")

plt.xlabel("sample size")
plt.ylabel("p-value")
plt.legend()
plt.show()
