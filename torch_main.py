import torch
from matplotlib import pyplot as plt
import tqdm
import numpy as np
from scipy.stats import norm

np.random.seed(0)
torch.manual_seed(0)

data = np.genfromtxt("data/newcomb.txt", delimiter=",", filling_values=np.nan)
data = data[~np.isnan(data)]  # Remove NaN values if any

# Step 2: Fit a normal distribution to the data
mu, std = norm.fit(data[data > 0])
fitted_samples = np.random.normal(mu, std, size=1000)

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

def mmd(x, y, sigma):
    # compare kernel MMD paper and code:
    # A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    # x shape [n, d] y shape [m, d]
    # n_perm number of bootstrap permutations to get p-value, pass none to not get p-value
    n, d = x.shape
    m, d2 = y.shape
    assert d == d2
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    # note that sigma should be squared in the RBF to match the Gretton et al heuristic
    k = torch.exp((-1 / (2 * sigma ** 2)) * dists ** 2) + torch.eye(n + m) * 1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd


dists = torch.pdist(torch.cat([x, y], dim=0)[:, None])
sigma = dists[:100].median() / 2
our_mmd = mmd(x[:, None], y[:, None], sigma)

N_X = len(x)
N_Y = len(y)
xy = torch.cat([x, y], dim=0)[:, None].double()

mmds = []
for i in tqdm.tqdm(range(1000)):
    xy = xy[torch.randperm(len(xy))]
    mmds.append(mmd(xy[:N_X], xy[N_X:], sigma).item())
mmds = torch.tensor(mmds)

print((our_mmd < mmds).float().mean())

plt.hist(data, bins=15, alpha=0.5, label="Data", density=True)
plt.plot(np.linspace(-60, 60, 100), norm.pdf(np.linspace(-60, 60, 100), mu, std), 'r-', lw=2, label="Fitted normal")
plt.show()
