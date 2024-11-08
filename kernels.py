import torch


# 1. Radial Basis Function (RBF) Kernel
def rbf_kernel(x, y, sigma=1.0, linear=False):
    if linear:
        return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))
    return torch.exp(-torch.cdist(x, y, p=2) ** 2 / (2 * sigma ** 2))


# 2. Laplacian Kernel
def laplacian_kernel(x, y, sigma=1.0, linear=False):
    if linear:
        return torch.exp(-torch.norm(x - y, p=1) / sigma)
    return torch.exp(-torch.cdist(x, y, p=1) / sigma)


# 3. Exponential Kernel
def exponential_kernel(x, y, sigma=1.0, linear=False):
    if linear:
        return torch.exp(-torch.norm(x - y, p=1) ** 2 / (2 * sigma ** 2))
    return torch.exp(-torch.cdist(x, y, p=1) / (2 * sigma ** 2))
