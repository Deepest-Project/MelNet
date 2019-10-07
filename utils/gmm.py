import torch

def get_pi_indices(pi):
    cumsum = torch.cumsum(pi, dim=3)
    rand = torch.rand(pi.shape[:-1] + (1,))
    indices = pi.shape[-1] - (cumsum < rand).sum(dim=3)
    return indices

def sample_gmm(mu, std, pi):
    indices = get_pi_indices(pi)
    mu = mu[indices]
    std = std[indices]
    return torch.normal(mu, std)