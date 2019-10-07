import torch

def get_pi_indices(pi):
    cumsum = torch.cumsum(pi.cpu(), dim=3)
    rand = torch.rand(pi.shape[:-1] + (1,))
    indices = pi.shape[-1] - (cumsum < rand).sum(dim=3)
    return indices.flatten().detach().numpy()

def sample_gmm(mu, std, pi):
    indices = get_pi_indices(pi)
    mu = mu.reshape(-1, mu.shape[-1])[indices].reshape_as(std)
    std = std.reshape(-1, std.shape[-1])[indices].reshape_as(mu)
    return torch.normal(mu, std).reshape_as(mu)