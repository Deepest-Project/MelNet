import torch
import numpy as np

def get_pi_indices(pi):
    cumsum = torch.cumsum(pi.cpu(), dim=-1)
    rand = torch.rand(pi.shape[:-1] + (1,))
    indices = (cumsum < rand).sum(dim=-1)
    return indices.flatten().detach().numpy()

def sample_gmm(mu, std, pi):
    std = std.exp()
    pi = pi.softmax(dim=-1)
    indices = get_pi_indices(pi)
    mu = mu.reshape(-1, mu.shape[-1])
    mu = mu[np.arange(mu.shape[0]), indices].reshape(std.shape[:-1])
    std = std.reshape(-1, std.shape[-1])
    std = std[np.arange(std.shape[0]), indices].reshape(mu.shape)
    return torch.normal(mu, std).reshape_as(mu).clamp(0.0, 1.0).to(mu.device)
