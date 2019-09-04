import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GMMLoss(nn.Module):
    def __init__(self):
        super(GMMLoss, self).__init__()

    def forward(self, x, mu, std, pi):
        x = x.unsqueeze(-1)
        distrib = (1.0 / np.sqrt(2*np.pi)) * \
            torch.exp(-0.5 * ((x - mu) / std) ** 2) / std
        distrib = torch.sum(pi * distrib, dim=3)
        loss = -1.0 * torch.log(distrib) # NLL
        return torch.mean(loss)
