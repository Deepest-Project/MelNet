import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GMMLoss(nn.Module):
    def __init__(self):
        super(GMMLoss, self).__init__()

    def forward(self, x, mu, std, pi):
        x = x.unsqueeze(-1)
        distrib = torch.exp(-((x - mu) / std) ** 2 / 2) / (std * np.sqrt(2 * np.pi))
        distrib = torch.sum(pi * distrib, dim=3)
<<<<<<< Updated upstream
        loss = -1.0 * torch.log(distrib + 1e-6) # NLL
        return torch.mean(loss)
=======
        loss = -torch.log(distrib).mean() # NLL
        return loss
>>>>>>> Stashed changes
