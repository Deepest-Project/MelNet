import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class GMMLoss(nn.Module):
    def __init__(self):
        super(GMMLoss, self).__init__()

    def forward(self, x, mu, std, pi):
        log_probs = Normal(mu, std + 1e-9).log_prob(x.unsqueeze(-1))
        log_probs = torch.logsumexp(log_probs + pi, -1)
        return -log_probs.mean()