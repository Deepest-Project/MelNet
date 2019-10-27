import torch
import torch.nn as nn
import torch.nn.functional as F


class GMMLoss(nn.Module):
    def __init__(self):
        super(GMMLoss, self).__init__()

    def forward(self, x, mu, std, pi):
        x = x.unsqueeze(-1)
        log_prob = torch.distributions.Normal(loc=mu, scale=std.exp()).log_prob(x)
        log_distrib = log_prob + F.log_softmax(pi, dim=-1)
        loss = -torch.logsumexp(log_distrib, dim=-1).mean()
        return loss
