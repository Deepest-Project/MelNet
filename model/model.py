import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnn import DelayedRNN


class MelNet(nn.Module):
    def __init__(self, hp):
        super(MelNet, self).__init__()
        num_hidden = hp.model.hidden
        self.hp = hp

        # TODO: fix this hard-coded value
        self.freq = 32

        self.W_t_0 = nn.Linear(1, num_hidden)
        self.W_f_0 = nn.Linear(1, num_hidden)
        self.W_c_0 = nn.Linear(self.freq, num_hidden)

        self.layers = nn.ModuleList([
            DelayedRNN(num_hidden) for _ in range(hp.model.tier)
        ])

        # Gaussian Mixture Model: eq. (2)
        self.K = hp.model.gmm
        self.W_theta = nn.Linear(num_hidden, 3*self.K)
        self.pi_softmax = nn.Softmax(dim=3)

    def forward(self, x):
        h_t = self.W_t_0(x.unsqueeze(-1))
        h_f = self.W_f_0(x.unsqueeze(-1))
        h_c = self.W_c_0(x)

        for layer in self.layers:
            h_t, h_f, h_c = layer(h_t, h_f, h_c)

        theta_hat = self.W_theta(h_f)

        mu = theta_hat[:, :, :, :self.K] # eq. (3)
        std = torch.exp(theta_hat[:, :, :, self.K:2*self.K]) # eq. (4)
        pi = self.pi_softmax(theta_hat[:, :, :, 2*self.K:]) # eq. (5)

        return mu, std, pi
 
    def get_loss(self, x, mu, std, pi):
        distrib = (1.0 / np.sqrt(2*np.pi)) * \
            torch.exp(-0.5 * ((x - mu) / std) ** 2) / std
        distrib = torch.sum(pi * distrib, dim=3)
        loss = -1.0 * torch.log(distrib) # NLL
        return torch.mean(loss)
