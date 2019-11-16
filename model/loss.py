import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class GMMLoss(nn.Module):
    def __init__(self):
        super(GMMLoss, self).__init__()

    def forward(self, x, mu, std, pi, audio_lengths):
        x = nn.utils.rnn.pack_padded_sequence(
            x.unsqueeze(-1).transpose(1, 2),
            audio_lengths,
            batch_first=True,
            enforce_sorted=False
        ).data
        mu = nn.utils.rnn.pack_padded_sequence(
            mu.transpose(1, 2),
            audio_lengths,
            batch_first=True,
            enforce_sorted=False
        ).data
        std = nn.utils.rnn.pack_padded_sequence(
            std.transpose(1, 2),
            audio_lengths,
            batch_first=True,
            enforce_sorted=False
        ).data
        pi = nn.utils.rnn.pack_padded_sequence(
            pi.transpose(1, 2),
            audio_lengths,
            batch_first=True,
            enforce_sorted=False
        ).data
        log_prob = Normal(loc=mu, scale=std.exp()).log_prob(x)
        log_distrib = log_prob + F.log_softmax(pi, dim=-1)
        loss = -torch.logsumexp(log_distrib, dim=-1).mean()
        return loss
