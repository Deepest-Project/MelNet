import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tier import Tier
from .loss import GMMLoss
from utils.constant import f_div, t_div


class MelNet(nn.Module):
    def __init__(self, hp):
        super(MelNet, self).__init__()
        self.hp = hp
        self.f_div = f_div[hp.model.tier]
        self.t_div = t_div[hp.model.tier]

        self.tiers = nn.ModuleList([None] +
            [Tier(hp=hp,
                freq=hp.audio.n_mels // self.f_div * f_div[tier],
                layers=hp.model.layers[tier-1],
                first=(tier==1))
            for tier in range(1, hp.model.tier+1)])

    def forward(self, x, tier_num):
        assert tier_num > 0, 'tier_num should be larger than 0, got %d' % tier_num

        mu, std, pi = self.tiers[tier_num](x)
        return mu, std, pi

    def sample(self):
        raise NotImplementedError
