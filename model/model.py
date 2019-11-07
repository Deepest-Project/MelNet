import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .tier import Tier
from .loss import GMMLoss
from utils.constant import f_div, t_div
from utils.hparams import load_hparam_str
from utils.tierutil import TierUtil


class MelNet(nn.Module):
    def __init__(self, hp, args, infer_hp):
        super(MelNet, self).__init__()
        self.hp = hp
        self.args = args
        self.infer_hp = infer_hp
        self.f_div = f_div[hp.model.tier]
        self.t_div = t_div[hp.model.tier]
        self.n_mels = hp.audio.n_mels

        self.tierutil = TierUtil(hp)

        self.tiers = nn.ModuleList([None] +
            [Tier(hp=hp,
                freq=hp.audio.n_mels // self.f_div * f_div[tier],
                layers=hp.model.layers[tier-1],
                tierN=tier)
            for tier in range(1, hp.model.tier+1)])

    def forward(self, x, tier_num):
        assert tier_num > 0, 'tier_num should be larger than 0, got %d' % tier_num

        return self.tiers[tier_num](x)

    def unconditional_sample(self):
        zeros = torch.zeros(1, self.n_mels//self.f_div, self.args.timestep//self.t_div).cuda()

        x = self.tiers[1].sample(zeros)
        for tier in range(2, self.hp.model.tier+1):
            temp = self.tiers[tier].sample(x)
            x = self.tierutil.interleave(x, temp, tier+1)
            del temp

        return x

    def load_tiers(self):
        for idx, chkpt_path in enumerate(self.infer_hp.checkpoints):
            checkpoint = torch.load(chkpt_path)
            hp = load_hparam_str(checkpoint['hp_str'])

            if self.hp != hp:
                print('Warning: hp different in file %s' % chkpt_path)
            
            checkpoint['model'] = OrderedDict({name[7:]: value for name, value in checkpoint['model'].items()})
            self.tiers[idx+1].load_state_dict(checkpoint['model'])

            del checkpoint

