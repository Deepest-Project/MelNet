import numpy as np

from .constant import f_div, t_div


class TierUtil():
    def __init__(self, hp):
        self.hp = hp
        self.n_mels = hp.audio.n_mels

        self.f_div = f_div[hp.model.tier]
        self.t_div = t_div[hp.model.tier]

        # when we perform stft, the number of time frames we get is:
        # self.T = int(hp.audio.sr * hp.audio.duration) // hp.audio.hop_length + 1
        # 10*22050 // 256 + 1 = 862 (blizzard)
        # 6*22050 // 256 + 1 = 517 (maestro)
        # 6*16000 // 180 + 1 = 534 (voxceleb2)
        # 10*16000 // 180 + 1 = 889 (tedlium3)        

    def cut_divide_tiers(self, x):
        x = x[:, :, :-(x.size(-1) % self.t_div)]
        B, M, T = x.size()
        assert M % self.f_div == 0, \
            'freq(mel) dimension should be divisible by %d, got %d.' \
            % (self.f_div, M)
        assert T % self.t_div == 0, \
            'time dimension should be divisible by %d, got %d.' \
            % (self.t_div, T)

        tiers = list()
        tiers.append(x) # final objective
        for i in range(self.hp.model.tier - 1, 1, -1):
            if i % 2 == 0: # make consistent with utils/constant.py
                tiers.append(x[:, 1::2, :])
                x = x[:, ::2, :]
            else:
                tiers.append(x[:, :, 1::2])
                x = x[:, :, ::2]
        tiers.append(x) # tier 1 (last)
        tiers = tiers[::-1]
        tiers.append(None)
        return tiers # [None, Tier1, Tier2, Tier3, ..., TierN, original]
