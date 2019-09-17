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
        self.f_div = f_div[hp.n_tiers+1]
        self.t_div = t_div[hp.n_tiers+1]

        self.tiers = nn.ModuleList([None] +
                                   [Tier(hp=hp,
                                         freq=hp.n_mels // self.f_div * f_div[tier],
                                         layers=hp.layers[tier-1],
                                         tierN=tier)
                                    for tier in range(1, hp.n_tiers+1)])

    def forward(self, x, tier_num, text=None):
        assert tier_num > 0, 'tier_num should be larger than 0, got %d' % tier_num

        mu, std, pi = self.tiers[tier_num](x, text)
        return mu, std, pi

    
       
    def interleave(self, tier_a, tier_b, axis):
        assert axis in ['time', 'freq'], "Axis shoud be either time or frequncy"
        
        if axis=='time':
            output = tier_a.new_zeros(tier_a.size(0), tier_a.size(1), 2*tier_a.size(2))
            for i in range(tier_a.size(2)):
                output[:,:,2*i] = tier_a[:,:,i]
                output[:,:,2*i+1] = tier_b[:,:,i]
        
        elif axis=='freq':
            output = tier_a.new_zeros(tier_a.size(0), 2*tier_a.size(1), tier_a.size(2))
            for i in range(tier_a.size(1)):
                output[:,2*i,:] = tier_a[:,i,:]
                output[:,2*i+1,::] = tier_b[:,i,:]
                
            return output
        
        
        
    def sample(self, n_sec, text=None, device=None):
        #autoregressively compute tier1
        n_slice = n_sec * self.hp.sr 
        x = torch.zeros(1, self.hp.n_mels//self.f_div, n_slice//self.t_div)
        
        if device is not None:
            x = x.to(device)
        
        tier1 = self.tiers[1].sample(x, text)
        tier2 = self.tiers[2].sample(tier1)
        tier1_2 = self.interleave(tier1, tier2, axis='freq')
        
        tier3 = self.tiers[3].sample( tier1_2 )
        tier1_2_3 = self.interleave(tier1_2, tier3, axis='time')
        
        tier4 = self.tiers[4].sample( tier1_2_3 )
        tier1_2_3_4 = self.interleave(tier1_2_3, tier4, axis='freq')
        
        tier5 = self.tiers[5].sample( tier1_2_3_4 )
        tier1_2_3_4_5 = self.interleave(tier1_2_3_4, tier5, axis='time')
        
        tier6 = self.tiers[6].sample( tier1_2_3_4_5 )
        
        output = self.interleave(tier1_2_3_4_5, tier6, axis='freq')
            
        return output
