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

    def forward(self, x, tier_num):
        assert tier_num > 0, 'tier_num should be larger than 0, got %d' % tier_num

        mu, std, pi = self.tiers[tier_num](x)
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
        
    def sample(self, n_sec, device=None):
        #autoregressively compute tier1
        n_slice = n_sec * self.hp.sr 
        x = torch.zeros(1, self.hp.n_mels//self.f_div, n_slice//self.t_div)
        
        if device is not None:
            x = x.to(device)
        
        tier1 = self.tiers[1].sample(x)
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


        










'''
import torch
import torch.nn as nn
from math import ceil
from module import Stack, UpperStack

class MelNet(nn.Module):
    def __init__(self, hp):
        super(MelNet, self).__init__()
        self.first_stack = Stack(hp, use_central=True)
        self.rest_stack = nn.ModuleList([Stack(hp, use_central=False) for _ in range(3)])
        self.upper_stack = nn.ModuleList([UpperStack(hp) for _ in range(3)])
            
        #self.pred_layer = nn.Sequential( nn.Linear(hp.hidden_dim, 1), nn.Sigmoid() )
        self.pred_layer = nn.Linear(hp.hidden_dim, 1)
        
        self.time_expand = nn.Linear(1, hp.hidden_dim)
        self.freq_expand = nn.Linear(1, hp.hidden_dim)
        
        self.use_central = hp.use_central
        if self.use_central==True:
            self.central_expand = nn.Linear( int(hp.n_mels*(0.5)**(ceil((hp.n_tiers-1)/2))) , hp.hidden_dim)
        
    def Tier1(self, x):
        B,F,T = x.size()
        GO_time, GO_freq = x.new_zeros(B,F,1), x.new_zeros(B,1,T)
        x_t, x_f = torch.cat([GO_time, x[:,:,1:]], dim=2).unsqueeze(-1), torch.cat([GO_freq, x[:,1:,:]], dim=1).unsqueeze(-1)
        x_t = self.time_expand(x_t)
        x_f = self.freq_expand(x_f)
        
        if self.use_central==True:
            x_c = self.central_expand(x.transpose(1,2))
        else:
            x_c = None
        
        time_out, freq_out = self.first_stack( x_t, x_f, x_c )
        for stack in self.rest_stack:
            time_out, freq_out = stack(time_out, freq_out)
            
        out = self.pred_layer(freq_out).squeeze(-1)
        
        return out
    
    def not_Tier1(self, x):
        raise NotImplementedError
    
    def interleave(self, tier_n, tier_m):
        raise NotImplementedError
    
    def forward(self, x):
        tier1 = self.Tier1(x)
        tier2 = self.not_Tier1(tier1)

        tier3 = self.not_Tier1(self.interleave(tier1, tier2), target=3)
        tier4 = self.not_Tier1(self.interleave(tier2, tier3), target=4)
        tier5 = self.not_Tier1(self.interleave(tier3, tier4), target=5)
        tier6 = self.not_Tier1(self.interleave(tier4, tier5), target=6)
        return tier1, tier2, tier3, tier4, tier5, tier6
    
    def infer(self):
        raise NotImplementedError
'''