import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnn import DelayedRNN


class Tier(nn.Module):
    def __init__(self, hp, freq, layers, tierN):
        super(Tier, self).__init__()
        self.TTS = hp.TTS
        self.tierN = tierN

        self.W_t_0 = nn.Linear(1, hp.hidden_dim)
        self.W_f_0 = nn.Linear(1, hp.hidden_dim)
        self.W_c_0 = nn.Linear(freq, hp.hidden_dim)

        self.layers = nn.ModuleList([
            DelayedRNN(hp, tierN) for _ in range(layers)
        ])

        # Gaussian Mixture Model: eq. (2)
        self.K = hp.K
        self.pi_softmax = nn.Softmax(dim=3)

        # map output to produce GMM parameter eq. (10)
        self.W_theta = nn.Linear(hp.hidden_dim, 3*self.K)

        if self.TTS==True:
            self.TextEncoder = nn.Sequential(nn.Embedding(hp.n_voca, hp.hidden_dim),
                                             nn.LSTM(input_size=hp.num_hidden, hidden_size=self.num_hidden, batch_first=True))
        
        
    def forward(self, x, text=None):
        if text is None:
            memory = None
        elif:
            memory = self.TextEncoder(text)
            
        if self.tierN == 1:
            h_t = self.W_t_0(F.pad(x, [1, -1, 0, 0, 0, 0]).unsqueeze(-1))
            h_f = self.W_f_0(F.pad(x, [0, 0, 1, -1, 0, 0]).unsqueeze(-1))
            h_c = self.W_c_0(F.pad(x, [1, -1, 0, 0, 0, 0]).transpose(1, 2))
        else:
            h_t = self.W_t_0(x.unsqueeze(-1))
            h_f = self.W_f_0(x.unsqueeze(-1))
            h_c = self.W_c_0(x.transpose(1, 2))

        # h_t, h_f: [B, M, T, D] / h_c: [B, T, D]
        for layer in self.layers:
            h_t, h_f, h_c = layer(h_t, h_f, h_c, memory)

        theta_hat = self.W_theta(h_f)

        mu = torch.sigmoid(theta_hat[:, :, :, :self.K]) # eq. (3)
        std = torch.exp(theta_hat[:, :, :, self.K:2*self.K]) # eq. (4)
        pi = self.pi_softmax(theta_hat[:, :, :, 2*self.K:]) # eq. (5)

        return mu, std, pi
    
    
    
    
    def sample(self, x, text=None):
        # x: [1, M, T] / B=1, M=mel, T=time
        if self.tierN == 1:
            n_slice = x.size(-1)
            x_t, x_f = x.clone(), x.clone()
            
            for i in range(n_slice):
                h_t = self.W_t_0(x_t.unsqueeze(-1))
                h_f = self.W_f_0(x_f.unsqueeze(-1))
                h_c = self.W_c_0(x_t.transpose(1, 2))
                
                for layer in self.layers:
                    h_t, h_f, h_c = layer(h_t, h_f, h_c)
                    
                theta_hat = self.W_theta(h_f)

                mu = torch.sigmoid(theta_hat[:, :, :, :self.K]) # eq. (3)
                pi = self.pi_softmax(theta_hat[:, :, :, 2*self.K:]) # eq. (5)
                
                mu = torch.sum(mu*pi, dim=3)
                
                x_t[:,:,i+1] = mu[:,:,i]
                x_f[:,i+1,:] = mu[:,i,:]
        
            return mu
        
        else:
            return self.forward(x)
        
        
        
        
