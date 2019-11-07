import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.set_printoptions(precision=1)

from .rnn import DelayedRNN
from text import symbols


class Attention(nn.Module):
    def __init__(self, hp):
        super(Attention, self).__init__()
        self.M = hp.model.gmm
        self.rnn_cell = nn.LSTMCell(input_size=2*hp.model.hidden, hidden_size=hp.model.hidden)
        self.W_g = nn.Linear(hp.model.hidden, 3*self.M)
        
    def attention(self, h_i, memory, ksi):
        phi_hat = self.W_g(h_i)
        ksi = ksi+torch.exp(phi_hat[:, :self.M])
        ksi.clamp_(max=memory.size(1)-1)
        
        beta = torch.exp( phi_hat[:, self.M:2*self.M] )
        alpha = F.softmax(phi_hat[:, 2*self.M:3*self.M], dim=-1)
        
        u = memory.new_tensor( range(memory.size(1)), dtype=torch.float )
        u_R = u + 0.5
        u_L = u - 0.5
        
        term1 = torch.sum(alpha.unsqueeze(-1)*
                          torch.reciprocal(1 + torch.exp((ksi.unsqueeze(-1) - u_R)
                                                         / beta.unsqueeze(-1))), dim=1)
        
        term2 = torch.sum(alpha.unsqueeze(-1)*
                          torch.reciprocal(1 + torch.exp((ksi.unsqueeze(-1) - u_L)
                                                         / beta.unsqueeze(-1))), dim=1)
        
        weights = (term1-term2).unsqueeze(1)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        context = torch.bmm(weights, memory)
        
        termination = 1 - torch.sum(alpha.unsqueeze(-1)*
                                    torch.reciprocal(1 + torch.exp((ksi.unsqueeze(-1) - u_R)
                                                                   / beta.unsqueeze(-1))),
                                    dim=1)

        return context, weights, termination, ksi # (B, 1, D), (B, 1, T), (B, T)

    
    
    def forward(self, input_h_c, memory, input_lengths):
        B, T, D = input_h_c.size()
        
        context = input_h_c.new_zeros(B, D)
        h_i, c_i  = input_h_c.new_zeros(B, D), input_h_c.new_zeros(B, D)
        ksi = input_h_c.new_zeros(B, self.M)
        
        contexts, weights = [], []
        for i in range(T):
            x = torch.cat([input_h_c[:, i], context.squeeze(1)], dim=-1)
            h_i, c_i = self.rnn_cell(x, (h_i, c_i))
            context, weight, termination, ksi = self.attention(h_i, memory, ksi)
            contexts.append(context)
            weights.append(weight)
            
            
        contexts = torch.cat(contexts, dim=1)
        alignment = torch.cat(weights, dim=1)
        termination = torch.gather(termination, 1, (input_lengths-1).unsqueeze(-1)) # 4

        return context, alignment, termination



class TTS(nn.Module):
    def __init__(self, hp, freq, layers, tierN):
        super(TTS, self).__init__()
        self.hp = hp
        assert tierN==1, 'TTS tier must be 1'
        self.tierN = tierN

        self.W_t_0 = nn.Linear(1, hp.model.hidden)
        self.W_f_0 = nn.Linear(1, hp.model.hidden)
        self.W_c_0 = nn.Linear(freq, hp.model.hidden)
        
        self.layers = nn.ModuleList([ DelayedRNN(hp) for _ in range(layers) ])

        # Gaussian Mixture Model: eq. (2)
        self.K = hp.model.gmm
        self.pi_softmax = nn.LogSoftmax(dim=3)

        # map output to produce GMM parameter eq. (10)
        self.W_theta = nn.Linear(hp.model.hidden, 3*self.K)
        
        self.TextEncoder = nn.Sequential(nn.Embedding(len(symbols), hp.model.hidden),
                                         nn.LSTM(input_size=hp.model.hidden,
                                                 hidden_size=hp.model.hidden//2, 
                                                 batch_first=True,
                                                 bidirectional=True)
                                        )
        
        self.attention = Attention(hp)

        
    def forward(self, x, text, input_lengths, output_lengths):
        # Extract memory
        memory, _ = self.TextEncoder(text)
        
        # x: [B, M, T] / B=batch, M=mel, T=time
        h_t = self.W_t_0(F.pad(x, [1, -1]).unsqueeze(-1))
        h_f = self.W_f_0(F.pad(x, [0, 0, 1, -1]).unsqueeze(-1))
        h_c = self.W_c_0(F.pad(x, [1, -1]).transpose(1, 2))
        
        # h_t, h_f: [B, M, T, D] / h_c: [B, T, D]
        for i, layer in enumerate(self.layers):
            if i!=(len(self.layers)//2):
                h_t, h_f, h_c = layer(h_t, h_f, h_c)
                
            else:
                h_c_temp, alignment, termination = self.attention(h_c,
                                                                  memory,
                                                                  input_lengths)
                
                h_t, h_f, h_c_temp = layer(h_t, h_f, h_c_temp, attention=True)
                h_c = h_c + h_c_temp

        theta_hat = self.W_theta(h_f)

        mu = theta_hat[:,:,:, :self.K] # eq. (3)
        std = torch.exp(theta_hat[:,:,:, self.K:2*self.K]) # eq. (4)
        pi = self.pi_softmax(theta_hat[:,:,:, 2*self.K:]) # eq. (5)

        ### MASKING ###
        idx = torch.arange(1, mu.size(-2)+1, device=mu.device)
        mask = (output_lengths.unsqueeze(-1) < idx.unsqueeze(0)).to(torch.bool) # B, T
        mask = mask.unsqueeze(1).unsqueeze(3)
        
        mu = torch.sigmoid(mu.masked_fill(mask, 0))
        std = std.masked_fill(mask, 1/np.sqrt(2 * np.pi))
            
        return mu, std, pi, alignment
    
    
    def sample(self, x, text, input_lengths):
        # Extract memory
        memory = self.TextEncoder(text)
        
        # x: [1, M, T] / B=1, M=mel, T=time
        x_t, x_f = x.clone(), x.clone()

        for i in range(x.size(-1)):
            h_t = self.W_t_0(x_t.unsqueeze(-1))
            h_f = self.W_f_0(x_f.unsqueeze(-1))
            h_c, alignment, termination = self.attention(self.W_c_0(F.pad(x, [1, -1]).transpose(1, 2)),
                                                         memory,
                                                         input_lengths)

            # h_t, h_f: [B, M, T, D] / h_c: [B, T, D]
            for i, layer in enumerate(self.layers):
                if i!=(len(self.layers)//2):
                    h_t, h_f, h_c = layer(h_t, h_f, h_c)

                else:
                    h_c, alignment, termination = self.attention(h_c,
                                                                 memory,
                                                                 input_lengths)

                    h_t, h_f, _ = layer(h_t, h_f, h_c, attention=True)

            theta_hat = self.W_theta(h_f)

            mu = torch.sigmoid(theta_hat[:, :, :, :self.K]) # eq. (3)
            pi = self.pi_softmax(theta_hat[:, :, :, 2*self.K:]) # eq. (5)

            mu = torch.sum(mu*pi, dim=3)

            x_t[:,:,i+1] = mu[:,:,i]
            x_f[:,i+1,:] = mu[:,i,:]

        return mu, std, pi, termination