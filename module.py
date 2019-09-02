import torch
import torch.nn as nn
from math import ceil

class Stack(nn.Module):
    def __init__(self, hp, use_central=False):
        super(Stack, self).__init__()
        self.time_rnn1 = nn.GRU(hp.hidden_dim, hp.hidden_dim, batch_first=True)
        self.time_rnn2 = nn.GRU(hp.hidden_dim, hp.hidden_dim, batch_first=True)
        self.time_rnn3 = nn.GRU(hp.hidden_dim, hp.hidden_dim, batch_first=True)
        self.freq_rnn = nn.GRU(hp.hidden_dim, hp.hidden_dim, batch_first=True)
        
        self.time_linear = nn.Linear(hp.hidden_dim*3, hp.hidden_dim)
        self.freq_linear = nn.Linear(hp.hidden_dim, hp.hidden_dim)
        
        self.use_central=use_central
        if use_central==True:
            self.central_rnn = nn.GRU(hp.hidden_dim, hp.hidden_dim, batch_first=True)
            self.central_linear = nn.Linear(hp.hidden_dim, hp.hidden_dim)
        
        
        
    def time_stack(self, x_t):
        B, F, T = x_t.size()[:-1]
        
        direction_right = []
        for i in range( F ):
            direction_right.append(self.time_rnn1(x_t[:,i])[0].unsqueeze(1))
        direction_right = torch.cat(direction_right, dim=1)
            
        direction_up = []
        for i in range( T ):
            direction_up.append(self.time_rnn2(x_t[:,:,i])[0].unsqueeze(2))
        direction_up = torch.cat(direction_up, dim=2)
            
        direction_down = []
        for i in range( T, 0, -1 ):
            direction_down.append(self.time_rnn3(x_t[:,:,i-1])[0].unsqueeze(2))
        direction_down = torch.cat(direction_down, dim=2)
        
        out = torch.cat([direction_right, direction_up, direction_down], dim=-1)
        
        out = self.time_linear(out) + x_t
        
        return out
        
    
    def freq_stack(self, x_t, x_f, x_c=None):
        B, F, T = x_t.size()[:-1]
        
        x_input = x_t + x_f
        if x_c is not None:
            x_input = x_input + x_c.unsqueeze(1)
        
        direction_up = []
        for i in range( T ):
            direction_up.append(self.freq_rnn(x_input[:,:,i])[0].unsqueeze(2))
        direction_up = torch.cat(direction_up, dim=2)
        
        out = self.freq_linear(direction_up) + x_f
        
        return out
    
    
    def cent_stack(self, x_c):
        B, T = x_c.size()[:-1]
        out = self.cent_rnn(x_c)
        out = self.cent_linear(out)
        out = out + x_c
        return out
        
        
    def forward(self, x_t, x_f, x_c=None):
        time_out = self.time_stack(x_t)
        freq_out = self.freq_stack(time_out, x_f, x_c)
        
        return time_out, freq_out
    
    
    
class UpperStack(nn.Module):
    def __init__(self, hp):
        super(UpperStack, self).__init__()
        self.time_rnn = nn.GRU(hp.hidden_dim, hp.hidden_dim, bidirectional=True, batch_first=True)
        self.freq_rnn = nn.GRU(hp.hidden_dim, hp.hidden_dim, bidirectional=True, batch_first=True)
        
        self.linear = nn.Linear(hp.hidden_dim*4, 1)
        
    def forward(self, x):
        time_out = self.time_rnn(x.transpose(1,2).contiguous())
        freq_out = self.freq_rnn(x)
        
        out = torch.cat([time_out, freq_out], dim=-1)
        out = self.linear(out)
        
        return out