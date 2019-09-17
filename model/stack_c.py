import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_module(nn.Module):
    def __init__(self, hp):
        super(Attention_module, self).__init__()
        self.rnn_cell = nn.LSTMCell(input_size=2*hp.num_hidden, hidden_size=hp.num_hidden, batch_first=True)
        self.num_hidden = hp.num_hidden
        
        self.W_g = nn.Linear(hp.num_hidden, 3*hp.M)

        
    def attention(self, h_i, memory):

        phi_hat = self.W_g(h_i)
        self.ksi_hat = self.ksi_hat + torch.exp(phi_hat[:, :self.M])
        self.beta_hat = torch.exp( phi_hat[:, self.M:2*self.M] )
        self.alpha_hat = F.softmax(phi_hat[:, 2*self.M:3*self.M], dim=-1)
        
        self.u = torch.LongTensor( range(memory.size(1)) )
        self.u_R = self.u + 0.5
        self.u_L = self.u - 0.5
        
        term1 = torch.sum(self.alpha_hat.unsqueeze(-1)*
                          torch.reciprocal(1 + torch.exp((self.ksi_hat.unsqueeze(-1) - self.u_R) / self.beta_hat.unsqueeze(-1))), dim=1)
        
        term2 = torch.sum(self.alpha_hat.unsqueeze(-1)*
                          torch.reciprocal(1 + torch.exp((self.ksi_hat.unsqueeze(-1) - self.u_L) / self.beta_hat.unsqueeze(-1))), dim=1)
        
        weights = (term1-term2).unsqueeze(1)
        
        
        context = torch.bmm(weights, memory)
        
        termination = 1 - torch.sum(self.alpha_hat.unsqueeze(-1)*
                                    torch.reciprocal(1 + torch.exp((self.ksi_hat.unsqueeze(-1) - self.u_R) / self.beta_hat.unsqueeze(-1))), dim=1)

        
        return context, weights, termination # (B, 1, D), (B, 1, T), (B, 1, T)

    
    
    def forward(self, input_h_c, memory, input_lengths):
        B, T, D = input_h_c.size()
        
        context = input_h_c.new_zeros(B, D)
        h_i, c_i  = input_h_c.new_zeros(B, D), input_h_c.new_zeros(B, D)
        
        contexts, weights, terminations = [], [], []
        for i in range(T):
            x = torch.cat([input_h_c[:, i], context], dim=-1)
            h_i, c_i = self.rnn_cell(x, (h_i, c_i))
            context, weight, termination = self.attention(h_i, memory)
            
            contexts.append(context)
            weights.append(weight)
            terminations.append(termination)
            
        contexts = torch.cat(contexts, dim=1)
        weights = torch.cat(weights, dim=1)
        terminations = torch.cat(terminations, dim=1)
        terminations = torch.gather(terminations, 2, input_lengths.unsqueeze(-1))

        return context, weights, terminations


    
class Stack_C(nn.Module):
    def __init__(self, hp):
        super(Stack_C, self).__init__()
        self.hp = hp
        self.TTS = hp.TTS
        
        if TTS==True:
            self.c_RNN = nn.LSTM(input_size=hp.num_hidden, hidden_size=hp.num_hidden, batch_first=True)
            
        else:
            self.c_RNN = Attention_module(hp)

    def forward(self, x, memory=None):
        if memory is None:
            h_c_temp, _ = self.c_RNN(input_h_c)
            
        elif:
            h_c_temp, _ = self.c_RNN(input_h_c, memory)
            
            
        return h_c_temp