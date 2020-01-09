import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constant import f_div


class DelayedRNN(nn.Module):
    def __init__(self, hp):
        super(DelayedRNN, self).__init__()
        self.num_hidden = hp.model.hidden

        self.t_delay_RNN_x = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            batch_first=True
        )
        self.t_delay_RNN_yz = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            batch_first=True,
            bidirectional=True
        )

        # use central stack only at initial tier
        self.c_RNN = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            batch_first=True
        )
        self.f_delay_RNN = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            batch_first=True
        )

        self.W_t = nn.Linear(3*self.num_hidden, self.num_hidden)
        self.W_c = nn.Linear(self.num_hidden, self.num_hidden)
        self.W_f = nn.Linear(self.num_hidden, self.num_hidden)
   
    def flatten_rnn(self):
        self.t_delay_RNN_x.flatten_parameters()
        self.t_delay_RNN_yz.flatten_parameters()
        self.c_RNN.flatten_parameters()
        self.f_delay_RNN.flatten_parameters()

    def forward(self, input_h_t, input_h_f, input_h_c, audio_lengths):
      
        self.flatten_rnn()
        # input_h_t, input_h_f: [B, M, T, D]
        # input_h_c: [B, T, D]
        B, M, T, D = input_h_t.size()

        ####### time-delayed stack #######
        # Fig. 2(a)-1 can be parallelized by viewing each horizontal line as batch
        h_t_x_temp = input_h_t.view(-1, T, D)
        h_t_x_packed = nn.utils.rnn.pack_padded_sequence(
            h_t_x_temp,
            audio_lengths.unsqueeze(1).repeat(1, M).reshape(-1),
            batch_first=True,
            enforce_sorted=False
        )
        h_t_x, _ = self.t_delay_RNN_x(h_t_x_packed)
        h_t_x, _ = nn.utils.rnn.pad_packed_sequence(
            h_t_x,
            batch_first=True,
            total_length=T
        )
        h_t_x = h_t_x.view(B, M, T, D)

        # Fig. 2(a)-2,3 can be parallelized by viewing each vertical line as batch,
        # using bi-directional version of GRU
        h_t_yz_temp = input_h_t.transpose(1, 2).contiguous() # [B, T, M, D]
        h_t_yz_temp = h_t_yz_temp.view(-1, M, D)
        h_t_yz, _ = self.t_delay_RNN_yz(h_t_yz_temp)
        h_t_yz = h_t_yz.view(B, T, M, 2*D)
        h_t_yz = h_t_yz.transpose(1, 2)

        h_t_concat = torch.cat((h_t_x, h_t_yz), dim=3)
        output_h_t = input_h_t + self.W_t(h_t_concat) # residual connection, eq. (6)

        ####### centralized stack #######
        h_c_temp = nn.utils.rnn.pack_padded_sequence(
            input_h_c,
            audio_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        h_c_temp, _ = self.c_RNN(h_c_temp)
        h_c_temp, _ = nn.utils.rnn.pad_packed_sequence(
            h_c_temp,
            batch_first=True,
            total_length=T
        )
            
        output_h_c = input_h_c + self.W_c(h_c_temp) # residual connection, eq. (11)
        h_c_expanded = output_h_c.unsqueeze(1)

        ####### frequency-delayed stack #######
        h_f_sum = input_h_f + output_h_t + h_c_expanded
        h_f_sum = h_f_sum.transpose(1, 2).contiguous() # [B, T, M, D]
        h_f_sum = h_f_sum.view(-1, M, D)

        h_f_temp, _ = self.f_delay_RNN(h_f_sum)
        h_f_temp = h_f_temp.view(B, T, M, D)
        h_f_temp = h_f_temp.transpose(1, 2) # [B, M, T, D]
        
        output_h_f = input_h_f + self.W_f(h_f_temp) # residual connection, eq. (8)

        return output_h_t, output_h_f, output_h_c
