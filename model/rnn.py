import torch
import torch.nn as nn
import torch.nn.functional as F


class DelayedRNN(nn.Module):
    def __init__(self, hp):
        super(DelayedRNN, self).__init__()
        self.num_hidden = hp.model.hidden

        # TODO: fix this hard-coded value
        self.freq = 32

        self.t_delay_RNN_x = nn.GRU(
            input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True)
        self.t_delay_RNN_y = nn.GRU(
            input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True)
        self.t_delay_RNN_z = nn.GRU(
            input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True)
        self.c_RNN = nn.GRU(
            input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True)
        self.f_delay_RNN = nn.GRU(
            input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True)

        self.W_t = nn.Linear(3*self.num_hidden, self.num_hidden)
        self.W_c = nn.Linear(self.num_hidden, self.num_hidden)
        self.W_f = nn.Linear(self.num_hidden, self.num_hidden)

    def forward(self, input_h_t, input_h_f, input_h_c):
        # time-delayed stack 
        h_t_x = torch.zeros(input_h_t.shape)

        for i in range(input_h_t.shape[2]): # line by line. TODO: parallelize
            h_t_x_slice, _ = self.t_delay_RNN_x(input_h_t[:, :, i, :])
            h_t_x[:, :, i, :] = h_t_x_slice

        h_t_y = torch.zeros(input_h_t.shape)
        h_t_z = torch.zeros(input_h_t.shape)
        for i in range(input_h_t.shape[1]): # line by line. TODO: parallelize
            h_t_y_slice, _ = self.t_delay_RNN_y(input_h_t[:, i, :, :])
            h_t_z_slice, _ = self.t_delay_RNN_z(input_h_t.flip(2)[:, i, :, :])
            # TODO: can use bidirectional=True
            h_t_y[:, i, :, :] = h_t_y_slice
            h_t_z[:, i, :, :] = h_t_z_slice.flip(1)

        h_t_concat = torch.cat((h_t_x, h_t_y, h_t_z), dim=3)
        output_h_t = input_h_t + self.W_t(h_t_concat) # residual connection, eq. (6)

        # centralized stack
        h_c_temp, _ = self.c_RNN(input_h_c)
        output_h_c = input_h_c + self.W_c(h_c_temp) # residual connection, eq. (11)

        # frequency-delayed stack
        h_c_expanded = output_h_c.unsqueeze(2).repeat(1, 1, self.freq, 1)
        h_f_sum = input_h_f + output_h_t + h_c_expanded

        h_f_temp = torch.zeros(input_h_f.shape)
        for i in range(h_f_sum.shape[1]):
            h_f_slice, _ = self.f_delay_RNN(h_f_sum[:, i, :, :])
            h_f_temp[:, i, :, :] = h_f_slice

        output_h_f = input_h_f + self.W_f(h_f_temp) # residual connection, eq. (8)

        return output_h_t, output_h_f, output_h_c
