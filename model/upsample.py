import torch
import torch.nn as nn


class UpsampleRNN(nn.Module):
    def __init__(self, hp):
        super(UpsampleRNN, self).__init__()
        self.num_hidden = hp.model.hidden

        self.rnn_x = nn.GRU(
            input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True, bidirectional=True
        )
        self.rnn_y = nn.GRU(
            input_size=self.num_hidden, hidden_size=self.num_hidden, batch_first=True, bidirectional=True
        )

        self.W = nn.Linear(4 * self.num_hidden, self.num_hidden)

    def flatten_parameters(self):
        self.rnn_x.flatten_parameters()
        self.rnn_y.flatten_parameters()

    def forward(self, inp):
        self.flatten_parameters()
        
        B, M, T, D = inp.size()

        x, _ = self.rnn_x(inp.view(-1, T, D))
        x = x.view(B, M, T, 2 * D)

        y, _ = self.rnn_y(inp.transpose(1, 2).contiguous().view(-1, M, D))
        y = y.view(B, T, M, 2 * D).transpose(1, 2).contiguous()

        z = torch.cat([x, y], dim=-1)

        output = inp + self.W(z)

        return output
