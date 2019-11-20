import torch
import torch.nn as nn


class UpsampleRNN(nn.Module):
    def __init__(self, hp):
        super(UpsampleRNN, self).__init__()
        self.num_hidden = hp.model.hidden

        self.rnn_x = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.rnn_y = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.W = nn.Linear(4 * self.num_hidden, self.num_hidden)

    def flatten_parameters(self):
        self.rnn_x.flatten_parameters()
        self.rnn_y.flatten_parameters()

    def forward(self, inp, audio_lengths):
        self.flatten_parameters()
        
        B, M, T, D = inp.size()

        inp_temp = inp.view(-1, T, D)
        inp_temp = nn.utils.rnn.pack_padded_sequence(
            inp_temp,
            audio_lengths.unsqueeze(1).repeat(1, M).reshape(-1),
            batch_first=True,
            enforce_sorted=False
        )
        x, _ = self.rnn_x(inp_temp)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x,
            batch_first=True,
            total_length=T
        )
        x = x.view(B, M, T, 2 * D)

        y, _ = self.rnn_y(inp.transpose(1, 2).contiguous().view(-1, M, D))
        y = y.view(B, T, M, 2 * D).transpose(1, 2).contiguous()

        z = torch.cat([x, y], dim=-1)

        output = inp + self.W(z)

        return output
