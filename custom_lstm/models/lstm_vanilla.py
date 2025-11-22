import math

import torch
import torch.nn as nn


class LSTMCellVanilla(nn.Module):
    """
    Vanilla LSTM cell implementation on PyTorch.
    """

    def __init__(self, input_size, hidden_size):
        super(LSTMCellVanilla, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_hf = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xf = nn.Parameter(torch.empty(input_size, hidden_size))
        self.b_f = nn.Parameter(torch.empty(hidden_size))

        self.W_hi = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xi = nn.Parameter(torch.empty(input_size, hidden_size))
        self.b_i = nn.Parameter(torch.empty(hidden_size))

        self.W_hc = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xc = nn.Parameter(torch.empty(input_size, hidden_size))
        self.b_c = nn.Parameter(torch.empty(hidden_size))

        self.W_ho = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_xo = nn.Parameter(torch.empty(input_size, hidden_size))
        self.b_o = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        h, c = hidden

        f = torch.sigmoid(torch.matmul(h, self.W_hf) + torch.matmul(x, self.W_xf) + self.b_f)
        i = torch.sigmoid(torch.matmul(h, self.W_hi) + torch.matmul(x, self.W_xi) + self.b_i)
        g = torch.tanh(torch.matmul(h, self.W_hc) + torch.matmul(x, self.W_xc) + self.b_c)
        o = torch.sigmoid(torch.matmul(h, self.W_ho) + torch.matmul(x, self.W_xo) + self.b_o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class LSTMVanilla(nn.Module):
    """
    Vanilla LSTM Network implementation on PyTorch.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMVanilla, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = LSTMCellVanilla(input_size, hidden_size)
        self.dropout = nn.Dropout(0)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, sequence):
        batch_size = sequence.size(0)
        seq_length = sequence.size(1)
        h_t = torch.zeros(batch_size, self.hidden_size, device=sequence.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=sequence.device)

        for i in range(seq_length):
            x = sequence[:, i, :]
            h_t, c_t = self.lstm(x, (h_t, c_t))

        output = self.dropout(h_t)
        output = self.linear(output)
        return output
        return output
