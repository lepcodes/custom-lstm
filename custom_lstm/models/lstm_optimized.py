import math

import torch
import torch.nn as nn


class LSTMCellVanillaOptimized(nn.Module):
    """
    Vanilla LSTM cell implementation on PyTorch with optimized number of tensor operations.
    """

    def __init__(self, input_size, hidden_size):
        super(LSTMCellVanillaOptimized, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b_xh = nn.Parameter(torch.empty(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.empty(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        h, c = hidden

        gates = torch.matmul(x, self.W_xh.t()) + self.b_xh + torch.matmul(h, self.W_hh.t()) + self.b_hh
        forget_gate, input_gate, output_gate, cell_gate = gates.chunk(4, 1)

        f = torch.sigmoid(forget_gate)
        i = torch.sigmoid(input_gate)
        g = torch.tanh(cell_gate)
        o = torch.sigmoid(output_gate)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class LSTMVanillaOptimized(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMVanillaOptimized, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = LSTMCellVanillaOptimized(input_size, hidden_size)
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
