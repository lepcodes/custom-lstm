import math

import torch
import torch.nn as nn

from custom_lstm.models.mlp import MLP


class LSTMCellCustom(nn.Module):
    """
    Vanilla LSTM cell implementation on PyTorch.
    """

    def __init__(self, input_size, hidden_size):
        super(LSTMCellCustom, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forget_mlp = MLP(input_size + hidden_size, hidden_size, [16, 8])

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

        f = self.forget_mlp(torch.cat((x, h), dim=1))
        i = torch.sigmoid(torch.matmul(h, self.W_hi) + torch.matmul(x, self.W_xi) + self.b_i)
        g = torch.tanh(torch.matmul(h, self.W_hc) + torch.matmul(x, self.W_xc) + self.b_c)
        o = torch.sigmoid(torch.matmul(h, self.W_ho) + torch.matmul(x, self.W_xo) + self.b_o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new, f


class LSTMCustomStateful(nn.Module):
    """
    Custom LSTM cell implementation on PyTorch with statefulness.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMCustomStateful, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = LSTMCellCustom(input_size, hidden_size)
        self.dropout = nn.Dropout(0)
        self.linear = nn.Linear(hidden_size, output_size)
        self.h_t = None
        self.c_t = None

    def forward(self, sequence):
        batch_size = sequence.size(0)
        seq_length = sequence.size(1)
        forget_gates = []

        if self.h_t is None:
            self.h_t = torch.zeros(batch_size, self.hidden_size, device=sequence.device)
            self.c_t = torch.zeros(batch_size, self.hidden_size, device=sequence.device)

        else:
            self.h_t = self.h_t.detach()
            self.c_t = self.c_t.detach()

        hidden_states = []
        for i in range(seq_length):
            x = sequence[:, i, :]
            self.h_t, self.c_t, f = self.lstm(x, (self.h_t, self.c_t))
            hidden_states.append(self.h_t)
            forget_gates.append(f)

        outputs = torch.stack(hidden_states, dim=1)
        outputs = self.dropout(outputs)
        predictions = self.linear(outputs)

        return predictions, torch.stack(forget_gates, dim=1)

    def reset_state(self):
        self.h_t = None
        self.c_t = None
        self.c_t = None
        self.c_t = None
