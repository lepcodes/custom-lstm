import math

import torch
import torch.nn as nn

from custom_lstm.models.mlp import MLP
from custom_lstm.models.base_model import AblationModel
from custom_lstm.models.telemetry import GateTelemetry


class LSTMCellCustom(nn.Module):
    """
    Custom LSTM cell implementation on PyTorch.
    """

    def __init__(self, input_size, hidden_size):
        super(LSTMCellCustom, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forget_mlp = MLP(input_size + hidden_size, hidden_size, [16, 16])

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

        f_raw, _ = self.forget_mlp(torch.cat((x, h), dim=1))
        f = torch.sigmoid(f_raw)
        i = torch.sigmoid(torch.matmul(h, self.W_hi) + torch.matmul(x, self.W_xi) + self.b_i)
        g = torch.tanh(torch.matmul(h, self.W_hc) + torch.matmul(x, self.W_xc) + self.b_c)
        o = torch.sigmoid(torch.matmul(h, self.W_ho) + torch.matmul(x, self.W_xo) + self.b_o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new, f, i


class LSTMCustom(AblationModel):
    """
    Vanilla LSTM Network implementation on PyTorch.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMCustom, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = LSTMCellCustom(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, sequence):
        batch_size = sequence.size(0)
        seq_length = sequence.size(1)
        h_t = torch.zeros(batch_size, self.hidden_size, device=sequence.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=sequence.device)

        forget_gates = []
        input_gates = []
        for t in range(seq_length):
            x = sequence[:, t, :]
            h_t, c_t, f, ig = self.lstm(x, (h_t, c_t))
            forget_gates.append(f)
            input_gates.append(ig)

        output = self.linear(h_t)
        telemetry = GateTelemetry(
            forget_gates=torch.stack(forget_gates, dim=1),
            input_gates=torch.stack(input_gates, dim=1),
        )
        return output, telemetry

    def reset_state(self):
        pass
