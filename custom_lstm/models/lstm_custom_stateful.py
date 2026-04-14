import torch
import torch.nn as nn

from custom_lstm.models.lstm_custom import LSTMCellCustom
from custom_lstm.models.base_model import AblationModel
from custom_lstm.models.telemetry import GateTelemetry


class LSTMCustomStateful(AblationModel):
    """
    Custom LSTM cell implementation on PyTorch with statefulness.
    """

    def __init__(self, input_size, hidden_size, output_size, forget_gate_layers=[]):
        super(LSTMCustomStateful, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = LSTMCellCustom(input_size, hidden_size, forget_gate_layers=forget_gate_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.h_t = None
        self.c_t = None

    def forward(self, sequence):
        batch_size = sequence.size(0)
        seq_length = sequence.size(1)

        if self.h_t is None:
            self.h_t = torch.zeros(batch_size, self.hidden_size, device=sequence.device)
            self.c_t = torch.zeros(batch_size, self.hidden_size, device=sequence.device)

        else:
            self.h_t = self.h_t.detach()
            self.c_t = self.c_t.detach()

        hidden_states = []
        forget_gates = []
        input_gates = []
        for t in range(seq_length):
            x = sequence[:, t, :]
            self.h_t, self.c_t, f, i = self.lstm(x, (self.h_t, self.c_t))
            hidden_states.append(self.h_t)
            forget_gates.append(f)
            input_gates.append(i)

        outputs = torch.stack(hidden_states, dim=1)
        predictions = self.linear(outputs)

        telemetry = GateTelemetry(
            forget_gates=torch.stack(forget_gates, dim=1),
            input_gates=torch.stack(input_gates, dim=1),
        )
        return predictions, telemetry

    def reset_state(self):
        self.h_t = None
        self.c_t = None
