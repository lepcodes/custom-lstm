import torch
import torch.nn as nn

from custom_lstm.models.lstm_vanilla import LSTMCellVanilla


class LSTMVanillaStateful(nn.Module):
    """
    Vanilla LSTM cell implementation on PyTorch with statefulness.
    """

    def __init__(self, input_size, hidden_size, output_size, return_forget=False):
        super(LSTMVanillaStateful, self).__init__()
        self.return_forget = return_forget
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = LSTMCellVanilla(input_size, hidden_size)
        self.dropout = nn.Dropout(0)
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
        for i in range(seq_length):
            x = sequence[:, i, :]
            self.h_t, self.c_t = self.lstm(x, (self.h_t, self.c_t))
            hidden_states.append(self.h_t)
        outputs = torch.stack(hidden_states, dim=1)
        outputs = self.dropout(outputs)
        predictions = self.linear(outputs)

        if self.return_forget:
            return predictions, (self.h_t, self.c_t)
        return predictions

    def reset_state(self):
        self.h_t = None
        self.c_t = None
        self.c_t = None
