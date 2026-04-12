import torch
import torch.nn as nn

from custom_lstm.models.lstm_vanilla import LSTMCellVanilla


class EncoderDecoderCustom(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EncoderDecoderCustom, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder = LSTMCellVanilla(input_size, hidden_size)
        self.decoder = LSTMCellVanilla(output_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, history_seq, target_len, teacher_forcing_ratio=0.0, target_seq=None):
        batch_size = history_seq.size(0)
        seq_len = history_seq.size(1)
        device = history_seq.device

        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=device)

        # ==========================================
        # PHASE 1: THE ENCODER
        # ==========================================
        for t in range(seq_len):
            x_t = history_seq[:, t, :]
            h_t, c_t, _ = self.encoder(x_t, (h_t, c_t))

        decoder_input = history_seq[:, -1, :]
        predictions = []
        decoder_forget_gates = []

        # ==========================================
        # PHASE 2: THE DECODER
        # ==========================================
        for t in range(target_len):
            h_t, c_t, f_t = self.decoder(decoder_input, (h_t, c_t))

            prediction = self.fc_out(h_t)

            predictions.append(prediction)
            decoder_forget_gates.append(f_t)

            if self.training and target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t, :]
            else:
                decoder_input = prediction

        predictions = torch.stack(predictions, dim=1)  # [batch, target_len, output_size]
        decoder_forget_gates = torch.stack(decoder_forget_gates, dim=1)  # [batch, target_len, hidden_size]

        return predictions, decoder_forget_gates
