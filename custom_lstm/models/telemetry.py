from typing import NamedTuple

import torch


class GateTelemetry(NamedTuple):
    """
    Typed container for LSTM gate activations exposed during forward pass.
    Enables the EW-ACF loss to route penalties per hidden unit.
    """
    forget_gates: torch.Tensor  # (batch, seq_len, hidden_size)
    input_gates: torch.Tensor   # (batch, seq_len, hidden_size)
