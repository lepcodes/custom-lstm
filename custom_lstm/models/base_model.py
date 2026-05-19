import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn

from custom_lstm.models.telemetry import GateTelemetry


class AblationModel(nn.Module, abc.ABC):
    """
    Abstract Base Class for all models used in the Ablation Study.
    Enforces a consistent interface for the Trainer Strategies.
    """

    @abc.abstractmethod
    def forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, Optional[GateTelemetry]]:
        """
        Forward pass of the model.
        Args:
            sequence: Input tensor of shape (batch, seq_len, features)
        Returns:
            predictions: Output tensor of shape (batch, seq_len, output_size)
            telemetry: Optional[GateTelemetry] — typed container with
                forget_gates and input_gates tensors, or None for non-LSTM models.
        """
        pass

    @abc.abstractmethod
    def reset_state(self) -> None:
        """
        Resets the internal state (e.g., hidden and cell states) of the model.
        Must be callable, even if the model (like an MLP) is stateless.
        """
        pass
