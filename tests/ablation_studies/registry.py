"""
Model Registration for the Ablation Study.

Import this module at startup to register all model classes with the ModelRegistry.
This keeps registration side-effects isolated from configuration and factory logic.
"""

from custom_lstm.models.registry import ModelRegistry
from custom_lstm.models.lstm_custom_stateful import LSTMCustomStateful
from custom_lstm.models.lstm_vanilla_stateful import LSTMVanillaStateful
from custom_lstm.models.mlp import MLP

from tests.ablation_studies.config import ArchitectureType


# ─── Model Registration (uses Enum values) ───────────────────────────────────

ModelRegistry.register(ArchitectureType.LSTM_VANILLA_PURE)(LSTMVanillaStateful)
ModelRegistry.register(ArchitectureType.LSTM_VANILLA_WINDOWED)(LSTMVanillaStateful)
ModelRegistry.register(ArchitectureType.LSTM_CUSTOM_PURE)(LSTMCustomStateful)
ModelRegistry.register(ArchitectureType.LSTM_CUSTOM_WINDOWED)(LSTMCustomStateful)
ModelRegistry.register(ArchitectureType.LSTM_VANILLA_PURE_NO_RECURRENCE)(LSTMVanillaStateful)
ModelRegistry.register(ArchitectureType.LSTM_VANILLA_WINDOWED_NO_RECURRENCE)(LSTMVanillaStateful)
ModelRegistry.register(ArchitectureType.SIMPLE_MLP)(MLP)
