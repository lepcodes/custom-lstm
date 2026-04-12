"""
Facade for the training module to maintain backwards compatibility.
"""

from custom_lstm.training.base import BaseTrainerStrategy, TrainingCallback
from custom_lstm.training.bp import BPTrainingStrategy
from custom_lstm.training.tbptt import TBPTTTrainerStrategy
from custom_lstm.training.ewacf_tbptt import EWACFTBPTTTrainerStrategy

__all__ = [
    "TrainingCallback",
    "BaseTrainerStrategy",
    "BPTrainingStrategy",
    "TBPTTTrainerStrategy",
    "EWACFTBPTTTrainerStrategy",
]
