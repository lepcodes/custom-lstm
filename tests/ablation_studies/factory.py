"""
Trainer & Criterion Factory for the Ablation Study.

Given a validated ExperimentConfig, build_trainer() assembles:
  1. The correct criterion (MSELoss or EWACFLoss with routing)
  2. The correct trainer strategy (TBPTT, BP, or EWACF-TBPTT)
  3. Injects an optional TrainingCallback for MLOps logging
"""

import torch
import torch.nn as nn

from custom_lstm.losses.acf_losses import EWACFLoss
from custom_lstm.models.base_model import AblationModel
from custom_lstm.training.trainer import (
    BaseTrainerStrategy,
    BPTrainingStrategy,
    EWACFTBPTTTrainerStrategy,
    TBPTTTrainerStrategy,
    TrainingCallback,
)

from tests.ablation_studies.config import (
    DataMode,
    ExperimentConfig,
    LossType,
    TrainerStrategyType,
)


def build_trainer(
    config: ExperimentConfig,
    model: AblationModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    callback: TrainingCallback | None = None,
) -> BaseTrainerStrategy:
    """
    Builds the correct TrainerStrategy and criterion from a validated ExperimentConfig.
    Owns criterion creation — callers do not need to instantiate a loss function.
    """
    # ── Build criterion ────────────────────────────────────────────────────
    if config.loss_type == LossType.MSE:
        criterion = nn.MSELoss()
    else:
        # We only support EWACF_BROADCAST now, routing is fixed to broadcast
        criterion = EWACFLoss(
            lambda_=config.ewacf_lambda,
            lag=config.ewacf_lag,
            alpha=config.ewacf_alpha,
            threshold=config.ewacf_threshold,
        )
        if config.data_mode == DataMode.WINDOWED:
            criterion.enforce_min_lag(config.window_size)

    # ── Build trainer strategy ─────────────────────────────────────────────
    if config.trainer_strategy == TrainerStrategyType.EWACF_TBPTT:
        return EWACFTBPTTTrainerStrategy(
            model=model, optimizer=optimizer, criterion=criterion,
            device=device, bptt_steps=config.bptt_steps, callback=callback,
        )
    elif config.trainer_strategy == TrainerStrategyType.TBPTT:
        return TBPTTTrainerStrategy(
            model=model, optimizer=optimizer, criterion=criterion,
            device=device, bptt_steps=config.bptt_steps, callback=callback,
        )
    elif config.trainer_strategy == TrainerStrategyType.STANDARD_BP:
        return BPTrainingStrategy(
            model=model, optimizer=optimizer, criterion=criterion,
            device=device, callback=callback,
        )
    else:
        raise ValueError(f"Unknown trainer strategy: {config.trainer_strategy}")
