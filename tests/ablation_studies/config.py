"""
Experiment Configuration & Resolution Pipeline
================================================

When you create an ExperimentConfig (from YAML or code), the pipeline
resolves as follows:

    ┌─────────────────────────────────────────────────────────────┐
    │  YOU SET (in YAML or constructor):                          │
    │    • architecture    (required)                             │
    │    • data_mode       (required)                             │
    │    • loss_type       (default: "mse")                       │
    │    • trainer_strategy (default: auto-resolved)              │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  AUTO-RESOLUTION (runs during validation):                  │
    │                                                             │
    │  1. trainer_strategy:                                       │
    │     • If you set it explicitly → validate it's compatible   │
    │       with architecture (checked against allowed set)       │
    │     • If loss_type is EW-ACF → force "ewacf_tbptt"         │
    │     • If loss_type is MSE    → use architecture's default   │
    │                                                             │
    │  2. criterion (built by build_trainer in factory.py):       │
    │     • loss_type="mse"              → nn.MSELoss()           │
    │     • loss_type="ewacf_broadcast"  → EWACFLoss(broadcast)   │
    │     • loss_type="ewacf_input_gate" → EWACFLoss(input_gate)  │
    │     • If data_mode="win" → enforce_min_lag(window_size)     │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  RESULTING PIPELINE:                                        │
    │    Model      = ModelRegistry.build(architecture)           │
    │    Criterion   = based on loss_type                         │
    │    Trainer     = based on trainer_strategy                  │
    │                                                             │
    │  Call config.describe_pipeline() to see the resolved state  │
    └─────────────────────────────────────────────────────────────┘

Quick Reference — The 6-Model Phase 2 Ablation Matrix:
    P-0: architecture=lstm_custom_pure,     loss_type=mse,              trainer=tbptt
    P-1: architecture=lstm_custom_pure,     loss_type=ewacf_broadcast,  trainer=ewacf_tbptt
    P-2: architecture=lstm_custom_pure,     loss_type=ewacf_input_gate, trainer=ewacf_tbptt
    W-0: architecture=lstm_custom_windowed, loss_type=mse,              trainer=tbptt
    W-1: architecture=lstm_custom_windowed, loss_type=ewacf_broadcast,  trainer=ewacf_tbptt
    W-2: architecture=lstm_custom_windowed, loss_type=ewacf_input_gate, trainer=ewacf_tbptt
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ─── Single Source of Truth: Architecture Names ───────────────────────────────

class ArchitectureType(str, Enum):
    LSTM_VANILLA_PURE = "lstm_vanilla_pure"
    LSTM_VANILLA_WINDOWED = "lstm_vanilla_windowed"
    LSTM_CUSTOM_PURE = "lstm_custom_pure"
    LSTM_CUSTOM_WINDOWED = "lstm_custom_windowed"
    SIMPLE_MLP = "simple_mlp"
    LSTM_VANILLA_PURE_NO_RECURRENCE = "lstm_vanilla_pure_no_recurrence"
    LSTM_VANILLA_WINDOWED_NO_RECURRENCE = "lstm_vanilla_windowed_no_recurrence"


class LossType(str, Enum):
    """Which loss function to use."""
    MSE = "mse"
    EWACF_BROADCAST = "ewacf_broadcast"
    EWACF_INPUT_GATE = "ewacf_input_gate"


class TrainerStrategyType(str, Enum):
    """Available training strategies."""
    TBPTT = "tbptt"
    STANDARD_BP = "standard_bp"
    EWACF_TBPTT = "ewacf_tbptt"


class DataMode(str, Enum):
    """Which tensor layout the architecture expects."""
    PURE = "pure"
    WINDOWED = "win"


# ─── Architecture → Strategy Compatibility Rules ─────────────────────────────
# Each architecture maps to a set of *allowed* strategies and a *default*.

ARCHITECTURE_STRATEGY_RULES: Dict[ArchitectureType, Dict] = {
    ArchitectureType.LSTM_VANILLA_PURE: {
        "allowed": {TrainerStrategyType.TBPTT, TrainerStrategyType.EWACF_TBPTT},
        "default": TrainerStrategyType.TBPTT,
    },
    ArchitectureType.LSTM_VANILLA_WINDOWED: {
        "allowed": {TrainerStrategyType.TBPTT, TrainerStrategyType.EWACF_TBPTT},
        "default": TrainerStrategyType.TBPTT,
    },
    ArchitectureType.LSTM_CUSTOM_PURE: {
        "allowed": {TrainerStrategyType.TBPTT, TrainerStrategyType.EWACF_TBPTT},
        "default": TrainerStrategyType.EWACF_TBPTT,
    },
    ArchitectureType.LSTM_CUSTOM_WINDOWED: {
        "allowed": {TrainerStrategyType.TBPTT, TrainerStrategyType.EWACF_TBPTT},
        "default": TrainerStrategyType.EWACF_TBPTT,
    },
    ArchitectureType.SIMPLE_MLP: {
        "allowed": {TrainerStrategyType.STANDARD_BP},
        "default": TrainerStrategyType.STANDARD_BP,
    },
    ArchitectureType.LSTM_VANILLA_PURE_NO_RECURRENCE: {
        "allowed": {TrainerStrategyType.STANDARD_BP, TrainerStrategyType.TBPTT},
        "default": TrainerStrategyType.STANDARD_BP,
    },
    ArchitectureType.LSTM_VANILLA_WINDOWED_NO_RECURRENCE: {
        "allowed": {TrainerStrategyType.STANDARD_BP, TrainerStrategyType.TBPTT},
        "default": TrainerStrategyType.STANDARD_BP,
    },
}


# ─── Experiment Configuration ─────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """
    Pydantic model that validates a YAML experiment configuration.
    All auto-resolution happens in a single resolve_pipeline() validator.
    Call describe_pipeline() after creation to see the fully resolved state.
    """
    experiment_name: str = Field(default="Thesis_Ablation", description="MLflow experiment name")
    run_name: Optional[str] = Field(default=None, description="MLflow run name, auto-generated if None")

    data_path: str = Field(..., description="Path to preprocessed CSV dataset")
    dataset_name: Optional[str] = Field(default=None, description="Auto-derived from data_path if None")

    architecture: ArchitectureType = Field(..., description="Which registered model to use")
    data_mode: DataMode = Field(..., description="Which tensor layout to use: 'pure' or 'win'")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed directly to the model constructor")
    trainer_strategy: Optional[TrainerStrategyType] = Field(default=None, description="Training strategy. Auto-resolved from architecture if None.")

    window_size: int = Field(default=10, ge=1)
    epochs: int = Field(default=100, ge=1)
    lr: float = Field(default=0.001, gt=0)
    bptt_steps: int = Field(default=50, ge=1)

    # ── EW-ACF Loss Configuration ─────────────────────────────────────────
    loss_type: LossType = Field(default=LossType.MSE, description="Loss function: 'mse', 'ewacf_broadcast', or 'ewacf_input_gate'")
    ewacf_alpha: float = Field(default=0.5, ge=0, description="Weight of the EW-ACF penalty term")
    ewacf_lambda: float = Field(default=0.5, ge=0, le=1, description="Exponential decay factor for running statistics")
    ewacf_lag: int = Field(default=1, ge=1, description="Lag for autocorrelation computation")
    ewacf_threshold: float = Field(default=0.1, ge=0, description="Irrelevance threshold below which penalty is zero")

    @model_validator(mode="after")
    def resolve_pipeline(self) -> "ExperimentConfig":
        """
        Single-pass resolution of all auto-derived fields.

        Resolution order:
          1. trainer_strategy  (from loss_type + architecture rules)
          2. dataset_name      (from data_path)
          3. experiment_name   (from dataset_name)
        """
        rules = ARCHITECTURE_STRATEGY_RULES[self.architecture]

        # — Step 1: Resolve trainer_strategy —
        if self.trainer_strategy is not None:
            # User explicitly set it → validate compatibility
            if self.trainer_strategy not in rules["allowed"]:
                allowed_list = [str(s) for s in rules["allowed"]]
                raise ValueError(
                    f"Strategy '{self.trainer_strategy}' is incompatible with "
                    f"architecture '{self.architecture}'. "
                    f"Allowed: {allowed_list}"
                )
        elif self.loss_type != LossType.MSE:
            # EW-ACF loss requires the specialized trainer
            resolved = TrainerStrategyType.EWACF_TBPTT
            if resolved not in rules["allowed"]:
                allowed_list = [str(s) for s in rules["allowed"]]
                raise ValueError(
                    f"loss_type='{self.loss_type}' requires strategy "
                    f"'{resolved}', but architecture "
                    f"'{self.architecture}' doesn't allow it. "
                    f"Allowed: {allowed_list}"
                )
            self.trainer_strategy = resolved
        else:
            # Pure MSE → use the architecture's default
            self.trainer_strategy = rules["default"]

        # — Step 2: Resolve names —
        dataset_stem = Path(self.data_path).stem
        if self.dataset_name is None:
            self.dataset_name = dataset_stem
        if self.experiment_name == "Thesis_Ablation":
            self.experiment_name = f"Phase1_{dataset_stem}"

        return self

    def describe_pipeline(self) -> str:
        """
        Returns a human-readable summary of the fully resolved experiment pipeline.
        Call this after config creation to verify what you'll actually get.
        """
        loss_detail = str(self.loss_type)
        if self.loss_type != LossType.MSE:
            loss_detail += (
                f" (α={self.ewacf_alpha}, λ={self.ewacf_lambda}, "
                f"lag={self.ewacf_lag}, θ={self.ewacf_threshold})"
            )

        lag_note = ""
        if self.loss_type != LossType.MSE and self.data_mode == DataMode.WINDOWED:
            effective_lag = max(self.ewacf_lag, self.window_size)
            lag_note = f"\n  Effective lag:   {effective_lag} (enforced >= window_size={self.window_size})"

        input_desc = f"window_size={self.window_size}" if self.data_mode == DataMode.WINDOWED else "1"

        return (
            f"{'─' * 60}\n"
            f"  Experiment Pipeline Summary\n"
            f"{'─' * 60}\n"
            f"  Architecture:    {self.architecture}\n"
            f"  Data mode:       {self.data_mode} (input_size={input_desc})\n"
            f"  Loss:            {loss_detail}\n"
            f"  Trainer:         {self.trainer_strategy}\n"
            f"  BPTT steps:      {self.bptt_steps}{lag_note}\n"
            f"  Dataset:         {self.dataset_name} ({self.data_path})\n"
            f"  Experiment:      {self.experiment_name}\n"
            f"{'─' * 60}"
        )

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
