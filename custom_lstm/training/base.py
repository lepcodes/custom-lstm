import abc
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
from optuna import Trial

from custom_lstm.models.base_model import AblationModel


@runtime_checkable
class TrainingCallback(Protocol):
    """
    Protocol for training loop callbacks.
    Implementations (e.g., MLflowCallback) are injected by the caller,
    keeping this module free of MLOps dependencies.
    """

    def on_epoch_end(self, epoch: int, metrics: dict) -> None: ...


class BaseTrainerStrategy(abc.ABC):
    """
    Abstract Base Trainer specifying the Template Method for the training loop.
    """

    def __init__(
        self,
        model: AblationModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        callback: TrainingCallback | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.callback = callback
        self.model.to(self.device)
        self.criterion.to(self.device)

    @abc.abstractmethod
    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor, **kwargs):
        pass

    def validate_epoch(self, X_val: torch.Tensor, y_val: torch.Tensor):
        """Default validation: evaluate criterion on full validation set."""
        self.model.eval()
        self.model.reset_state()

        with torch.no_grad():
            y_pred, telemetry = self.model(X_val)
            val_loss = self.criterion(y_pred, y_val).item()

        metrics = {"val_loss": val_loss}

        if telemetry is not None and getattr(telemetry, "forget_gates", None) is not None:
            # Variance along the sequence dimension (dim=1), averaged across batch and hidden units
            metrics["val_fg_variance"] = telemetry.forget_gates.var(dim=1).mean().item()

        return metrics

    def train(
        self,
        epochs: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        optuna_trial: Trial | None = None,
        patience: int = None,
        **kwargs,
    ):
        """
        The Template Method that calls the training and validation epochs.
        Supports early stopping via the patience parameter and Optuna pruning.
        Returns the best validation loss observed during training and restores the best model weights.
        """
        import copy

        best_val_loss = float("inf")
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(X_train, y_train, **kwargs)
            val_metrics = self.validate_epoch(X_val, y_val)

            metrics_to_log = {}
            if isinstance(train_metrics, dict):
                metrics_to_log.update(train_metrics)
                train_mse_print = train_metrics.get("train_mse", train_metrics.get("train_loss", 0.0))
            else:
                metrics_to_log["train_loss"] = train_metrics
                train_mse_print = train_metrics

            if isinstance(val_metrics, dict):
                metrics_to_log.update(val_metrics)
                val_loss = val_metrics.get("val_loss", 0.0)
            else:
                val_loss = val_metrics
                metrics_to_log["val_loss"] = val_loss

            if self.callback:
                self.callback.on_epoch_end(epoch, metrics_to_log)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch:>3}/{epochs}  |  Train MSE: {train_mse_print:.5f}  |  Val MSE: {val_loss:.5f}")

            if optuna_trial is not None:
                import optuna

                optuna_trial.report(val_loss, epoch)
                if optuna_trial.should_prune():
                    print(f"Trial pruned at epoch {epoch}")
                    raise optuna.exceptions.TrialPruned()

            if patience is not None and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return best_val_loss
