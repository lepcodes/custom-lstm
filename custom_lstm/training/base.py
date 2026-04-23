import abc
from typing import Protocol, runtime_checkable
from optuna import Trial

import torch
import torch.nn as nn

from custom_lstm.models.base_model import AblationModel


@runtime_checkable
class TrainingCallback(Protocol):
    """
    Protocol for training loop callbacks.
    Implementations (e.g., MLflowCallback) are injected by the caller,
    keeping this module free of MLOps dependencies.
    """

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float) -> None: ...


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

    @abc.abstractmethod
    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor, **kwargs):
        pass

    def validate_epoch(self, X_val: torch.Tensor, y_val: torch.Tensor):
        """Default validation: evaluate criterion on full validation set."""
        self.model.eval()
        self.model.reset_state()

        with torch.no_grad():
            y_pred, _ = self.model(X_val)
            val_loss = self.criterion(y_pred, y_val).item()

        return val_loss

    def train(
        self,
        epochs: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        optuna_trial: Trial | None = None,
        patience: int = None,
        **kwargs
    ):
        """
        The Template Method that calls the training and validation epochs.
        Supports early stopping via the patience parameter and Optuna pruning.
        Returns the best validation loss observed during training.
        """
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(X_train, y_train, **kwargs)
            val_loss = self.validate_epoch(X_val, y_val)

            if self.callback:
                self.callback.on_epoch_end(epoch, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch:>3}/{epochs}  |  Train MSE: {train_loss:.5f}  |  Val MSE: {val_loss:.5f}")

            if optuna_trial is not None:
                import optuna
                optuna_trial.report(val_loss, epoch)
                if optuna_trial.should_prune():
                    print(f"Trial pruned at epoch {epoch}")
                    raise optuna.exceptions.TrialPruned()

            if patience is not None and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        return best_val_loss
