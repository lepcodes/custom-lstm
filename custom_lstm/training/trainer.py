import abc
from typing import Protocol, runtime_checkable

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

    def train(self, epochs: int, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, patience: int = None, **kwargs):
        """
        The Template Method that calls the training and validation epochs.
        Supports early stopping via the patience parameter.
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

            if patience is not None and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        return best_val_loss


class TBPTTTrainerStrategy(BaseTrainerStrategy):
    """
    Concrete Trainer Strategy implementing Truncated Backpropagation Through Time (TBPTT).
    """

    def __init__(
        self,
        model: AblationModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        bptt_steps: int,
        callback: TrainingCallback | None = None,
    ):
        super().__init__(model, optimizer, criterion, device, callback)
        self.bptt_steps = bptt_steps

    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor, **kwargs):
        self.model.train()
        self.model.reset_state()

        epoch_loss = 0.0
        total_samples = 0
        total_steps = X_train.size(1)

        for i in range(0, total_steps, self.bptt_steps):
            X_batch = X_train[:, i : i + self.bptt_steps, :]
            y_batch = y_train[:, i : i + self.bptt_steps, :]
            chunk_size = X_batch.size(1)

            y_pred, _ = self.model(X_batch)

            loss = self.criterion(y_pred, y_batch)

            if torch.isnan(loss):
                print(f"NAN loss detected at batch start={i}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * chunk_size
            total_samples += chunk_size

        return epoch_loss / total_samples


class BPTrainingStrategy(BaseTrainerStrategy):
    def __init__(
        self,
        model: AblationModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        callback: TrainingCallback | None = None,
    ):
        super().__init__(model, optimizer, criterion, device, callback)

    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor, **kwargs):
        self.model.train()
        self.model.reset_state()

        y_pred, _ = self.model(X_train)
        loss = self.criterion(y_pred, y_train)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class EWACFTBPTTTrainerStrategy(BaseTrainerStrategy):
    """
    TBPTT Trainer Strategy for EW-ACF regularized training.
    The criterion IS the EWACFLoss (which computes MSE + penalty internally).
    Logs train_mse, train_penalty, and train_loss (total) separately.
    Validates with pure MSE (penalty is a training-only regularizer).
    """

    def __init__(
        self,
        model: AblationModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        bptt_steps: int,
        callback: TrainingCallback | None = None,
    ):
        super().__init__(model, optimizer, criterion, device, callback)
        self.bptt_steps = bptt_steps
        self._val_mse = nn.MSELoss()

    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor, **kwargs):
        self.model.train()
        self.model.reset_state()
        self.criterion.reset_state()

        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_penalty = 0.0
        total_samples = 0
        total_steps = X_train.size(1)

        for i in range(0, total_steps, self.bptt_steps):
            X_batch = X_train[:, i : i + self.bptt_steps, :]
            y_batch = y_train[:, i : i + self.bptt_steps, :]
            chunk_size = X_batch.size(1)

            y_pred, telemetry = self.model(X_batch)
            total_loss, mse_val, penalty_val = self.criterion(
                y_pred, y_batch, X_batch,
                telemetry.forget_gates, telemetry.input_gates,
            )

            if torch.isnan(total_loss):
                print(f"NAN loss detected at batch start={i}")

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += total_loss.item() * chunk_size
            epoch_mse += mse_val.item() * chunk_size
            epoch_penalty += penalty_val.item() * chunk_size
            total_samples += chunk_size

        return epoch_loss / total_samples

    def validate_epoch(self, X_val: torch.Tensor, y_val: torch.Tensor):
        """Validate with pure MSE — the penalty is a training-only regularizer."""
        self.model.eval()
        self.model.reset_state()

        with torch.no_grad():
            y_pred, _ = self.model(X_val)
            val_loss = self._val_mse(y_pred, y_val).item()

        return val_loss
