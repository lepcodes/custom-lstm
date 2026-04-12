import torch
import torch.nn as nn

from custom_lstm.models.base_model import AblationModel
from custom_lstm.training.base import BaseTrainerStrategy, TrainingCallback


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
