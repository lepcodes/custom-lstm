import torch
import torch.nn as nn

from custom_lstm.models.base_model import AblationModel
from custom_lstm.training.base import BaseTrainerStrategy, TrainingCallback


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
        """
        Training Loop for Truncated Backpropagation Through Time
        """
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

        return {"train_loss": epoch_loss / total_samples}

    def validate_epoch(self, X_val: torch.Tensor, y_val: torch.Tensor):
        """Validate with chunking to prevent OOM and maintain state symmetry with training."""
        self.model.eval()
        self.model.reset_state()

        epoch_loss = 0.0
        epoch_fg_variance = 0.0
        total_samples = 0
        total_steps = X_val.size(1)

        with torch.no_grad():
            for i in range(0, total_steps, self.bptt_steps):
                X_batch = X_val[:, i : i + self.bptt_steps, :]
                y_batch = y_val[:, i : i + self.bptt_steps, :]
                chunk_size = X_batch.size(1)

                y_pred, telemetry = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)

                epoch_loss += loss.item() * chunk_size

                if telemetry is not None and getattr(telemetry, "forget_gates", None) is not None:
                    chunk_variance = telemetry.forget_gates.var(dim=1).mean().item()
                    epoch_fg_variance += chunk_variance * chunk_size

                total_samples += chunk_size

        metrics = {"val_loss": epoch_loss / total_samples}
        if epoch_fg_variance > 0.0:
            metrics["val_fg_variance"] = epoch_fg_variance / total_samples

        return metrics
