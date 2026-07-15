import torch
import torch.nn as nn

from custom_lstm.models.base_model import AblationModel
from custom_lstm.training.base import BaseTrainerStrategy, TrainingCallback
from custom_lstm.utils import EWACFEngine


class EWACFTBPTTTrainerStrategy(BaseTrainerStrategy):
    """
    TBPTT Trainer Strategy for EW-ACF regularized training.
    The trainer strategy OWNS the EWACFEngine to compute the autocorrelation signal.
    """

    def __init__(
        self,
        model: AblationModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        bptt_steps: int,
        lambda_: float = 0.5,
        lags: list[int] = [1],
        callback: TrainingCallback | None = None,
    ):
        super().__init__(model, optimizer, criterion, device, callback)
        self.bptt_steps = bptt_steps
        self.acf_engine = EWACFEngine(lambda_=lambda_, lags=lags)
        self.acf_engine.to(device)

    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor, **kwargs):
        self.model.train()
        self.model.reset_state()
        self.acf_engine.reset_state()

        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_penalty = 0.0
        total_samples = 0
        total_steps = X_train.size(1)

        for i in range(0, total_steps, self.bptt_steps):
            X_batch = X_train[:, i : i + self.bptt_steps, :]
            y_batch = y_train[:, i : i + self.bptt_steps, :]
            chunk_size = X_batch.size(1)

            # 1. Compute Autocorrelation Signal
            autocorrelation = self.acf_engine(X_batch)

            # 2. Forward pass
            y_pred, telemetry = self.model(X_batch)

            # 3. Compute Loss
            total_loss, mse_val, penalty_val = self.criterion(y_pred, y_batch, telemetry.forget_gates, autocorrelation)

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

        return {
            "train_loss": epoch_loss / total_samples,
            "train_mse": epoch_mse / total_samples,
            "train_penalty": epoch_penalty / total_samples,
        }

    def validate_epoch(self, X_val: torch.Tensor, y_val: torch.Tensor):
        """Validate with chunking and ACF computation."""
        self.model.eval()
        self.model.reset_state()
        self.acf_engine.reset_state()

        epoch_mse = 0.0
        epoch_penalty = 0.0
        epoch_total = 0.0
        epoch_fg_variance = 0.0

        total_samples = 0
        total_steps = X_val.size(1)

        with torch.no_grad():
            for i in range(0, total_steps, self.bptt_steps):
                X_batch = X_val[:, i : i + self.bptt_steps, :]
                y_batch = y_val[:, i : i + self.bptt_steps, :]
                chunk_size = X_batch.size(1)

                autocorrelation = self.acf_engine(X_batch)
                y_pred, telemetry = self.model(X_batch)
                total_loss, mse_val, penalty_val = self.criterion(y_pred, y_batch, telemetry.forget_gates, autocorrelation)

                epoch_mse += mse_val.item() * chunk_size
                epoch_penalty += penalty_val.item() * chunk_size
                epoch_total += total_loss.item() * chunk_size

                # Safely accumulate the temporal variance per chunk
                if telemetry is not None and getattr(telemetry, "forget_gates", None) is not None:
                    chunk_variance = telemetry.forget_gates.var(dim=1).mean().item()
                    epoch_fg_variance += chunk_variance * chunk_size

                total_samples += chunk_size

        metrics = {
            "val_loss": epoch_mse / total_samples,  # Pure MSE for early stopping
            "val_mse": epoch_mse / total_samples,
            "val_penalty": epoch_penalty / total_samples,
            "val_total": epoch_total / total_samples,
        }

        # Only add variance to metrics if it was calculated
        if epoch_fg_variance > 0.0:
            metrics["val_fg_variance"] = epoch_fg_variance / total_samples

        return metrics
