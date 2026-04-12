import torch
import torch.nn as nn

from custom_lstm.models.base_model import AblationModel
from custom_lstm.training.base import BaseTrainerStrategy, TrainingCallback


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
