"""
Training callbacks for MLOps integration.

These implement the TrainingCallback protocol defined in custom_lstm.training.trainer,
keeping MLflow (and any other tracking tool) out of the core PyTorch library.
"""

import mlflow


class MLflowCallback:
    """Logs per-epoch metrics to an active MLflow run."""

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float) -> None:
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
