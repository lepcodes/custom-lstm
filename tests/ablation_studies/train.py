import argparse
import random

import mlflow
import numpy as np
import torch
import torch.optim as optim

from custom_lstm.models.registry import ModelRegistry
from tests.ablation_studies.callbacks import MLflowCallback
from tests.ablation_studies.config import ExperimentConfig
from tests.ablation_studies.data_loader import load_data
from tests.ablation_studies.factory import build_trainer
import tests.ablation_studies.model_setup  # noqa: F401 — triggers model registration


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(config: ExperimentConfig, patience: int = None) -> dict:
    """
    Runs a single experiment within an active MLflow experiment.
    Returns dict with 'best_val_loss' and 'run_id'.
    """
    run_name = config.run_name or f"{config.architecture}_{config.dataset_name}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(config.model_dump())

        print(f"Loading data from {config.data_path}...")
        data_dict, scaler = load_data(config.data_path, window_size=config.window_size)

        mode = config.data_mode
        mode_train = data_dict.train.get_by_mode(mode)
        mode_val = data_dict.val.get_by_mode(mode)
        X_train, y_train = mode_train.X, mode_train.Y
        X_val, y_val = mode_val.X, mode_val.Y

        print(f"Building {config.architecture}...")
        model = ModelRegistry.build(config.architecture, **config.model_kwargs)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {n_params}")

        device = torch.device("cpu")
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        callback = MLflowCallback()

        print(config.describe_pipeline())
        trainer = build_trainer(config, model, optimizer, device, callback=callback)

        best_val_loss = trainer.train(
            epochs=config.epochs,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            patience=patience,
        )

        mlflow.log_metric("best_val_loss", best_val_loss)

        return {"best_val_loss": best_val_loss, "run_id": run.info.run_id}


def main():
    seed_everything(42)
    parser = argparse.ArgumentParser(description="Master Thesis: Ablation Study Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML experiment config")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    print(config.describe_pipeline())
    mlflow.set_experiment(config.experiment_name)
    run_experiment(config)


if __name__ == "__main__":
    main()
