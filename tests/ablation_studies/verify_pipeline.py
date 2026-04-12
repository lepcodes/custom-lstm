import torch

from custom_lstm.models.lstm_vanilla_stateful import LSTMVanillaStateful
from custom_lstm.models.mlp import MLP
from custom_lstm.models.registry import ModelRegistry
from custom_lstm.training.trainer import TBPTTTrainerStrategy

import tests.ablation_studies.registry  # noqa: F401 — triggers model registration


def verify_pipeline():
    print(f"Registered Models: {ModelRegistry._registry.keys()}")

    # 1. Test Instantiation from Registry
    mlp_model = ModelRegistry.build("simple_mlp", input_size=10, output_size=1, hidden_layers=[16, 16])
    lstm_model = ModelRegistry.build("lstm_vanilla_pure", input_size=1, hidden_size=32, output_size=1)

    assert isinstance(mlp_model, MLP)
    assert isinstance(lstm_model, LSTMVanillaStateful)
    print("Models successfully instantiated from Registry.")

    # 2. Test Dummy Execution via Trainer Strategy (no MLflow needed)
    device = torch.device("cpu")

    # Dummy sequences: Batch=2, SeqLength=50, Features=1
    X_train = torch.randn(2, 50, 1)
    y_train = torch.randn(2, 50, 1)
    X_val = torch.randn(2, 20, 1)
    y_val = torch.randn(2, 20, 1)

    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    trainer = TBPTTTrainerStrategy(model=lstm_model, optimizer=optimizer, criterion=criterion, device=device, bptt_steps=10)

    print("Starting mock training loop via Strategy...")
    trainer.train(epochs=2, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    print("TBPTTTrainerStrategy successfully executed training loop.")
    print("Pipeline verification completed successfully.")


if __name__ == "__main__":
    verify_pipeline()
