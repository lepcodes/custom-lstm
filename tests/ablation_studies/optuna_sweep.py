"""
Bayesian Hyperparameter Search using Optuna.
Usage: python optuna_sweep.py --config search/optuna/optuna_lstm_custom_ewacf_broadcast.yaml
"""

import argparse
from pathlib import Path

import mlflow
import optuna
import yaml

from tests.ablation_studies.config import ExperimentConfig
from tests.ablation_studies.train import run_experiment, seed_everything


def objective(trial: optuna.Trial, sweep_cfg):
    """
    Optuna objective function.
    """
    arch = sweep_cfg["architecture"]
    data_mode = sweep_cfg["data_mode"]
    loss_type = sweep_cfg.get("loss_type", "mse")

    # Fixed parameters
    output_size = sweep_cfg.get("output_size", 1)
    epochs = sweep_cfg.get("epochs", 200)
    patience = sweep_cfg.get("patience", 30)
    data_path = sweep_cfg["data_path"]

    # Sample from search space
    search_space = sweep_cfg.get("search_space", {})

    sampled_params = {}
    for param_name, param_config in search_space.items():
        param_type = param_config.get("type")
        if param_type == "int":
            sampled_params[param_name] = trial.suggest_int(param_name, param_config["low"], param_config["high"], step=param_config.get("step", 1))
        elif param_type == "float":
            sampled_params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"], log=param_config.get("log", False))
        elif param_type == "categorical":
            choices = param_config["choices"]
            choice_idx = trial.suggest_categorical(f"{param_name}_idx", range(len(choices)))
            sampled_params[param_name] = choices[choice_idx]

    # Build model_kwargs
    window_size = sampled_params.get("window_size", sweep_cfg.get("window_size", 10))
    input_size = window_size if data_mode == "win" else 1

    model_kwargs = {"input_size": input_size, "output_size": output_size}
    if "hidden_size" in sampled_params:
        model_kwargs["hidden_size"] = sampled_params["hidden_size"]
    if "hidden_layers" in sampled_params:
        model_kwargs["hidden_layers"] = sampled_params["hidden_layers"]
    if "forget_gate_layers" in sampled_params:
        model_kwargs["forget_gate_layers"] = sampled_params["forget_gate_layers"]

    if "no_recurrence" in arch:
        model_kwargs["sever_recurrence"] = True

    # ── Build descriptive run name (Arch + Loss, no hyperparams) ───────────
    run_name = f"Trial_{trial.number}_{arch}_{loss_type}"

    # ── Build Config ────────────────────────────────────────────────────────
    config = ExperimentConfig(
        data_path=data_path,
        architecture=arch,
        data_mode=data_mode,
        model_kwargs=model_kwargs,
        experiment_name=sweep_cfg.get("experiment_name", "Optuna_Ablation"),
        run_name=run_name,
        lr=sampled_params.get("lr", sweep_cfg.get("lr", 0.001)),
        epochs=epochs,
        window_size=window_size,
        bptt_steps=sampled_params.get("bptt_steps", sweep_cfg.get("bptt_steps", 50)),
        loss_type=loss_type,
        ewacf_alpha=sampled_params.get("ewacf_alpha", sweep_cfg.get("ewacf_alpha", 0.5)),
        ewacf_lambda=sampled_params.get("ewacf_lambda", sweep_cfg.get("ewacf_lambda", 0.5)),
        ewacf_lag=sampled_params.get("ewacf_lag", sweep_cfg.get("ewacf_lag", 1)),
        ewacf_threshold=sampled_params.get("ewacf_threshold", sweep_cfg.get("ewacf_threshold", 0.1)),
    )

    seed_everything(42)

    print(f"\n{'─' * 70}")
    print(f"  Trial {trial.number}: {sampled_params}")
    print(f"{'─' * 70}")

    result = run_experiment(config, patience=patience, optuna_trial=trial)

    return result["best_val_loss"]


def main():
    parser = argparse.ArgumentParser(description="Optuna Bayesian Hyperparameter Search")
    parser.add_argument("--config", type=str, required=True, help="Path to Optuna sweep YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        sweep_cfg = yaml.safe_load(f)

    dataset_stem = Path(sweep_cfg["data_path"]).stem
    experiment_name = f"{sweep_cfg.get('experiment_name', 'Optuna_Ablation')}_{dataset_stem}"

    mlflow.set_experiment(experiment_name)

    study = optuna.create_study(
        study_name=experiment_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50, interval_steps=5),
        # pruner=optuna.pruners.NopPruner(),
    )

    n_trials = sweep_cfg.get("n_trials", 40)

    print(f"\n{'=' * 70}")
    print(f"  OPTUNA SWEEP START: {sweep_cfg['architecture']} on {dataset_stem}")
    print(f"  Trials:  {n_trials}")
    print(f"  Experiment Name: {experiment_name}")
    print(f"{'=' * 70}\n")

    study.optimize(lambda trial: objective(trial, sweep_cfg), n_trials=n_trials)

    # ── Print summary (Mirroring sweep.py style) ───────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  OPTUNA SWEEP COMPLETE: {sweep_cfg['architecture']} on {dataset_stem}")
    print(f"{'=' * 70}")
    print("\n  Results (sorted by val_loss):\n")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for t in sorted(completed_trials, key=lambda x: x.value if x.value is not None else float("inf")):
        marker = "  ★ BEST" if t.number == study.best_trial.number else ""
        print(f"    Val MSE: {t.value:.6f}  |  Trial {t.number:02d}: {t.params}{marker}")

    print(f"\n{'=' * 70}")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Val Loss: {study.best_trial.value:.6f}")
    print("Best Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
