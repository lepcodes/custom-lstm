"""
Hyperparameter Grid Search Sweep for the Ablation Study.
Usage: python sweep.py --config configs/sweep_lstm_vanilla_pure.yaml
"""

import argparse
import itertools
from pathlib import Path

import mlflow
import yaml

from tests.ablation_studies.train import run_experiment, seed_everything
from tests.ablation_studies.config import ExperimentConfig

# Keys from the grid that map to model_kwargs (everything else maps to ExperimentConfig)
MODEL_KWARGS_KEYS = {"hidden_size", "hidden_layers", "forget_gate_layers"}


def build_experiment_config(sweep_cfg: dict, grid_combo: dict) -> ExperimentConfig:
    """Builds an ExperimentConfig from the sweep's fixed values + one grid combination."""
    arch = sweep_cfg["architecture"]
    data_mode = sweep_cfg["data_mode"]

    # ── Build model_kwargs ─────────────────────────────────────────────────
    model_kwargs = {"output_size": sweep_cfg.get("output_size", 1)}

    window_size = grid_combo.get("window_size", sweep_cfg.get("window_size", 10))
    model_kwargs["input_size"] = window_size if data_mode == "win" else 1

    for key in MODEL_KWARGS_KEYS:
        if key in grid_combo:
            model_kwargs[key] = grid_combo[key]
        elif key in sweep_cfg:
            model_kwargs[key] = sweep_cfg[key]

    if "no_recurrence" in arch:
        model_kwargs["sever_recurrence"] = True

    # ── Build descriptive run name ─────────────────────────────────────────
    tag_map = {"hidden_size": "hs", "hidden_layers": "hl", "forget_gate_layers": "fgl", "window_size": "ws", "bptt_steps": "bptt", "lr": "lr"}
    parts = [arch]
    for k, v in grid_combo.items():
        tag = tag_map.get(k, k)
        if k == "forget_gate_layers":
            val_str = "x".join(str(x) for x in v) if v else "std"
        else:
            val_str = "x".join(str(x) for x in v) if isinstance(v, list) else str(v)
        parts.append(f"{tag}{val_str}")
    run_name = "_".join(parts)

    # ── EW-ACF loss configuration (from fixed sweep values or grid) ────────
    loss_type = grid_combo.get("loss_type", sweep_cfg.get("loss_type", "mse"))
    ewacf_alpha = grid_combo.get("ewacf_alpha", sweep_cfg.get("ewacf_alpha", 0.5))
    ewacf_lambda = grid_combo.get("ewacf_lambda", sweep_cfg.get("ewacf_lambda", 0.5))
    ewacf_lag = grid_combo.get("ewacf_lag", sweep_cfg.get("ewacf_lag", 1))
    ewacf_threshold = grid_combo.get("ewacf_threshold", sweep_cfg.get("ewacf_threshold", 0.1))

    # ── Build config ───────────────────────────────────────────────────────
    return ExperimentConfig(
        data_path=sweep_cfg["data_path"],
        architecture=arch,
        data_mode=data_mode,
        model_kwargs=model_kwargs,
        run_name=run_name,
        lr=grid_combo.get("lr", sweep_cfg.get("lr", 0.001)),
        epochs=sweep_cfg.get("epochs", 200),
        window_size=window_size,
        bptt_steps=grid_combo.get("bptt_steps", sweep_cfg.get("bptt_steps", 50)),
        loss_type=loss_type,
        ewacf_alpha=ewacf_alpha,
        ewacf_lambda=ewacf_lambda,
        ewacf_lag=ewacf_lag,
        ewacf_threshold=ewacf_threshold,
    )


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Grid Search")
    parser.add_argument("--config", type=str, required=False, help="Path to sweep YAML config")
    args = parser.parse_args()
    if args.config is None:
        args.config = "C:/Users/Luis/Documents/ML-AI-Projects/custom-lstm/tests/ablation_studies/search/sweep_simple_mlp.yaml"
    with open(args.config) as f:
        sweep_cfg = yaml.safe_load(f)

    grid = sweep_cfg.pop("grid")
    patience = sweep_cfg.pop("patience", 20)

    # ── Build cartesian product of grid values ─────────────────────────────
    grid_keys = list(grid.keys())
    grid_values = [v if isinstance(v, list) else [v] for v in grid.values()]
    combinations = list(itertools.product(*grid_values))

    dataset_stem = Path(sweep_cfg["data_path"]).stem
    arch = sweep_cfg["architecture"]

    print(f"\n{'=' * 70}")
    print(f"  SWEEP: {arch} on {dataset_stem}")
    print(f"  Grid:  {grid}")
    print(f"  Runs:  {len(combinations)}  |  Patience: {patience}")
    print(f"{'=' * 70}")

    # ── Resolve Experiment Name ─────────────────────────────────────────────
    # Build a dummy config with default values just to resolve the experiment name
    # which is consistent for the whole sweep.
    dummy_config = ExperimentConfig(
        data_path=sweep_cfg["data_path"],
        architecture=arch,
        data_mode=sweep_cfg["data_mode"],
        experiment_name=sweep_cfg.get("experiment_name", "Thesis_Ablation")
    )
    mlflow.set_experiment(dummy_config.experiment_name)

    best_val_loss = float("inf")
    best_combo = None
    best_run_id = None
    results = []

    for i, combo_values in enumerate(combinations, 1):
        combo = dict(zip(grid_keys, combo_values))

        print(f"\n{'─' * 70}")
        print(f"  Run {i}/{len(combinations)}: {combo}")
        print(f"{'─' * 70}")

        seed_everything(42)
        config = build_experiment_config(sweep_cfg, combo)
        result = run_experiment(config, patience=patience)

        val_loss = result["best_val_loss"]
        run_id = result["run_id"]
        results.append({"combo": combo, "val_loss": val_loss, "run_id": run_id})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_combo = combo
            best_run_id = run_id

    # ── Tag best run in MLflow ─────────────────────────────────────────────
    if best_run_id:
        client = mlflow.MlflowClient()
        client.set_tag(best_run_id, "best_in_sweep", "true")

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SWEEP COMPLETE: {arch} on {dataset_stem}")
    print(f"{'=' * 70}")
    print("\n  Results (sorted by val_loss):\n")
    for r in sorted(results, key=lambda x: x["val_loss"]):
        marker = "  ★ BEST" if r["combo"] == best_combo else ""
        print(f"    Val MSE: {r['val_loss']:.6f}  |  {r['combo']}{marker}")
    print()


if __name__ == "__main__":
    main()
