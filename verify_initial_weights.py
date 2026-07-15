import torch
import yaml
from tests.ablation_studies.train import seed_everything
from custom_lstm.models.registry import ModelRegistry
import tests.ablation_studies.model_setup  # Registers the models

def get_initial_model(config_path):
    with open(config_path) as f:
        sweep_cfg = yaml.safe_load(f)
    
    # Extract the first grid combination (which defines the architecture size)
    grid = sweep_cfg["grid"]
    combo = {k: v[0] if isinstance(v, list) else v for k, v in grid.items()}
    
    arch = sweep_cfg["architecture"]
    data_mode = sweep_cfg["data_mode"]
    
    # Replicate `sweep.py` model_kwargs construction
    model_kwargs = {"output_size": sweep_cfg.get("output_size", 1)}
    window_size = combo.get("window_size", 15)
    model_kwargs["input_size"] = window_size if data_mode == "win" else 1
    
    for key in ["hidden_size", "hidden_layers", "forget_gate_layers"]:
        if key in combo:
            model_kwargs[key] = combo[key]
            
    if "no_recurrence" in arch:
        model_kwargs["sever_recurrence"] = True
        
    # --- 1. RESET SEED ---
    # This is what guarantees identical weights
    seed_everything(42)
    
    # --- 2. BUILD MODEL ---
    # Because no RNG is consumed between the seed reset and here, 
    # the matrices will be initialized identically.
    model = ModelRegistry.build(arch, **model_kwargs)
    return model

if __name__ == "__main__":
    mse_config_path = "tests/ablation_studies/search/phase2_b_ablation/sweep_lstm_custom_windowed_mse.yaml"
    ewacf_config_path = "tests/ablation_studies/search/phase2_b_ablation/sweep_lstm_custom_windowed_ewacf_broadcast.yaml"
    
    print("Building MSE Baseline Model...")
    model_mse = get_initial_model(mse_config_path)
    
    print("Building EW-ACF Ablation Model...")
    model_ewacf = get_initial_model(ewacf_config_path)
    
    print("\nComparing Weights...")
    all_match = True
    for (name1, param1), (name2, param2) in zip(model_mse.named_parameters(), model_ewacf.named_parameters()):
        if name1 != name2:
            print(f"❌ Parameter name mismatch: {name1} vs {name2}")
            all_match = False
            break
            
        # Check if the tensors are exactly equal
        is_equal = torch.equal(param1.data, param2.data)
        
        if is_equal:
            print(f"  [Match] {name1:<20} | Shape: {list(param1.shape)}")
        else:
            print(f"❌ [MISMATCH] {name1}")
            all_match = False
            
    print("-" * 50)
    if all_match:
        print("✅ SUCCESS: All initial weights are perfectly identical byte-for-byte!")
    else:
        print("❌ FAILED: Weights differ.")
