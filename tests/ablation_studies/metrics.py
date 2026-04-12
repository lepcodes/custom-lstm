import mlflow

# 1. Pull the entire experiment into a Pandas DataFrame
df = mlflow.search_runs(experiment_names=["Thesis_Ablation_Phase_1"])

# 2. Filter down to the exact metrics and parameters you care about
# (e.g., tags.architecture, metrics.test_mse, metrics.train_mse)
summary_df = df[["params.architecture", "params.dataset_name", "metrics.train_loss", "metrics.val_loss"]]

# 3. Pivot the table so Architectures are rows and Datasets are columns
pivot_df = summary_df.pivot(index="params.architecture", columns="params.dataset_name")

# 4. Generate the raw LaTeX string automatically
latex_table = pivot_df.to_latex(float_format="%.5f")

print(latex_table)
