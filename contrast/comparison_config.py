"""Editable defaults for the official-source missing-view benchmark."""

# The program first checks this path relative to run_missing_comparisons.py.
# It must contain labels_tr.csv, labels_te.csv and 1..3_tr/te.csv.
DATA_FOLDER = "ROSMAP"
DATASET_NAME = "ROSMAP"       # e.g. "BRCA"; None uses the folder name
OUTPUT_DIR = "comparison_results/ROSMAP"

# Author repositories.  Recent baselines are imported directly from these
# folders; their model/loss source files are hashed into the experiment manifest.
DIMV_REPO = (r"D:\对比算法\机器学习模型在缺失数据下的可解释性"
             r"\DIMVImputation-main\DIMVImputation-main")
CLCLSA_REPO = r"D:\CLCLSA-main\CLCLSA-main"
TMONET_REPO = r"D:\TMO-Net-master\TMO-Net-master"
INTEGRAO_REPO = r"D:\IntegrAO-main\IntegrAO-main"
MOGEDN_REPO = r"D:\MOGEDN-master\MOGEDN-master"

# The proposed model is imported at model/loss level only.  Relative paths are
# resolved from run_missing_comparisons.py, so this points to D:\EEEEEE\C2.py
# when the comparison runner is stored in D:\EEEEEE\contrast\.
RUSI_MODEL_FILE = r"..\C2.py"

# Fixed manuscript settings for the proposed method.  Adjust these here when a
# dataset uses a pre-declared setting (for example, a dataset-specific dropout
# or lambda_al); do not tune them on outer-test results.
RUSI_PARAMS = {
    "hidden_dims": [128],
    "prediction_dims": [64, 32],
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 50,
    "dropout": 0.2,
    "temperature": 0.2,
    "lambda_cil": 1.0,
    "contrastive_update_interval": 5,
    "lambda_al": 0.1,
    "lambda_imputation": 0.2,
    "cross_sample_k": 10,
    "knn_base_temperature": 1.0,
    "router_temperature": 1.0,
    "router_update_interval": 10,
    "support_mode": "masked",
    "use_cross_sample_impute": True,
    "use_prototype_bank": True,
    "use_action_router": True,
}

# Set a method to False to disable it.  Command-line --methods overrides this.
METHOD_SWITCHES = {
    # Proposed method (included in the same paired protocol)
    "rusi": False,

    # Five classic comparators
    "zero_mask": True,       # direct: zero fill plus missing-view mask
    "mean": True,            # imputation
    "knn": True,             # imputation
    "mice": True,            # imputation
    "svd": True,             # fast low-rank SVD imputation

    # Five recent comparators
    "dimv": False,            # imputation, 2023
    "clclsa": True,          # direct incomplete-view handling, 2024
    "tmonet": True,          # direct incomplete-view handling, 2024
    "integrao": True,        # direct incomplete-view handling, 2025
    "mogedn": True,          # direct incomplete-view handling, 2025
}

# --missing-rate or --missing-rates overrides this list for one run.
MISSING_RATES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
