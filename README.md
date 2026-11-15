# RUSI

**Adaptive Selective Imputation with Reliability–Utility Routing for Incomplete Multi-Omics Classification**

![RUSI architecture for selective imputation and multi-omics classification](./RUSI.png)

RUSI classifies patients from three omics assays when one or two assays are missing. For each patient, it makes two decisions: how to estimate a missing assay's latent representation, and whether including that estimate helps classification. A patient with two missing assays can therefore receive a prediction based on the observed assay alone or on the observed assay and one recovered representation.

## Method

Each assay has its own feature-weighting layer, encoder, and auxiliary classifier. The encoders map assays with different feature counts into a common latent space. For a missing assay, RUSI builds candidate representations from cross-omics predictors, neighboring training samples, and class prototypes. A fourth expert contributes a zero vector so that the model can reduce the contribution of uncertain estimates.

A shared router learns expert weights from reconstruction errors under simulated masking. Its action head learns from the classification losses of the available choices: skip imputation or recover one of the missing views. At inference, the classifier combines the observed and selected representations using the learned routing policy.

## Quick start

Run the commands from the directory containing `run_nested_cv.py`. The CPU examples below were tested on Windows with Python 3.9.13 and the following packages:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.24.4 scikit-learn==1.0.2 tqdm==4.64.1
```

To check the full workflow, run a two-by-two nested cross-validation on ROSMAP with one training epoch and 20% of patients missing at least one assay:

```bash
python run_nested_cv.py --data_folder ROSMAP --missing_rate 0.2 --outer_folds 2 --inner_folds 2 --num_epoch 1 --warmup_epochs 0 --test_interval 1 --device cpu --use_amp false
```

This produces one held-out prediction for each of the 351 ROSMAP samples. To run BRCA at the same missing-patient rate with the default ten outer folds, five inner folds, and 1,200-epoch training budget on a CUDA-enabled installation:

```bash
python run_nested_cv.py --data_folder BRCA --missing_rate 0.2 --device cuda
```

## Data and simulated missingness

Each dataset directory contains three pairs of headerless numerical feature matrices, `1_tr.csv` through `3_tr.csv` and their `_te.csv` counterparts, plus `labels_tr.csv` and `labels_te.csv`. Rows represent patients and columns represent features. Within each source split, all three matrices and the integer labels must use the same patient order. Each assay must use the same feature order across the two source splits. The loader expects finite values and generates missingness using a separate binary mask.

| Dataset | Source `_tr` rows | Source `_te` rows | Patients in nested CV | Classes |
| --- | ---: | ---: | ---: | ---: |
| BRCA | 612 | 263 | 875 | 5 |
| GBM | 219 | 55 | 274 | 4 |
| KIPAN | 460 | 198 | 658 | 3 |
| LGG | 357 | 153 | 510 | 2 |
| ROSMAP | 245 | 106 | 351 | 2 |

The BRCA task uses five PAM50 breast cancer subtypes. ROSMAP distinguishes Alzheimer's disease from normal controls. See the [BRCA label definitions](BRCA/README.md) and [ROSMAP label definitions](ROSMAP/README.md).

In the default `sample` mode, `--missing_rate 0.2 --two_missing_ratio 0.5` makes approximately 20% of patients incomplete. Half of those patients lose one assay and half lose two, subject to rounding. In `entry` mode, the rate applies to all patient–assay pairs. Its maximum is two thirds because each patient retains at least one assay. The default missing rate of zero uses complete data.

## Evaluation protocol

1. Merge the supplied `_tr` and `_te` rows for each assay. Generate one cohort-wide availability mask for the requested missing rate and reuse it across folds.
2. Split the merged cohort into stratified outer folds. Within each outer training partition, use stratified inner folds to select the epoch with the highest mean validation accuracy. Ties select the earliest epoch.
3. Fit each assay's min–max scaling on its observed rows in the current training partition. The default `--support_mode masked` also restricts the reference data used for neighbor and prototype estimates to the masked training inputs.
4. Train a fresh model on the full outer training partition for the selected number of epochs, then score its held-out fold. Each patient appears in an outer test fold exactly once.

Binary tasks report accuracy, F1, and ROC AUC. Multiclass tasks report accuracy, weighted F1, and macro-F1. The results distinguish the mean and standard deviation across outer folds from metrics computed on the pooled out-of-fold probabilities.

## Experiment outputs

The main runner writes an experiment directory under `./exp`. Change the parent directory with `--exp`.

| File | Contents |
| --- | --- |
| `config.json`, `original_source_indices.npz` | Run settings and the correspondence between merged rows and the original splits. |
| `fixed_merged_mask.npy` | Patient-by-assay availability mask for a simulated missingness run. |
| `outer_fold_test_metrics.csv` | Held-out metrics for each outer fold. |
| `nested_oof_probabilities.npy` | One row of predicted class probabilities per patient, in merged cohort order. |
| `nested_cv_summary.csv`, `summary.json` | Aggregate outer-fold results and pooled metrics. |

Each outer-fold directory also records the inner validation results used for epoch selection, the final model weights, and the held-out class probabilities.

## Comparison experiments

The [comparison runner](contrast/run_missing_comparisons.py) evaluates selected methods with the same folds and missingness masks. For a short CPU run with the training-fold mean imputation baseline:

```bash
python contrast/run_missing_comparisons.py --data-folder ROSMAP --methods mean --missing-rate 0.2 --outer-folds 2 --inner-folds 2 --epochs 1 --warmup-epochs 0 --rusi-warmup-epochs 0 --eval-interval 1 --device cpu --output-dir comparison_results/ROSMAP
```

The benchmark uses hyphenated option names and has its own [configuration file](contrast/comparison_config.py). Several optional methods load external implementations. Set the corresponding repository paths before enabling them. The benchmark's default RUSI model path points to `C2.py` from an earlier layout. The standalone RUSI workflow above uses the included `model.py` directly.

## Code guide

| File | Role |
| --- | --- |
| [run_nested_cv.py](run_nested_cv.py) | Command-line entry point. Run with `--help` for the full option list. |
| [data_utils.py](data_utils.py) | Loads the three assays, validates labels, and simulates missing views. |
| [model.py](model.py) and [modules.py](modules.py) | Implement the RUSI model (class `CLUECL3`), candidate estimators, router, and classifier. |
| [trainer.py](trainer.py) | Runs nested cross-validation, selects training epochs, and saves predictions. |
| [utils.py](utils.py) | Provides random seed control, contrastive loss, and numerical utilities. |

