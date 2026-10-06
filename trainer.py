"""Training, validation, nested cross-validation and result writing."""

import csv
import json
import math
import os
import pprint
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from data_utils import apply_existing_mask, describe_mask, get_mask, load_raw_data
from model import CLUECL3
from utils import set_random_seed


class CLCLSA_Trainer(object):

    def __init__(self, params):
        self.params = dict(params)
        self.device_obj = torch.device(self.params['device'])
        if (self.device_obj.type == 'cuda'
                and not torch.cuda.is_available()):
            raise RuntimeError(
                "CUDA was requested but is not available; use --device=cpu")
        if not os.path.isdir(self.params['data_folder']):
            raise ValueError(
                f"data_folder does not exist: "
                f"{self.params['data_folder']!r}")
        missing_rate = self.params.get('missing_rate', 0.0)
        if not (0.0 <= missing_rate < 1.0):
            raise ValueError("missing_rate must be in [0, 1)")
        rate_mode = self.params.get('missing_rate_mode', 'sample')
        if rate_mode not in {'sample', 'entry'}:
            raise ValueError("missing_rate_mode must be 'sample' or 'entry'")
        two_ratio = self.params.get('two_missing_ratio', 0.5)
        if not (0.0 <= two_ratio <= 1.0):
            raise ValueError("two_missing_ratio must be in [0, 1]")
        if rate_mode == 'entry' and missing_rate > 2.0 / 3.0:
            raise ValueError(
                "entry missing_rate cannot exceed 2/3 when every patient "
                "must retain at least one of three views")
        if self.params.get('support_mode', 'masked') not in {'masked', 'complete'}:
            raise ValueError("support_mode must be 'masked' or 'complete'")
        if self.params.get('cross_sample_k', 5) <= 0:
            raise ValueError("cross_sample_k must be a positive integer")
        if self.params.get('knn_base_temperature', 0.2) <= 0:
            raise ValueError("knn_base_temperature must be positive")
        if self.params.get('router_update_interval', 10) <= 0:
            raise ValueError("router_update_interval must be a positive integer")
        if self.params.get('contrastive_update_interval', 5) <= 0:
            raise ValueError("contrastive_update_interval must be a positive integer")
        if self.params.get('outer_folds', 10) < 2:
            raise ValueError("outer_folds must be at least 2")
        if self.params.get('inner_folds', 5) < 2:
            raise ValueError("inner_folds must be at least 2")
        if self.params.get('num_epoch', 0) <= 0:
            raise ValueError("num_epoch must be a positive integer")
        if self.params.get('test_interval', 0) <= 0:
            raise ValueError("test_interval must be a positive integer")
        if self.params.get('warmup_epochs', 0) < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if (self.params.get('warmup_epochs', 0)
                > self.params['num_epoch']):
            raise ValueError(
                "warmup_epochs cannot exceed num_epoch")
        if self.params.get('lr', 1e-4) <= 0:
            raise ValueError("lr must be positive")
        if not (0.0 <= self.params.get('dropout', 0.5) < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if self.params.get('max_grad_norm', 1.0) <= 0:
            raise ValueError("max_grad_norm must be positive")
        hidden_dim = self.params.get('hidden_dim', [])
        if not hidden_dim or any(dim <= 0 for dim in hidden_dim):
            raise ValueError("hidden_dim must contain positive integers")
        if hidden_dim[0] < 2:
            raise ValueError(
                "hidden_dim[0] must be at least 2")
        prediction = self.params.get('prediction', {})
        if set(prediction) != {0, 1, 2}:
            raise ValueError("prediction must define branches 0, 1, and 2")
        if any(not dims or any(dim <= 0 for dim in dims)
               for dims in prediction.values()):
            raise ValueError(
                "every prediction branch must contain positive hidden sizes")
        for name in ('lambda_al', 'lambda_imputation', 'lambda_cil'):
            if self.params.get(name, 0.0) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.params.get('temperature', 0.2) < 0:
            raise ValueError("temperature must be non-negative; 0 disables InfoNCE")
        if self.params.get('router_temperature', 0.5) <= 0:
            raise ValueError("router_temperature must be positive")
        if self.device_obj.type == 'cuda':
            # Allows TensorFloat-32 kernels for any FP32 matmul that remains
            # outside AMP.  This changes only low-order rounding, not structure.
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
        self.amp_enabled = (
            self.params.get('use_amp', False)
            and self.device_obj.type == 'cuda')
        self.__init_dataset__()

    def __init_dataset__(self):
        """Merge the supplied train/test files and build outer CV splits."""
        (train_views, test_views, labels_train, labels_test,
         class_values) = load_raw_data(self.params['data_folder'])
        self.original_train_size = int(labels_train.size)
        self.original_test_size = int(labels_test.size)
        self.data_all_raw = [
            np.concatenate((train_view, test_view), axis=0)
            for train_view, test_view in zip(train_views, test_views)
        ]
        self.labels_all_np = np.concatenate(
            (labels_train, labels_test)).astype(np.int64, copy=False)
        self.params['class_values'] = [int(value) for value in class_values]
        self.dim_list = [view.shape[1] for view in self.data_all_raw]
        self.num_class = int(class_values.size)

        outer_folds = self.params['outer_folds']
        class_counts = np.bincount(
            self.labels_all_np, minlength=self.num_class)
        if class_counts.min() < outer_folds:
            raise ValueError(
                "nested outer CV requires every class to contain at least "
                f"outer_folds={outer_folds} samples; class counts are "
                f"{class_counts.tolist()}")

        outer_splitter = StratifiedKFold(
            n_splits=outer_folds, shuffle=True,
            random_state=self.params['seed'])
        self.outer_indices = list(outer_splitter.split(
            np.zeros(self.labels_all_np.size), self.labels_all_np))

        inner_folds = self.params['inner_folds']
        for outer_id, (outer_train_idx, _) in enumerate(
                self.outer_indices, start=1):
            inner_counts = np.bincount(
                self.labels_all_np[outer_train_idx],
                minlength=self.num_class)
            if inner_counts.min() < inner_folds:
                raise ValueError(
                    f"outer fold {outer_id} cannot run inner_folds="
                    f"{inner_folds}; outer-training class counts are "
                    f"{inner_counts.tolist()}")

        # Generate one reproducible missingness realization for the merged
        # cohort.  Subsetting this mask keeps every patient in the same state
        # whenever it appears in an inner or outer split.
        self.fixed_all_mask = None
        if self.params['missing_rate'] > 0:
            numpy_state = np.random.get_state()
            try:
                np.random.seed(self.params['seed'] + 500_000)
                self.fixed_all_mask = get_mask(
                    3, self.labels_all_np.size,
                    self.params['missing_rate'],
                    rate_mode=self.params['missing_rate_mode'],
                    two_missing_ratio=self.params['two_missing_ratio'])
            finally:
                np.random.set_state(numpy_state)

        print(f"[x] number of classes = {self.num_class}")
        print(f"[x] merged samples = {self.labels_all_np.size} "
              f"(original train={self.original_train_size}, "
              f"original test={self.original_test_size})")
        print(f"[x] strict nested CV: outer={outer_folds}, "
              f"inner={inner_folds}")

    def _initialize_optimization(self, total_epochs, run_seed,
                                 schedule_total_epochs=None):
        """Create a fresh model, optimizer, scheduler and AMP scaler."""
        total_epochs = int(total_epochs)
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        schedule_total_epochs = int(
            total_epochs if schedule_total_epochs is None
            else schedule_total_epochs)
        if schedule_total_epochs < total_epochs:
            raise ValueError(
                "schedule_total_epochs cannot be smaller than total_epochs")
        set_random_seed(int(run_seed))
        self.model = CLUECL3(
            self.dim_list, self.params['hidden_dim'], self.num_class,
            self.params['dropout'], self.params['prediction']
        ).to(self.device_obj)
        optimizer_kwargs = {
            "lr": self.params['lr'],
            "weight_decay": 1e-4,
        }
        if self.device_obj.type == 'cuda':
            try:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), fused=True,
                    **optimizer_kwargs)
            except (TypeError, RuntimeError):
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), **optimizer_kwargs)
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), **optimizer_kwargs)

        warmup_epochs = min(
            int(self.params['warmup_epochs']), schedule_total_epochs)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            progress = (
                (epoch - warmup_epochs)
                / max(schedule_total_epochs - warmup_epochs, 1))
            return 0.5 * (
                1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda)
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if self.amp_enabled else None)

    def _prepare_split(self, train_idx, run_path, run_name,
                       validation_idx=None, test_idx=None):
        """Normalize from train_idx only and materialize one leakage-safe split."""
        sample_count = self.labels_all_np.size

        def validate_indices(name, indices, required):
            if indices is None:
                if required:
                    raise ValueError(f"{name} cannot be None")
                return None
            array = np.asarray(indices)
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"{name} must be a non-empty 1D array")
            if not np.issubdtype(array.dtype, np.integer):
                if (not np.issubdtype(array.dtype, np.number)
                        or not np.isfinite(array).all()
                        or not np.equal(array, np.round(array)).all()):
                    raise TypeError(f"{name} must contain integer indices")
            array = array.astype(np.int64, copy=False)
            if array.min() < 0 or array.max() >= sample_count:
                raise IndexError(
                    f"{name} contains indices outside [0, {sample_count})")
            if np.unique(array).size != array.size:
                raise ValueError(f"{name} contains duplicate indices")
            return array

        train_idx = validate_indices("train_idx", train_idx, required=True)
        validation_idx = validate_indices(
            "validation_idx", validation_idx, required=False)
        test_idx = validate_indices("test_idx", test_idx, required=False)

        named_splits = [
            ("train_idx", train_idx),
            ("validation_idx", validation_idx),
            ("test_idx", test_idx),
        ]
        for left_position, (left_name, left_idx) in enumerate(named_splits):
            if left_idx is None:
                continue
            for right_name, right_idx in named_splits[left_position + 1:]:
                if (right_idx is not None
                        and np.intersect1d(
                            left_idx, right_idx, assume_unique=True).size):
                    raise ValueError(
                        f"{left_name} and {right_name} must be disjoint")
        os.makedirs(run_path, exist_ok=True)

        normalization = {}
        train_views, validation_views, test_views = [], [], []
        eps = 1e-10
        for view_id, raw_view in enumerate(self.data_all_raw, start=1):
            # In missing-view experiments, a hidden training view is not
            # allowed to affect even the preprocessing statistics.  Fit each
            # view's min/max only from rows where that view is observed in the
            # current optimization split.  This also keeps validation and
            # outer-test information completely outside preprocessing.
            if self.fixed_all_mask is None:
                observed_train = np.ones(train_idx.size, dtype=bool)
            else:
                observed_train = (
                    self.fixed_all_mask[train_idx, view_id - 1] > 0.5)
            if not observed_train.any():
                raise ValueError(
                    f"{run_name}: view {view_id} has no observed training "
                    "sample, so leakage-free normalization is impossible")
            fitting_data = raw_view[train_idx[observed_train]]
            train_min = fitting_data.min(axis=0, keepdims=True)
            scale = ((fitting_data - train_min).max(
                axis=0, keepdims=True) + eps)

            def transform(indices):
                values = ((raw_view[indices] - train_min) / scale).astype(
                    np.float32, copy=False)
                return torch.from_numpy(values).to(self.device_obj)

            train_views.append(transform(train_idx))
            if validation_idx is not None:
                validation_views.append(transform(validation_idx))
            if test_idx is not None:
                test_views.append(transform(test_idx))
            normalization[f"view_{view_id}_min"] = train_min
            normalization[f"view_{view_id}_scale"] = scale
            normalization[f"view_{view_id}_observed_train_count"] = (
                np.asarray(observed_train.sum(), dtype=np.int64))

        np.savez(os.path.join(run_path, "normalization.npz"),
                 **normalization)
        index_payload = {"train_indices": train_idx}
        if validation_idx is not None:
            index_payload["validation_indices"] = validation_idx
        if test_idx is not None:
            index_payload["test_indices"] = test_idx
        np.savez(os.path.join(run_path, "indices.npz"), **index_payload)

        labels_train_np = self.labels_all_np[train_idx]
        self.labels_tr_tensor = torch.from_numpy(
            labels_train_np).to(self.device_obj)
        self.data_tr_list = train_views
        self.data_tr_original = (
            tuple(train_views)
            if (self.params['missing_rate'] > 0
                and self.params['support_mode'] == 'complete')
            else None)

        if validation_idx is not None:
            self.labels_val_np = self.labels_all_np[validation_idx]
            self.data_val_list = validation_views
        else:
            self.labels_val_np = None
            self.data_val_list = None
        if test_idx is not None:
            self.labels_test_np = self.labels_all_np[test_idx]
            self.data_test_list = test_views
        else:
            self.labels_test_np = None
            self.data_test_list = None

        if self.fixed_all_mask is not None:
            self.data_tr_list, self.mask_train = apply_existing_mask(
                self.data_tr_list, self.fixed_all_mask[train_idx],
                self.device_obj, split_name=f"{run_name} train mask")
            if validation_idx is not None:
                self.data_val_list, self.mask_val = apply_existing_mask(
                    self.data_val_list,
                    self.fixed_all_mask[validation_idx], self.device_obj,
                    split_name=f"{run_name} validation mask")
            else:
                self.mask_val = None
            if test_idx is not None:
                self.data_test_list, self.mask_test = apply_existing_mask(
                    self.data_test_list, self.fixed_all_mask[test_idx],
                    self.device_obj, split_name=f"{run_name} test mask")
            else:
                self.mask_test = None
        else:
            self.mask_train = None
            self.mask_val = None
            self.mask_test = None

    def _metric_names(self):
        return (["ACC", "F1", "AUC"] if self.num_class == 2
                else ["ACC", "F1_weighted", "F1_macro"])

    @staticmethod
    def _write_csv_rows(filename, fieldnames, rows):
        """Write a complete UTF-8 CSV with a deterministic column order."""
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _append_csv_rows(filename, fieldnames, rows):
        """Append rows while writing the header exactly once."""
        if not rows:
            return
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        write_header = not os.path.isfile(filename) or os.path.getsize(filename) == 0
        with open(filename, "a", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _release_training_state(self):
        """Release one fold before constructing the next independent model."""
        for attribute in (
                "model", "optimizer", "scheduler", "data_tr_list",
                "data_val_list", "data_test_list", "data_tr_original",
                "labels_tr_tensor", "labels_val_np", "labels_test_np",
                "mask_train", "mask_val", "mask_test", "_current_epoch"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        self.scaler = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_inner_fold(self, outer_id, inner_id, train_idx,
                        validation_idx, inner_path):
        """Record validation metrics on the common inner-fold epoch grid.

        Every inner model runs for the same maximum number of epochs.  This is
        necessary because selecting the epoch with the highest five-fold mean
        validation ACC requires all five folds to be evaluated at the same
        candidate epochs.  The outer test indices are never passed here.
        """
        run_name = f"outer {outer_id} inner {inner_id}"
        run_seed = (self.params['seed']
                    + outer_id * 10_000 + inner_id * 100)
        self._prepare_split(
            train_idx, inner_path, run_name,
            validation_idx=validation_idx, test_idx=None)
        self._initialize_optimization(
            self.params['num_epoch'], run_seed)

        validation_history = []
        print(f"\n[x] {run_name}: train={len(train_idx)}, "
              f"validation={len(validation_idx)}, seed={run_seed}")

        for epoch in tqdm(
                range(self.params['num_epoch']),
                desc=f"Outer {outer_id} / Inner {inner_id}"):
            self._current_epoch = epoch
            should_validate = (
                (epoch + 1) % self.params['test_interval'] == 0
                or epoch == self.params['num_epoch'] - 1)
            self.train_epoch(print_loss=should_validate)
            if not should_validate:
                continue
            validation_probabilities = self.validation_epoch()
            if not np.isfinite(validation_probabilities).all():
                raise FloatingPointError(
                    f"{run_name}, epoch {epoch + 1} produced non-finite "
                    "validation probabilities")
            validation_metrics = self._score_probabilities(
                self.labels_val_np, validation_probabilities,
                f"Outer {outer_id} inner {inner_id} "
                f"validation epoch {epoch + 1}")
            validation_history.append({
                "epoch": int(epoch + 1),
                "metrics": validation_metrics,
            })

        if not validation_history:
            raise FloatingPointError(
                f"{run_name} did not record any validation checkpoint")
        best_entry = max(
            validation_history,
            key=lambda item: (item["metrics"][0], -item["epoch"]))
        result = {
            "inner_fold": int(inner_id),
            "train_indices": np.asarray(train_idx).tolist(),
            "validation_indices": np.asarray(validation_idx).tolist(),
            "diagnostic_single_fold_best_epoch": best_entry["epoch"],
            "diagnostic_single_fold_best_metrics": best_entry["metrics"],
            "validation_history": validation_history,
        }
        with open(os.path.join(inner_path, "metrics.json"), "w",
                  encoding="utf-8") as file_pointer:
            json.dump(result, file_pointer, indent=4)
        self._release_training_state()
        return result

    def _select_outer_epoch(self, inner_results, outer_path):
        """Choose the epoch maximizing mean validation ACC across inner folds."""
        if len(inner_results) != self.params['inner_folds']:
            raise ValueError(
                "inner_results count does not match inner_folds")
        history_maps = []
        for result in inner_results:
            history = result.get("validation_history", [])
            if not history:
                raise ValueError("an inner fold has an empty validation history")
            history_maps.append({
                int(entry["epoch"]): np.asarray(
                    entry["metrics"], dtype=np.float64)
                for entry in history
            })

        common_epochs = set(history_maps[0])
        for history_map in history_maps[1:]:
            common_epochs.intersection_update(history_map)
        if not common_epochs:
            raise RuntimeError(
                "inner folds have no common validation epoch")

        epoch_curve = []
        for epoch in sorted(common_epochs):
            fold_metrics = np.stack(
                [history_map[epoch] for history_map in history_maps],
                axis=0)
            if not np.isfinite(fold_metrics).all():
                raise FloatingPointError(
                    f"non-finite inner metrics at epoch {epoch}")
            mean_metrics = fold_metrics.mean(axis=0)
            sample_std = (
                fold_metrics.std(axis=0, ddof=1)
                if fold_metrics.shape[0] > 1
                else np.zeros(fold_metrics.shape[1], dtype=np.float64))
            epoch_curve.append({
                "epoch": int(epoch),
                "fold_metrics": fold_metrics.tolist(),
                "mean_metrics": mean_metrics.tolist(),
                "sample_std_metrics": sample_std.tolist(),
            })

        # Highest mean ACC wins.  An exact tie is resolved in favour of the
        # earlier epoch to reduce unnecessary training and overfitting risk.
        best_mean_acc = max(entry["mean_metrics"][0]
                            for entry in epoch_curve)
        tolerance = 1e-12
        selected_entry = min(
            (entry for entry in epoch_curve
             if abs(entry["mean_metrics"][0] - best_mean_acc)
             <= tolerance),
            key=lambda entry: entry["epoch"])
        selected_epoch = int(selected_entry["epoch"])

        selection = {
            "criterion": "maximum_five_fold_mean_validation_ACC",
            "tie_break": "earliest_epoch",
            "selected_epoch": selected_epoch,
            "selected_mean_metrics": selected_entry["mean_metrics"],
            "selected_sample_std_metrics": (
                selected_entry["sample_std_metrics"]),
            "epoch_curve": epoch_curve,
        }
        with open(os.path.join(outer_path, "inner_epoch_selection.json"),
                  "w", encoding="utf-8") as file_pointer:
            json.dump(selection, file_pointer, indent=4)

        metric_names = self._metric_names()
        curve_fields = ["epoch", "selected"]
        for inner_id in range(1, self.params['inner_folds'] + 1):
            curve_fields.extend(
                [f"inner_{inner_id}_{name}" for name in metric_names])
        curve_fields.extend(
            [f"mean_{name}" for name in metric_names])
        curve_fields.extend(
            [f"sample_std_{name}" for name in metric_names])
        curve_rows = []
        for entry in epoch_curve:
            row = {
                "epoch": entry["epoch"],
                "selected": int(entry["epoch"] == selected_epoch),
            }
            for inner_zero, metrics in enumerate(
                    entry["fold_metrics"]):
                for name, value in zip(metric_names, metrics):
                    row[f"inner_{inner_zero + 1}_{name}"] = value
            for name, value in zip(
                    metric_names, entry["mean_metrics"]):
                row[f"mean_{name}"] = value
            for name, value in zip(
                    metric_names, entry["sample_std_metrics"]):
                row[f"sample_std_{name}"] = value
            curve_rows.append(row)
        self._write_csv_rows(
            os.path.join(outer_path, "inner_epoch_selection.csv"),
            curve_fields, curve_rows)
        return selected_epoch, selection

    def _fit_outer_model(self, outer_id, train_idx, test_idx,
                         selected_epoch, outer_path):
        """Retrain on all nine outer-training folds, then test exactly once."""
        final_path = os.path.join(outer_path, "final_model")
        run_name = f"outer {outer_id} final"
        run_seed = self.params['seed'] + outer_id * 10_000 + 9_999
        self._prepare_split(
            train_idx, final_path, run_name,
            validation_idx=None, test_idx=test_idx)
        # Preserve the same learning-rate trajectory used by the inner runs.
        # Changing the cosine horizon to selected_epoch would make retraining
        # follow a different schedule from the one that selected that epoch.
        self._initialize_optimization(
            selected_epoch, run_seed,
            schedule_total_epochs=self.params['num_epoch'])
        print(f"\n[x] {run_name}: train={len(train_idx)}, "
              f"test={len(test_idx)}, epochs={selected_epoch}, "
              f"seed={run_seed}")
        for epoch in tqdm(
                range(selected_epoch), desc=f"Outer {outer_id} final fit"):
            self._current_epoch = epoch
            print_loss = (
                (epoch + 1) % self.params['test_interval'] == 0
                or epoch == selected_epoch - 1)
            self.train_epoch(print_loss=print_loss)

        self.save_checkpoint(final_path, filename="final_model.pt")
        # This is the only point at which this outer test fold is evaluated.
        test_probabilities = self.test_epoch()
        if not np.isfinite(test_probabilities).all():
            raise FloatingPointError(
                f"outer fold {outer_id} produced non-finite test output")
        test_metrics = self._score_probabilities(
            self.labels_test_np, test_probabilities,
            f"Outer fold {outer_id} held-out test")
        np.save(os.path.join(final_path, "test_probabilities.npy"),
                test_probabilities)
        self._release_training_state()
        return test_probabilities, test_metrics

    def train(self):
        """Run outer-10/inner-5 stratified nested cross-validation."""
        dataset_name = os.path.basename(os.path.normpath(
            self.params['data_folder'])) or "dataset"
        exp_name = os.path.join(
            self.params['exp'],
            f"{dataset_name}_nested_"
            f"{self.params['outer_folds']}x{self.params['inner_folds']}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
        os.makedirs(exp_name, exist_ok=True)
        with open(os.path.join(exp_name, 'config.json'), 'w',
                  encoding='utf-8') as file_pointer:
            json.dump(self.params, file_pointer, indent=4)
        np.savez(
            os.path.join(exp_name, "original_source_indices.npz"),
            original_train_indices=np.arange(self.original_train_size),
            original_test_indices=np.arange(
                self.original_train_size, self.labels_all_np.size))
        if self.fixed_all_mask is not None:
            np.save(os.path.join(exp_name, "fixed_merged_mask.npy"),
                    self.fixed_all_mask)

        print(f"[x] AMP mixed precision: "
              f"{'ON' if self.amp_enabled else 'OFF'}")
        print(f"[x] Gradient clipping: "
              f"max_norm={self.params['max_grad_norm']}")

        nested_oof_probabilities = np.full(
            (self.labels_all_np.size, self.num_class),
            np.nan, dtype=np.float64)
        outer_metrics = []
        outer_results = []
        metric_names = self._metric_names()

        inner_epoch_csv = os.path.join(
            exp_name, "inner_fold_epoch_metrics.csv")
        inner_best_csv = os.path.join(
            exp_name, "inner_fold_best_metrics.csv")
        outer_test_csv = os.path.join(
            exp_name, "outer_fold_test_metrics.csv")
        inner_epoch_fields = [
            "outer_fold", "inner_fold", "epoch",
            "train_size", "validation_size",
        ] + [f"validation_{name}" for name in metric_names]
        inner_best_fields = [
            "outer_fold", "inner_fold", "best_epoch",
            "train_size", "validation_size",
        ] + [f"best_validation_{name}" for name in metric_names]
        outer_test_fields = [
            "outer_fold", "selected_epoch",
            "outer_train_size", "outer_test_size",
        ]
        outer_test_fields += [
            f"selected_inner_mean_{name}" for name in metric_names]
        outer_test_fields += [
            f"selected_inner_sample_std_{name}" for name in metric_names]
        outer_test_fields += [
            f"outer_test_{name}" for name in metric_names]
        # Create the files and headers before training so partial results are
        # still readable if a long nested-CV run is interrupted later.
        self._write_csv_rows(
            inner_epoch_csv, inner_epoch_fields, [])
        self._write_csv_rows(
            inner_best_csv, inner_best_fields, [])
        self._write_csv_rows(
            outer_test_csv, outer_test_fields, [])

        for outer_zero, (outer_train_idx, outer_test_idx) in enumerate(
                self.outer_indices):
            outer_id = outer_zero + 1
            outer_path = os.path.join(exp_name, f"outer_{outer_id}")
            os.makedirs(outer_path, exist_ok=True)
            np.savez(
                os.path.join(outer_path, "outer_indices.npz"),
                train_indices=outer_train_idx,
                test_indices=outer_test_idx)
            print("\n" + "=" * 76)
            print(f"  OUTER FOLD {outer_id}/{self.params['outer_folds']}: "
                  f"train={outer_train_idx.size}, "
                  f"held-out test={outer_test_idx.size}")
            print("=" * 76)

            inner_splitter = StratifiedKFold(
                n_splits=self.params['inner_folds'], shuffle=True,
                random_state=self.params['seed'] + 1000 * outer_id)
            inner_results = []
            outer_train_labels = self.labels_all_np[outer_train_idx]
            for inner_zero, (inner_train_rel, inner_val_rel) in enumerate(
                    inner_splitter.split(
                        np.zeros(outer_train_idx.size),
                        outer_train_labels)):
                inner_id = inner_zero + 1
                inner_train_idx = outer_train_idx[inner_train_rel]
                inner_val_idx = outer_train_idx[inner_val_rel]
                inner_path = os.path.join(
                    outer_path, f"inner_{inner_id}")
                inner_result = self._run_inner_fold(
                    outer_id, inner_id, inner_train_idx,
                    inner_val_idx, inner_path)
                inner_results.append(inner_result)
                inner_epoch_rows = []
                for entry in inner_result["validation_history"]:
                    row = {
                        "outer_fold": outer_id,
                        "inner_fold": inner_id,
                        "epoch": entry["epoch"],
                        "train_size": inner_train_idx.size,
                        "validation_size": inner_val_idx.size,
                    }
                    for name, value in zip(
                            metric_names, entry["metrics"]):
                        row[f"validation_{name}"] = value
                    inner_epoch_rows.append(row)
                self._append_csv_rows(
                    inner_epoch_csv, inner_epoch_fields,
                    inner_epoch_rows)

                best_row = {
                    "outer_fold": outer_id,
                    "inner_fold": inner_id,
                    "best_epoch": inner_result[
                        "diagnostic_single_fold_best_epoch"],
                    "train_size": inner_train_idx.size,
                    "validation_size": inner_val_idx.size,
                }
                for name, value in zip(
                        metric_names,
                        inner_result[
                            "diagnostic_single_fold_best_metrics"]):
                    best_row[f"best_validation_{name}"] = value
                self._append_csv_rows(
                    inner_best_csv, inner_best_fields, [best_row])

            selected_epoch, epoch_selection = self._select_outer_epoch(
                inner_results, outer_path)
            selected_mean_acc = (
                epoch_selection["selected_mean_metrics"][0])
            selected_std_acc = (
                epoch_selection["selected_sample_std_metrics"][0])
            print(f"\n[x] Outer fold {outer_id}: selected_epoch="
                  f"{selected_epoch}; inner mean validation ACC="
                  f"{selected_mean_acc:.5f} +/- {selected_std_acc:.5f}")

            test_probabilities, test_metrics = self._fit_outer_model(
                outer_id, outer_train_idx, outer_test_idx,
                selected_epoch, outer_path)
            nested_oof_probabilities[outer_test_idx] = test_probabilities
            outer_metrics.append(test_metrics)
            outer_csv_row = {
                "outer_fold": outer_id,
                "selected_epoch": selected_epoch,
                "outer_train_size": outer_train_idx.size,
                "outer_test_size": outer_test_idx.size,
            }
            for name, value in zip(
                    metric_names,
                    epoch_selection["selected_mean_metrics"]):
                outer_csv_row[f"selected_inner_mean_{name}"] = value
            for name, value in zip(
                    metric_names,
                    epoch_selection[
                        "selected_sample_std_metrics"]):
                outer_csv_row[
                    f"selected_inner_sample_std_{name}"] = value
            for name, value in zip(metric_names, test_metrics):
                outer_csv_row[f"outer_test_{name}"] = value
            self._append_csv_rows(
                outer_test_csv, outer_test_fields, [outer_csv_row])
            outer_result = {
                "outer_fold": outer_id,
                "outer_train_indices": outer_train_idx.tolist(),
                "outer_test_indices": outer_test_idx.tolist(),
                "selected_epoch": selected_epoch,
                "inner_epoch_selection": epoch_selection,
                "inner_results": inner_results,
                "outer_test_metrics": test_metrics,
            }
            outer_results.append(outer_result)
            with open(os.path.join(outer_path, "metrics.json"), "w",
                      encoding="utf-8") as file_pointer:
                json.dump(outer_result, file_pointer, indent=4)

        if not np.isfinite(nested_oof_probabilities).all():
            missing_rows = np.where(
                ~np.isfinite(nested_oof_probabilities).all(axis=1))[0]
            raise RuntimeError(
                "nested outer-fold predictions are incomplete for rows "
                f"{missing_rows.tolist()}")

        outer_array = np.asarray(outer_metrics, dtype=np.float64)
        outer_means = np.nanmean(outer_array, axis=0)
        outer_stds = np.nanstd(outer_array, axis=0, ddof=1)

        print("\n" + "=" * 68)
        print(f"  Nested {self.params['outer_folds']}x"
              f"{self.params['inner_folds']} Cross-Validation Summary")
        print("=" * 68)
        for name, mean, std in zip(
                metric_names, outer_means, outer_stds):
            print(f"  Outer held-out test {name}: "
                  f"{mean:.5f} +/- {std:.5f}")

        pooled_nested_metrics = self._score_probabilities(
            self.labels_all_np, nested_oof_probabilities,
            "Pooled nested outer-fold predictions")

        np.save(
            os.path.join(exp_name, "nested_oof_probabilities.npy"),
            nested_oof_probabilities)
        summary_fields = ["statistic"] + metric_names
        summary_rows = []
        for statistic, values in (
                ("outer_fold_mean", outer_means),
                ("outer_fold_sample_std", outer_stds),
                ("pooled_nested_oof", pooled_nested_metrics)):
            row = {"statistic": statistic}
            for name, value in zip(metric_names, values):
                row[name] = value
            summary_rows.append(row)
        self._write_csv_rows(
            os.path.join(exp_name, "nested_cv_summary.csv"),
            summary_fields, summary_rows)
        summary = {
            "protocol": "stratified_nested_cross_validation",
            "metric_names": metric_names,
            "outer_folds": int(self.params['outer_folds']),
            "inner_folds": int(self.params['inner_folds']),
            "epoch_selection": (
                "maximum_five_fold_mean_validation_ACC; "
                "ties_resolved_by_earliest_epoch"),
            "outer_test_mean": outer_means.tolist(),
            "outer_test_sample_std": outer_stds.tolist(),
            "pooled_nested_oof_metrics": pooled_nested_metrics,
            "outer_results": outer_results,
        }
        with open(os.path.join(exp_name, "summary.json"), "w",
                  encoding="utf-8") as file_pointer:
            json.dump(summary, file_pointer, indent=4)
        return summary, exp_name

    def train_epoch(self, print_loss=False):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # ---------- AMP 混合精度前向传播 ----------
        if self.scaler is not None:
            with torch.amp.autocast('cuda'):
                loss, _, loss_dict = self._forward_pass(record_loss=print_loss)
        else:
            loss, _, loss_dict = self._forward_pass(record_loss=print_loss)

        if print_loss:
            pprint.pprint(loss_dict)

        # ---------- 反向传播 + 梯度裁剪 + 参数更新 ----------
        loss = torch.mean(loss)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            # 梯度裁剪：防止高缺失率场景下梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=self.params.get('max_grad_norm', 1.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=self.params.get('max_grad_norm', 1.0))
            self.optimizer.step()

        # scheduler.step 在 optimizer.step 之后调用（LambdaLR 推荐顺序）
        self.scheduler.step()

    def _forward_pass(self, record_loss=False):
        """统一前向传播入口，减少 train_epoch 中的重复代码"""
        if self.params['missing_rate'] > 0:
            epoch = getattr(self, '_current_epoch', 0)
            router_interval = self.params.get('router_update_interval', 10)
            contrastive_interval = self.params.get(
                'contrastive_update_interval', 5)
            update_router = (epoch % router_interval == 0)
            update_contrastive = (epoch % contrastive_interval == 0)
            # [support_mode] 'masked' is the leakage-safe default;
            # 'complete' is an explicitly declared reference-panel protocol.
            _sm = self.params.get('support_mode', 'masked')  # [FIX] safer default
            _support = (self.data_tr_original
                        if _sm == 'complete'
                        else self.data_tr_list)
            _sup_mask = self.mask_train if _sm == 'masked' else None
            return self.model.train_missing_cg(
                self.data_tr_list, self.mask_train, self.labels_tr_tensor,
                aux_loss=self.params['lambda_al'] > 0,
                lambda_al=self.params['lambda_al'],
                original_data_list=_support,
                support_mask=_sup_mask,
                use_cross_sample_impute=self.params['use_cross_sample_impute'],
                use_prototype_bank=self.params['use_prototype_bank'],
                lambda_imputation=self.params['lambda_imputation'],
                cross_sample_k=self.params['cross_sample_k'],
                knn_base_temperature=self.params['knn_base_temperature'],
                support_labels=self.labels_tr_tensor,
                contrastive_loss=(self.params['temperature'] > 0 and
                                  self.params.get('lambda_cil', 1.0) > 0 and
                                  update_contrastive),
                temperature=self.params['temperature'],
                lambda_cil=self.params.get('lambda_cil', 1.0),
                router_temperature=self.params.get('router_temperature', 0.5),
                compute_router_supervision=update_router,
                use_action_router=self.params['use_action_router'],
                record_loss=record_loss)
        else:
            return self.model(self.data_tr_list, self.labels_tr_tensor,
                              aux_loss=self.params['lambda_al'] > 0,
                              lambda_al=self.params['lambda_al'],
                              record_loss=record_loss)

    def _predict_epoch(self, data_list, mask=None):
        self.model.eval()
        with torch.inference_mode(), torch.amp.autocast(
                self.device_obj.type, enabled=self.scaler is not None):
            if self.params['missing_rate'] > 0:
                _sm = self.params.get('support_mode', 'masked')  # [FIX] safer default
                _support = (self.data_tr_original
                            if _sm == 'complete'
                            else self.data_tr_list)
                _sup_mask = self.mask_train if _sm == 'masked' else None
                logit = self.model.infer_on_missing(
                    data_list,
                    mask,
                    support_data_list=_support,
                    support_labels=self.labels_tr_tensor,
                    support_mask=_sup_mask,
                    use_cross_sample_impute=self.params['use_cross_sample_impute'],
                    use_prototype_bank=self.params['use_prototype_bank'],
                    cross_sample_k=self.params['cross_sample_k'],
                    knn_base_temperature=self.params['knn_base_temperature'],
                    exclude_self=False,
                    router_temperature=self.params.get(
                        'router_temperature', 0.5),
                    use_action_router=(
                        self.params['use_action_router']
                        and self.params['lambda_imputation'] > 0),
                )
            else:
                logit = self.model.infer(data_list)
            prob = F.softmax(logit, dim=1).cpu().numpy()
        return prob

    def validation_epoch(self):
        return self._predict_epoch(
            self.data_val_list,
            getattr(self, 'mask_val', None))

    def test_epoch(self):
        return self._predict_epoch(
            self.data_test_list,
            getattr(self, 'mask_test', None))

    def _score_probabilities(self, labels, prob, split_name):
        predictions = prob.argmax(1)
        acc = accuracy_score(labels, predictions)
        if self.num_class == 2:
            f1 = f1_score(labels, predictions, zero_division=0)
            auc = (roc_auc_score(labels, prob[:, 1])
                   if np.unique(labels).size == 2 else float('nan'))
            metrics = [float(acc), float(f1), float(auc)]
            print(f"\n{split_name}: ACC={acc:.5f}, F1={f1:.5f}, "
                  f"AUC={auc:.5f}")
        else:
            f1w = f1_score(
                labels, predictions, average='weighted', zero_division=0)
            f1m = f1_score(
                labels, predictions, average='macro', zero_division=0)
            metrics = [float(acc), float(f1w), float(f1m)]
            print(f"\n{split_name}: ACC={acc:.5f}, "
                  f"F1_weighted={f1w:.5f}, F1_macro={f1m:.5f}")
        return metrics

    def save_checkpoint(self, checkpoint_path, filename="checkpoint.pt"):
        os.makedirs(checkpoint_path, exist_ok=True)
        filename = os.path.join(checkpoint_path, filename)
        torch.save(self.model.state_dict(), filename)
