#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RUSI plus classic/recent missing-view comparators under nested CV.

Protocol
--------
1. Merge the externally supplied ``*_tr.csv`` and ``*_te.csv`` files.
2. Generate one cohort-level missing-view mask for each missing rate.  Every
   enabled method receives exactly the same mask.
3. Outer stratified CV provides untouched test folds (10 by default).
4. Inner stratified CV selects the epoch maximizing mean validation ACC
   across the five inner folds; ties select the earliest epoch.
5. Fit normalization/imputation on the current training rows only.
6. Retrain on the complete outer-training fold for the selected number of
   epochs, and evaluate the outer-test fold exactly once.

Binary output: ACC, F1, AUC.  Multiclass output: ACC, weighted-F1, macro-F1.
Use ``--methods`` to switch individual comparators on/off.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from comparison_config import (
    CLCLSA_REPO as CONFIG_CLCLSA_REPO,
    DATA_FOLDER as CONFIG_DATA_FOLDER,
    DATASET_NAME as CONFIG_DATASET_NAME,
    DIMV_REPO as CONFIG_DIMV_REPO,
    INTEGRAO_REPO as CONFIG_INTEGRAO_REPO,
    METHOD_SWITCHES,
    MISSING_RATES,
    MOGEDN_REPO as CONFIG_MOGEDN_REPO,
    OUTPUT_DIR as CONFIG_OUTPUT_DIR,
    RUSI_MODEL_FILE as CONFIG_RUSI_MODEL_FILE,
    RUSI_PARAMS as CONFIG_RUSI_PARAMS,
    TMONET_REPO as CONFIG_TMONET_REPO,
)
from missing_comparison_core import (
    METHODS, FoldNormalizer, build_adapter, score_probabilities,
    seed_everything,
)
from official_model_loader import audit_selected_sources


METRICS = ["ACC", "F1", "AUC", "F1_weighted", "F1_macro"]
PROTOCOL_VERSION = "paired_nested_cv_v8_official_sources_deterministic_2026_08_19"
FOLD_FIELDS = [
    "dataset", "task_type", "missing_rate", "two_missing_ratio", "method",
    "method_year", "method_category", "method_category_cn",
    "outer_fold", "selected_epoch",
    *METRICS, "duration_seconds", "status", "implementation_source", "notes",
]
SUMMARY_FIELDS = [
    "dataset", "task_type", "missing_rate", "two_missing_ratio", "method",
    "method_year", "method_category", "method_category_cn", "statistic", *METRICS,
    "duration_seconds", "successful_outer_folds", "status",
    "implementation_source", "notes",
]
MAIN_TABLE_FIELDS = [
    "dataset", "task_type", "missing_rate", "two_missing_ratio", "method",
    "method_year", "method_category", "method_category_cn",
    "ACC_mean", "ACC_std", "F1_mean", "F1_std", "AUC_mean", "AUC_std",
    "F1_weighted_mean", "F1_weighted_std", "F1_macro_mean", "F1_macro_std",
    "duration_seconds", "successful_outer_folds", "status", "notes",
]


def category_cn(category: str) -> str:
    if category == "imputation":
        return "插补方法"
    if category == "selective_imputation":
        return "选择性插补方法（本文）"
    return "直接处理缺失数据"


def rate_column(rate: float) -> str:
    text = f"{rate:.10g}"
    if "." not in text:
        text += ".0"
    return f"r={text}"


def parse_float_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    if any(not 0.0 <= x < 1.0 for x in values):
        raise argparse.ArgumentTypeError("missing rates must be in [0, 1)")
    return values


def parse_str_list(text: str) -> List[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def parse_alpha_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(x < 0 for x in values):
        raise argparse.ArgumentTypeError("DIMV alphas must be non-negative")
    return values


def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError(
            "all dimensions must be positive integers")
    return values


def dataset_name(path: Path) -> str:
    return path.resolve().name or "dataset"


REQUIRED_DATA_FILES = {
    "labels_tr.csv", "labels_te.csv",
    "1_tr.csv", "1_te.csv", "2_tr.csv", "2_te.csv", "3_tr.csv", "3_te.csv",
}


def is_data_folder(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file()
                                 for name in REQUIRED_DATA_FILES)


def resolve_data_folder(raw_value: str | None, script_dir: Path,
                        *, allow_discovery: bool = True) -> Path:
    """Resolve configured/CLI data path, then auto-discover if necessary."""
    attempted: List[Path] = []
    if raw_value:
        raw = Path(raw_value).expanduser()
        candidates = ([raw] if raw.is_absolute() else [
            script_dir / raw,          # D:\...\contrast\BRCA
            script_dir.parent / raw,   # D:\...\BRCA
            Path.cwd() / raw,          # preserve the main model's cwd behavior
        ])
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate not in attempted:
                attempted.append(candidate)
            if is_data_folder(candidate):
                return candidate
        if not allow_discovery:
            tried = ", ".join(str(path) for path in attempted)
            raise FileNotFoundError(
                "the explicitly requested dataset folder is invalid. A valid "
                f"folder must contain {sorted(REQUIRED_DATA_FILES)}. Checked: {tried}")

    discovered = set()
    # Search both the comparator directory and its project parent.  This
    # supports layouts such as D:\EEEEEE\contrast\script.py + D:\EEEEEE\BRCA.
    for search_root in dict.fromkeys((script_dir, script_dir.parent, Path.cwd())):
        if is_data_folder(search_root):
            discovered.add(search_root.resolve())
        for label_file in search_root.rglob("labels_tr.csv"):
            candidate = label_file.parent.resolve()
            if is_data_folder(candidate):
                discovered.add(candidate)
    if len(discovered) == 1:
        selected = next(iter(discovered))
        print(f"[auto] data folder: {selected}")
        return selected
    if len(discovered) > 1:
        choices = "\n  ".join(str(path) for path in sorted(discovered))
        raise ValueError(
            "Multiple datasets were found. Set DATA_FOLDER in comparison_config.py "
            f"or pass --data-folder. Candidates:\n  {choices}")
    tried = ", ".join(str(path) for path in attempted) or str(script_dir)
    raise FileNotFoundError(
        "No valid dataset folder was found. A dataset folder must contain "
        f"{sorted(REQUIRED_DATA_FILES)}. Checked: {tried}. Set DATA_FOLDER in "
        "comparison_config.py or pass --data-folder.")


def resolve_model_file(raw_value: str, script_dir: Path) -> Path:
    """Resolve a model file relative to the runner, its parent, or the cwd."""
    raw = Path(raw_value).expanduser()
    candidates = ([raw] if raw.is_absolute() else [
        script_dir / raw, script_dir.parent / raw, Path.cwd() / raw,
    ])
    attempted: List[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in attempted:
            attempted.append(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "RUSI model file was not found. Checked: "
        + ", ".join(str(path) for path in attempted))


def rate_tag(rate: float) -> str:
    return f"mr_{rate:.4f}".replace(".", "p")


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]],
                     fields: Sequence[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def build_main_table(summary_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Build one publication-friendly row per method and missing rate."""
    grouped: Dict[tuple, List[Mapping[str, object]]] = {}
    for row in summary_rows:
        key = (row.get("dataset", ""), row.get("task_type", ""),
               str(row.get("missing_rate", "")),
               str(row.get("two_missing_ratio", "")), row.get("method", ""))
        grouped.setdefault(key, []).append(row)
    table: List[Dict[str, object]] = []
    for rows in grouped.values():
        mean = next((r for r in rows if r.get("statistic") == "outer_fold_mean"), None)
        std = next((r for r in rows if r.get("statistic") == "outer_fold_sample_std"), None)
        failed = next((r for r in rows if r.get("statistic") == "failed"), None)
        base = mean or failed or rows[0]
        out: Dict[str, object] = {
            "dataset": base.get("dataset", ""),
            "task_type": base.get("task_type", ""),
            "missing_rate": base.get("missing_rate", ""),
            "two_missing_ratio": base.get("two_missing_ratio", ""),
            "method": base.get("method", ""),
            "method_year": base.get("method_year", ""),
            "method_category": base.get("method_category", ""),
            "method_category_cn": base.get("method_category_cn", ""),
            "duration_seconds": base.get("duration_seconds", ""),
            "successful_outer_folds": base.get("successful_outer_folds", ""),
            "status": base.get("status", ""),
            "notes": base.get("notes", ""),
        }
        for metric in METRICS:
            out[f"{metric}_mean"] = mean.get(metric, "") if mean else ""
            out[f"{metric}_std"] = std.get(metric, "") if std else ""
        table.append(out)
    return sorted(table, key=lambda r: (
        str(r["dataset"]), float(r["missing_rate"]), str(r["method"])))


def write_final_metric_tables(output_dir: Path,
                              summary_rows: Sequence[Mapping[str, object]],
                              dataset: str, task_type: str,
                              print_tables: bool = False) -> List[Path]:
    """Write screenshot-style wide tables: methods x missing rates."""
    relevant = [r for r in summary_rows
                if str(r.get("dataset", "")) == dataset]
    rates = sorted({float(r["missing_rate"]) for r in relevant
                    if r.get("missing_rate", "") != ""})
    metrics = (["ACC", "F1", "AUC"] if task_type == "binary"
               else ["ACC", "F1_weighted", "F1_macro"])
    method_order = {info.display_name: i for i, info in enumerate(METHODS.values())}
    category_by_method = {
        str(r.get("method", "")): str(r.get("method_category", ""))
        for r in relevant}
    method_names = sorted({str(r.get("method", "")) for r in relevant
                           if r.get("method", "")},
                          key=lambda name: (
                              0 if category_by_method.get(name) == "imputation" else 1,
                              method_order.get(name, 10_000)))
    safe_dataset = re.sub(r"[^0-9A-Za-z._-]+", "_", dataset).strip("_") or "dataset"
    written: List[Path] = []
    for metric in metrics:
        fields = ["dataset", "method", "type"]
        fields += [rate_column(rate) for rate in rates]
        fields.append("note")
        rows: List[Dict[str, object]] = []
        for method_name in method_names:
            method_rows = [r for r in relevant if r.get("method") == method_name]
            base = next((r for r in method_rows
                         if r.get("statistic") == "outer_fold_mean"), method_rows[0])
            category = str(base.get("method_category", ""))
            if category == "imputation":
                type_label = "imputation (插补)"
            elif category == "selective_imputation":
                type_label = "selective imputation (本文)"
            else:
                type_label = "direct (直接处理)"
            row: Dict[str, object] = {
                "dataset": dataset,
                "method": method_name,
                "type": type_label,
                "note": base.get("notes", ""),
            }
            for rate in rates:
                candidates = [r for r in method_rows
                              if float(r.get("missing_rate", -1)) == rate]
                mean = next((r for r in candidates
                             if r.get("statistic") == "outer_fold_mean"), None)
                failed = next((r for r in candidates
                               if r.get("statistic") == "failed"), None)
                value: object = ""
                if mean is not None and mean.get(metric, "") != "":
                    value = f"{float(mean[metric]):.4f}"
                elif failed is not None:
                    value = "FAILED"
                row[rate_column(rate)] = value
            rows.append(row)
        csv_path = output_dir / f"{safe_dataset}_{metric}_final_results.csv"
        write_csv_atomic(csv_path, rows, fields)
        written.append(csv_path)

        # A plain-text copy mirrors the layout shown in the user's example.
        text_path = output_dir / f"{safe_dataset}_{metric}_final_results.txt"
        widths = {field: max(len(field), *(len(str(row.get(field, "")))
                                           for row in rows)) for field in fields}
        lines = [f"--- dataset={dataset} | {metric} ---",
                 "  ".join(field.ljust(widths[field]) for field in fields),
                 "  ".join("-" * widths[field] for field in fields)]
        lines.extend("  ".join(str(row.get(field, "")).ljust(widths[field])
                               for field in fields) for row in rows)
        text_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        written.append(text_path)
        if print_tables:
            print("\n" + "\n".join(lines))
    return written


def finite_or_blank(value: float):
    return "" if not np.isfinite(value) else float(value)


def validate_probabilities(prob: np.ndarray, n_rows: int, num_classes: int,
                           context: str) -> np.ndarray:
    """Reject malformed baseline output instead of silently scoring it."""
    prob = np.asarray(prob, dtype=np.float64)
    expected = (n_rows, num_classes)
    if prob.shape != expected:
        raise ValueError(
            f"{context} returned probability shape {prob.shape}, expected {expected}")
    if not np.isfinite(prob).all():
        raise FloatingPointError(f"{context} returned NaN or infinite probabilities")
    if np.any(prob < -1e-7):
        raise ValueError(f"{context} returned negative probabilities")
    row_sums = prob.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError(f"{context} probabilities do not sum to one")
    return prob


def config_hash(cfg: Mapping[str, object]) -> str:
    # Methods and rates may be added to the same compatible experiment later.
    # They do not alter a particular method/rate checkpoint's training rules.
    ignored = {
        "resume", "overwrite", "methods", "missing_rates", "missing_rate",
        "output_dir", "fail_fast", "dry_run", "official_source_sha256",
        "dimv_repo", "clclsa_repo", "tmonet_repo", "integrao_repo",
        "mogedn_repo",
    }
    # Proposed-method-only settings must not invalidate already completed
    # baseline checkpoints.  They are bound specifically to RUSI below.
    clean = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()
             if k not in ignored and not k.startswith("rusi_")}
    raw = json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def method_run_hash(cfg: Mapping[str, object], method: str) -> str:
    """Bind one checkpoint to the common protocol and that method's source."""
    official = cfg.get("official_source_sha256", {})
    source_hashes = (official.get(method, {})
                     if isinstance(official, Mapping) else {})
    payload = {
        "configuration_sha256": str(cfg["config_hash"]),
        "method": method,
        "official_source_sha256": source_hashes,
    }
    if method == "rusi":
        payload["method_configuration"] = {
            key: value for key, value in cfg.items()
            if key.startswith("rusi_") and key != "rusi_model_file"
        }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def verify_existing_manifest(path: Path, expected_hash: str,
                             resume: bool, overwrite: bool) -> None:
    """Prevent fold CSV/checkpoint mixtures from incompatible experiments."""
    if overwrite or not resume:
        return
    if not path.exists():
        stale = [path.parent / name for name in (
            "comparison_outer_folds.csv", "comparison_summary.csv")
                 if (path.parent / name).exists()]
        if stale:
            raise RuntimeError(
                "existing result CSV files have no compatible manifest; "
                "use a new output directory or rerun with --overwrite")
        return
    try:
        with path.open("r", encoding="utf-8") as fh:
            previous = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read existing manifest {path}; use --overwrite") from exc
    previous_hash = str(previous.get("configuration_sha256", ""))
    if previous_hash != expected_hash:
        raise RuntimeError(
            "existing results use a different data/training protocol; "
            "use a new output directory or rerun with --overwrite")


def validate_splits(y: np.ndarray, outer_folds: int, inner_folds: int) -> None:
    counts = np.bincount(y)
    if counts.min() < outer_folds:
        raise ValueError(
            f"smallest class has {counts.min()} samples, fewer than outer_folds={outer_folds}")
    # Worst outer-training count is floor(c * (K-1)/K).
    worst = int(np.floor(counts.min() * (outer_folds - 1) / outer_folds))
    if worst < inner_folds:
        raise ValueError(
            f"an outer-training split may have only {worst} samples in the smallest "
            f"class, fewer than inner_folds={inner_folds}")


def save_cv_indices(y: np.ndarray, cfg: Mapping[str, object], path: Path) -> None:
    """Persist the exact paired outer/inner patient partitions for auditing."""
    payload: Dict[str, np.ndarray] = {}
    outer = StratifiedKFold(
        n_splits=int(cfg["outer_folds"]), shuffle=True,
        random_state=int(cfg["seed"]))
    for outer_id, (outer_train, outer_test) in enumerate(
            outer.split(np.zeros(len(y)), y), 1):
        payload[f"outer_{outer_id}_train"] = outer_train.astype(np.int64)
        payload[f"outer_{outer_id}_test"] = outer_test.astype(np.int64)
        inner = StratifiedKFold(
            n_splits=int(cfg["inner_folds"]), shuffle=True,
            random_state=int(cfg["seed"]) + 1000 * outer_id)
        for inner_id, (train_rel, val_rel) in enumerate(
                inner.split(np.zeros(len(outer_train)), y[outer_train]), 1):
            payload[f"outer_{outer_id}_inner_{inner_id}_train"] = (
                outer_train[train_rel].astype(np.int64))
            payload[f"outer_{outer_id}_inner_{inner_id}_validation"] = (
                outer_train[val_rel].astype(np.int64))
    np.savez_compressed(path, **payload)


def load_raw_data(data_folder: Path):
    """The same data-loading contract used by the user's main model."""
    def load_label_vector(filename: Path) -> np.ndarray:
        values = np.asarray(np.loadtxt(filename, delimiter=","))
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim == 1:
            return values
        if values.ndim == 2 and 1 in values.shape:
            return values.reshape(-1)
        raise ValueError(
            f"{filename.name} must contain exactly one label row or column")

    labels_train_raw = load_label_vector(data_folder / "labels_tr.csv")
    labels_test_raw = load_label_vector(data_folder / "labels_te.csv")
    for name, values in (("labels_tr.csv", labels_train_raw),
                         ("labels_te.csv", labels_test_raw)):
        if values.size == 0:
            raise ValueError(f"{name} is empty")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinity")
        if not np.equal(values, np.round(values)).all():
            raise ValueError(f"{name} must contain integer labels")

    labels_train_raw = labels_train_raw.astype(np.int64)
    labels_test_raw = labels_test_raw.astype(np.int64)
    class_values = np.unique(np.concatenate(
        (labels_train_raw, labels_test_raw)))
    if class_values.size < 2:
        raise ValueError("merged data must contain at least two classes")
    labels_train = np.searchsorted(
        class_values, labels_train_raw).astype(np.int64)
    labels_test = np.searchsorted(
        class_values, labels_test_raw).astype(np.int64)

    train_views, test_views = [], []
    for view in range(1, 4):
        train_view = np.loadtxt(
            data_folder / f"{view}_tr.csv", delimiter=",", ndmin=2)
        test_view = np.loadtxt(
            data_folder / f"{view}_te.csv", delimiter=",", ndmin=2)
        if not (np.isfinite(train_view).all()
                and np.isfinite(test_view).all()):
            raise ValueError(f"view {view} contains NaN or infinity")
        if train_view.shape[1] != test_view.shape[1]:
            raise ValueError(f"view {view} train/test feature counts differ")
        if train_view.shape[0] != labels_train.shape[0]:
            raise ValueError(f"view {view} training rows do not match labels")
        if test_view.shape[0] != labels_test.shape[0]:
            raise ValueError(f"view {view} test rows do not match labels")
        train_views.append(train_view)
        test_views.append(test_view)
    return train_views, test_views, labels_train, labels_test, class_values


def load_merged(data_folder: Path):
    train_views, test_views, labels_train, labels_test, class_values = (
        load_raw_data(data_folder))
    views = [np.concatenate((train_view, test_view), axis=0).astype(
        np.float32, copy=False)
        for train_view, test_view in zip(train_views, test_views)]
    y = np.concatenate((labels_train, labels_test)).astype(
        np.int64, copy=False)
    return views, y, class_values


def generate_shared_mask(n: int, rate: float, mode: str,
                         two_missing_ratio: float, seed: int) -> np.ndarray:
    """Create an exact, reproducible three-view mask without model imports."""
    rng = np.random.RandomState(seed)
    mask = np.ones((n, 3), dtype=np.float32)
    if n == 0 or rate == 0:
        return mask
    if mode == "sample":
        n_incomplete = int(round(n * rate))
        incomplete = rng.choice(n, size=n_incomplete, replace=False)
        n_two = int(round(n_incomplete * two_missing_ratio))
        for position, row in enumerate(incomplete):
            count = 2 if position < n_two else 1
            mask[row, rng.choice(3, size=count, replace=False)] = 0.0
    else:
        if rate > 2.0 / 3.0:
            raise ValueError("entry missing rate cannot exceed 2/3")
        n_missing = int(round(n * 3 * rate))
        protected = rng.randint(0, 3, size=n)
        rows = np.repeat(np.arange(n), 3)
        views = np.tile(np.arange(3), n)
        eligible = views != protected[rows]
        erows, eviews = rows[eligible], views[eligible]
        chosen = rng.choice(len(erows), size=n_missing, replace=False)
        mask[erows[chosen], eviews[chosen]] = 0.0
    if mask.shape != (n, 3) or not np.all(np.isin(mask, [0, 1])):
        raise RuntimeError("invalid missing-view mask")
    if np.any(mask.sum(axis=1) < 1):
        raise RuntimeError("mask contains a patient with no available view")
    return mask


def subset_views(views: Sequence[np.ndarray], idx: np.ndarray) -> List[np.ndarray]:
    return [x[idx] for x in views]


def prepare_split(views: Sequence[np.ndarray], mask: np.ndarray,
                  train_idx: np.ndarray, eval_idx: np.ndarray):
    train_raw = subset_views(views, train_idx)
    eval_raw = subset_views(views, eval_idx)
    train_mask, eval_mask = mask[train_idx], mask[eval_idx]
    normalizer = FoldNormalizer().fit(train_raw, train_mask)
    return (normalizer.transform(train_raw), train_mask,
            normalizer.transform(eval_raw), eval_mask)


def evaluation_epochs(max_epochs: int, interval: int) -> List[int]:
    points = list(range(interval, max_epochs + 1, interval))
    if max_epochs not in points:
        points.append(max_epochs)
    return points


def make_user_compatible_scheduler(adapter, cfg: Mapping[str, object]):
    """Warmup + cosine schedule used by the user's main nested-CV model."""
    if not hasattr(adapter, "optim"):
        raise AttributeError("method adapter does not expose its optimizer")
    total = int(cfg["epochs"])
    warmup = min(int(getattr(
        adapter, "scheduler_warmup_epochs", cfg["warmup_epochs"])), total)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(adapter.optim, lr_lambda)


def run_inner_selection(method: str, views: Sequence[np.ndarray], y: np.ndarray,
                        mask: np.ndarray, outer_train: np.ndarray,
                        outer_fold: int, cfg: Mapping[str, object],
                        device: torch.device) -> Tuple[int, Dict[int, float]]:
    epochs = evaluation_epochs(int(cfg["epochs"]), int(cfg["eval_interval"]))
    scores: Dict[int, List[float]] = {e: [] for e in epochs}
    inner = StratifiedKFold(
        n_splits=int(cfg["inner_folds"]), shuffle=True,
        random_state=int(cfg["seed"]) + 1000 * outer_fold)
    y_outer = y[outer_train]
    for inner_fold, (tr_local, val_local) in enumerate(inner.split(outer_train, y_outer), 1):
        tr_idx, val_idx = outer_train[tr_local], outer_train[val_local]
        fold_seed = (int(cfg["seed"]) + outer_fold * 10_000
                     + inner_fold * 100)
        seed_everything(fold_seed)
        local_cfg = dict(cfg); local_cfg["seed"] = fold_seed
        tr_x, tr_m, val_x, val_m = prepare_split(views, mask, tr_idx, val_idx)
        adapter = build_adapter(method, tr_x, tr_m, len(np.unique(y)), device, local_cfg)
        scheduler = make_user_compatible_scheduler(adapter, cfg)
        point_set = set(epochs)
        for epoch in range(1, int(cfg["epochs"]) + 1):
            adapter.train_epoch(tr_x, tr_m, y[tr_idx])
            scheduler.step()
            if epoch in point_set:
                prob = validate_probabilities(
                    adapter.predict(val_x, val_m), len(val_idx),
                    len(np.unique(y)),
                    f"{method} outer={outer_fold} inner={inner_fold} epoch={epoch}")
                scores[epoch].append(float(np.mean(prob.argmax(1) == y[val_idx])))
        del adapter, scheduler, tr_x, val_x
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    means = {e: float(np.mean(values)) for e, values in scores.items()}
    best_mean_acc = max(means.values())
    best = min(e for e in epochs
               if abs(means[e] - best_mean_acc) <= 1e-12)
    return int(best), means


def train_outer_and_predict(method: str, views: Sequence[np.ndarray], y: np.ndarray,
                            mask: np.ndarray, train_idx: np.ndarray,
                            test_idx: np.ndarray, selected_epoch: int,
                            outer_fold: int, cfg: Mapping[str, object],
                            device: torch.device) -> np.ndarray:
    fold_seed = int(cfg["seed"]) + outer_fold * 10_000 + 9_999
    seed_everything(fold_seed)
    local_cfg = dict(cfg); local_cfg["seed"] = fold_seed
    tr_x, tr_m, te_x, te_m = prepare_split(views, mask, train_idx, test_idx)
    adapter = build_adapter(method, tr_x, tr_m, len(np.unique(y)), device, local_cfg)
    scheduler = make_user_compatible_scheduler(adapter, cfg)
    for _ in range(selected_epoch):
        adapter.train_epoch(tr_x, tr_m, y[train_idx])
        scheduler.step()
    prob = validate_probabilities(
        adapter.predict(te_x, te_m), len(test_idx), len(np.unique(y)),
        f"{method} outer={outer_fold} final")
    del adapter, scheduler, tr_x, te_x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return prob


def fold_row(data_name: str, task_type: str, rate: float, two_ratio: float, method: str,
             outer_fold: int, epoch: int, metrics: Mapping[str, float],
             duration: float, status: str = "ok", notes: str = "") -> Dict[str, object]:
    info = METHODS[method]
    row: Dict[str, object] = {
        "dataset": data_name, "task_type": task_type, "missing_rate": rate,
        "two_missing_ratio": two_ratio, "method": info.display_name,
        "method_year": info.year, "method_category": info.category,
        "method_category_cn": category_cn(info.category),
        "outer_fold": outer_fold, "selected_epoch": epoch,
        "duration_seconds": round(duration, 3), "status": status,
        "implementation_source": info.source,
        "notes": info.notes + ((" " + notes) if notes else ""),
    }
    row.update({m: finite_or_blank(float(metrics.get(m, np.nan))) for m in METRICS})
    return row


def summary_rows(data_name: str, task_type: str, rate: float, two_ratio: float, method: str,
                 fold_rows: Sequence[Mapping[str, object]], oof_y: np.ndarray,
                 oof_prob: np.ndarray, num_classes: int,
                 total_duration: float) -> List[Dict[str, object]]:
    info = METHODS[method]
    ok = [r for r in fold_rows if r.get("status") == "ok"]
    rows: List[Dict[str, object]] = []
    for statistic in ("outer_fold_mean", "outer_fold_sample_std"):
        values = {}
        for metric in METRICS:
            nums = np.array([float(r[metric]) for r in ok if r.get(metric, "") != ""], dtype=float)
            if nums.size:
                values[metric] = float(np.mean(nums) if statistic.endswith("mean")
                                       else (np.std(nums, ddof=1) if nums.size > 1 else np.nan))
            else:
                values[metric] = np.nan
        rows.append(_summary_row(data_name, task_type, rate, two_ratio, info, statistic,
                                 values, total_duration, len(ok),
                                 "ok" if len(ok) else "failed"))
    valid = np.isfinite(oof_prob).all(axis=1)
    pooled = score_probabilities(oof_y[valid], oof_prob[valid], num_classes) if valid.any() else {}
    rows.append(_summary_row(
        data_name, task_type, rate, two_ratio, info, "pooled_nested_oof", pooled,
        total_duration, len(ok), "ok" if valid.all() else "partial"))
    return rows


def _summary_row(data_name, task_type, rate, two_ratio, info, statistic, values,
                 duration, folds, status):
    row = {
        "dataset": data_name, "task_type": task_type, "missing_rate": rate,
        "two_missing_ratio": two_ratio, "method": info.display_name,
        "method_year": info.year, "method_category": info.category,
        "method_category_cn": category_cn(info.category),
        "statistic": statistic, "duration_seconds": round(duration, 3),
        "successful_outer_folds": folds, "status": status,
        "implementation_source": info.source, "notes": info.notes,
    }
    row.update({m: finite_or_blank(float(values.get(m, np.nan))) for m in METRICS})
    return row


def fold_row_matches(row: Mapping[str, object], cfg: Mapping[str, object],
                     method: str, rate: float) -> bool:
    try:
        return (
            str(row.get("dataset", "")) == str(cfg["dataset"])
            and str(row.get("task_type", "")) == str(cfg["task_type"])
            and str(row.get("method", "")) == METHODS[method].display_name
            and math.isclose(float(row.get("missing_rate", -1)), rate,
                             rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(
                float(row.get("two_missing_ratio", -1)),
                float(cfg["two_missing_ratio"]),
                rel_tol=0.0, abs_tol=1e-12)
        )
    except (TypeError, ValueError):
        return False


def run_method_rate(method: str, rate: float, views: Sequence[np.ndarray],
                    y: np.ndarray, mask: np.ndarray, cfg: Mapping[str, object],
                    out_dir: Path, all_fold_rows: List[Dict[str, object]],
                    all_summary_rows: List[Dict[str, object]]) -> None:
    data_name = str(cfg["dataset"])
    rate_cfg = dict(cfg)
    rate_cfg["active_missing_rate"] = float(rate)
    num_classes = len(np.unique(y))
    device = torch.device(str(cfg["device"]))
    checkpoint = out_dir / f"checkpoint_{method}_{rate_tag(rate)}.npz"
    expected_checkpoint_hash = method_run_hash(cfg, method)
    oof_prob = np.full((len(y), num_classes), np.nan, dtype=np.float64)
    completed: set[int] = set()
    selected_epochs = np.zeros(int(cfg["outer_folds"]), dtype=np.int64)
    duration_so_far = 0.0
    resumed_checkpoint = False
    if bool(cfg["resume"]) and checkpoint.is_file():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved["config_hash"].item()) != expected_checkpoint_hash:
                raise RuntimeError(
                    f"checkpoint configuration mismatch: {checkpoint}; "
                    "use --overwrite")
            saved_prob = np.asarray(saved["oof_prob"], dtype=np.float64)
            saved_epochs = np.asarray(saved["selected_epochs"], dtype=np.int64)
            if saved_prob.shape != oof_prob.shape:
                raise RuntimeError(f"checkpoint probability shape mismatch: {checkpoint}")
            if saved_epochs.shape != selected_epochs.shape:
                raise RuntimeError(f"checkpoint epoch shape mismatch: {checkpoint}")
            oof_prob[:] = saved_prob
            selected_epochs[:] = saved_epochs
            completed = set(int(x) for x in saved["completed_folds"].tolist())
            duration_so_far = float(saved["duration_seconds"].item())
        valid_fold_ids = set(range(1, int(cfg["outer_folds"]) + 1))
        if not completed.issubset(valid_fold_ids):
            raise RuntimeError(f"checkpoint contains invalid fold IDs: {checkpoint}")
        resumed_checkpoint = True

    outer = StratifiedKFold(
        n_splits=int(cfg["outer_folds"]), shuffle=True,
        random_state=int(cfg["seed"]))
    matching_rows = [r for r in all_fold_rows
                     if fold_row_matches(r, cfg, method, rate)]
    all_fold_rows[:] = [r for r in all_fold_rows
                        if not fold_row_matches(r, cfg, method, rate)]
    rows_by_fold: Dict[int, Dict[str, object]] = {}
    if resumed_checkpoint:
        for row in matching_rows:
            try:
                fold_id = int(row.get("outer_fold", 0))
            except (TypeError, ValueError):
                continue
            if fold_id in completed and row.get("status") == "ok":
                rows_by_fold[fold_id] = dict(row)

    outer_splits = list(outer.split(np.arange(len(y)), y))
    # A valid checkpoint is authoritative.  Reconstruct any missing CSV row
    # from its saved out-of-fold probabilities instead of duplicating folds.
    for fold_id in sorted(completed):
        if fold_id not in rows_by_fold:
            _, test_idx = outer_splits[fold_id - 1]
            if not np.isfinite(oof_prob[test_idx]).all():
                raise RuntimeError(
                    f"checkpoint fold {fold_id} has incomplete predictions")
            metrics = score_probabilities(y[test_idx], oof_prob[test_idx], num_classes)
            rows_by_fold[fold_id] = fold_row(
                data_name, str(cfg["task_type"]), rate,
                float(cfg["two_missing_ratio"]), method, fold_id,
                int(selected_epochs[fold_id - 1]), metrics, 0.0,
                notes="CSV row reconstructed from a validated checkpoint.")
    method_rows = [rows_by_fold[k] for k in sorted(rows_by_fold)]
    all_fold_rows.extend(method_rows)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, 1):
        if outer_fold in completed:
            print(f"  [resume] outer fold {outer_fold}/{cfg['outer_folds']}")
            continue
        start = time.perf_counter()
        print(f"  outer fold {outer_fold}/{cfg['outer_folds']}: inner selection", flush=True)
        selected, curve = run_inner_selection(
            method, views, y, mask, train_idx, outer_fold, rate_cfg, device)
        print(f"    selected epoch={selected}, mean inner ACC={curve[selected]:.6f}")
        prob = train_outer_and_predict(
            method, views, y, mask, train_idx, test_idx, selected,
            outer_fold, rate_cfg, device)
        oof_prob[test_idx] = prob
        selected_epochs[outer_fold - 1] = selected
        metrics = score_probabilities(y[test_idx], prob, num_classes)
        elapsed = time.perf_counter() - start
        row = fold_row(data_name, str(cfg["task_type"]), rate,
                       float(cfg["two_missing_ratio"]), method,
                       outer_fold, selected, metrics, elapsed)
        method_rows.append(row)
        all_fold_rows.append(row)
        completed.add(outer_fold)
        duration_so_far += elapsed
        np.savez_compressed(
            checkpoint, config_hash=np.array(expected_checkpoint_hash),
            oof_prob=oof_prob, selected_epochs=selected_epochs,
            completed_folds=np.array(sorted(completed), dtype=np.int64),
            duration_seconds=np.array(duration_so_far))
        write_csv_atomic(out_dir / "comparison_outer_folds.csv", all_fold_rows, FOLD_FIELDS)
        print("    " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                               if np.isfinite(v)), flush=True)
    duration = duration_so_far
    all_summary_rows.extend(summary_rows(
        data_name, str(cfg["task_type"]), rate,
        float(cfg["two_missing_ratio"]), method, method_rows,
        y, oof_prob, num_classes, duration))
    write_csv_atomic(out_dir / "comparison_summary.csv", all_summary_rows, SUMMARY_FIELDS)
    write_csv_atomic(out_dir / "comparison_main_table.csv",
                     build_main_table(all_summary_rows), MAIN_TABLE_FIELDS)


def load_existing_rows(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Nested-CV missing-view comparison benchmark")
    p.add_argument("--data-folder", default=None,
                   help=("Folder containing labels_tr/te.csv and 1..3_tr/te.csv; "
                         "defaults to comparison_config.py, then auto-discovery"))
    p.add_argument("--dataset-name", default=None,
                   help="Name printed in every final table; defaults to the data-folder name")
    p.add_argument("--output-dir", default=None,
                   help="Defaults to OUTPUT_DIR in comparison_config.py")
    p.add_argument("--dimv-repo", default=CONFIG_DIMV_REPO,
                   help="Official DIMVImputation repository root")
    p.add_argument("--clclsa-repo", default=CONFIG_CLCLSA_REPO,
                   help="Official CLCLSA/CLUECL3 repository root")
    p.add_argument("--tmonet-repo", default=CONFIG_TMONET_REPO,
                   help="Official TMO-Net repository root")
    p.add_argument("--integrao-repo", default=CONFIG_INTEGRAO_REPO,
                   help="Official IntegrAO repository root")
    p.add_argument("--mogedn-repo", default=CONFIG_MOGEDN_REPO,
                   help="Official MOGEDN repository root")
    p.add_argument("--rusi-model-file", default=CONFIG_RUSI_MODEL_FILE,
                   help="RUSI main Python file; only CLUECL3 and its losses are imported")
    p.add_argument("--methods", type=parse_str_list, default=None,
                   help=("Comma list overriding comparison_config.py; keys: "
                         + ",".join(METHODS)))
    p.add_argument("--missing-rates", type=parse_float_list,
                   default=None,
                   help="Comma list overriding comparison_config.py")
    p.add_argument("--missing-rate", type=float, default=None,
                   help="Run one manually specified rate; overrides --missing-rates")
    p.add_argument("--missing-rate-mode", choices=["sample", "entry"], default="sample")
    p.add_argument("--two-missing-ratio", type=float, default=0.5)
    p.add_argument("--outer-folds", type=int, default=10)
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--eval-interval", type=int, default=25)
    p.add_argument("--seed", type=int, default=28)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use CUDA mixed precision for AMP-safe neural baselines")
    p.add_argument(
        "--fused-adamw", action=argparse.BooleanOptionalAction, default=True,
        help="Use fused AdamW when supported by the installed CUDA PyTorch")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    # Proposed-method settings.  Defaults mirror the current RUSI main program;
    # keep these fixed to the manuscript settings for an unbiased comparison.
    p.add_argument("--rusi-hidden-dims", type=parse_int_list,
                   default=list(CONFIG_RUSI_PARAMS["hidden_dims"]))
    p.add_argument("--rusi-prediction-dims", type=parse_int_list,
                   default=list(CONFIG_RUSI_PARAMS["prediction_dims"]))
    p.add_argument("--rusi-lr", type=float, default=CONFIG_RUSI_PARAMS["lr"])
    p.add_argument("--rusi-weight-decay", type=float,
                   default=CONFIG_RUSI_PARAMS["weight_decay"])
    p.add_argument("--rusi-warmup-epochs", type=int,
                   default=CONFIG_RUSI_PARAMS["warmup_epochs"])
    p.add_argument("--rusi-dropout", type=float,
                   default=CONFIG_RUSI_PARAMS["dropout"])
    p.add_argument("--rusi-temperature", type=float,
                   default=CONFIG_RUSI_PARAMS["temperature"])
    p.add_argument("--rusi-lambda-cil", type=float,
                   default=CONFIG_RUSI_PARAMS["lambda_cil"])
    p.add_argument("--rusi-contrastive-update-interval", type=int,
                   default=CONFIG_RUSI_PARAMS["contrastive_update_interval"])
    p.add_argument("--rusi-lambda-al", type=float,
                   default=CONFIG_RUSI_PARAMS["lambda_al"])
    p.add_argument("--rusi-lambda-imputation", type=float,
                   default=CONFIG_RUSI_PARAMS["lambda_imputation"])
    p.add_argument("--rusi-cross-sample-k", type=int,
                   default=CONFIG_RUSI_PARAMS["cross_sample_k"])
    p.add_argument("--rusi-knn-base-temperature", type=float,
                   default=CONFIG_RUSI_PARAMS["knn_base_temperature"])
    p.add_argument("--rusi-router-temperature", type=float,
                   default=CONFIG_RUSI_PARAMS["router_temperature"])
    p.add_argument("--rusi-router-update-interval", type=int,
                   default=CONFIG_RUSI_PARAMS["router_update_interval"])
    p.add_argument("--rusi-support-mode", choices=["masked", "complete"],
                   default=CONFIG_RUSI_PARAMS["support_mode"])
    p.add_argument("--rusi-use-cross-sample-impute",
                   action=argparse.BooleanOptionalAction,
                   default=CONFIG_RUSI_PARAMS["use_cross_sample_impute"])
    p.add_argument("--rusi-use-prototype-bank",
                   action=argparse.BooleanOptionalAction,
                   default=CONFIG_RUSI_PARAMS["use_prototype_bank"])
    p.add_argument("--rusi-use-action-router",
                   action=argparse.BooleanOptionalAction,
                   default=CONFIG_RUSI_PARAMS["use_action_router"])
    p.add_argument("--knn-neighbors", type=int, default=5)
    p.add_argument("--imputer-max-iter", type=int, default=5)
    p.add_argument("--mice-nearest-features", type=int, default=50)
    p.add_argument("--tmonet-kl-weight", type=float, default=0.01,
                   help="Official TMO-Net KL coefficient")
    p.add_argument("--tmonet-objective-weight", type=float, default=1.0,
                   help="Weight of the applicable official TMO-Net self objective")
    p.add_argument("--mogedn-recon-weight", type=float, default=0.5,
                   help="Official MOGEDN phase-3 latent reconstruction coefficient")
    p.add_argument("--graph-k", type=int, default=10)
    p.add_argument("--integrao-align-weight", type=float, default=1.0,
                   help="Official IntegrAO alignment-loss coefficient")
    # CLCLSA official defaults from main_clcl.py.  They are method-specific
    # architecture/loss settings, not parameters of the proposed method.
    p.add_argument("--clclsa-hidden-dim", type=int, default=300)
    p.add_argument("--clclsa-prediction-dims", type=parse_int_list,
                   default=[64, 32])
    p.add_argument("--clclsa-lambda-al", type=float, default=1.0)
    p.add_argument("--clclsa-lambda-co", type=float, default=0.02)
    p.add_argument("--clclsa-lambda-cl", type=float, default=0.05)
    p.add_argument("--dimv-cv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dimv-n-jobs", type=int, default=1,
                   help="DIMV covariance workers; keep 1 on Windows to avoid process/page-file exhaustion")
    p.add_argument("--dimv-alpha", type=float, default=0.1)
    p.add_argument("--dimv-alphas", type=parse_alpha_list, default=[0.01, 0.1, 1.0])
    p.add_argument("--dimv-cv-percent", type=float, default=30.0)
    p.add_argument("--dimv-corr-threshold", type=float, default=0.1)
    p.add_argument("--dimv-largest-features", type=int, default=20)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action="store_true",
                   help="Stop the entire benchmark when one optional method fails")
    p.add_argument("--dry-run", action="store_true")
    return p


def validate_args(args: argparse.Namespace) -> None:
    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown method keys: {unknown}; valid keys are {list(METHODS)}")
    if not args.methods:
        raise ValueError("--methods cannot be empty")
    if not (0 <= args.two_missing_ratio <= 1):
        raise ValueError("--two-missing-ratio must be in [0,1]")
    if args.missing_rate is not None and not 0 <= args.missing_rate < 1:
        raise ValueError("--missing-rate must be in [0,1)")
    if (not args.missing_rates
            or any(not 0 <= rate < 1 for rate in args.missing_rates)):
        raise ValueError("all configured missing rates must be in [0,1)")
    if len(set(args.missing_rates)) != len(args.missing_rates):
        raise ValueError("configured missing rates must not contain duplicates")
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("outer/inner folds must both be at least 2")
    if args.epochs < 1 or args.eval_interval < 1:
        raise ValueError("epochs/eval-interval must be positive")
    if not 0 <= args.warmup_epochs <= args.epochs:
        raise ValueError("warmup-epochs must be between 0 and epochs")
    if args.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be positive and weight-decay non-negative")
    if args.hidden_dim <= 0 or args.latent_dim <= 0:
        raise ValueError("hidden-dim and latent-dim must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    if not args.rusi_hidden_dims or args.rusi_hidden_dims[0] < 2:
        raise ValueError("rusi-hidden-dims must start with an integer >= 2")
    if not args.rusi_prediction_dims:
        raise ValueError("rusi-prediction-dims cannot be empty")
    if args.rusi_lr <= 0 or args.rusi_weight_decay < 0:
        raise ValueError("RUSI lr must be positive and weight decay non-negative")
    if not 0 <= args.rusi_dropout < 1:
        raise ValueError("rusi-dropout must be in [0,1)")
    if not 0 <= args.rusi_warmup_epochs <= args.epochs:
        raise ValueError("rusi-warmup-epochs must be between 0 and epochs")
    for name in ("rusi_cross_sample_k", "rusi_router_update_interval",
                 "rusi_contrastive_update_interval"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in ("rusi_weight_decay", "rusi_temperature", "rusi_lambda_cil",
                 "rusi_lambda_al", "rusi_lambda_imputation"):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    for name in ("rusi_knn_base_temperature", "rusi_router_temperature"):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.dimv_n_jobs == 0:
        raise ValueError("dimv-n-jobs cannot be zero")
    positive_names = (
        "knn_neighbors", "imputer_max_iter", "mice_nearest_features",
        "graph_k", "dimv_largest_features", "clclsa_hidden_dim",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if not args.clclsa_prediction_dims:
        raise ValueError("clclsa-prediction-dims cannot be empty")
    nonnegative_names = (
        "tmonet_kl_weight", "tmonet_objective_weight",
        "mogedn_recon_weight", "integrao_align_weight", "dimv_alpha",
        "dimv_corr_threshold", "clclsa_lambda_al", "clclsa_lambda_co",
        "clclsa_lambda_cl",
    )
    for name in nonnegative_names:
        if float(getattr(args, name)) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    if not 0 < args.dimv_cv_percent <= 100:
        raise ValueError("dimv-cv-percent must be in (0,100]")
    if args.missing_rate_mode == "entry" and max(args.missing_rates) > 2 / 3:
        raise ValueError("entry missing rate cannot exceed 2/3 with one retained view")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu")


def configure_torch_runtime(args: argparse.Namespace) -> torch.device:
    """Validate and report the effective compute device without silent fallback."""
    device = torch.device(str(args.device))
    if device.type != "cuda":
        return device
    # Resolve ``cuda`` to the current CUDA device and fail early for a bad index.
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"requested cuda:{index}, but only {torch.cuda.device_count()} CUDA device(s) exist")
    torch.cuda.set_device(index)
    # High matmul precision permits TF32 acceleration on Ampere-or-newer GPUs.
    # It affects only numerical execution, not folds, masks, objectives, or labels.
    torch.set_float32_matmul_precision("high")
    return torch.device(f"cuda:{index}")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.methods is None:
        unknown_switches = sorted(set(METHOD_SWITCHES) - set(METHODS))
        if unknown_switches:
            raise ValueError(f"comparison_config.py has unknown methods: {unknown_switches}")
        args.methods = [key for key in METHODS if METHOD_SWITCHES.get(key, False)]
    if args.missing_rate is not None:
        args.missing_rates = [float(args.missing_rate)]
    elif args.missing_rates is None:
        args.missing_rates = [float(x) for x in MISSING_RATES]
    validate_args(args)
    effective_device = configure_torch_runtime(args)
    args.device = str(effective_device)
    script_dir = Path(__file__).resolve().parent
    if "rusi" in args.methods:
        args.rusi_model_file = str(resolve_model_file(
            args.rusi_model_file, script_dir))
    data_folder = resolve_data_folder(
        args.data_folder if args.data_folder is not None else CONFIG_DATA_FOLDER,
        script_dir, allow_discovery=args.data_folder is None)
    output_value = args.output_dir or CONFIG_OUTPUT_DIR
    output_path = Path(output_value).expanduser()
    out_dir = (output_path if output_path.is_absolute()
               else script_dir / output_path).resolve()
    internal_dir = out_dir / "_internal"
    out_dir.mkdir(parents=True, exist_ok=True)
    internal_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for pattern in ("checkpoint_*.npz", "shared_mask_*.npz"):
            for p in internal_dir.glob(pattern):
                p.unlink()
        for name in ("comparison_outer_folds.csv", "comparison_summary.csv",
                     "comparison_main_table.csv", "experiment_manifest.json",
                     "failures.json", "paired_nested_cv_indices.npz"):
            p = internal_dir / name
            if p.exists(): p.unlink()
        for p in out_dir.glob("*_final_results.csv"):
            p.unlink()
        for p in out_dir.glob("*_final_results.txt"):
            p.unlink()
        # Remove legacy root-level detailed tables created by older versions.
        for name in ("comparison_outer_folds.csv", "comparison_summary.csv",
                     "comparison_main_table.csv", "failures.json",
                     "experiment_manifest.json"):
            legacy = out_dir / name
            if legacy.exists():
                legacy.unlink()

    views, y, class_values = load_merged(data_folder)
    validate_splits(y, args.outer_folds, args.inner_folds)
    cfg = vars(args).copy()
    cfg["data_folder"] = str(data_folder)
    cfg["output_dir"] = str(out_dir)
    cfg["dataset"] = str(
        args.dataset_name or CONFIG_DATASET_NAME or dataset_name(data_folder))
    cfg["task_type"] = "binary" if len(class_values) == 2 else "multiclass"
    cfg["protocol_version"] = PROTOCOL_VERSION
    # Disabled methods need neither an installed repository nor dependencies.
    # Per-method checkpoint hashes below still bind enabled methods to the exact
    # official source files used by that method.
    official_sources = audit_selected_sources(args.methods, cfg)
    cfg["official_source_sha256"] = {
        method: record["files"] for method, record in official_sources.items()
    }
    cfg["config_hash"] = config_hash(cfg)
    manifest_path = internal_dir / "experiment_manifest.json"
    verify_existing_manifest(
        manifest_path, str(cfg["config_hash"]), bool(args.resume),
        bool(args.overwrite))
    if bool(args.resume) and manifest_path.is_file() and not bool(args.overwrite):
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        previous_audit = previous_manifest.get("official_source_audit", {})
        if isinstance(previous_audit, dict):
            official_sources = {**previous_audit, **official_sources}
            cfg["official_source_sha256"] = {
                method: record.get("files", {})
                for method, record in official_sources.items()
                if isinstance(record, dict)
            }
    save_cv_indices(y, cfg, internal_dir / "paired_nested_cv_indices.npz")
    manifest = {
        "configuration": cfg, "configuration_sha256": cfg["config_hash"],
        "samples": len(y), "classes_encoded": len(class_values),
        "original_class_values": class_values.tolist(),
        "view_dimensions": [x.shape[1] for x in views],
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (torch.cuda.get_device_name(effective_device)
                        if effective_device.type == "cuda" else None),
        "method_metadata": {k: METHODS[k].__dict__ for k in METHODS},
        "official_source_audit": official_sources,
        "paired_fairness_protocol": {
            "outer_random_state": int(args.seed),
            "inner_random_state": "seed + 1000 * outer_fold_id",
            "mask_random_state": "seed + 500000 for every missing rate",
            "normalization": "observed-training-only per-feature Min-Max",
            "epoch_selection": "maximum inner-fold mean ACC; 1e-12 tie -> earliest",
            "outer_test_use": "evaluated once after outer-train refit",
            "cv_indices_file": "paired_nested_cv_indices.npz",
        },
        "compute_policy": {
            "cuda": (
                "all PyTorch neural models, tensor operations, and graph-neighbor "
                "construction when --device is cuda"),
            "amp": (
                "enabled only for model paths whose official operations are stable "
                "in mixed precision"),
            "fused_optimizer": "fused AdamW requested on supported CUDA builds",
            "fold_cache": (
                "fixed validation/test preprocessing and graph structures are cached; "
                "model predictions are never cached"),
            "cpu_required": (
                "scikit-learn imputers, the official DIMV imputer, metrics, and I/O"),
        },
        "scientific_warning": (
            "Recent baselines import model/loss source directly from the author "
            "repositories recorded above. Their original test-set checkpoint "
            "selection, transductive graphs, external checkpoints, or cohort-wide "
            "missingness assumptions are intentionally replaced by the common "
            "leakage-free nested-CV adapters. Describe them as official model/loss "
            "implementations under a unified protocol adaptation, not as exact "
            "end-to-end reproductions of each paper's original experiment."),
    }
    write_json_atomic(manifest_path, manifest)

    total_fits = len(args.methods) * len(args.missing_rates) * (
        args.outer_folds * args.inner_folds + args.outer_folds)
    print("=" * 78)
    print(f"dataset={cfg['dataset']}  n={len(y)}  classes={len(class_values)}")
    print(f"views={[x.shape[1] for x in views]}")
    print(f"methods={args.methods}")
    print(f"missing_rates={args.missing_rates}")
    print(f"nested CV={args.outer_folds} outer x {args.inner_folds} inner")
    print(f"maximum model fits={total_fits:,}; epochs per inner fit={args.epochs}")
    if effective_device.type == "cuda":
        print(f"compute={effective_device} ({torch.cuda.get_device_name(effective_device)})")
        print(f"AMP={'on' if args.amp else 'off'}; fused AdamW="
              f"{'requested' if args.fused_adamw else 'off'}")
        print("GPU: neural models and torch graph-neighbor search; "
              "CPU: sklearn imputers, DIMV imputation, metrics and file I/O")
        print("speed cache: fixed validation/test imputations, CUDA tensors and "
              "inductive graphs are reused within each fit")
    else:
        print("compute=cpu (CUDA is not active; neural training will be slower)")
    print("=" * 78)
    if args.dry_run:
        print("dry-run complete; no model was trained")
        return

    fold_path = internal_dir / "comparison_outer_folds.csv"
    summary_path = internal_dir / "comparison_summary.csv"
    valid_method_names = {info.display_name for info in METHODS.values()}
    all_fold_rows = load_existing_rows(fold_path) if args.resume else []
    all_summary_rows = load_existing_rows(summary_path) if args.resume else []
    # Migration from the previous method set: keep valid completed baselines,
    # but do not let obsolete Iterative-RF rows leak into regenerated tables.
    all_fold_rows = [row for row in all_fold_rows
                     if str(row.get("method", "")) in valid_method_names]
    all_summary_rows = [row for row in all_summary_rows
                        if str(row.get("method", "")) in valid_method_names]
    failure_path = internal_dir / "failures.json"
    failures: List[Dict[str, object]] = []
    if args.resume and failure_path.is_file():
        try:
            with failure_path.open("r", encoding="utf-8") as fh:
                loaded_failures = json.load(fh)
            if isinstance(loaded_failures, list):
                failures = [dict(x) for x in loaded_failures
                            if isinstance(x, dict)
                            and str(x.get("method", "")) in METHODS]
        except (OSError, json.JSONDecodeError):
            failures = []
    for rate in args.missing_rates:
        # The user's main code is launched once per rate and always uses this
        # seed, so every comparator must do the same for exact paired masks.
        mask_seed = args.seed + 500000
        mask = generate_shared_mask(
            len(y), rate, args.missing_rate_mode,
            args.two_missing_ratio, mask_seed)
        np.savez_compressed(internal_dir / f"shared_mask_{rate_tag(rate)}.npz",
                            mask=mask, seed=np.array(mask_seed), labels=y)
        actual_sample = float(np.mean(mask.sum(1) < 3))
        actual_entry = float(np.mean(mask == 0))
        print(f"\nmissing_rate={rate:.2f} (incomplete patients={actual_sample:.4f}, "
              f"missing entries={actual_entry:.4f})")
        for method in args.methods:
            # Rebuild summaries deterministically instead of duplicating rows.
            all_summary_rows = [r for r in all_summary_rows
                                if not fold_row_matches(r, cfg, method, rate)]
            failures = [f for f in failures
                        if not (str(f.get("method", "")) == method
                                and math.isclose(
                                    float(f.get("missing_rate", -1)), rate,
                                    rel_tol=0.0, abs_tol=1e-12))]
            print(f"\n[{method}] {METHODS[method].display_name}")
            try:
                run_method_rate(method, rate, views, y, mask, cfg, internal_dir,
                                all_fold_rows, all_summary_rows)
                write_final_metric_tables(
                    out_dir, all_summary_rows, str(cfg["dataset"]),
                    str(cfg["task_type"]), print_tables=False)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                print(f"  FAILED: {msg}", file=sys.stderr)
                traceback.print_exc()
                failures.append({"method": method, "missing_rate": rate, "error": msg})
                info = METHODS[method]
                failed = _summary_row(
                    cfg["dataset"], cfg["task_type"], rate,
                    args.two_missing_ratio, info,
                    "failed", {}, 0.0, 0, "failed")
                failed["notes"] = info.notes + " Error: " + msg
                all_summary_rows.append(failed)
                write_csv_atomic(summary_path, all_summary_rows, SUMMARY_FIELDS)
                write_csv_atomic(internal_dir / "comparison_main_table.csv",
                                 build_main_table(all_summary_rows), MAIN_TABLE_FIELDS)
                write_final_metric_tables(
                    out_dir, all_summary_rows, str(cfg["dataset"]),
                    str(cfg["task_type"]), print_tables=False)
                if args.fail_fast:
                    raise

    write_csv_atomic(fold_path, all_fold_rows, FOLD_FIELDS)
    write_csv_atomic(summary_path, all_summary_rows, SUMMARY_FIELDS)
    main_table_path = internal_dir / "comparison_main_table.csv"
    write_csv_atomic(main_table_path, build_main_table(all_summary_rows),
                     MAIN_TABLE_FIELDS)
    final_tables = write_final_metric_tables(
        out_dir, all_summary_rows, str(cfg["dataset"]), str(cfg["task_type"]),
        print_tables=True)
    write_json_atomic(failure_path, failures)
    print("\nCompleted.")
    print("Final result tables:")
    for path in final_tables:
        print(f"  {path}")
    if failures:
        print(f"{len(failures)} method/rate runs failed; see _internal/failures.json")


if __name__ == "__main__":
    main()
