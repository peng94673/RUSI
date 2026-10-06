"""Data loading, missing-view simulation and mask utilities."""

import os

import numpy as np
import torch


def load_raw_data(data_folder):
    """Load raw arrays so every fold can fit its own normalization."""
    def load_label_vector(filename):
        values = np.asarray(np.loadtxt(filename, delimiter=","))
        if values.ndim == 0:
            return values.reshape(1)
        if values.ndim == 1:
            return values
        if values.ndim == 2 and 1 in values.shape:
            return values.reshape(-1)
        raise ValueError(
            f"{os.path.basename(filename)} must contain exactly one label "
            "row or column")

    labels_train_raw = load_label_vector(
        os.path.join(data_folder, "labels_tr.csv"))
    labels_test_raw = load_label_vector(
        os.path.join(data_folder, "labels_te.csv"))
    for name, values in (
            ("labels_tr.csv", labels_train_raw),
            ("labels_te.csv", labels_test_raw)):
        if values.size == 0:
            raise ValueError(f"{name} is empty")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinity")
        if not np.equal(values, np.round(values)).all():
            raise ValueError(f"{name} must contain integer labels")

    labels_train_raw = labels_train_raw.astype(np.int64)
    labels_test_raw = labels_test_raw.astype(np.int64)
    # The two supplied files are merged before nested CV, so label encoding
    # must be defined from their union rather than from the old training file.
    class_values = np.unique(np.concatenate(
        (labels_train_raw, labels_test_raw)))
    if class_values.size < 2:
        raise ValueError("merged data must contain at least two classes")

    # CrossEntropyLoss requires contiguous labels in [0, C).
    labels_train = np.searchsorted(
        class_values, labels_train_raw).astype(np.int64)
    labels_test = np.searchsorted(
        class_values, labels_test_raw).astype(np.int64)

    train_views, test_views = [], []
    for view in range(1, 4):
        train_view = np.loadtxt(
            os.path.join(data_folder, f"{view}_tr.csv"),
            delimiter=",", ndmin=2)
        test_view = np.loadtxt(
            os.path.join(data_folder, f"{view}_te.csv"),
            delimiter=",", ndmin=2)
        if not (np.isfinite(train_view).all()
                and np.isfinite(test_view).all()):
            raise ValueError(
                f"view {view} contains NaN or infinity")
        if train_view.shape[1] != test_view.shape[1]:
            raise ValueError(
                f"view {view} train/test feature counts differ")
        if train_view.shape[0] != labels_train.shape[0]:
            raise ValueError(
                f"view {view} training rows do not match labels")
        if test_view.shape[0] != labels_test.shape[0]:
            raise ValueError(
                f"view {view} test rows do not match labels")
        train_views.append(train_view)
        test_views.append(test_view)

    return (
        train_views, test_views, labels_train, labels_test,
        class_values)

def get_mask(view_num, alldata_len, missing_rate, rate_mode="sample",
             two_missing_ratio=0.5):
    """Generate a binary mask while keeping at least one view per patient.

    ``rate_mode='sample'``: ``missing_rate`` is the exact fraction of patients
    with at least one missing view. ``two_missing_ratio`` controls how many of
    those patients miss two views in the three-view setting.

    ``rate_mode='entry'``: ``missing_rate`` is the exact fraction of missing
    sample-view entries, up to integer rounding.
    """
    if view_num < 2:
        raise ValueError(f"view_num must be >= 2, got {view_num}")
    if alldata_len < 0:
        raise ValueError(f"alldata_len must be >= 0, got {alldata_len}")
    if not (0.0 <= missing_rate < 1.0):
        raise ValueError(f"missing_rate must be in [0, 1), got {missing_rate}")
    if rate_mode not in {"sample", "entry"}:
        raise ValueError("rate_mode must be either 'sample' or 'entry'")
    if not (0.0 <= two_missing_ratio <= 1.0):
        raise ValueError("two_missing_ratio must be in [0, 1]")

    mask = np.ones((alldata_len, view_num), dtype=np.float32)
    if alldata_len == 0 or missing_rate == 0:
        return mask

    if rate_mode == "sample":
        n_incomplete = int(round(alldata_len * missing_rate))
        incomplete_rows = np.random.choice(
            alldata_len, size=n_incomplete, replace=False)
        max_missing = min(2, view_num - 1)
        # Allocate the requested two-missing proportion deterministically up to
        # integer rounding. Independent Bernoulli draws made small validation
        # sets much harder or easier than the test set by chance.
        n_two_missing = (int(round(n_incomplete * two_missing_ratio))
                         if max_missing >= 2 else 0)
        for position, row in enumerate(incomplete_rows):
            n_missing = 2 if position < n_two_missing else 1
            missing_views = np.random.choice(
                view_num, size=n_missing, replace=False)
            mask[row, missing_views] = 0.0
    else:
        max_rate = (view_num - 1) / view_num
        if missing_rate > max_rate:
            raise ValueError(
                f"entry missing_rate={missing_rate} is infeasible when every "
                f"patient must retain a view; maximum is {max_rate:.6f}")
        n_missing = int(round(alldata_len * view_num * missing_rate))
        protected = np.random.randint(0, view_num, size=alldata_len)
        rows = np.repeat(np.arange(alldata_len), view_num)
        views = np.tile(np.arange(view_num), alldata_len)
        eligible = views != protected[rows]
        eligible_rows = rows[eligible]
        eligible_views = views[eligible]
        chosen = np.random.choice(
            eligible_rows.size, size=n_missing, replace=False)
        mask[eligible_rows[chosen], eligible_views[chosen]] = 0.0

    if np.any(mask.sum(axis=1) == 0):
        raise RuntimeError("internal error: generated an all-zero mask row")
    return mask

def describe_mask(mask, name="mask"):
    """Print and return the actual missingness statistics."""
    arr = np.asarray(mask)
    missing_per_sample = (arr == 0).sum(axis=1)
    stats = {
        "entry_missing_rate": float((arr == 0).mean()),
        "incomplete_patient_rate": float((missing_per_sample > 0).mean()),
        "one_missing_rate": float((missing_per_sample == 1).mean()),
        "two_missing_rate": float((missing_per_sample == 2).mean()),
    }
    print(
        f"[x] {name}: entry_missing={stats['entry_missing_rate']:.4f}, "
        f"incomplete_patients={stats['incomplete_patient_rate']:.4f}, "
        f"one_missing={stats['one_missing_rate']:.4f}, "
        f"two_missing={stats['two_missing_rate']:.4f}")
    return stats

def apply_existing_mask(data_list, mask_np, device, split_name="mask"):
    """Apply one previously generated mask to all three views."""
    describe_mask(mask_np, split_name)
    mask = torch.from_numpy(mask_np).to(device)
    masked = [
        data * mask[:, view:view + 1]
        for view, data in enumerate(data_list)
    ]
    return masked, mask
