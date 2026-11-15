"""Leakage-safe adapters for the missing-view comparison benchmark.

Recent model bodies and objectives are imported from the authors' repositories
through :mod:`official_model_loader`.  This file owns only the protocol bridge:
fold-local fitting, patient-specific masks and inductive held-out inference.
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.extmath import randomized_svd

from official_model_loader import (
    load_clclsa, load_dimv, load_integrao, load_mogedn, load_rusi,
    load_tmonet,
)


ArrayList = List[np.ndarray]
# Importing this experimental module registers IterativeImputer in sklearn.
enable_iterative_imputer


@dataclass(frozen=True)
class MethodInfo:
    key: str
    display_name: str
    year: str
    category: str
    source: str
    notes: str


METHODS: Dict[str, MethodInfo] = {
    "rusi": MethodInfo(
        "rusi", "RUSI (ours)", "proposed", "selective_imputation",
        "user RUSI model source (SHA-256 audited)",
        "Reliability- and utility-guided multi-expert selective imputation; "
        "same paired masks and nested-CV protocol as every comparator."),
    "zero_mask": MethodInfo(
        "zero_mask", "Zero-fill + mask", "classic", "direct",
        "local common classifier",
        "No imputation; zero-filled blocks and the 3-view mask are inputs."),
    "mean": MethodInfo(
        "mean", "Mean imputation", "classic", "imputation",
        "scikit-learn SimpleImputer", "Train-fold feature means only."),
    "knn": MethodInfo(
        "knn", "KNN imputation", "classic", "imputation",
        "scikit-learn KNNImputer", "Imputer fitted on the train fold only."),
    "mice": MethodInfo(
        "mice", "MICE", "classic", "imputation",
        "scikit-learn IterativeImputer + BayesianRidge",
        "Iterative chained equations; train-fold fit only."),
    "svd": MethodInfo(
        "svd", "SVD imputation", "classic", "imputation",
        "local train-fold low-rank SVD imputer",
        "Training-fold low-rank basis; held-out rows use observed features only."),
    "dimv": MethodInfo(
        "dimv", "DIMV", "2023", "imputation",
        "official DIMVImputation repository source",
        "Official conditional-expectation imputer; fitted on optimization-fold data only."),
    "clclsa": MethodInfo(
        "clclsa", "CLCLSA", "2024", "direct",
        "official CLCLSA/CLUECL3 repository source",
        "Official CLUECL3 model and internal losses; nested-CV checkpoint selection."),
    "tmonet": MethodInfo(
        "tmonet", "TMO-Net", "2024", "direct",
        "official TMO-Net repository source (MIT)",
        "Official body/head and KL/reconstruction terms with patient-mask adaptation; "
        "no external checkpoint or complete-cohort adversarial pretraining."),
    "integrao": MethodInfo(
        "integrao", "IntegrAO", "2025", "direct",
        "official IntegrAO repository source (MIT)",
        "Official GraphSAGE and objective; inductive graphs prevent held-out-to-train messages."),
    "mogedn": MethodInfo(
        "mogedn", "MOGEDN", "2025", "direct",
        "official MOGEDN repository source",
        "Official GCN/decoder/VCDN and phase-3 focal/MSE terms; fold-local training "
        "without external checkpoints."),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Match the user's main nested-CV implementation exactly.  This is slower
    # than cuDNN benchmarking on some GPUs, but prevents method/fold results
    # from changing because a different convolution/kernel algorithm was picked.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def score_probabilities(y: np.ndarray, prob: np.ndarray,
                        num_classes: int) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    pred = prob.argmax(axis=1)
    out: Dict[str, float] = {"ACC": float(accuracy_score(y, pred))}
    if num_classes == 2:
        out["F1"] = float(f1_score(y, pred, average="binary", zero_division=0))
        try:
            out["AUC"] = float(roc_auc_score(y, prob[:, 1]))
        except ValueError:
            out["AUC"] = float("nan")
        out["F1_weighted"] = float("nan")
        out["F1_macro"] = float("nan")
    else:
        out["F1"] = float("nan")
        out["AUC"] = float("nan")
        out["F1_weighted"] = float(
            f1_score(y, pred, average="weighted", zero_division=0))
        out["F1_macro"] = float(
            f1_score(y, pred, average="macro", zero_division=0))
    return out


class FoldNormalizer:
    """User-model-compatible per-view Min-Max normalization.

    Statistics are fitted from observed rows of the current optimization split
    only, exactly matching the leakage-safe preprocessing in the main model.
    """

    def __init__(self) -> None:
        self.means: ArrayList = []
        self.stds: ArrayList = []

    def fit(self, views: Sequence[np.ndarray], mask: np.ndarray) -> "FoldNormalizer":
        self.means, self.stds = [], []
        for v, x in enumerate(views):
            observed = mask[:, v].astype(bool)
            if not observed.any():
                raise ValueError(f"training split contains no observed samples for view {v + 1}")
            fitting = np.asarray(x[observed], dtype=np.float64)
            train_min = fitting.min(axis=0)
            scale = (fitting - train_min).max(axis=0) + 1e-10
            self.means.append(train_min.astype(np.float32))
            self.stds.append(scale.astype(np.float32))
        return self

    def transform(self, views: Sequence[np.ndarray]) -> ArrayList:
        return [((np.asarray(x, dtype=np.float32) - m) / s).astype(np.float32)
                for x, m, s in zip(views, self.means, self.stds)]


def masked_concat(views: Sequence[np.ndarray], mask: np.ndarray,
                  nan_missing: bool) -> np.ndarray:
    blocks = []
    for v, x in enumerate(views):
        z = np.array(x, dtype=np.float64, copy=True)
        missing = ~mask[:, v].astype(bool)
        z[missing, :] = np.nan if nan_missing else 0.0
        blocks.append(z)
    return np.concatenate(blocks, axis=1)


class CommonMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 hidden: int, dropout: float) -> None:
        super().__init__()
        h2 = max(16, hidden // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FastSVDImputer:
    """Fast leakage-safe low-rank imputer for block-missing multi-omics.

    ``fit`` learns feature means and a randomized-SVD basis from the current
    optimization fold only.  ``transform`` estimates each held-out row's latent
    scores from that row's observed features; it never refits on evaluation
    rows or uses other evaluation samples.
    """

    def __init__(self, rank: int, power_iterations: int,
                 refinement_steps: int, ridge: float, seed: int) -> None:
        self.rank = int(rank)
        self.power_iterations = int(power_iterations)
        self.refinement_steps = int(refinement_steps)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self.mean_: Optional[np.ndarray] = None
        self.minimum_: Optional[np.ndarray] = None
        self.maximum_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "FastSVDImputer":
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("SVD imputer expects a 2-D matrix")
        observed = np.isfinite(values)
        counts = observed.sum(axis=0)
        if np.any(counts == 0):
            bad = np.where(counts == 0)[0][:10].tolist()
            raise ValueError(f"training fold has entirely missing features: {bad}")
        self.mean_ = np.nanmean(values, axis=0)
        self.minimum_ = np.nanmin(values, axis=0)
        self.maximum_ = np.nanmax(values, axis=0)
        missing = ~observed
        filled = np.where(observed, values, self.mean_[None, :])
        maximum_rank = min(values.shape[0] - 1, values.shape[1] - 1)
        if maximum_rank < 1:
            raise ValueError("SVD imputer needs at least two rows and two features")
        effective_rank = min(self.rank, maximum_rank)
        for step in range(self.refinement_steps + 1):
            centered = filled - self.mean_[None, :]
            _, _, components = randomized_svd(
                centered, n_components=effective_rank,
                n_iter=self.power_iterations,
                random_state=self.seed + step)
            self.components_ = np.asarray(components, dtype=np.float64)
            if step < self.refinement_steps and missing.any():
                reconstructed = self._reconstruct(values)
                filled[missing] = reconstructed[missing]
        return self

    def _check_fitted(self) -> None:
        if any(value is None for value in (
                self.mean_, self.minimum_, self.maximum_, self.components_)):
            raise RuntimeError("SVD imputer is not fitted")

    def _reconstruct(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.mean_ is not None and self.components_ is not None
        values = np.asarray(x, dtype=np.float64)
        if values.shape[1] != self.mean_.shape[0]:
            raise ValueError("SVD transform feature count differs from fitted data")
        observed = np.isfinite(values)
        centered = np.where(observed, values - self.mean_[None, :], 0.0)
        reconstruction = np.tile(self.mean_, (len(values), 1))

        # Missing-view simulation yields only a handful of observation patterns.
        # Solve one small ridge system per pattern rather than per patient.
        unique_patterns, inverse = np.unique(observed, axis=0, return_inverse=True)
        identity = np.eye(self.components_.shape[0], dtype=np.float64)
        for pattern_id, pattern in enumerate(unique_patterns):
            rows = np.where(inverse == pattern_id)[0]
            columns = np.where(pattern)[0]
            if columns.size == 0:
                continue
            basis_observed = self.components_[:, columns]
            gram = basis_observed @ basis_observed.T + self.ridge * identity
            right = centered[np.ix_(rows, columns)] @ basis_observed.T
            try:
                scores = np.linalg.solve(gram, right.T).T
            except np.linalg.LinAlgError:
                scores = right @ np.linalg.pinv(gram)
            reconstruction[rows] = scores @ self.components_ + self.mean_[None, :]
        return reconstruction

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.minimum_ is not None and self.maximum_ is not None
        values = np.asarray(x, dtype=np.float64)
        missing = ~np.isfinite(values)
        if not missing.any():
            return values.copy()
        reconstruction = self._reconstruct(values)
        reconstruction = np.clip(
            reconstruction, self.minimum_[None, :], self.maximum_[None, :])
        result = values.copy()
        result[missing] = reconstruction[missing]
        return result


class BaseAdapter:
    info: MethodInfo

    def _configure_training(self, device: torch.device,
                            cfg: Mapping[str, object],
                            *, amp_safe: bool = True) -> None:
        """Configure a common CUDA/AMP training path for every neural baseline."""
        self.device = device
        self.max_grad_norm = float(cfg["max_grad_norm"])
        self.amp_enabled = bool(cfg.get("amp", True)) and device.type == "cuda" and amp_safe
        self.scaler = (
            torch.amp.GradScaler("cuda") if self.amp_enabled else None)

    @staticmethod
    def _query_key(views: ArrayList, mask: np.ndarray) -> tuple:
        """Identity key for the fixed validation/test cohort used by nested CV.

        During one model fit the runner evaluates the same validation arrays at
        every checkpoint.  Caching their deterministic preprocessing avoids
        repeated CPU imputation, host-to-device copies, and graph construction
        without caching model outputs (which must change as training proceeds).
        """
        return (
            id(mask), tuple(mask.shape), str(mask.dtype),
            tuple((id(x), tuple(x.shape), str(x.dtype)) for x in views),
        )

    def _autocast(self):
        if self.amp_enabled:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def _backward_step(self, loss: torch.Tensor) -> None:
        """Backward, unscale, clip, and update without changing the objective."""
        if self.scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            self.optim.step()
            return
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optim)
        self.scaler.update()

    @staticmethod
    def _make_adamw(parameters, device: torch.device,
                    cfg: Mapping[str, object]) -> torch.optim.Optimizer:
        kwargs = {
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg["weight_decay"]),
        }
        # Fused AdamW reduces optimizer-launch overhead on supported CUDA builds.
        if device.type == "cuda" and bool(cfg.get("fused_adamw", True)):
            try:
                return torch.optim.AdamW(parameters, fused=True, **kwargs)
            except (TypeError, RuntimeError):
                pass
        return torch.optim.AdamW(parameters, **kwargs)

    def train_epoch(self, views: ArrayList, mask: np.ndarray,
                    y: np.ndarray) -> float:
        raise NotImplementedError

    def predict(self, views: ArrayList, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RUSIAdapter(BaseAdapter):
    """Fold-level bridge to the user's RUSI model and original joint objective.

    This deliberately imports only ``CLUECL3`` from the user's program.  The
    program's top-level trainer is not used because the benchmark already owns
    the outer/inner cross-validation loops.
    """

    def __init__(self, train_views: ArrayList, train_mask: np.ndarray,
                 num_classes: int, device: torch.device,
                 cfg: Mapping[str, object]) -> None:
        self.info = METHODS["rusi"]
        self.cfg = dict(cfg)
        self._configure_training(device, cfg, amp_safe=True)
        self.scheduler_warmup_epochs = int(cfg["rusi_warmup_epochs"])
        module, _ = load_rusi(cfg["rusi_model_file"])
        hidden_dims = [int(x) for x in cfg["rusi_hidden_dims"]]
        prediction_dims = [int(x) for x in cfg["rusi_prediction_dims"]]
        prediction = {view: list(prediction_dims) for view in range(3)}
        self.model = module.CLUECL3(
            [int(x.shape[1]) for x in train_views], hidden_dims,
            int(num_classes), float(cfg["rusi_dropout"]), prediction,
        ).to(device)

        self.optim = self._make_adamw_with_rusi_lr(cfg)
        self.active_missing_rate = float(cfg.get("active_missing_rate", 0.0))
        self.support_mode = str(cfg["rusi_support_mode"])
        self.train_mask_np = np.asarray(train_mask, dtype=np.float32)
        self.train_mask = torch.as_tensor(
            self.train_mask_np, dtype=torch.float32, device=device)
        self.train_original = [
            torch.as_tensor(x, dtype=torch.float32, device=device)
            for x in train_views
        ]
        self.train_data = [
            x * self.train_mask[:, view:view + 1]
            for view, x in enumerate(self.train_original)
        ]
        self.train_y: Optional[torch.Tensor] = None
        self.current_epoch = 0
        self._query_cache: Dict[tuple, tuple[List[torch.Tensor], torch.Tensor]] = {}

    def _make_adamw_with_rusi_lr(
            self, cfg: Mapping[str, object]) -> torch.optim.Optimizer:
        local_cfg = dict(cfg)
        local_cfg["lr"] = float(cfg["rusi_lr"])
        local_cfg["weight_decay"] = float(cfg["rusi_weight_decay"])
        return self._make_adamw(self.model.parameters(), self.device, local_cfg)

    def train_epoch(self, views: ArrayList, mask: np.ndarray,
                    y: np.ndarray) -> float:
        del views, mask
        if self.train_y is None:
            self.train_y = torch.as_tensor(
                y, dtype=torch.long, device=self.device)
        self.model.train()
        self.optim.zero_grad(set_to_none=True)
        epoch = self.current_epoch
        router_interval = int(self._cfg("rusi_router_update_interval"))
        contrastive_interval = int(
            self._cfg("rusi_contrastive_update_interval"))
        with self._autocast():
            if self.active_missing_rate > 0:
                support = (self.train_original if self.support_mode == "complete"
                           else self.train_data)
                support_mask = (None if self.support_mode == "complete"
                                else self.train_mask)
                loss, _, _ = self.model.train_missing_cg(
                    self.train_data, self.train_mask, self.train_y,
                    aux_loss=float(self._cfg("rusi_lambda_al")) > 0,
                    lambda_al=float(self._cfg("rusi_lambda_al")),
                    original_data_list=support,
                    support_mask=support_mask,
                    use_cross_sample_impute=bool(
                        self._cfg("rusi_use_cross_sample_impute")),
                    use_prototype_bank=bool(
                        self._cfg("rusi_use_prototype_bank")),
                    lambda_imputation=float(
                        self._cfg("rusi_lambda_imputation")),
                    cross_sample_k=int(self._cfg("rusi_cross_sample_k")),
                    knn_base_temperature=float(
                        self._cfg("rusi_knn_base_temperature")),
                    support_labels=self.train_y,
                    contrastive_loss=(
                        float(self._cfg("rusi_temperature")) > 0
                        and float(self._cfg("rusi_lambda_cil")) > 0
                        and epoch % contrastive_interval == 0),
                    temperature=float(self._cfg("rusi_temperature")),
                    lambda_cil=float(self._cfg("rusi_lambda_cil")),
                    router_temperature=float(
                        self._cfg("rusi_router_temperature")),
                    compute_router_supervision=(epoch % router_interval == 0),
                    use_action_router=bool(
                        self._cfg("rusi_use_action_router")),
                    record_loss=False,
                )
            else:
                loss, _, _ = self.model(
                    self.train_data, self.train_y,
                    aux_loss=float(self._cfg("rusi_lambda_al")) > 0,
                    lambda_al=float(self._cfg("rusi_lambda_al")),
                    record_loss=False,
                )
            loss = loss.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"RUSI produced a non-finite loss at epoch {epoch + 1}")
        self._backward_step(loss)
        self.current_epoch += 1
        return float(loss.detach())

    def _cfg(self, name: str) -> object:
        return self.cfg[name]

    @torch.inference_mode()
    def predict(self, views: ArrayList, mask: np.ndarray) -> np.ndarray:
        self.model.eval()
        key = self._query_key(views, mask)
        cached = self._query_cache.get(key)
        if cached is None:
            mask_tensor = torch.as_tensor(
                mask, dtype=torch.float32, device=self.device)
            query = [
                torch.as_tensor(x, dtype=torch.float32, device=self.device)
                * mask_tensor[:, view:view + 1]
                for view, x in enumerate(views)
            ]
            cached = (query, mask_tensor)
            self._query_cache[key] = cached
        query, mask_tensor = cached
        with self._autocast():
            if self.active_missing_rate > 0:
                support = (self.train_original if self.support_mode == "complete"
                           else self.train_data)
                support_mask = (None if self.support_mode == "complete"
                                else self.train_mask)
                logits = self.model.infer_on_missing(
                    query, mask_tensor,
                    support_data_list=support,
                    support_labels=self.train_y,
                    support_mask=support_mask,
                    use_cross_sample_impute=bool(
                        self._cfg("rusi_use_cross_sample_impute")),
                    use_prototype_bank=bool(
                        self._cfg("rusi_use_prototype_bank")),
                    cross_sample_k=int(self._cfg("rusi_cross_sample_k")),
                    knn_base_temperature=float(
                        self._cfg("rusi_knn_base_temperature")),
                    exclude_self=False,
                    router_temperature=float(
                        self._cfg("rusi_router_temperature")),
                    use_action_router=(
                        bool(self._cfg("rusi_use_action_router"))
                        and float(self._cfg("rusi_lambda_imputation")) > 0),
                )
            else:
                logits = self.model.infer(query)
            return F.softmax(logits, dim=1).float().cpu().numpy()


class ImputerMLPAdapter(BaseAdapter):
    def __init__(self, key: str, train_views: ArrayList, train_mask: np.ndarray,
                 num_classes: int, device: torch.device, cfg: Mapping[str, object]) -> None:
        self.info = METHODS[key]
        self.key = key
        self._configure_training(device, cfg)
        self.append_mask = key == "zero_mask"
        self._identity_imputation = (
            not self.append_mask
            and float(cfg.get("active_missing_rate", -1.0)) == 0.0)
        self.imputer = (None if self._identity_imputation
                        else self._make_imputer(key, cfg))
        train_x = masked_concat(train_views, train_mask, nan_missing=not self.append_mask)
        if self.append_mask:
            train_x = np.concatenate([train_x, train_mask.astype(np.float64)], axis=1)
        else:
            if not self._identity_imputation:
                self.imputer.fit(train_x)
            train_x = self._safe_transform(train_x)
        self.train_x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
        self.model = CommonMLP(
            self.train_x.shape[1], num_classes, int(cfg["hidden_dim"]),
            float(cfg["dropout"])).to(device)
        self.optim = self._make_adamw(self.model.parameters(), device, cfg)
        self.num_classes = num_classes
        self._cached_feature_key: Optional[tuple] = None
        self._cached_features: Optional[torch.Tensor] = None

    @staticmethod
    def _make_imputer(key: str, cfg: Mapping[str, object]):
        if key == "zero_mask":
            return None
        if key == "mean":
            return SimpleImputer(strategy="mean")
        if key == "knn":
            return KNNImputer(n_neighbors=int(cfg["knn_neighbors"]), weights="distance")
        if key == "mice":
            return IterativeImputer(
                estimator=BayesianRidge(), max_iter=int(cfg["imputer_max_iter"]),
                n_nearest_features=int(cfg["mice_nearest_features"]),
                initial_strategy="mean", skip_complete=True,
                random_state=int(cfg["seed"]), tol=1e-3)
        if key == "svd":
            return FastSVDImputer(
                rank=int(cfg.get("svd_rank", 20)),
                power_iterations=int(cfg.get("svd_power_iterations", 2)),
                refinement_steps=int(cfg.get("svd_refinement_steps", 1)),
                ridge=float(cfg.get("svd_ridge", 1e-4)), seed=int(cfg["seed"]))
        raise KeyError(key)

    def _safe_transform(self, x: np.ndarray) -> np.ndarray:
        if self._identity_imputation:
            if np.isnan(x).any():
                raise RuntimeError(
                    f"{self.key} identity branch received missing evaluation data")
            return np.asarray(x, dtype=np.float64)
        out = np.asarray(self.imputer.transform(x), dtype=np.float64)
        if out.shape[1] != x.shape[1]:
            raise RuntimeError("imputer changed the number of features")
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _features(self, views: ArrayList, mask: np.ndarray) -> torch.Tensor:
        key = self._query_key(views, mask)
        if key == self._cached_feature_key and self._cached_features is not None:
            return self._cached_features
        x = masked_concat(views, mask, nan_missing=not self.append_mask)
        if self.append_mask:
            x = np.concatenate([x, mask.astype(np.float64)], axis=1)
        else:
            x = self._safe_transform(x)
        features = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self._cached_feature_key = key
        self._cached_features = features
        return features

    def train_epoch(self, views: ArrayList, mask: np.ndarray, y: np.ndarray) -> float:
        self.model.train()
        target = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.optim.zero_grad(set_to_none=True)
        with self._autocast():
            loss = F.cross_entropy(self.model(self.train_x), target)
        self._backward_step(loss)
        return float(loss.detach())

    @torch.inference_mode()
    def predict(self, views: ArrayList, mask: np.ndarray) -> np.ndarray:
        self.model.eval()
        with self._autocast():
            prob = F.softmax(self.model(self._features(views, mask)), dim=1)
        return prob.float().cpu().numpy()


class DIMVAdapter(ImputerMLPAdapter):
    def __init__(self, train_views: ArrayList, train_mask: np.ndarray,
                 num_classes: int, device: torch.device, cfg: Mapping[str, object]) -> None:
        self.info = METHODS["dimv"]
        self.key = "dimv"
        self._configure_training(device, cfg)
        self.append_mask = False
        dimv_cls, self.source_record = load_dimv(cfg["dimv_repo"])
        self.imputer = dimv_cls()
        train_x = masked_concat(train_views, train_mask, nan_missing=True)
        # Even a complete optimization fold must fit DIMV: with a small
        # missing rate it is possible that this fold is complete while its
        # validation/query fold contains missing views.
        self._identity_imputation = (
            float(cfg.get("active_missing_rate", -1.0)) == 0.0)
        # DIMV's DPER implementation uses process-based parallelism.  On
        # Windows, n_jobs=-1 may spawn enough Python/Torch processes to exhaust
        # the page file, so it has a separate conservative default.
        self._dimv_largest_features = int(cfg["dimv_largest_features"])
        if not self._identity_imputation:
            self.imputer.fit(train_x, initializing=False,
                             n_jobs=int(cfg["dimv_n_jobs"]))
            covariance_size = int(self.imputer.cov_no_zeros.shape[0])
            if covariance_size < 1:
                raise ValueError("DIMV found no usable non-constant feature")
            self._dimv_largest_features = min(
                self._dimv_largest_features, covariance_size)
            if bool(cfg["dimv_cv"]) and np.isnan(train_x).any():
                self.imputer.cross_validate(
                    alphas=list(cfg["dimv_alphas"]),
                    train_percent=float(cfg["dimv_cv_percent"]),
                    features_corr_threshold=float(cfg["dimv_corr_threshold"]),
                    mlargest_features=self._dimv_largest_features)
        self._dimv_cfg = cfg
        train_x = self._safe_transform(train_x)
        self.train_x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
        self.model = CommonMLP(
            self.train_x.shape[1], num_classes, int(cfg["hidden_dim"]),
            float(cfg["dropout"])).to(device)
        self.optim = self._make_adamw(self.model.parameters(), device, cfg)
        self.num_classes = num_classes
        self._cached_feature_key: Optional[tuple] = None
        self._cached_features: Optional[torch.Tensor] = None

    def _safe_transform(self, x: np.ndarray) -> np.ndarray:
        if self._identity_imputation:
            if np.isnan(x).any():
                raise RuntimeError(
                    "DIMV was fitted on complete data but received missing evaluation data")
            return np.asarray(x, dtype=np.float64)
        alpha = (self.imputer.best_alpha
                 if self.imputer.best_alpha is not None
                 else float(self._dimv_cfg.get("dimv_alpha", 0.1)))
        out = self.imputer.transform(
            x, alpha=alpha, cross_validation=False,
            features_corr_threshold=float(self._dimv_cfg["dimv_corr_threshold"]),
            mlargest_features=self._dimv_largest_features)
        return np.nan_to_num(np.asarray(out), nan=0.0, posinf=0.0, neginf=0.0)


def _to_tensors(views: ArrayList, device: torch.device) -> List[torch.Tensor]:
    return [torch.as_tensor(x, dtype=torch.float32, device=device) for x in views]


class CLCLSAAdapter(BaseAdapter):
    def __init__(self, train_views: ArrayList, train_mask: np.ndarray,
                 num_classes: int, device: torch.device, cfg: Mapping[str, object]) -> None:
        self.info = METHODS["clclsa"]
        h = int(cfg["clclsa_hidden_dim"])
        prediction = [int(x) for x in cfg["clclsa_prediction_dims"]]
        cluecl3_cls, self.source_record = load_clclsa(cfg["clclsa_repo"])
        self.model = cluecl3_cls(
            [x.shape[1] for x in train_views], [h], num_classes,
            float(cfg["dropout"]), {v: prediction for v in range(3)}).to(device)
        # The mutual-information term contains products and logarithms that can
        # underflow in FP16.  It still runs on CUDA, but in stable FP32.
        self._configure_training(device, cfg, amp_safe=False)
        self.optim = self._make_adamw(self.model.parameters(), device, cfg)
        self.train_x = _to_tensors(train_views, device)
        self.train_mask = torch.as_tensor(train_mask, dtype=torch.long, device=device)
        self.train_y: Optional[torch.Tensor] = None
        self.all_complete = bool(np.all(train_mask == 1))
        if not self.all_complete and int((train_mask.sum(axis=1) == 3).sum()) < 2:
            raise ValueError("CLCLSA needs at least two complete training samples")
        self.lambda_al = float(cfg["clclsa_lambda_al"])
        self.lambda_co = float(cfg["clclsa_lambda_co"])
        self.lambda_cl = float(cfg["clclsa_lambda_cl"])
        self._cached_query_key: Optional[tuple] = None
        self._cached_query: Optional[Tuple[List[torch.Tensor], torch.Tensor]] = None

    def _query_tensors(self, views: ArrayList, mask: np.ndarray):
        key = self._query_key(views, mask)
        if key != self._cached_query_key or self._cached_query is None:
            self._cached_query = (
                _to_tensors(views, self.device),
                torch.as_tensor(mask, dtype=torch.long, device=self.device),
            )
            self._cached_query_key = key
        return self._cached_query

    def train_epoch(self, views: ArrayList, mask: np.ndarray, y: np.ndarray) -> float:
        self.model.train()
        self.train_y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.optim.zero_grad(set_to_none=True)
        with self._autocast():
            if self.all_complete:
                # This is the official complete-data branch used by the
                # CLCLSA trainer when missing_rate == 0.
                loss, _, _ = self.model(
                    self.train_x, self.train_y,
                    aux_loss=self.lambda_al > 0,
                    lambda_al=self.lambda_al)
            else:
                loss, _, _ = self.model.train_missing_cg(
                    self.train_x, self.train_mask, self.train_y, self.device,
                    aux_loss=self.lambda_al > 0,
                    lambda_al=self.lambda_al,
                    cross_omics_loss=self.lambda_co > 0,
                    lambda_col=self.lambda_co,
                    constrastive_loss=self.lambda_cl > 0,
                    lambda_cl=self.lambda_cl)
        if not torch.isfinite(loss):
            raise FloatingPointError("CLCLSA produced a non-finite loss")
        self._backward_step(loss)
        return float(loss.detach())

    @torch.inference_mode()
    def predict(self, views: ArrayList, mask: np.ndarray) -> np.ndarray:
        self.model.eval()
        x, m = self._query_tensors(views, mask)
        with self._autocast():
            logits = (self.model.infer(x) if np.all(mask == 1)
                      else self.model.infer_on_missing(x, m, self.device))
        return F.softmax(logits.float(), dim=1).cpu().numpy()


class TMONetWrapper(nn.Module):
    """Official TMO-Net body/head with patient-specific mask dispatch."""

    def __init__(self, official_module, dims: Sequence[int], latent: int,
                 num_classes: int, hidden: int, kl_weight: float) -> None:
        super().__init__()
        # Instantiate the author's downstream class so both its TMO_Net body
        # and its classification head are the original implementations.
        official = official_module.DownStream_predictor(
            3, list(dims), latent, [hidden], [hidden], None,
            {"output_dim": num_classes}, ["gaussian"] * 3, False,
            {"view1": 0, "view2": 1, "view3": 2}, kl_weight)
        self.body = official.cross_encoders
        self.head = official.downstream_predictor
        self.latent = latent

    def embedding(self, xs: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        n = mask.shape[0]
        latent = self.latent
        result = xs[0].new_zeros((n, 3 * latent))
        # Encode each available source view in one sufficiently large batch.
        # This avoids BatchNorm failures caused by a rare mask pattern that
        # contains only one training sample.
        encoded: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for src in range(3):
            src_idx = torch.where(mask[:, src] > 0)[0]
            if src_idx.numel() == 0:
                continue
            for target in range(3):
                z, mu, lv = self.body.encoders[src][target](xs[src][src_idx])
                encoded[(src, target)] = (src_idx, z, mu, lv)
        patterns = torch.unique(mask, dim=0)
        for pattern in patterns:
            idx = torch.where(torch.all(mask == pattern, dim=1))[0]
            available = torch.where(pattern > 0)[0].tolist()
            target_z = []
            for target in range(3):
                cross_mu, cross_lv = [], []
                for src in available:
                    src_idx, _, all_mu, all_lv = encoded[(src, target)]
                    position = torch.searchsorted(src_idx, idx)
                    mu, lv = all_mu[position], all_lv[position]
                    if src != target:
                        cross_mu.append(mu)
                        cross_lv.append(lv)
                if target in available:
                    src_idx, _, all_mu, _ = encoded[(target, target)]
                    own_mu = all_mu[torch.searchsorted(src_idx, idx)]
                    if cross_mu:
                        poe_mu, _ = _product_of_experts_safe(cross_mu, cross_lv)
                        z = (own_mu + poe_mu) / 2.0
                    else:
                        z = own_mu
                else:
                    poe_mu, _ = _product_of_experts_safe(cross_mu, cross_lv)
                    z = poe_mu
                target_z.append(z)
            result[idx] = torch.cat(target_z, dim=1)
        return result

    def forward(self, xs: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(xs, mask))


def _product_of_experts_safe(mus: List[torch.Tensor], logvars: List[torch.Tensor]):
    if not mus:
        raise ValueError("TMO-Net received a sample with no available view")
    precision = [torch.exp(-lv).clamp(max=1e6) for lv in logvars]
    denom = torch.stack(precision).sum(0).clamp_min(1e-8)
    mu = torch.stack([m * p for m, p in zip(mus, precision)]).sum(0) / denom
    return mu, -torch.log(denom)


class TMONetAdapter(BaseAdapter):
    def __init__(self, train_views: ArrayList, train_mask: np.ndarray,
                 num_classes: int, device: torch.device, cfg: Mapping[str, object]) -> None:
        self.info = METHODS["tmonet"]
        self._configure_training(device, cfg)
        official_module, loss_module, self.source_record = load_tmonet(
            cfg["tmonet_repo"])
        self.official_loss = loss_module
        self.model = TMONetWrapper(
            official_module, [x.shape[1] for x in train_views], int(cfg["latent_dim"]),
            num_classes, int(cfg["hidden_dim"]),
            float(cfg["tmonet_kl_weight"])).to(device)
        self.optim = self._make_adamw(self.model.parameters(), device, cfg)
        self.train_x = _to_tensors(train_views, device)
        self.train_mask = torch.as_tensor(train_mask, dtype=torch.long, device=device)
        self.recon_weight = float(cfg["tmonet_objective_weight"])
        self.kl_weight = float(cfg["tmonet_kl_weight"])
        self._cached_query_key: Optional[tuple] = None
        self._cached_query: Optional[Tuple[List[torch.Tensor], torch.Tensor]] = None

    def _query_tensors(self, views: ArrayList, mask: np.ndarray):
        key = self._query_key(views, mask)
        if key != self._cached_query_key or self._cached_query is None:
            self._cached_query = (
                _to_tensors(views, self.device),
                torch.as_tensor(mask, dtype=torch.long, device=self.device),
            )
            self._cached_query_key = key
        return self._cached_query

    def _reconstruction(self) -> torch.Tensor:
        loss = self.train_x[0].new_zeros(())
        terms = 0
        for v in range(3):
            idx = self.train_mask[:, v].bool()
            if idx.any():
                z, mu, logvar = self.model.body.encoders[v][v](self.train_x[v][idx])
                recon = self.model.body.self_decoders[v](z)
                # These are the author's exact KL and Gaussian reconstruction
                # functions.  They are evaluated only on genuinely observed
                # rows; missing outer/inner rows never become targets.
                loss = loss + self.official_loss.reconstruction_loss(
                    recon, self.train_x[v][idx], 1.0, "gaussian")
                loss = loss + self.kl_weight * self.official_loss.KL_loss(
                    mu, logvar, 1.0)
                terms += 1
        return loss / max(terms, 1)

    def train_epoch(self, views: ArrayList, mask: np.ndarray, y: np.ndarray) -> float:
        self.model.train()
        target = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.optim.zero_grad(set_to_none=True)
        with self._autocast():
            logits = self.model(self.train_x, self.train_mask)
            loss = (F.cross_entropy(logits, target)
                    + self.recon_weight * self._reconstruction())
        self._backward_step(loss)
        return float(loss.detach())

    @torch.inference_mode()
    def predict(self, views: ArrayList, mask: np.ndarray) -> np.ndarray:
        self.model.eval()
        x, m = self._query_tensors(views, mask)
        with self._autocast():
            logits = self.model(x, m)
        return F.softmax(logits.float(), dim=1).cpu().numpy()


def knn_sparse_adjacency(x: torch.Tensor, observed: torch.Tensor, k: int) -> torch.Tensor:
    """Symmetric normalized sparse adjacency; unavailable nodes have self-loops."""
    n = x.shape[0]
    obs_idx = torch.where(observed)[0]
    self_nodes = torch.arange(n, device=x.device)
    rows = [self_nodes]
    cols = [self_nodes]
    if obs_idx.numel() > 1:
        z = x[obs_idx]
        kk = min(int(k), z.shape[0] - 1)
        distance = torch.cdist(z, z)
        distance.fill_diagonal_(float("inf"))
        neighbors = distance.topk(kk, largest=False, dim=1).indices
        source_local = torch.arange(z.shape[0], device=x.device).repeat_interleave(kk)
        source = obs_idx[source_local]
        target = obs_idx[neighbors.reshape(-1)]
        rows.extend((source, target))
        cols.extend((target, source))
    row = torch.cat(rows)
    col = torch.cat(cols)
    idx = torch.stack((row, col))
    values = torch.ones(row.numel(), dtype=x.dtype, device=x.device)
    adj = torch.sparse_coo_tensor(idx, values, (n, n)).coalesce()
    degree = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1.0)
    r, c = adj.indices()
    norm = adj.values() / torch.sqrt(degree[r] * degree[c])
    return torch.sparse_coo_tensor(adj.indices(), norm, (n, n)).coalesce()


def inductive_sparse_adjacency(reference_x: torch.Tensor,
                               reference_observed: torch.Tensor,
                               query_x: torch.Tensor,
                               query_observed: torch.Tensor,
                               k: int) -> torch.Tensor:
    """Train-reference graph plus one-way reference-to-query messages."""
    nr, nq = reference_x.shape[0], query_x.shape[0]
    self_nodes = torch.arange(nr + nq, device=reference_x.device)
    rows, cols = [self_nodes], [self_nodes]
    ref_ids = torch.where(reference_observed)[0]
    if ref_ids.numel() > 1:
        z_ref = reference_x[ref_ids]
        kk = min(int(k), z_ref.shape[0] - 1)
        distance = torch.cdist(z_ref, z_ref)
        distance.fill_diagonal_(float("inf"))
        neighbors = distance.topk(kk, largest=False, dim=1).indices
        source_local = torch.arange(z_ref.shape[0], device=reference_x.device).repeat_interleave(kk)
        source = ref_ids[source_local]
        target = ref_ids[neighbors.reshape(-1)]
        rows.extend((source, target)); cols.extend((target, source))
    if ref_ids.numel() >= 1:
        q_ids = torch.where(query_observed)[0]
        if q_ids.numel():
            kk = min(int(k), ref_ids.numel())
            neighbors = torch.cdist(query_x[q_ids], reference_x[ref_ids]).topk(
                kk, largest=False, dim=1).indices
            query_target = (nr + q_ids).repeat_interleave(kk)
            reference_source = ref_ids[neighbors.reshape(-1)]
            # adj[row=target, col=source]: query aggregates references only.
            rows.append(query_target); cols.append(reference_source)
    row = torch.cat(rows); col = torch.cat(cols)
    idx = torch.stack((row, col))
    values = torch.ones(row.numel(), dtype=reference_x.dtype, device=reference_x.device)
    adj = torch.sparse_coo_tensor(idx, values, (nr + nq, nr + nq)).coalesce()
    degree = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1.0)
    r, c = adj.indices()
    norm = adj.values() / torch.sqrt(degree[r] * degree[c])
    return torch.sparse_coo_tensor(adj.indices(), norm, adj.shape).coalesce()


class MOGEDNWrapper(nn.Module):
    def __init__(self, module, dims: Sequence[int], latent: int, hidden: int,
                 num_classes: int, dropout: float) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([
            module.GCNEncoder({str(v + 1): dims[v]}, [hidden], latent, dropout)
            for v in range(3)])
        self.decoder = module.ConcatLatentDecoder(
            latent, ["1", "2", "3"], [hidden], {str(v + 1): latent for v in range(3)})
        self.small = nn.ModuleList([module.ClassifierSingle(latent, num_classes) for _ in range(3)])
        self.fusion = module.VCDNClassifier(3, num_classes, hidden)

    def forward(self, xs: List[torch.Tensor], mask: torch.Tensor,
                adjs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        raw = [self.encoders[v](xs[v] * mask[:, v:v + 1], adjs[v]) for v in range(3)]
        completed = []
        recon = []
        for target in range(3):
            sources = [raw[v] * mask[:, v:v + 1] for v in range(3) if v != target]
            predicted = self.decoder(torch.cat(sources, dim=1), str(target + 1))
            recon.append(predicted)
            m = mask[:, target:target + 1]
            completed.append(m * raw[target] + (1.0 - m) * predicted)
        logits = [self.small[v](completed[v]) for v in range(3)]
        return self.fusion(logits), raw, recon


class MOGEDNAdapter(BaseAdapter):
    def __init__(self, train_views: ArrayList, train_mask: np.ndarray,
                 num_classes: int, device: torch.device, cfg: Mapping[str, object]) -> None:
        self.info = METHODS["mogedn"]
        # Sparse GCN kernels may reject mixed Float/Half operands on some
        # PyTorch/CUDA builds.  Keep this method on GPU in stable FP32.
        self._configure_training(device, cfg, amp_safe=False)
        official_models, official_utils, self.source_record = load_mogedn(
            cfg["mogedn_repo"])
        self.official_utils = official_utils
        self.model = MOGEDNWrapper(
            official_models, [x.shape[1] for x in train_views], int(cfg["latent_dim"]),
            int(cfg["hidden_dim"]), num_classes, float(cfg["dropout"])).to(device)
        self.optim = self._make_adamw(self.model.parameters(), device, cfg)
        self.train_x = _to_tensors(train_views, device)
        self.train_mask = torch.as_tensor(train_mask, dtype=torch.float32, device=device)
        self.num_classes = int(num_classes)
        self.k = int(cfg["graph_k"])
        self.train_adj = [knn_sparse_adjacency(self.train_x[v], self.train_mask[:, v].bool(), self.k)
                          for v in range(3)]
        self.recon_weight = float(cfg["mogedn_recon_weight"])
        self._cached_query_key: Optional[tuple] = None
        self._cached_query = None

    def _query_graph(self, views: ArrayList, mask: np.ndarray):
        key = self._query_key(views, mask)
        if key != self._cached_query_key or self._cached_query is None:
            query_x = _to_tensors(views, self.device)
            query_m = torch.as_tensor(
                mask, dtype=torch.float32, device=self.device)
            x = [torch.cat([self.train_x[v], query_x[v]], dim=0)
                 for v in range(3)]
            m = torch.cat([self.train_mask, query_m], dim=0)
            adj = [inductive_sparse_adjacency(
                self.train_x[v], self.train_mask[:, v].bool(), query_x[v],
                query_m[:, v].bool(), self.k) for v in range(3)]
            self._cached_query = (x, m, adj)
            self._cached_query_key = key
        return self._cached_query

    def train_epoch(self, views: ArrayList, mask: np.ndarray, y: np.ndarray) -> float:
        self.model.train()
        target = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.optim.zero_grad(set_to_none=True)
        logits, raw, recon = self.model(self.train_x, self.train_mask, self.train_adj)
        rloss = logits.new_zeros(())
        terms = 0
        for v in range(3):
            usable = self.train_mask[:, v].bool() & (self.train_mask.sum(1) >= 2)
            if usable.any():
                rloss = rloss + F.mse_loss(recon[v][usable], raw[v][usable].detach())
                terms += 1
        counts = torch.bincount(target, minlength=self.num_classes).float()
        class_weight = target.numel() / (
            self.num_classes * counts.clamp_min(1.0))
        sample_weight_np = self.official_utils.cal_sample_weight(
            np.asarray(y), self.num_classes)
        sample_weight = torch.as_tensor(
            sample_weight_np, dtype=logits.dtype, device=self.device)
        focal = self.official_utils.FocalLossWithWeights(
            gamma=2, class_weight=class_weight)
        loss_cls = focal(logits, target, sample_weight=sample_weight)
        # MOGEDN's phase-3 objective is focal classification plus latent MSE.
        loss = loss_cls + self.recon_weight * rloss / max(terms, 1)
        self._backward_step(loss)
        return float(loss.detach())

    @torch.inference_mode()
    def predict(self, views: ArrayList, mask: np.ndarray) -> np.ndarray:
        self.model.eval()
        x, m, adj = self._query_graph(views, mask)
        logits, _, _ = self.model(x, m, adj)
        logits = logits[len(self.train_mask):]
        return F.softmax(logits, dim=1).cpu().numpy()


def _pyg_edge_index(x: torch.Tensor, k: int) -> torch.Tensor:
    n = x.shape[0]
    if n == 1:
        return torch.zeros((2, 1), dtype=torch.long, device=x.device)
    kk = min(int(k), n - 1)
    distance = torch.cdist(x, x)
    distance.fill_diagonal_(float("inf"))
    neighbors = distance.topk(kk, largest=False, dim=1).indices
    source = torch.arange(n, device=x.device).repeat_interleave(kk)
    target = neighbors.reshape(-1)
    self_nodes = torch.arange(n, device=x.device)
    return torch.stack((torch.cat((self_nodes, source, target)),
                        torch.cat((self_nodes, target, source))))


class IntegrAOAdapter(BaseAdapter):
    """Leakage-free official GraphSAGE architecture adaptation.

    Each split constructs modality graphs only from samples passed to that
    split.  This deliberately does not call IntegrAO's tutorial-level
    transductive predictor, which diffuses a graph containing the held-out
    cohort.
    """

    def __init__(self, train_views: ArrayList, train_mask: np.ndarray,
                 num_classes: int, device: torch.device, cfg: Mapping[str, object]) -> None:
        self.info = METHODS["integrao"]
        # GraphSAGE remains on CUDA, while FP32 avoids version-specific AMP
        # failures in torch-geometric scatter/sparse kernels.
        self._configure_training(device, cfg, amp_safe=False)
        official_model, official_losses, self.source_record = load_integrao(
            cfg["integrao_repo"])
        self.official_losses = official_losses
        self.model = official_model.IntegrAO(
            [x.shape[1] for x in train_views], int(cfg["hidden_dim"]),
            int(cfg["latent_dim"]), num_layers=2, num_classes=num_classes).to(device)
        self.optim = self._make_adamw(self.model.parameters(), device, cfg)
        self.train_x_raw = train_views
        self.train_mask_raw = train_mask
        self.k = int(cfg["graph_k"])
        self.align_weight = float(cfg["integrao_align_weight"])
        self.x_dict, self.edge_dict, self.ids = self._graphs(train_views, train_mask)
        self.p_dict = {
            v: self._training_probability(self.x_dict[v]) for v in self.x_dict
        }
        self._cached_query_key: Optional[tuple] = None
        self._cached_query = None

    def _query_graphs(self, views: ArrayList, mask: np.ndarray):
        key = self._query_key(views, mask)
        if key != self._cached_query_key or self._cached_query is None:
            self._cached_query = self._inductive_graphs(views, mask)
            self._cached_query_key = key
        return self._cached_query

    def _training_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Fold-local affinity input for IntegrAO's official t-SNE KL loss."""
        with torch.no_grad():
            distance = torch.cdist(x, x).pow(2)
            positive = distance[distance > 0]
            scale = (positive.median() if positive.numel()
                     else distance.new_tensor(1.0)).clamp_min(1e-8)
            affinity = torch.exp(-distance / scale)
            affinity.fill_diagonal_(0.0)
            n = x.shape[0]
            if n > 1:
                kk = min(self.k, n - 1)
                keep = torch.zeros_like(affinity, dtype=torch.bool)
                index = affinity.topk(kk, dim=1).indices
                keep.scatter_(1, index, True)
                affinity = torch.where(keep | keep.T, affinity,
                                       torch.zeros_like(affinity))
            total = affinity.sum().clamp_min(1e-12)
            return (affinity / total).clamp_min(1e-12)

    def _graphs(self, views: ArrayList, mask: np.ndarray):
        x_dict, edges, ids = {}, {}, {}
        for v in range(3):
            present = np.where(mask[:, v] == 1)[0]
            if present.size == 0:
                raise ValueError(f"IntegrAO split has no nodes for view {v + 1}")
            x = torch.as_tensor(views[v][present], dtype=torch.float32, device=self.device)
            x_dict[v] = x
            edges[v] = _pyg_edge_index(x, self.k)
            ids[v] = [int(i) for i in present]
        return x_dict, edges, ids

    def _inductive_graphs(self, views: ArrayList, mask: np.ndarray):
        """Add query nodes with reference->query edges and no query->reference edges."""
        x_dict, edges, ids = {}, {}, {}
        n_ref = len(self.train_mask_raw)
        for v in range(3):
            ref_rows = np.where(self.train_mask_raw[:, v] == 1)[0]
            query_rows = np.where(mask[:, v] == 1)[0]
            if query_rows.size == 0:
                # The global mask guarantees one view per patient, not one
                # patient per view.  An empty query domain can simply be absent.
                if ref_rows.size == 0:
                    continue
                x = torch.as_tensor(self.train_x_raw[v][ref_rows], dtype=torch.float32,
                                    device=self.device)
                x_dict[v] = x
                edges[v] = _pyg_edge_index(x, self.k)
                ids[v] = [int(i) for i in ref_rows]
                continue
            ref_x = torch.as_tensor(self.train_x_raw[v][ref_rows], dtype=torch.float32,
                                    device=self.device)
            query_x = torch.as_tensor(views[v][query_rows], dtype=torch.float32,
                                      device=self.device)
            x = torch.cat([ref_x, query_x], dim=0)
            nr = len(ref_rows)
            src, dst = [], []
            # PyG edge_index is source -> target.
            ref_edge = _pyg_edge_index(ref_x, self.k)
            src.extend(ref_edge[0].tolist()); dst.extend(ref_edge[1].tolist())
            kk = min(self.k, nr)
            qnbr = torch.cdist(query_x, ref_x).topk(
                kk, largest=False, dim=1).indices
            query_nodes = nr + torch.arange(len(query_x), device=self.device)
            src.extend(query_nodes.tolist()); dst.extend(query_nodes.tolist())
            src.extend(qnbr.reshape(-1).tolist())
            dst.extend(query_nodes.repeat_interleave(kk).tolist())
            x_dict[v] = x
            edges[v] = torch.tensor([src, dst], dtype=torch.long, device=self.device)
            ids[v] = ([int(i) for i in ref_rows]
                      + [n_ref + int(i) for i in query_rows])
        return x_dict, edges, ids

    def train_epoch(self, views: ArrayList, mask: np.ndarray, y: np.ndarray) -> float:
        self.model.train()
        self.optim.zero_grad(set_to_none=True)
        z_all, _, logits, id_list = self.model(self.x_dict, self.edge_dict, self.ids)
        target = torch.as_tensor([int(y[i]) for i in id_list], dtype=torch.long, device=self.device)
        # Official IntegrAO objective: per-domain t-SNE KL + pairwise alignment
        # MSE + supervised cross entropy.  The affinity matrices above are
        # constructed from optimization-fold rows only.
        kl = logits.new_zeros(())
        for view, embedding in z_all.items():
            kl = kl + self.official_losses.tsne_loss(
                self.p_dict[view], embedding)
        loss = F.cross_entropy(logits, target) + kl
        align = logits.new_zeros(())
        terms = 0
        for a in range(2):
            pos_a = {sid: j for j, sid in enumerate(self.ids[a])}
            for b in range(a + 1, 3):
                pos_b = {sid: j for j, sid in enumerate(self.ids[b])}
                common = sorted(set(pos_a).intersection(pos_b))
                if common:
                    ia = torch.tensor([pos_a[s] for s in common], device=self.device)
                    ib = torch.tensor([pos_b[s] for s in common], device=self.device)
                    align = align + F.mse_loss(z_all[a][ia], z_all[b][ib])
                    terms += 1
        loss = loss + self.align_weight * align / max(terms, 1)
        self._backward_step(loss)
        return float(loss.detach())

    @torch.inference_mode()
    def predict(self, views: ArrayList, mask: np.ndarray) -> np.ndarray:
        self.model.eval()
        x, e, ids = self._query_graphs(views, mask)
        _, _, logits, id_list = self.model(x, e, ids)
        prob = F.softmax(logits, dim=1).cpu().numpy()
        out = np.zeros((len(mask), prob.shape[1]), dtype=np.float64)
        n_ref = len(self.train_mask_raw)
        for row, sid in enumerate(id_list):
            if int(sid) >= n_ref:
                out[int(sid) - n_ref] = prob[row]
        if np.any(out.sum(axis=1) == 0):
            raise RuntimeError("IntegrAO failed to return one prediction per patient")
        return out


def build_adapter(key: str, train_views: ArrayList, train_mask: np.ndarray,
                  num_classes: int, device: torch.device,
                  cfg: Mapping[str, object]) -> BaseAdapter:
    if key == "rusi":
        return RUSIAdapter(train_views, train_mask, num_classes, device, cfg)
    if key in {"zero_mask", "mean", "knn", "mice", "svd"}:
        return ImputerMLPAdapter(key, train_views, train_mask, num_classes, device, cfg)
    if key == "dimv":
        return DIMVAdapter(train_views, train_mask, num_classes, device, cfg)
    if key == "clclsa":
        return CLCLSAAdapter(train_views, train_mask, num_classes, device, cfg)
    if key == "tmonet":
        return TMONetAdapter(train_views, train_mask, num_classes, device, cfg)
    if key == "mogedn":
        return MOGEDNAdapter(train_views, train_mask, num_classes, device, cfg)
    if key == "integrao":
        return IntegrAOAdapter(train_views, train_mask, num_classes, device, cfg)
    raise KeyError(f"unknown method: {key}")
