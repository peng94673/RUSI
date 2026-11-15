"""Audited loaders for the authors' unmodified baseline implementations.

This module never copies a model into the benchmark.  It imports Python files
from the user-supplied repositories and records SHA-256 hashes of every source
file that defines a model or objective.  The adapters in
``missing_comparison_core.py`` own only the leakage-free data/CV bridge.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Mapping, Sequence


class OfficialSourceError(RuntimeError):
    """Raised when an author repository or its dependencies are unavailable."""


@dataclass(frozen=True)
class SourceRecord:
    method: str
    repository: str
    files: Mapping[str, str]
    adaptation: str


_CACHE: Dict[tuple[str, str], ModuleType] = {}


def _repo(path: object, method: str) -> Path:
    root = Path(str(path)).expanduser().resolve()
    if not root.is_dir():
        raise OfficialSourceError(
            f"{method} official repository was not found: {root}. "
            "Set its *_REPO value in comparison_config.py or pass the matching "
            "--*-repo option.")
    return root


def _required(root: Path, relative_files: Sequence[str], method: str) -> list[Path]:
    files = [root / rel for rel in relative_files]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise OfficialSourceError(
            f"{method} repository is incomplete; missing: {', '.join(missing)}")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(method: str, root: Path, relative_files: Sequence[str],
            adaptation: str) -> SourceRecord:
    files = _required(root, relative_files, method)
    return SourceRecord(
        method=method,
        repository=str(root),
        files={str(path.relative_to(root)): _sha256(path) for path in files},
        adaptation=adaptation,
    )


def _load_file(module_name: str, path: Path) -> ModuleType:
    key = (module_name, str(path.resolve()))
    if key in _CACHE:
        return _CACHE[key]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise OfficialSourceError(f"cannot create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    _CACHE[key] = module
    return module


def load_rusi(model_file: object) -> tuple[ModuleType, SourceRecord]:
    """Load the user's RUSI implementation without executing its CLI entrypoint.

    The comparison runner supplies the folds and masks.  Only the model classes,
    selective-imputation logic and losses are imported from this file.
    """
    path = Path(str(model_file)).expanduser().resolve()
    if not path.is_file():
        raise OfficialSourceError(
            f"RUSI model file was not found: {path}. Set RUSI_MODEL_FILE in "
            "comparison_config.py or pass --rusi-model-file.")
    module = _load_file("_rusi_user_model", path)
    if not hasattr(module, "CLUECL3"):
        raise OfficialSourceError(
            f"RUSI model file does not define CLUECL3: {path}")
    record = SourceRecord(
        method="RUSI (ours)",
        repository=str(path.parent),
        files={path.name: _sha256(path)},
        adaptation=(
            "user model body/selective-imputation/router/losses with the common "
            "leakage-safe fold adapter and paired nested-CV protocol"),
    )
    return module, record


def _import_from_root(root: Path, qualified_name: str) -> ModuleType:
    """Import an author's package while keeping its own absolute imports valid."""
    root_text = str(root)
    added = root_text not in sys.path
    if added:
        sys.path.insert(0, root_text)
    try:
        return importlib.import_module(qualified_name)
    finally:
        if added:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass


def _assert_from(module: ModuleType, root: Path, method: str) -> None:
    source = Path(str(getattr(module, "__file__", ""))).resolve()
    try:
        source.relative_to(root.resolve())
    except ValueError as exc:
        raise OfficialSourceError(
            f"{method} resolved to {source}, outside configured repository {root}. "
            "Restart Python after correcting the repository path/module cache.") from exc


def load_dimv(repo_path: object) -> tuple[type, SourceRecord]:
    root = _repo(repo_path, "DIMV")
    rel = [
        "src/DIMVImputation.py", "src/conditional_expectation.py",
        "src/dpers.py", "src/utils.py",
    ]
    record = _record(
        "DIMV", root, rel,
        "fit and internal alpha CV use optimization-fold rows only; the common "
        "classifier and nested-CV protocol are external to DIMV.")
    src = root / "src"
    src_text = str(src)
    added = src_text not in sys.path
    if added:
        sys.path.insert(0, src_text)
    try:
        module = importlib.import_module("DIMVImputation")
    finally:
        if added:
            try:
                sys.path.remove(src_text)
            except ValueError:
                pass
    _assert_from(module, root, "DIMV")
    cls = getattr(module, "DIMVImputation", None)
    if cls is None:
        raise OfficialSourceError("official DIMVImputation class was not found")
    return cls, record


def load_clclsa(repo_path: object) -> tuple[type, SourceRecord]:
    root = _repo(repo_path, "CLCLSA/CLUECL3")
    rel = [
        "networks/models/clcl.py", "networks/models/common_layers.py",
        "networks/models/losses.py",
    ]
    record = _record(
        "CLCLSA/CLUECL3", root, rel,
        "official model methods and their internal objectives are used; outer-test "
        "checkpoint selection in the author trainer is replaced by inner-CV selection.")
    module = _import_from_root(root, "networks.models.clcl")
    _assert_from(module, root, "CLCLSA/CLUECL3")
    cls = getattr(module, "CLUECL3", None)
    if cls is None:
        raise OfficialSourceError("official CLUECL3 class was not found")
    return cls, record


def load_tmonet(repo_path: object) -> tuple[ModuleType, ModuleType, SourceRecord]:
    root = _repo(repo_path, "TMO-Net")
    rel = ["model/TMO_Net_model.py", "util/loss_function.py"]
    record = _record(
        "TMO-Net", root, rel,
        "official encoders/decoders and official KL/reconstruction objectives are "
        "used with the official downstream head; patient-specific mask grouping is "
        "the leakage-free adaptation. No external pretrained checkpoint is used.")
    # The author's classification model imports ``lifelines`` only because the
    # same loss file also contains an unused survival c-index helper.  Do not
    # force that unrelated package on classification-only experiments.
    try:
        importlib.import_module("lifelines.utils")
    except ModuleNotFoundError:
        lifelines = ModuleType("lifelines")
        lifelines_utils = ModuleType("lifelines.utils")

        def _survival_only(*_args, **_kwargs):
            raise OfficialSourceError(
                "TMO-Net's survival helper needs lifelines; it is not used by "
                "this classification benchmark")

        lifelines_utils.concordance_index = _survival_only
        lifelines.utils = lifelines_utils
        sys.modules.setdefault("lifelines", lifelines)
        sys.modules.setdefault("lifelines.utils", lifelines_utils)
    model_module = _import_from_root(root, "model.TMO_Net_model")
    loss_module = _import_from_root(root, "util.loss_function")
    _assert_from(model_module, root, "TMO-Net")
    _assert_from(loss_module, root, "TMO-Net loss")
    return model_module, loss_module, record


def load_mogedn(repo_path: object) -> tuple[ModuleType, ModuleType, SourceRecord]:
    root = _repo(repo_path, "MOGEDN")
    rel = ["models.py", "utils.py"]
    record = _record(
        "MOGEDN", root, rel,
        "official GCN/decoder/classifier/VCDN modules and focal/reconstruction "
        "objectives are used; graphs are rebuilt per optimization fold and query "
        "nodes receive one-way messages from training nodes only.")
    suffix = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    models = _load_file(f"official_mogedn_models_{suffix}", root / "models.py")
    utils = _load_file(f"official_mogedn_utils_{suffix}", root / "utils.py")
    return models, utils, record


def load_integrao(repo_path: object) -> tuple[ModuleType, ModuleType, SourceRecord]:
    root = _repo(repo_path, "IntegrAO")
    rel = ["integrao/IntegrAO_supervised.py", "integrao/supervised_train.py"]
    record = _record(
        "IntegrAO", root, rel,
        "official GraphSAGE body and KL/alignment/classification objectives are "
        "used; transductive author graphs are replaced by training-only graphs "
        "and one-way train-to-query inductive graphs.")
    try:
        model = _import_from_root(root, "integrao.IntegrAO_supervised")
        losses = _import_from_root(root, "integrao.supervised_train")
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown dependency"
        raise OfficialSourceError(
            f"IntegrAO dependency '{missing}' is unavailable. Install the author's "
            "requirements (notably torch-geometric and snfpy) in the active environment.") from exc
    _assert_from(model, root, "IntegrAO")
    _assert_from(losses, root, "IntegrAO loss")
    return model, losses, record


def audit_selected_sources(methods: Iterable[str], cfg: Mapping[str, object]) -> dict:
    """Hash selected official sources without importing optional dependencies."""
    records: Dict[str, dict] = {}
    for key in methods:
        if key == "rusi":
            path = Path(str(cfg["rusi_model_file"])).expanduser().resolve()
            if not path.is_file():
                raise OfficialSourceError(
                    f"RUSI model file was not found: {path}. Set "
                    "RUSI_MODEL_FILE or pass --rusi-model-file.")
            rec = SourceRecord(
                method="RUSI (ours)", repository=str(path.parent),
                files={path.name: _sha256(path)},
                adaptation=(
                    "user model body/selective-imputation/router/losses with the "
                    "common leakage-safe fold adapter and paired nested CV"),
            )
        elif key == "dimv":
            root = _repo(cfg["dimv_repo"], "DIMV")
            rec = _record("DIMV", root, [
                "src/DIMVImputation.py", "src/conditional_expectation.py",
                "src/dpers.py", "src/utils.py"],
                "fold-local official imputation plus common downstream classifier")
        elif key == "clclsa":
            root = _repo(cfg["clclsa_repo"], "CLCLSA/CLUECL3")
            rec = _record("CLCLSA/CLUECL3", root, [
                "networks/models/clcl.py", "networks/models/common_layers.py",
                "networks/models/losses.py"],
                "official model/loss with inner-CV rather than test-set selection")
        elif key == "tmonet":
            root = _repo(cfg["tmonet_repo"], "TMO-Net")
            rec = _record("TMO-Net", root, [
                "model/TMO_Net_model.py", "util/loss_function.py"],
                "official body/loss with patient-mask dispatch and no external checkpoint")
        elif key == "mogedn":
            root = _repo(cfg["mogedn_repo"], "MOGEDN")
            rec = _record("MOGEDN", root, ["models.py", "utils.py"],
                          "official modules/loss with fold-local inductive graphs")
        elif key == "integrao":
            root = _repo(cfg["integrao_repo"], "IntegrAO")
            rec = _record("IntegrAO", root, [
                "integrao/IntegrAO_supervised.py",
                "integrao/supervised_train.py"],
                "official GraphSAGE/loss with fold-local inductive graphs")
        else:
            continue
        records[key] = asdict(rec)
    return records
