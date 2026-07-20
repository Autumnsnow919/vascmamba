"""Rigorous VascMamba-Hybrid training on real per-view BiomedCLIP features.

The original experiment expanded one session-averaged B-mode embedding and one
session-averaged ULM embedding into four identical "views".  This entry point
requires genuine per-view features instead and evaluates them with nested
cross-validation: the outer fold is never used for early stopping or threshold
selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


SEED = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SelectiveSSM(nn.Module):
    """Small selective state-space layer for short per-view sequences.

    This remains a lightweight research implementation rather than the official
    ``mamba_ssm`` kernel.  Unlike the previous version, every declared parameter
    contributes to the forward pass: ``D`` supplies the direct skip term and the
    formerly ignored extra ``x_proj`` channel now modulates delta. Parameter
    shapes stay compatible with historical Hybrid checkpoints.
    """

    def __init__(self, d_model: int = 32, d_state: int = 4, d_conv: int = 2,
                 expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(self.d_inner, 1)

        a = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, d_state)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        x_in, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :length]
        x_conv = self.act(x_conv).transpose(1, 2)

        projected = self.x_proj(x_conv)
        b_ssm = projected[..., :self.d_state]
        c_ssm = projected[..., self.d_state:2 * self.d_state]
        dt_residual = projected[..., 2 * self.d_state:]
        dt = F.softplus(self.dt_proj(x_conv) + dt_residual)

        # A is shared across inner channels to preserve legacy checkpoint shapes.
        a = -torch.exp(self.A_log).to(dtype=x.dtype)
        a_bar = torch.exp(dt.unsqueeze(-1) * a.unsqueeze(0).unsqueeze(0))
        b_bar = (
            dt.unsqueeze(-1)
            * b_ssm.unsqueeze(2)
            * x_conv.unsqueeze(-1)
        )

        h = x.new_zeros(batch, self.d_inner, self.d_state)
        outputs = []
        for t in range(length):
            h = a_bar[:, t] * h + b_bar[:, t]
            outputs.append((h * c_ssm[:, t].unsqueeze(1)).sum(dim=-1))

        y = torch.stack(outputs, dim=1)
        y = y + x_conv * self.D.to(dtype=x.dtype)
        y = y * self.act(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int = 32, d_state: int = 4, d_conv: int = 2,
                 expand: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class VascMambaHybrid(nn.Module):
    """Fuse four genuine paired B-mode/ULM view embeddings.

    Density ordering is disabled by default because input order should only be
    changed when the ordering hypothesis is explicitly under ablation.  When it
    is enabled, B-mode, ULM, density and validity masks are sorted together so
    modality pairing is preserved.
    """

    def __init__(self, bc_dim: int = 512, d_model: int = 32, d_state: int = 4,
                 n_layers: int = 1, n_views: int = 4,
                 order_by_density: bool = False):
        super().__init__()
        self.n_views = n_views
        self.seq_len = n_views * 2
        self.order_by_density = order_by_density

        self.bmode_proj = nn.Sequential(
            nn.Linear(bc_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.ulm_proj = nn.Sequential(
            nn.Linear(bc_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        self.mod_emb = nn.Parameter(torch.zeros(1, 2, d_model))
        self.register_buffer(
            "mod_ids", torch.tensor([0, 1] * n_views, dtype=torch.long),
            persistent=False,
        )

        self.mamba = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv=2, expand=2)
             for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def _validate_inputs(self, bmode_feats: torch.Tensor,
                         ulm_feats: torch.Tensor,
                         ulm_density: torch.Tensor | None,
                         view_mask: torch.Tensor | None) -> None:
        expected = (bmode_feats.shape[0], self.n_views)
        if bmode_feats.ndim != 3 or ulm_feats.shape != bmode_feats.shape:
            raise ValueError("B-mode and ULM features must both have shape (B,V,D)")
        if bmode_feats.shape[1] != self.n_views:
            raise ValueError(f"expected {self.n_views} views, got {bmode_feats.shape[1]}")
        if ulm_density is not None and ulm_density.shape != expected:
            raise ValueError("density must have shape (B,V)")
        if view_mask is not None and view_mask.shape != expected:
            raise ValueError("view_mask must have shape (B,V)")

    def forward(self, bmode_feats: torch.Tensor, ulm_feats: torch.Tensor,
                ulm_density: torch.Tensor | None = None,
                view_mask: torch.Tensor | None = None) -> torch.Tensor:
        self._validate_inputs(bmode_feats, ulm_feats, ulm_density, view_mask)
        batch = bmode_feats.shape[0]
        if view_mask is None:
            view_mask = torch.ones(
                batch, self.n_views, device=bmode_feats.device, dtype=torch.bool
            )

        if self.order_by_density:
            if ulm_density is None:
                raise ValueError("density is required when order_by_density=True")
            sort_idx = ulm_density.argsort(dim=1, descending=True)
            feat_idx = sort_idx.unsqueeze(-1).expand_as(bmode_feats)
            bmode_feats = bmode_feats.gather(1, feat_idx)
            ulm_feats = ulm_feats.gather(1, feat_idx)
            ulm_density = ulm_density.gather(1, sort_idx)
            view_mask = view_mask.gather(1, sort_idx)

        b_tokens = self.bmode_proj(bmode_feats)
        u_tokens = self.ulm_proj(ulm_feats)
        tokens = torch.stack((b_tokens, u_tokens), dim=2).flatten(1, 2)
        token_mask = view_mask.unsqueeze(-1).expand(-1, -1, 2).flatten(1, 2)

        tokens = tokens + self.mod_emb[0, self.mod_ids].unsqueeze(0)
        tokens = tokens + self.pos_emb
        tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)
        for layer in self.mamba:
            tokens = layer(tokens)
            tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)

        denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1).to(tokens.dtype)
        pooled = tokens.sum(dim=1) / denom
        return self.head(pooled)


class FeatDataset(Dataset):
    def __init__(self, bmode: torch.Tensor, ulm: torch.Tensor,
                 density: torch.Tensor, labels: torch.Tensor,
                 valid: torch.Tensor | None = None):
        self.bmode = bmode
        self.ulm = ulm
        self.density = density
        self.labels = labels
        self.valid = torch.ones_like(density, dtype=torch.bool) if valid is None else valid.bool()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.bmode[index], self.ulm[index], self.density[index],
            self.valid[index], self.labels[index],
        )


@dataclass
class TrainConfig:
    d_model: int = 32
    d_state: int = 4
    n_layers: int = 1
    order_by_density: bool = False
    epochs: int = 100
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 5e-3
    patience: int = 20
    inner_folds: int = 3
    ensemble_size: int = 3
    class_weight_power: float = 1.0
    threshold_objective: str = "f1"
    recall_floor: float = 0.90


def class_weights(labels: torch.Tensor, device: torch.device,
                  power: float = 1.0) -> torch.Tensor:
    """Return softened inverse-frequency weights.

    ``power=1`` exactly balances both classes, while ``power=0`` is ordinary
    cross-entropy.  Intermediate values are available as an explicit ablation;
    the default remains fully balanced for comparability with prior runs.
    """
    if not 0.0 <= power <= 1.0:
        raise ValueError("class_weight_power must be in [0, 1]")
    counts = torch.bincount(labels.long(), minlength=2).float()
    if (counts == 0).any():
        raise ValueError("both classes are required in each training split")
    balanced = counts.sum() / (2.0 * counts)
    return balanced.pow(power).to(device)


def make_loader(bmode: torch.Tensor, ulm: torch.Tensor, density: torch.Tensor,
                valid: torch.Tensor, labels: torch.Tensor, indices: np.ndarray,
                batch_size: int, shuffle: bool) -> DataLoader:
    ds = FeatDataset(
        bmode[indices], ulm[indices], density[indices], labels[indices], valid[indices]
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probabilities, labels = [], []
    for bmode, ulm, density, valid, target in loader:
        logits = model(
            bmode.to(device), ulm.to(device), density.to(device), valid.to(device)
        )
        probabilities.append(logits.softmax(dim=-1)[:, 1].cpu())
        labels.append(target)
    return torch.cat(probabilities).numpy(), torch.cat(labels).numpy()


def _threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    """All decision boundaries that can change a prediction, plus 0.5."""
    unique = np.unique(np.asarray(probabilities, dtype=float))
    if len(unique) < 2:
        return np.asarray([0.5], dtype=float)
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return np.unique(np.clip(np.r_[0.0, midpoints, 0.5, 1.0], 0.0, 1.0))


def select_operating_threshold(labels: np.ndarray, probabilities: np.ndarray,
                               objective: str = "clinical",
                               recall_floor: float = 0.90) -> float:
    """Select a threshold only from inner out-of-fold predictions.

    ``clinical`` maximizes accuracy while requiring malignant sensitivity to
    remain above ``recall_floor``.  Precision and F1 break ties.  This avoids
    the previous behaviour where a tiny, single validation split could choose
    a threshold below the all-malignant operating point.  Alternative
    objectives are exposed for preregistered ablations.
    """
    if objective not in {"clinical", "f1", "balanced_accuracy"}:
        raise ValueError(f"unknown threshold objective: {objective}")
    if not 0.0 <= recall_floor <= 1.0:
        raise ValueError("recall_floor must be in [0, 1]")

    rows = []
    for threshold in _threshold_candidates(probabilities):
        metrics = classification_metrics(labels, probabilities, float(threshold))
        rows.append(metrics)

    if objective == "clinical":
        feasible = [row for row in rows if row["sensitivity"] >= recall_floor]
        pool = feasible or rows
        key = lambda row: (
            row["accuracy"], row["precision"], row["f1"],
            row["sensitivity"], -abs(row["threshold"] - 0.5),
        )
    elif objective == "f1":
        pool = rows
        key = lambda row: (
            row["f1"], row["accuracy"], row["precision"],
            row["sensitivity"], -abs(row["threshold"] - 0.5),
        )
    else:
        pool = rows
        key = lambda row: (
            row["balanced_accuracy"], row["f1"], row["accuracy"],
            -abs(row["threshold"] - 0.5),
        )
    return float(max(pool, key=key)["threshold"])


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Backward-compatible alias with deterministic accuracy tie-breaking."""
    return select_operating_threshold(labels, probabilities, objective="f1")


def classification_metrics_from_predictions(
        labels: np.ndarray, probabilities: np.ndarray,
        pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, pred),
        "balanced_accuracy": balanced_accuracy_score(labels, pred),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
        "precision": precision_score(labels, pred, pos_label=1, zero_division=0),
        "sensitivity": recall_score(labels, pred, pos_label=1, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(labels, pred, zero_division=0),
        "mcc": matthews_corrcoef(labels, pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray,
                           threshold: float) -> dict[str, float]:
    pred = (probabilities >= threshold).astype(np.int64)
    return {
        "threshold": threshold,
        **classification_metrics_from_predictions(labels, probabilities, pred),
    }


def fit_inner_split(model: VascMambaHybrid, bmode: torch.Tensor,
                    ulm: torch.Tensor, density: torch.Tensor, valid: torch.Tensor,
                    labels: torch.Tensor, train_idx: np.ndarray,
                    val_idx: np.ndarray, config: TrainConfig,
                    device: torch.device) -> tuple[VascMambaHybrid, float, dict]:
    train_loader = make_loader(
        bmode, ulm, density, valid, labels, train_idx,
        config.batch_size, shuffle=True,
    )
    val_loader = make_loader(
        bmode, ulm, density, valid, labels, val_idx,
        config.batch_size, shuffle=False,
    )
    weights = class_weights(
        labels[train_idx], device, power=config.class_weight_power
    )
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs * len(train_loader)), eta_min=1e-6
    )

    best_loss = math.inf
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    for epoch in range(config.epochs):
        model.train()
        for b, u, d, mask, target in train_loader:
            b, u, d = b.to(device), u.to(device), d.to(device)
            mask, target = mask.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(b, u, d, mask)
            loss = F.cross_entropy(logits, target, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for b, u, d, mask, target in val_loader:
                b, u, d = b.to(device), u.to(device), d.to(device)
                mask, target = mask.to(device), target.to(device)
                loss = F.cross_entropy(model(b, u, d, mask), target, weight=weights)
                val_loss_sum += loss.item() * len(target)
                n_val += len(target)
        val_loss = val_loss_sum / max(n_val, 1)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs > config.patience:
            break

    model.load_state_dict(best_state)
    val_prob, val_labels = predict(model, val_loader, device)
    threshold = select_operating_threshold(
        val_labels, val_prob,
        objective=config.threshold_objective,
        recall_floor=config.recall_floor,
    )
    diagnostics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "inner_metrics": classification_metrics(val_labels, val_prob, threshold),
    }
    return model, threshold, diagnostics


def fit_fixed_epochs(model: VascMambaHybrid, bmode: torch.Tensor,
                     ulm: torch.Tensor, density: torch.Tensor,
                     valid: torch.Tensor, labels: torch.Tensor,
                     train_idx: np.ndarray, epochs: int, config: TrainConfig,
                     device: torch.device) -> VascMambaHybrid:
    """Refit on the complete outer-training fold after model selection."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    loader = make_loader(
        bmode, ulm, density, valid, labels, train_idx,
        config.batch_size, shuffle=True,
    )
    weights = class_weights(
        labels[train_idx], device, power=config.class_weight_power
    )
    optimizer = AdamW(model.parameters(), lr=config.lr,
                      weight_decay=config.weight_decay)
    # Match the inner-training learning-rate trajectory.  Using ``epochs`` as
    # T_max here would decay to eta_min much earlier than the model-selection
    # runs and make the selected epoch non-transferable.
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs * len(loader)), eta_min=1e-6
    )
    for _ in range(epochs):
        model.train()
        for b, u, d, mask, target in loader:
            b, u, d = b.to(device), u.to(device), d.to(device)
            mask, target = mask.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(b, u, d, mask), target, weight=weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
    return model


def select_with_inner_oof(bmode: torch.Tensor, ulm: torch.Tensor,
                          density: torch.Tensor, valid: torch.Tensor,
                          labels: torch.Tensor, outer_train: np.ndarray,
                          config: TrainConfig, device: torch.device,
                          seed: int) -> tuple[int, float, dict]:
    """Estimate epochs and threshold using every outer-training patient once."""
    if config.inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    y_outer = labels[outer_train].numpy()
    splitter = StratifiedKFold(
        n_splits=config.inner_folds, shuffle=True, random_state=seed
    )
    inner_probability = np.full(len(outer_train), np.nan, dtype=np.float32)
    best_epochs = []
    for inner_fold, (rel_train, rel_val) in enumerate(
        splitter.split(np.arange(len(outer_train)), y_outer), start=1
    ):
        inner_train = outer_train[rel_train]
        inner_val = outer_train[rel_val]
        seed_everything(seed + inner_fold)
        model = VascMambaHybrid(
            d_model=config.d_model,
            d_state=config.d_state,
            n_layers=config.n_layers,
            n_views=bmode.shape[1],
            order_by_density=config.order_by_density,
        ).to(device)
        model, _, diagnostics = fit_inner_split(
            model, bmode, ulm, density, valid, labels,
            inner_train, inner_val, config, device,
        )
        val_loader = make_loader(
            bmode, ulm, density, valid, labels, inner_val,
            config.batch_size, shuffle=False,
        )
        probability, target = predict(model, val_loader, device)
        if not np.array_equal(target, y_outer[rel_val]):
            raise RuntimeError("inner OOF labels are misaligned")
        inner_probability[rel_val] = probability
        best_epochs.append(int(diagnostics["best_epoch"]) + 1)

    if np.isnan(inner_probability).any():
        raise RuntimeError("inner cross-validation did not cover every patient")
    selected_epochs = max(1, int(np.median(best_epochs)))
    threshold = select_operating_threshold(
        y_outer, inner_probability,
        objective=config.threshold_objective,
        recall_floor=config.recall_floor,
    )
    diagnostics = {
        "inner_best_epochs": best_epochs,
        "selected_epochs": selected_epochs,
        "inner_oof_metrics": classification_metrics(
            y_outer, inner_probability, threshold
        ),
    }
    return selected_epochs, threshold, diagnostics


def nested_cross_validate(bmode: torch.Tensor, ulm: torch.Tensor,
                          density: torch.Tensor, valid: torch.Tensor,
                          labels: torch.Tensor, config: TrainConfig,
                          output_dir: Path, device: torch.device,
                          seed: int = SEED) -> list[dict]:
    if config.ensemble_size < 1:
        raise ValueError("ensemble_size must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    y_np = labels.numpy()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []
    oof_probability = np.full(len(labels), np.nan, dtype=np.float32)
    oof_fold = np.full(len(labels), -1, dtype=np.int64)
    oof_threshold = np.full(len(labels), np.nan, dtype=np.float32)

    for fold, (outer_train, outer_test) in enumerate(
        outer.split(np.arange(len(labels)), y_np), start=1
    ):
        selected_epochs, threshold, diagnostics = select_with_inner_oof(
            bmode, ulm, density, valid, labels, outer_train,
            config, device, seed=seed + fold * 100,
        )
        test_loader = make_loader(
            bmode, ulm, density, valid, labels, outer_test,
            config.batch_size, shuffle=False,
        )
        ensemble_probability = []
        ensemble_states = []
        target = None
        for member in range(config.ensemble_size):
            member_seed = seed + fold * 1000 + member
            seed_everything(member_seed)
            model = VascMambaHybrid(
                d_model=config.d_model,
                d_state=config.d_state,
                n_layers=config.n_layers,
                n_views=bmode.shape[1],
                order_by_density=config.order_by_density,
            ).to(device)
            model = fit_fixed_epochs(
                model, bmode, ulm, density, valid, labels,
                outer_train, selected_epochs, config, device,
            )
            member_probability, member_target = predict(model, test_loader, device)
            if target is None:
                target = member_target
            elif not np.array_equal(target, member_target):
                raise RuntimeError("ensemble member labels are misaligned")
            ensemble_probability.append(member_probability)
            ensemble_states.append(copy.deepcopy(model.state_dict()))
        probability = np.mean(ensemble_probability, axis=0)
        assert target is not None
        oof_probability[outer_test] = probability
        oof_fold[outer_test] = fold
        oof_threshold[outer_test] = threshold
        metrics = classification_metrics(target, probability, threshold)
        metrics.update({"fold": fold, **diagnostics})
        results.append(metrics)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "ensemble_state_dicts": ensemble_states,
                "threshold": threshold,
                "config": asdict(config),
                "fold": fold,
                "outer_train_indices": outer_train,
                "outer_test_indices": outer_test,
            },
            output_dir / f"fold_{fold}.pt",
        )
        print(
            f"fold={fold} acc={metrics['accuracy']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} "
            f"auc={metrics['roc_auc']:.4f} precision={metrics['precision']:.4f} "
            f"recall={metrics['sensitivity']:.4f} f1={metrics['f1']:.4f} "
            f"threshold={threshold:.2f}"
        )

    with (output_dir / "nested_cv_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if np.isnan(oof_probability).any() or (oof_fold < 0).any():
        raise RuntimeError("outer cross-validation did not produce exactly one prediction per sample")
    np.savez_compressed(
        output_dir / "nested_cv_oof.npz",
        y=y_np,
        probability=oof_probability,
        fold=oof_fold,
        threshold=oof_threshold,
    )
    strict_pred = (oof_probability >= oof_threshold).astype(np.int64)
    metric_keys = [
        "accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "precision",
        "sensitivity", "specificity", "f1", "mcc",
    ]
    summary = {
        "fold_mean": {
            key: float(np.mean([result[key] for result in results]))
            for key in metric_keys
        },
        "fold_sd": {
            key: float(np.std([result[key] for result in results]))
            for key in metric_keys
        },
        "pooled_strict": classification_metrics_from_predictions(
            y_np, oof_probability, strict_pred
        ),
        "threshold_mean": float(oof_threshold.mean()),
        "threshold_sd": float(oof_threshold.std()),
        "majority_class_accuracy": float(np.max(np.bincount(y_np)) / len(y_np)),
        "note": (
            "pooled_strict uses only fold-specific thresholds selected from inner OOF; "
            "the outer labels never select a threshold"
        ),
    }
    with (output_dir / "nested_cv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return results


def train_hybrid(model, X_bmode, X_ulm, X_density, y, idx_train, idx_val,
                 epochs=100, lr=5e-4, batch_size=32, X_valid=None):
    """Compatibility wrapper for older experiment scripts.

    This function still evaluates its supplied validation indices, so it must
    not be used to claim an unbiased test result.  New experiments should call
    :func:`nested_cross_validate`.
    """
    valid = torch.ones_like(X_density, dtype=torch.bool) if X_valid is None else X_valid
    config = TrainConfig(
        d_model=model.head[1].in_features,
        order_by_density=model.order_by_density,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    device = next(model.parameters()).device
    model, threshold, _ = fit_inner_split(
        model, X_bmode, X_ulm, X_density, valid, y,
        np.asarray(idx_train), np.asarray(idx_val), config, device,
    )
    loader = make_loader(
        X_bmode, X_ulm, X_density, valid, y, np.asarray(idx_val),
        batch_size, shuffle=False,
    )
    probability, target = predict(model, loader, device)
    metrics = classification_metrics(target, probability, threshold)
    return {
        "acc": metrics["accuracy"], "auc": metrics["roc_auc"],
        "recall": metrics["sensitivity"], "f1": metrics["f1"],
        "threshold": threshold,
    }


def load_per_view_features(path: Path):
    data = np.load(path)
    required = {"X_bmode", "X_ulm", "density", "y"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
    bmode = torch.from_numpy(data["X_bmode"]).float()
    ulm = torch.from_numpy(data["X_ulm"]).float()
    density = torch.from_numpy(data["density"]).float()
    labels = torch.from_numpy(data["y"]).long()
    valid = (
        torch.from_numpy(data["valid"]).bool()
        if "valid" in data.files else torch.ones_like(density, dtype=torch.bool)
    )
    if bmode.ndim != 3 or bmode.shape != ulm.shape:
        raise ValueError("X_bmode and X_ulm must have shape (N,V,512)")
    if bmode.shape[:2] != density.shape or density.shape != valid.shape:
        raise ValueError("density/valid must match the (N,V) feature dimensions")
    if len(labels) != len(bmode):
        raise ValueError("labels and feature arrays have different sample counts")
    # Expanded session means are exactly identical along the view axis.  Refuse
    # such files so the rigorous entry point cannot silently reproduce the old
    # fake-token experiment.  Repeated B-mode alone is allowed because some ULM
    # exports legitimately share one grayscale background.
    b_repeated = (bmode - bmode[:, :1]).abs().amax(dim=(1, 2)) == 0
    u_repeated = (ulm - ulm[:, :1]).abs().amax(dim=(1, 2)) == 0
    d_repeated = (density - density[:, :1]).abs().amax(dim=1) == 0
    fake_fraction = (b_repeated & u_repeated & d_repeated).float().mean().item()
    if fake_fraction > 0.95:
        raise ValueError(
            "features appear to be expanded session means: more than 95% of "
            "samples have identical B-mode, ULM and density values across views"
        )
    return bmode, ulm, density, valid, labels


def summarize(results: list[dict]) -> None:
    keys = [
        "accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "precision",
        "sensitivity", "specificity", "f1", "mcc",
    ]
    print("\nNested outer-fold results (mean ± fold SD)")
    for key in keys:
        values = np.asarray([result[key] for result in results], dtype=float)
        print(f"  {key:18s} {values.mean():.4f} ± {values.std():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=Path,
        default=Path("/root/medic_data/biomedclip_perview_features.npz"),
        help="NPZ with X_bmode/X_ulm=(N,V,512), density/valid=(N,V), y=(N,)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("hybrid_nested_outputs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument(
        "--class-weight-power", type=float, default=1.0,
        help="0=unweighted CE, 1=fully balanced CE; intermediate values are ablations",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=["clinical", "f1", "balanced_accuracy"], default="f1",
    )
    parser.add_argument(
        "--recall-floor", type=float, default=0.90,
        help="Minimum malignant recall used by the clinical threshold objective",
    )
    parser.add_argument(
        "--order-by-density", action="store_true",
        help="Ablation only: sort paired B-mode/ULM views by ULM density",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    bmode, ulm, density, valid, labels = load_per_view_features(args.features)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        order_by_density=args.order_by_density,
        inner_folds=args.inner_folds,
        ensemble_size=args.ensemble_size,
        class_weight_power=args.class_weight_power,
        threshold_objective=args.threshold_objective,
        recall_floor=args.recall_floor,
    )
    print(
        f"samples={len(labels)} views={bmode.shape[1]} feature_dim={bmode.shape[2]} "
        f"valid_views={int(valid.sum())}/{valid.numel()} device={device}"
    )
    results = nested_cross_validate(
        bmode, ulm, density, valid, labels, config,
        args.output_dir, device, seed=args.seed,
    )
    summarize(results)


if __name__ == "__main__":
    main()
