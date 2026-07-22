"""Gradient-free retrieval classification for paired B-mode/ULM view sets.

The frozen BiomedCLIP embeddings form a labelled key-value cache.  A query
patient is compared with each cached patient by symmetric set matching over
their valid paired views.  Class-balanced top-k cache evidence is fused with
nearest-prototype and class-conditional subspace scores.  All configuration and
threshold selection uses inner OOF predictions; outer labels are read once.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
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


SEED = 42


@dataclass(frozen=True)
class RetrievalConfig:
    k: int
    bmode_weight: float
    density_penalty: float
    prototype_weight: float
    subspace_weight: float
    beta: float = 10.0


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def masked_mean(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    weight = valid[..., None].astype(np.float32)
    return (x * weight).sum(axis=1) / np.maximum(weight.sum(axis=1), 1.0)


def load_features(path: Path):
    data = np.load(path)
    required = {"X_bmode", "X_ulm", "density", "y"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
    bmode = data["X_bmode"].astype(np.float32)
    ulm = data["X_ulm"].astype(np.float32)
    density = data["density"].astype(np.float32)
    labels = data["y"].astype(np.int64)
    valid = (
        data["valid"].astype(bool)
        if "valid" in data.files else np.ones_like(density, dtype=bool)
    )
    if bmode.ndim != 3 or bmode.shape != ulm.shape:
        raise ValueError("X_bmode and X_ulm must have shape (N,V,D)")
    if bmode.shape[:2] != density.shape or density.shape != valid.shape:
        raise ValueError("density/valid must match the (N,V) feature dimensions")
    if not valid.any(axis=1).all():
        raise ValueError("each patient must contain at least one valid view")
    return l2_normalize(bmode), l2_normalize(ulm), density, valid, labels


def patient_similarity_matrix(bmode: np.ndarray, ulm: np.ndarray,
                              density: np.ndarray, valid: np.ndarray,
                              bmode_weight: float,
                              density_penalty: float) -> np.ndarray:
    """Symmetric Chamfer matching between two four-view patient sets."""
    n = len(bmode)
    result = np.empty((n, n), dtype=np.float32)
    for query in range(n):
        b_sim = np.einsum("vd,nwd->nvw", bmode[query], bmode)
        u_sim = np.einsum("vd,nwd->nvw", ulm[query], ulm)
        d_gap = np.abs(density[query][None, :, None] - density[:, None, :])
        pair = (
            bmode_weight * b_sim
            + (1.0 - bmode_weight) * u_sim
            - density_penalty * d_gap
        )
        pair_valid = valid[query][None, :, None] & valid[:, None, :]
        pair = np.where(pair_valid, pair, -np.inf)

        query_best = pair.max(axis=2)
        query_score = (
            np.where(valid[query][None, :], query_best, 0.0).sum(axis=1)
            / valid[query].sum()
        )
        reference_best = pair.max(axis=1)
        reference_score = (
            np.where(valid, reference_best, 0.0).sum(axis=1)
            / valid.sum(axis=1)
        )
        result[query] = 0.5 * (query_score + reference_score)
    return result


def _soft_topk(values: np.ndarray, k: int, beta: float) -> float:
    k = min(k, len(values))
    top = np.partition(values, len(values) - k)[-k:]
    scaled = beta * top
    maximum = scaled.max()
    return float((maximum + np.log(np.exp(scaled - maximum).mean())) / beta)


def cache_scores(similarity: np.ndarray, labels: np.ndarray,
                 query_idx: np.ndarray, bank_idx: np.ndarray,
                 k: int, beta: float) -> np.ndarray:
    scores = np.empty(len(query_idx), dtype=np.float32)
    benign_bank = bank_idx[labels[bank_idx] == 0]
    malignant_bank = bank_idx[labels[bank_idx] == 1]
    if len(benign_bank) == 0 or len(malignant_bank) == 0:
        raise ValueError("both classes are required in every memory bank")
    for out_index, query in enumerate(query_idx):
        benign = benign_bank[benign_bank != query]
        malignant = malignant_bank[malignant_bank != query]
        scores[out_index] = (
            _soft_topk(similarity[query, malignant], k, beta)
            - _soft_topk(similarity[query, benign], k, beta)
        )
    return scores


def patient_vectors(bmode: np.ndarray, ulm: np.ndarray,
                    valid: np.ndarray) -> np.ndarray:
    b_global = l2_normalize(masked_mean(bmode, valid))
    u_global = l2_normalize(masked_mean(ulm, valid))
    return l2_normalize(np.concatenate((b_global, u_global), axis=1))


def prototype_scores(vectors: np.ndarray, labels: np.ndarray,
                     query_idx: np.ndarray, bank_idx: np.ndarray) -> np.ndarray:
    benign = l2_normalize(vectors[bank_idx[labels[bank_idx] == 0]].mean(axis=0))
    malignant = l2_normalize(vectors[bank_idx[labels[bank_idx] == 1]].mean(axis=0))
    return vectors[query_idx] @ malignant - vectors[query_idx] @ benign


def _subspace(vectors: np.ndarray, rank: int = 16):
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[:min(rank, len(vectors) - 1)]


def _residual(vectors: np.ndarray, mean: np.ndarray,
              components: np.ndarray) -> np.ndarray:
    centered = vectors - mean
    projected = (centered @ components.T) @ components
    return np.linalg.norm(centered - projected, axis=1)


def subspace_scores(vectors: np.ndarray, labels: np.ndarray,
                    query_idx: np.ndarray, bank_idx: np.ndarray) -> np.ndarray:
    benign = vectors[bank_idx[labels[bank_idx] == 0]]
    malignant = vectors[bank_idx[labels[bank_idx] == 1]]
    benign_mean, benign_components = _subspace(benign)
    malignant_mean, malignant_components = _subspace(malignant)
    query = vectors[query_idx]
    # Larger means closer to the malignant subspace than the benign subspace.
    return (
        _residual(query, benign_mean, benign_components)
        - _residual(query, malignant_mean, malignant_components)
    )


def metrics(labels: np.ndarray, scores: np.ndarray,
            threshold: float | np.ndarray) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "threshold": (
            float(threshold) if np.asarray(threshold).ndim == 0 else None
        ),
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    unique = np.unique(scores)
    candidates = (
        np.asarray([0.0]) if len(unique) < 2
        else np.r_[unique[0] - 1e-6, (unique[:-1] + unique[1:]) / 2,
                   unique[-1] + 1e-6]
    )
    rows = [metrics(labels, scores, threshold) for threshold in candidates]
    return max(
        rows,
        key=lambda row: (
            row["f1"], row["accuracy"], row["precision"], row["recall"]
        ),
    )["threshold"]


def configurations() -> list[RetrievalConfig]:
    configs = []
    for k in (1, 3, 5):
        for bmode_weight in (0.25, 0.50):
            for density_penalty in (0.0, 0.25):
                configs.extend((
                    RetrievalConfig(k, bmode_weight, density_penalty, 0.0, 0.0),
                    RetrievalConfig(k, bmode_weight, density_penalty, 0.5, 0.0),
                    RetrievalConfig(k, bmode_weight, density_penalty, 0.5, 0.25),
                ))
    return configs


def combined_scores(config: RetrievalConfig, similarity_cache: dict,
                    vectors: np.ndarray, labels: np.ndarray,
                    query_idx: np.ndarray, bank_idx: np.ndarray) -> np.ndarray:
    similarity = similarity_cache[(config.bmode_weight, config.density_penalty)]
    score = cache_scores(
        similarity, labels, query_idx, bank_idx, config.k, config.beta
    )
    if config.prototype_weight:
        score += config.prototype_weight * prototype_scores(
            vectors, labels, query_idx, bank_idx
        )
    if config.subspace_weight:
        score += config.subspace_weight * subspace_scores(
            vectors, labels, query_idx, bank_idx
        )
    return score


def nested_cross_validate(bmode: np.ndarray, ulm: np.ndarray,
                          density: np.ndarray, valid: np.ndarray,
                          labels: np.ndarray, output_dir: Path,
                          seed: int = SEED):
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = configurations()
    similarity_cache = {
        (weight, penalty): patient_similarity_matrix(
            bmode, ulm, density, valid, weight, penalty
        )
        for weight in (0.25, 0.50)
        for penalty in (0.0, 0.25)
    }
    vectors = patient_vectors(bmode, ulm, valid)
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof_score = np.full(len(labels), np.nan, dtype=np.float32)
    oof_threshold = np.full(len(labels), np.nan, dtype=np.float32)
    results = []

    for fold, (outer_train, outer_test) in enumerate(
        outer.split(np.arange(len(labels)), labels), start=1
    ):
        inner = StratifiedKFold(3, shuffle=True, random_state=seed + fold)
        config_predictions = {
            config: np.full(len(outer_train), np.nan, dtype=np.float32)
            for config in configs
        }
        for rel_train, rel_val in inner.split(outer_train, labels[outer_train]):
            inner_train = outer_train[rel_train]
            inner_val = outer_train[rel_val]
            for config in configs:
                config_predictions[config][rel_val] = combined_scores(
                    config, similarity_cache, vectors, labels,
                    inner_val, inner_train,
                )

        candidates = []
        for config, score in config_predictions.items():
            if np.isnan(score).any():
                raise RuntimeError("inner OOF retrieval did not cover every patient")
            threshold = select_threshold(labels[outer_train], score)
            candidate_metrics = metrics(labels[outer_train], score, threshold)
            candidates.append((
                candidate_metrics["roc_auc"], candidate_metrics["f1"],
                candidate_metrics["accuracy"], config, threshold,
                candidate_metrics,
            ))
        _, _, _, best_config, threshold, inner_metrics = max(
            candidates, key=lambda item: item[:3]
        )

        test_score = combined_scores(
            best_config, similarity_cache, vectors, labels,
            outer_test, outer_train,
        )
        fold_metrics = metrics(labels[outer_test], test_score, threshold)
        fold_metrics.update({
            "fold": fold,
            "config": asdict(best_config),
            "inner_oof_metrics": inner_metrics,
        })
        results.append(fold_metrics)
        oof_score[outer_test] = test_score
        oof_threshold[outer_test] = threshold
        print(
            f"fold={fold} acc={fold_metrics['accuracy']:.4f} "
            f"auc={fold_metrics['roc_auc']:.4f} "
            f"precision={fold_metrics['precision']:.4f} "
            f"recall={fold_metrics['recall']:.4f} "
            f"f1={fold_metrics['f1']:.4f} k={best_config.k}"
        )

    with (output_dir / "retrieval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    np.savez_compressed(
        output_dir / "retrieval_oof.npz",
        y=labels, score=oof_score, threshold=oof_threshold,
    )
    keys = [
        "accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "precision",
        "recall", "specificity", "f1", "mcc",
    ]
    summary = {
        "fold_mean": {
            key: float(np.mean([row[key] for row in results])) for key in keys
        },
        "fold_sd": {
            key: float(np.std([row[key] for row in results])) for key in keys
        },
        "majority_class_accuracy": float(
            np.max(np.bincount(labels)) / len(labels)
        ),
        "pooled_strict": metrics(labels, oof_score, oof_threshold),
        "method": "gradient-free paired-view cache + prototype + subspace",
    }
    with (output_dir / "retrieval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=Path,
        default=Path("/root/medic_data/biomedclip_perview_features.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("retrieval_outputs"))
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    bmode, ulm, density, valid, labels = load_features(args.features)
    print(
        f"samples={len(labels)} views={bmode.shape[1]} dim={bmode.shape[2]} "
        "mode=gradient-free nested-CV"
    )
    nested_cross_validate(
        bmode, ulm, density, valid, labels, args.output_dir, seed=args.seed
    )


if __name__ == "__main__":
    main()
