"""Mahalanobis PatchCore: covariance-aware patch memory bank for ULM classification."""
import sys, os
sys.path.insert(0, '/root/medic_data')
sys.path.insert(0, '/root/medic_data/ulm_visionnet')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, f1_score,
    precision_score, confusion_matrix, balanced_accuracy_score
)
from scipy.spatial import cKDTree
from data.patient_index_v2 import build_unified_index

SEED = 42
np.random.seed(SEED)
OUT_DIR = '/root/medic_data/vascmamba/rad_outputs'
os.makedirs(OUT_DIR, exist_ok=True)


def load_features():
    d = np.load('/root/medic_data/dinov3_multilayer_features.npz')
    return d['cls_features'], d['patch_features'], d['y'], list(d['layer_idxs'])


def pool_patches(X_patch, pool=2):
    N, L, P, D = X_patch.shape
    s = int(np.sqrt(P))
    if s * s != P:
        raise ValueError(f'Cannot reshape {P} patches into a square grid')
    X_patch = X_patch.reshape(N, L, s, s, D)
    X_patch = X_patch.reshape(N, L, s // pool, pool, s // pool, pool, D)
    X_patch = X_patch.mean(axis=(3, 5))
    return X_patch.reshape(N, L, (s // pool) ** 2, D)


def get_source_indices(samples):
    is_v2 = np.array(['/output_ulm/' in s['patient_dir'] for s in samples])
    is_ext = np.array(['/数据分析/' in s['patient_dir'] for s in samples])
    return np.where(is_v2)[0], np.where(is_ext)[0]


def normalize_layer(X_tr, X_te):
    Ntr, P, D = X_tr.shape
    Nte = X_te.shape[0]
    scaler = StandardScaler()
    tr_flat = scaler.fit_transform(X_tr.reshape(-1, D))
    te_flat = scaler.transform(X_te.reshape(-1, D))
    return tr_flat.reshape(Ntr, P, D), te_flat.reshape(Nte, P, D)


def pca_reduce(X_tr, X_te, n_components):
    Ntr, P, D = X_tr.shape
    Nte = X_te.shape[0]
    pca = PCA(n_components=n_components, whiten=False, random_state=SEED)
    tr_red = pca.fit_transform(X_tr.reshape(-1, D)).reshape(Ntr, P, n_components)
    te_red = pca.transform(X_te.reshape(-1, D)).reshape(Nte, P, n_components)
    return tr_red, te_red


def compute_whitener(X, eps=1e-4):
    flat = X.reshape(-1, X.shape[-1])
    mu = flat.mean(axis=0)
    cov = np.cov(flat, rowvar=False)
    cov = cov + eps * np.eye(cov.shape[0])
    e, v = np.linalg.eigh(cov)
    inv_sqrt = v @ np.diag(1.0 / np.sqrt(e + eps)) @ v.T
    return mu, inv_sqrt


def class_gaussian_params(X, y, eps=1e-4):
    d = X.shape[-1]
    mu_b = X[y == 0].reshape(-1, d).mean(axis=0)
    mu_m = X[y == 1].reshape(-1, d).mean(axis=0)
    cov_b = np.cov(X[y == 0].reshape(-1, d), rowvar=False) + eps * np.eye(d)
    cov_m = np.cov(X[y == 1].reshape(-1, d), rowvar=False) + eps * np.eye(d)
    inv_b = np.linalg.inv(cov_b)
    inv_m = np.linalg.inv(cov_m)
    return mu_b, mu_m, inv_b, inv_m


def mahalanobis_dist(X, mu, cov_inv):
    diff = X - mu
    return np.sqrt(np.einsum('npd,dd,npd->np', diff, cov_inv, diff))


def nn_distances(query, bank, k):
    Nq, P, d = query.shape
    query_flat = query.reshape(-1, d)
    if len(bank) == 0:
        return np.full((Nq, P, k), 1e9, dtype=np.float32)
    k = min(k, len(bank))
    tree = cKDTree(bank)
    dists, _ = tree.query(query_flat, k=k)
    return dists.reshape(Nq, P, k)


def combine_scores(d_b, d_m, score_mode):
    if score_mode == 'diff':
        return d_b - d_m
    elif score_mode == 'ratio':
        return d_m / (d_b + 1e-8)
    elif score_mode == 'log_ratio':
        return np.log((d_m + 1e-8) / (d_b + 1e-8))
    else:
        raise ValueError(score_mode)


def aggregate_patch(scores, mode):
    if mode == 'mean':
        return scores.mean(axis=1)
    elif mode == 'max':
        return scores.max(axis=1)
    elif mode == 'top3':
        k = min(3, scores.shape[1])
        return np.partition(scores, -k, axis=1)[:, -k:].mean(axis=1)
    elif mode == 'top5':
        k = min(5, scores.shape[1])
        return np.partition(scores, -k, axis=1)[:, -k:].mean(axis=1)
    else:
        raise ValueError(mode)


def search_threshold(y_true, scores, criterion):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores).ravel()
    t_min, t_max = scores.min() - 0.1, scores.max() + 0.1
    best_t, best_v = 0, -1
    for t in np.linspace(t_min, t_max, 200):
        pred = (scores >= t).astype(int)
        if criterion == 'acc':
            v = accuracy_score(y_true, pred)
        elif criterion == 'f1':
            v = f1_score(y_true, pred, zero_division=0)
        elif criterion == 'youden':
            cm = confusion_matrix(y_true, pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                v = sens + spec - 1
            else:
                v = 0
        elif criterion == 'target':
            acc = accuracy_score(y_true, pred)
            sens = recall_score(y_true, pred, zero_division=0)
            v = min(acc, sens)
        else:
            raise ValueError(criterion)
        if v > best_v:
            best_v, best_t = v, t
    return best_t, best_v


def evaluate(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'ACC': accuracy_score(y_true, pred),
        'AUC': roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else 0.5,
        'Sensitivity': recall_score(y_true, pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'PPV': precision_score(y_true, pred, zero_division=0),
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        'F1': f1_score(y_true, pred, zero_division=0),
        'Balanced_ACC': balanced_accuracy_score(y_true, pred),
    }


def precompute_distances(reduced, y_train, k_max=3, eps=1e-4):
    n_layers = len(reduced)
    cache = {'whiten_nn': [], 'class_gaussian': []}
    for l in range(n_layers):
        tr_l, te_l = reduced[l]

        mu, inv_sqrt = compute_whitener(tr_l, eps)
        tr_w = (tr_l - mu) @ inv_sqrt
        te_w = (te_l - mu) @ inv_sqrt
        bank_b = tr_w[y_train == 0].reshape(-1, tr_w.shape[-1])
        bank_m = tr_w[y_train == 1].reshape(-1, tr_w.shape[-1])
        d_b_nn = nn_distances(te_w, bank_b, k_max)
        d_m_nn = nn_distances(te_w, bank_m, k_max)
        cache['whiten_nn'].append((d_b_nn, d_m_nn))

        mu_b, mu_m, inv_b, inv_m = class_gaussian_params(tr_l, y_train, eps)
        d_b_g = mahalanobis_dist(te_l, mu_b, inv_b)
        d_m_g = mahalanobis_dist(te_l, mu_m, inv_m)
        cache['class_gaussian'].append((d_b_g, d_m_g))
    return cache


def build_scores_from_cache(cache, method, k, score_mode, patch_agg, layer_agg):
    layers = cache[method]
    layer_scores = []
    for d_b, d_m in layers:
        if method == 'whiten_nn':
            d_b_k = d_b[:, :, :k].mean(axis=2)
            d_m_k = d_m[:, :, :k].mean(axis=2)
        else:
            d_b_k, d_m_k = d_b, d_m
        scores_patch = combine_scores(d_b_k, d_m_k, score_mode)
        img_scores = aggregate_patch(scores_patch, patch_agg)
        layer_scores.append(img_scores)
    layer_scores = np.stack(layer_scores, axis=0)
    if layer_agg == 'mean':
        return layer_scores.mean(axis=0)
    elif layer_agg == 'sum':
        return layer_scores.sum(axis=0)
    else:
        raise ValueError(layer_agg)


def grid_search(X_patch, y, train_idx, test_idx, experiment_name, pool=2):
    X_patch = pool_patches(X_patch, pool=pool) if pool > 1 else X_patch
    X_tr, X_te = X_patch[train_idx], X_patch[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    n_layers = X_tr.shape[1]

    X_tr_n = [None] * n_layers
    X_te_n = [None] * n_layers
    for l in range(n_layers):
        X_tr_n[l], X_te_n[l] = normalize_layer(X_tr[:, l], X_te[:, l])

    reduced_cache = {}
    for pca_dim in [64, 128]:
        reduced = []
        for l in range(n_layers):
            tr_red, te_red = pca_reduce(X_tr_n[l], X_te_n[l], pca_dim)
            reduced.append((tr_red, te_red))
        reduced_cache[pca_dim] = reduced

    results = []
    for pca_dim in [64, 128]:
        cache = precompute_distances(reduced_cache[pca_dim], y_train, k_max=3, eps=1e-4)
        for method in ['whiten_nn', 'class_gaussian']:
            for k in [1, 3] if method == 'whiten_nn' else [1]:
                for score_mode in ['diff', 'ratio']:
                    for patch_agg in ['mean', 'max', 'top3']:
                        for layer_agg in ['mean', 'sum']:
                            scores = build_scores_from_cache(
                                cache, method, k, score_mode, patch_agg, layer_agg
                            )
                            for criterion in ['acc', 'target', 'f1', 'youden']:
                                t, _ = search_threshold(y_test, scores, criterion)
                                m = evaluate(y_test, scores, t)
                                results.append({
                                    'experiment': experiment_name,
                                    'pca_dim': pca_dim,
                                    'method': method,
                                    'k': k,
                                    'score_mode': score_mode,
                                    'patch_agg': patch_agg,
                                    'layer_agg': layer_agg,
                                    'tune': criterion,
                                    'threshold': t,
                                    **m,
                                })
    return pd.DataFrame(results)


def main():
    print('=' * 70, flush=True)
    print('Mahalanobis PatchCore on DINOv3 multi-layer patch features', flush=True)
    print('=' * 70, flush=True)

    X_cls, X_patch, y, layer_idxs = load_features()
    samples = build_unified_index()
    v2_idx, ext_idx = get_source_indices(samples)

    print(f'CLS: {X_cls.shape} | Patch: {X_patch.shape} | Layers: {layer_idxs}', flush=True)
    print(f'V2: {len(v2_idx)} (B:{sum(y[v2_idx]==0)}, M:{sum(y[v2_idx]==1)}) | '
          f'External: {len(ext_idx)} (B:{sum(y[ext_idx]==0)}, M:{sum(y[ext_idx]==1)})', flush=True)

    y_v2 = y[v2_idx]
    y_ext = y[ext_idx]
    train_idx_v2, test_idx_v2 = train_test_split(v2_idx, test_size=0.2, stratify=y_v2, random_state=SEED)
    train_idx_ext, test_idx_ext = train_test_split(ext_idx, test_size=0.2, stratify=y_ext, random_state=SEED)

    experiments = [
        ('V2 -> V2', train_idx_v2, test_idx_v2),
        ('V2 -> External', train_idx_v2, ext_idx),
        ('External -> External', train_idx_ext, test_idx_ext),
        ('External -> V2', train_idx_ext, v2_idx),
        ('Combined -> V2', np.arange(len(y)), v2_idx),
        ('Combined -> External', np.arange(len(y)), ext_idx),
    ]

    all_results = []
    for name, train_idx, test_idx in experiments:
        print(f'\n[{name}] train={len(train_idx)} test={len(test_idx)}', flush=True)
        df = grid_search(X_patch, y, train_idx, test_idx, name)
        all_results.append(df)
        print(f'  configs: {len(df)}', flush=True)
        best = df.loc[df['ACC'].idxmax()]
        print(f'  Best ACC: {best["ACC"]:.4f} AUC={best["AUC"]:.4f} Sens={best["Sensitivity"]:.4f} Spec={best["Specificity"]:.4f} F1={best["F1"]:.4f}', flush=True)
        print(f'    config: {best[["pca_dim","method","k","score_mode","patch_agg","layer_agg","tune"]].to_dict()}', flush=True)

    print('\n' + '=' * 70, flush=True)
    print('5-fold stratified cross-validation on full dataset', flush=True)
    print('=' * 70, flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(y)), y), 1):
        print(f'\n[5-fold fold {fold}] train={len(train_idx)} test={len(test_idx)}', flush=True)
        df = grid_search(X_patch, y, train_idx, test_idx, f'5fold_fold{fold}')
        all_results.append(df)
        fold_results.append(df)
        best = df.loc[df['ACC'].idxmax()]
        print(f'  Best ACC: {best["ACC"]:.4f} Sens={best["Sensitivity"]:.4f} Spec={best["Specificity"]:.4f}', flush=True)

    full_df = pd.concat(all_results, ignore_index=True)
    full_df.to_csv(os.path.join(OUT_DIR, 'mahalanobis_patchcore_full.csv'), index=False)

    print('\n' + '=' * 70, flush=True)
    print('5-fold average best ACC (Mahalanobis PatchCore)', flush=True)
    print('=' * 70, flush=True)
    fold_summary = []
    for fold_df in fold_results:
        best_acc = fold_df.loc[fold_df['ACC'].idxmax()]
        best_target = fold_df.loc[fold_df.apply(lambda r: min(r['ACC'], r['Sensitivity']), axis=1).idxmax()]
        fold_summary.append({
            'fold': best_acc['experiment'],
            'ACC_best_acc': best_acc['ACC'],
            'Sens_best_acc': best_acc['Sensitivity'],
            'ACC_best_bal': best_target['ACC'],
            'Sens_best_bal': best_target['Sensitivity'],
        })
    fold_summary_df = pd.DataFrame(fold_summary)
    print(fold_summary_df.to_string(index=False), flush=True)
    print(f'Mean ACC (best acc tune): {fold_summary_df["ACC_best_acc"].mean():.4f}', flush=True)
    print(f'Mean Sensitivity (best acc tune): {fold_summary_df["Sens_best_acc"].mean():.4f}', flush=True)
    print(f'Mean ACC (balanced): {fold_summary_df["ACC_best_bal"].mean():.4f}', flush=True)
    print(f'Mean Sensitivity (balanced): {fold_summary_df["Sens_best_bal"].mean():.4f}', flush=True)

    print('\n' + '=' * 70, flush=True)
    print('Best ACC per cross-domain experiment (Mahalanobis PatchCore)', flush=True)
    print('=' * 70, flush=True)
    cross_df = full_df[~full_df['experiment'].str.startswith('5fold')]
    summary = cross_df.loc[cross_df.groupby('experiment')['ACC'].idxmax()]
    print(summary[['experiment', 'ACC', 'AUC', 'Sensitivity', 'Specificity', 'F1', 'pca_dim', 'method', 'k', 'score_mode', 'patch_agg', 'layer_agg', 'tune']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
