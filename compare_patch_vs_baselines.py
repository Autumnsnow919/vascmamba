#!/usr/bin/env python3
"""Head-to-head comparison: BC-SVM vs VascMamba-Hybrid vs PatchVascMamba.

Same protocol for all: 5-fold StratifiedKFold(seed=42), per-fold F1 threshold
search, OOF probabilities pooled for subset analysis.

Extra: recall on the low-density malignant subset (necrosis concern) —
malignant sessions with mean ULM vessel density below the malignant median.
"""
import sys, os, numpy as np
sys.path.insert(0, '/root/medic_data'); sys.path.insert(0, '/root/medic_data/ulm_visionnet')
sys.path.insert(0, '/root/medic_data/vascmamba')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score

SEED = 42
DEVICE = torch.device('cuda')


class FeatDataset(Dataset):
    def __init__(self, bm, um, d, y):
        self.bm, self.um, self.d, self.y = bm, um, d, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.bm[i], self.um[i], self.d[i], self.y[i]


def train_torch_oof(model, X_b, X_u, X_d, y, ti, vi, epochs=100, lr=5e-4, bs=32):
    """Train, return (best metrics, OOF probs on vi at best epoch)."""
    train_ds = FeatDataset(X_b[ti], X_u[ti], X_d[ti], y[ti])
    val_ds = FeatDataset(X_b[vi], X_u[vi], X_d[vi], y[vi])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sched = CosineAnnealingLR(opt, T_max=epochs * len(train_loader), eta_min=1e-6)

    best_acc, best_m, best_probs, patience = 0, None, None, 0
    for ep in range(epochs):
        model.train()
        for bm, um, d, lbl in train_loader:
            bm, um, d, lbl = bm.to(DEVICE), um.to(DEVICE), d.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            w = torch.tensor([3.0, 1.0], device=DEVICE)
            loss = F.cross_entropy(model(bm, um, d), lbl, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()

        model.eval()
        probs = []
        with torch.no_grad():
            for bm, um, d, lbl in val_loader:
                bm, um, d = bm.to(DEVICE), um.to(DEVICE), d.to(DEVICE)
                probs.append(F.softmax(model(bm, um, d), -1)[:, 1].cpu())
        probs = torch.cat(probs).numpy()
        labels = y[vi].numpy()

        bf1, bm_best = 0, None
        for t in np.arange(0.05, 0.95, 0.02):
            pred = (probs >= t).astype(int)
            f1v = f1_score(labels, pred, zero_division=0)
            if f1v > bf1:
                bf1 = f1v
                bm_best = {'acc': accuracy_score(labels, pred), 'auc': roc_auc_score(labels, probs),
                           'recall': recall_score(labels, pred, zero_division=0), 'f1': f1v, 't': t}
        if bm_best['acc'] > best_acc + 0.005:
            best_acc, best_m, best_probs, patience = bm_best['acc'], bm_best, probs, 0
        else:
            patience += 1
        if patience > 20:
            break
    if best_m is None:  # never improved: use last epoch
        best_m, best_probs = bm_best, probs
    return best_m, best_probs


def run_torch_model(name, make_model, X_b, X_u, X_d, y, skf):
    oof = np.zeros(len(y))
    folds = []
    for fi, (ti, vi) in enumerate(skf.split(np.arange(len(y)), y.numpy())):
        torch.manual_seed(SEED + fi); np.random.seed(SEED + fi)
        m = make_model().to(DEVICE)
        metrics, probs = train_torch_oof(m, X_b, X_u, X_d, y, ti, vi)
        oof[vi] = probs
        folds.append(metrics)
        print(f'  [{name}] fold {fi+1}: acc={metrics["acc"]:.4f} recall={metrics["recall"]:.4f}')
    return oof, folds


def run_svm(X, y, skf):
    oof = np.zeros(len(y))
    for fi, (ti, vi) in enumerate(skf.split(np.arange(len(y)), y.numpy())):
        sc = StandardScaler().fit(X[ti])
        clf = SVC(kernel='rbf', C=1.0, gamma='auto', class_weight={0: 3, 1: 1},
                  probability=True, random_state=SEED).fit(sc.transform(X[ti]), y.numpy()[ti])
        oof[vi] = clf.predict_proba(sc.transform(X[vi]))[:, 1]
    return oof


def best_f1_threshold(labels, probs):
    bt, bf = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.02):
        f1v = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1v > bf: bf, bt = f1v, t
    return bt


def summarize(name, oof, y_np, skf):
    """Per-fold threshold tuning (same as training recipes), pooled AUC."""
    accs, recs, f1s = [], [], []
    for ti, vi in skf.split(np.arange(len(y_np)), y_np):
        t = best_f1_threshold(y_np[vi], oof[vi])
        pred = (oof[vi] >= t).astype(int)
        accs.append(accuracy_score(y_np[vi], pred))
        recs.append(recall_score(y_np[vi], pred, zero_division=0))
        f1s.append(f1_score(y_np[vi], pred, zero_division=0))
    return {'name': name, 'acc': (np.mean(accs), np.std(accs)), 'recall': np.mean(recs),
            'f1': np.mean(f1s), 'auc': roc_auc_score(y_np, oof),
            'folds': [f'{a:.3f}' for a in accs]}


if __name__ == '__main__':
    y_np = np.load('/root/medic_data/biomedclip_features.npz')['y'].astype(int)
    N = len(y_np)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ── 1. BC-SVM on session-level 1024D ──
    print('[1/4] BC-SVM (1024D session features)...')
    bc = np.load('/root/medic_data/biomedclip_features.npz')
    X_bc = torch.from_numpy(bc['X']).float()
    oof_svm = run_svm(X_bc.numpy(), torch.from_numpy(y_np).long(), skf)

    # ── 2. VascMamba-Hybrid (session CLS, expanded) ──
    print('[2/4] VascMamba-Hybrid...')
    from hybrid import VascMambaHybrid
    vasc = np.load('/root/medic_data/vascular_features.npz')
    X_b_h = X_bc[:, :512].unsqueeze(1).expand(-1, 4, -1).contiguous()
    X_u_h = X_bc[:, 512:].unsqueeze(1).expand(-1, 4, -1).contiguous()
    X_d_h = torch.from_numpy(vasc['X_vasc'][:, 0]).float().unsqueeze(1).expand(-1, 4).contiguous()
    y_t = torch.from_numpy(y_np).long()
    oof_hyb, _ = run_torch_model('Hybrid', lambda: VascMambaHybrid(d_model=32, d_state=4, n_layers=1),
                                 X_b_h, X_u_h, X_d_h, y_t, skf)

    # ── 3. PatchVascMamba (full patch tokens) ──
    print('[3/4] PatchVascMamba...')
    from patch_vascmamba import PatchVascMamba
    pt = np.load('/root/medic_data/biomedclip_patch_tokens.npz')
    X_b_p = torch.from_numpy(pt['X_bmode']).float()
    X_u_p = torch.from_numpy(pt['X_ulm']).float()
    X_d_p = torch.from_numpy(pt['density']).float()
    oof_pat, _ = run_torch_model('Patch', lambda: PatchVascMamba(d_model=64, d_state=4, n_layers=1),
                                 X_b_p, X_u_p, X_d_p, y_t, skf)

    # ── Summary ──
    rows = [summarize('BC-SVM (1024D)', oof_svm, y_np, skf),
            summarize('VascMamba-Hybrid', oof_hyb, y_np, skf),
            summarize('PatchVascMamba', oof_pat, y_np, skf)]

    print('\n' + '=' * 78)
    print(f'{"Method":<22s} {"Acc":>15s} {"AUC":>7s} {"Recall":>7s} {"F1":>7s}  Per-fold acc')
    print('-' * 78)
    for r in rows:
        print(f'{r["name"]:<22s} {r["acc"][0]:.4f} ± {r["acc"][1]:.4f}  {r["auc"]:.4f}  '
              f'{r["recall"]:.4f}  {r["f1"]:.4f}  {r["folds"]}')

    # ── Low-density malignant subset (necrosis concern) ──
    dens_mean = X_d_p.mean(dim=1).numpy()
    mal = y_np == 1
    med = np.median(dens_mean[mal])
    subset = mal & (dens_mean < med)
    print('\n' + '=' * 78)
    print(f'Low-density malignant subset (density_mean < median {med:.4f}): n={subset.sum()}')
    print('-' * 78)
    for name, oof in [('BC-SVM', oof_svm), ('VascMamba-Hybrid', oof_hyb), ('PatchVascMamba', oof_pat)]:
        # per-fold thresholds, pooled subset predictions
        pred_all = np.zeros(N, dtype=int)
        for ti, vi in skf.split(np.arange(N), y_np):
            t = best_f1_threshold(y_np[vi], oof[vi])
            pred_all[vi] = (oof[vi] >= t).astype(int)
        rec = recall_score(y_np[subset], pred_all[subset], zero_division=0)
        print(f'  {name:<22s} subset recall = {rec:.4f}  ({int(pred_all[subset].sum())}/{subset.sum()} detected)')

    np.savez('/root/medic_data/vascmamba/compare_oof_probs.npz',
             oof_svm=oof_svm, oof_hyb=oof_hyb, oof_pat=oof_pat, y=y_np, density=dens_mean)
    print('\nOOF probs saved to vascmamba/compare_oof_probs.npz')
