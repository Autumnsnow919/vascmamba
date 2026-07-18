"""VascMamba-Hybrid with REAL per-view tokens.

Baseline hybrid.py expands view-averaged features into 4 identical fake tokens.
This version feeds the 8 genuinely distinct tokens (4 B-mode + 4 ULM views).
"""
import sys, os, numpy as np
sys.path.insert(0, '/root/medic_data'); sys.path.insert(0, '/root/medic_data/ulm_visionnet')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda')

from vascmamba.hybrid import VascMambaHybrid, train_hybrid

if __name__ == '__main__':
    print('=' * 70)
    print('VascMamba-Hybrid: REAL per-view tokens (8 distinct tokens)')
    print('=' * 70)

    d = np.load('/root/medic_data/biomedclip_perview_features.npz')
    X_bmode_full = torch.from_numpy(d['X_bmode']).float()   # (241, 4, 512)
    X_ulm_full = torch.from_numpy(d['X_ulm']).float()       # (241, 4, 512)
    X_density_full = torch.from_numpy(d['density']).float() # (241, 4)
    y_all = torch.from_numpy(d['y']).long()

    model = VascMambaHybrid(d_model=32, d_state=4, n_layers=1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Hybrid model: {n_params/1000:.0f}K params')

    N = len(y_all)
    y_np = y_all.numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    for fi, (ti, vi) in enumerate(skf.split(np.arange(N), y_np)):
        print(f'\n--- Fold {fi+1}/5 | Train={len(ti)} Val={len(vi)} ---')
        m = VascMambaHybrid(d_model=32, d_state=4, n_layers=1).to(DEVICE)
        metrics = train_hybrid(m, X_bmode_full, X_ulm_full, X_density_full, y_all,
                               ti, vi, epochs=100, lr=5e-4)
        fold_results.append(metrics)
        print(f'  fold {fi+1}: acc={metrics["acc"]:.4f} f1={metrics["f1"]:.4f} auc={metrics["auc"]:.4f}')

    accs = [r['acc'] for r in fold_results]
    aucs = [r['auc'] for r in fold_results]
    print('\n' + '=' * 70)
    print('REAL per-view VascMamba RESULTS')
    print('=' * 70)
    print(f'  Acc:    {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'  AUC:    {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    print(f'  Recall: {np.mean([r["recall"] for r in fold_results]):.4f}')
    print(f'  F1:     {np.mean([r["f1"] for r in fold_results]):.4f}')
    print(f'  Per-fold: {[f"{a:.3f}" for a in accs]}')
    print(f'\n  Baseline (fake tokens, view-averaged): 0.8800')
