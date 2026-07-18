"""Multi-seed comparison: fake-token (view-averaged) vs real per-view VascMamba.

Same protocol as hybrid.py (val-tuned threshold), 5 seeds x 5 folds.
"""
import sys, os, numpy as np
sys.path.insert(0, '/root/medic_data'); sys.path.insert(0, '/root/medic_data/ulm_visionnet')

import torch
from sklearn.model_selection import StratifiedKFold

from vascmamba.hybrid import VascMambaHybrid, train_hybrid

def run(X_bm, X_um, X_d, y, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []
    for ti, vi in skf.split(np.arange(len(y)), y.numpy()):
        m = VascMambaHybrid(d_model=32, d_state=4, n_layers=1).to('cuda')
        r = train_hybrid(m, X_bm, X_um, X_d, y, ti, vi, epochs=100, lr=5e-4)
        accs.append(r['acc']); f1s.append(r['f1']); aucs.append(r['auc'])
    return accs, f1s, aucs


if __name__ == '__main__':
    dp = np.load('/root/medic_data/biomedclip_perview_features.npz')
    da = np.load('/root/medic_data/biomedclip_features.npz')
    y = torch.from_numpy(dp['y']).long()

    real_bm = torch.from_numpy(dp['X_bmode']).float()
    real_um = torch.from_numpy(dp['X_ulm']).float()
    real_d = torch.from_numpy(dp['density']).float()

    X_bc = torch.from_numpy(da['X']).float()
    fake_bm = X_bc[:, :512].unsqueeze(1).expand(-1, 4, -1).clone()
    fake_um = X_bc[:, 512:].unsqueeze(1).expand(-1, 4, -1).clone()
    dv = np.load('/root/medic_data/vascular_features.npz')
    fake_d = torch.from_numpy(dv['X_vasc'][:, 0]).float().unsqueeze(1).expand(-1, 4).clone()

    seeds = [42, 1, 2, 3, 4]
    out = {}
    for name, (bm, um, dd) in [('fake', (fake_bm, fake_um, fake_d)),
                               ('real', (real_bm, real_um, real_d))]:
        all_acc, all_f1, all_auc = [], [], []
        for s in seeds:
            a, f, u = run(bm, um, dd, y, s)
            all_acc.append(np.mean(a)); all_f1.append(np.mean(f)); all_auc.append(np.mean(u))
            print(f'{name} seed={s}: ACC={np.mean(a):.4f} F1={np.mean(f):.4f} AUC={np.mean(u):.4f}', flush=True)
        out[name] = (all_acc, all_f1, all_auc)

    print('\n' + '=' * 60)
    for k, (a, f, u) in out.items():
        print(f'{k:5s} ACC={np.mean(a):.4f}±{np.std(a):.4f} F1={np.mean(f):.4f} AUC={np.mean(u):.4f}')
