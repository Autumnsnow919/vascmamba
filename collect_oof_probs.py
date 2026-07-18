"""Collect OOF probs for fake/real/topotoken x 3 seeds, then evaluate ensembles."""
import sys, os, numpy as np
sys.path.insert(0, '/root/medic_data'); sys.path.insert(0, '/root/medic_data/ulm_visionnet')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from vascmamba.hybrid import VascMambaHybrid
from vascmamba.topotoken import TopoTokenMamba

DEVICE = torch.device('cuda')


class DS(Dataset):
    def __init__(self, *tensors):
        self.t = tensors
    def __len__(self): return len(self.t[0])
    def __getitem__(self, i): return tuple(x[i] for x in self.t)


def train_probs(model, tensors, y, ti, vi, epochs=100, lr=5e-4, bs=32):
    tr = DataLoader(DS(*[t[ti] for t in tensors], y[ti]), batch_size=bs, shuffle=True)
    va = DataLoader(DS(*[t[vi] for t in tensors], y[vi]), batch_size=bs)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sched = CosineAnnealingLR(opt, T_max=epochs * len(tr), eta_min=1e-6)
    best_acc, best_p, patience = 0, None, 0
    for ep in range(epochs):
        model.train()
        for *feats, lbl in tr:
            feats = [f.to(DEVICE) for f in feats]; lbl = lbl.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(*feats), lbl,
                                   weight=torch.tensor([3.0, 1.0], device=DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
        model.eval()
        probs = []
        with torch.no_grad():
            for *feats, lbl in va:
                probs.append(F.softmax(model(*[f.to(DEVICE) for f in feats]), -1)[:, 1].cpu())
        probs = torch.cat(probs).numpy()
        yv = y[vi].numpy()
        from sklearn.metrics import accuracy_score
        acc = max(accuracy_score(yv, (probs >= t).astype(int)) for t in np.arange(0.05, 0.95, 0.02))
        if acc > best_acc + 0.005:
            best_acc, best_p, patience = acc, probs, 0
        else:
            patience += 1
        if patience > 20:
            break
    return best_p


if __name__ == '__main__':
    dp = np.load('/root/medic_data/biomedclip_perview_features.npz')
    da = np.load('/root/medic_data/biomedclip_features.npz')
    y = torch.from_numpy(dp['y']).long()
    N = len(y)

    real_bm = torch.from_numpy(dp['X_bmode']).float()
    real_um = torch.from_numpy(dp['X_ulm']).float()
    real_d = torch.from_numpy(dp['density']).float()

    X_bc = torch.from_numpy(da['X']).float()
    fake_bm = X_bc[:, :512].unsqueeze(1).expand(-1, 4, -1).clone()
    fake_um = X_bc[:, 512:].unsqueeze(1).expand(-1, 4, -1).clone()
    dv = np.load('/root/medic_data/vascular_features.npz')
    dt = np.load('/root/medic_data/tda_features.npz')
    fake_d = torch.from_numpy(dv['X_vasc'][:, 0]).float().unsqueeze(1).expand(-1, 4).clone()
    X_h = torch.from_numpy(np.hstack([dv['X_vasc'], dt['X_tda']])).float()

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(N), y.numpy()))
    probs = {k: np.zeros(N) for k in ['fake', 'real', 'topo']}

    for fi, (ti, vi) in enumerate(folds):
        sc = StandardScaler().fit(X_h[ti].numpy())
        Xh = torch.from_numpy(sc.transform(X_h.numpy())).float()
        for seed in [42, 1, 2]:
            torch.manual_seed(seed); np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            m1 = VascMambaHybrid(d_model=32, d_state=4, n_layers=1).to(DEVICE)
            p = train_probs(m1, (fake_bm, fake_um, fake_d), y, ti, vi)
            probs['fake'][vi] += p / 3.0
            m2 = VascMambaHybrid(d_model=32, d_state=4, n_layers=1).to(DEVICE)
            p = train_probs(m2, (real_bm, real_um, real_d), y, ti, vi)
            probs['real'][vi] += p / 3.0
            m3 = TopoTokenMamba(hand_dim=80).to(DEVICE)
            p = train_probs(m3, (real_bm, real_um, Xh, real_d), y, ti, vi)
            probs['topo'][vi] += p / 3.0
        print(f'fold {fi+1} done', flush=True)

    np.savez('/root/medic_data/vascmamba/oof_probs.npz', y=y.numpy(), **probs)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    yn = y.numpy()
    print('\nOOF (seed-ensembled) performance:')
    combos = {**{k: probs[k] for k in probs},
              'fake+real': (probs['fake'] + probs['real']) / 2,
              'real+topo': (probs['real'] + probs['topo']) / 2,
              'all3': (probs['fake'] + probs['real'] + probs['topo']) / 3}
    for k, p in combos.items():
        best_f, best_t = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.02):
            f = f1_score(yn, (p >= t).astype(int))
            if f > best_f: best_f, best_t = f, t
        pred = (p >= best_t).astype(int)
        print(f'{k:10s} ACC={accuracy_score(yn, pred):.4f} F1={best_f:.4f} AUC={roc_auc_score(yn, p):.4f}')
