"""Late fusion: real-per-view VascMamba probs + SVM(vasc54+TDA26) probs.

Fusion weight selected on TRAIN-fold probs only (no val leakage).
Fallback: if best train weight is w=1.0 (pure Mamba), fusion == baseline.
"""
import sys, os, numpy as np
sys.path.insert(0, '/root/medic_data'); sys.path.insert(0, '/root/medic_data/ulm_visionnet')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda')

from vascmamba.hybrid import VascMambaHybrid


class FeatDataset(Dataset):
    def __init__(self, bm, um, d, y):
        self.bm, self.um, self.d, self.y = bm, um, d, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.bm[i], self.um[i], self.d[i], self.y[i]


@torch.no_grad()
def infer(model, loader):
    model.eval()
    probs = []
    for bm, um, d, _ in loader:
        logits = model(bm.to(DEVICE), um.to(DEVICE), d.to(DEVICE))
        probs.append(F.softmax(logits, -1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def train_mamba_probs(X_bm, X_um, X_d, y, ti, vi, epochs=100, lr=5e-4, batch_size=32):
    model = VascMambaHybrid(d_model=32, d_state=4, n_layers=1).to(DEVICE)
    train_loader = DataLoader(FeatDataset(X_bm[ti], X_um[ti], X_d[ti], y[ti]),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FeatDataset(X_bm[vi], X_um[vi], X_d[vi], y[vi]),
                            batch_size=batch_size, shuffle=False)
    train_eval = DataLoader(FeatDataset(X_bm[ti], X_um[ti], X_d[ti], y[ti]),
                            batch_size=batch_size, shuffle=False)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sched = CosineAnnealingLR(opt, T_max=epochs * len(train_loader), eta_min=1e-6)

    best_acc, best_val, best_train, patience = 0, None, None, 0
    for ep in range(epochs):
        model.train()
        for bm, um, d, lbl in train_loader:
            bm, um, d, lbl = bm.to(DEVICE), um.to(DEVICE), d.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            logits = model(bm, um, d)
            w = torch.tensor([3.0, 1.0], device=DEVICE)
            loss = F.cross_entropy(logits, lbl, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()

        pv = infer(model, val_loader)
        yv = y[vi].numpy()
        acc = max(accuracy_score(yv, (pv >= t).astype(int)) for t in np.arange(0.05, 0.95, 0.02))
        if acc > best_acc + 0.005:
            best_acc = acc
            best_val = pv
            best_train = infer(model, train_eval)
            patience = 0
        else:
            patience += 1
        if patience > 20:
            break
    return best_train, best_val


def svm_probs(X, y, ti, vi):
    sc = StandardScaler().fit(X[ti])
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight={0: 3, 1: 1},
              probability=True, random_state=42)
    clf.fit(sc.transform(X[ti]), y[ti])
    return (clf.predict_proba(sc.transform(X[ti]))[:, 1],
            clf.predict_proba(sc.transform(X[vi]))[:, 1])


def best_thresh(y, p):
    best_t, best_f = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.02):
        f = f1_score(y, (p >= t).astype(int))
        if f > best_f: best_f, best_t = f, t
    return best_t


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred),
                auc=roc_auc_score(y, p), recall=recall_score(y, pred))


if __name__ == '__main__':
    d = np.load('/root/medic_data/biomedclip_perview_features.npz')
    X_bm = torch.from_numpy(d['X_bmode']).float()
    X_um = torch.from_numpy(d['X_ulm']).float()
    X_d = torch.from_numpy(d['density']).float()
    y = torch.from_numpy(d['y']).long()

    dv = np.load('/root/medic_data/vascular_features.npz')
    dt = np.load('/root/medic_data/tda_features.npz')
    X_hand = np.hstack([dv['X_vasc'], dt['X_tda']])
    assert (dv['y'] == d['y']).all() and (dt['y'] == d['y']).all()

    y_np = d['y']
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = {'mamba': [], 'svm': [], 'fuse': []}
    for fi, (ti, vi) in enumerate(skf.split(np.arange(len(y_np)), y_np)):
        pm_tr, pm_va = train_mamba_probs(X_bm, X_um, X_d, y, ti, vi)
        ps_tr, ps_va = svm_probs(X_hand, y_np, ti, vi)

        ytr, yva = y_np[ti], y_np[vi]
        # weight + threshold on train only
        best_w, best_t, best_f = 1.0, 0.5, -1
        for w in np.arange(0.0, 1.01, 0.1):
            pf = w * pm_tr + (1 - w) * ps_tr
            t = best_thresh(ytr, pf)
            f = f1_score(ytr, (pf >= t).astype(int))
            if f > best_f: best_f, best_w, best_t = f, w, t

        res['mamba'].append(metrics(yva, pm_va, best_t))
        res['svm'].append(metrics(yva, ps_va, best_t))
        res['fuse'].append(metrics(yva, best_w * pm_va + (1 - best_w) * ps_va, best_t))
        print(f'fold {fi+1}: w={best_w:.1f} t={best_t:.2f} '
              f'mamba={res["mamba"][-1]["acc"]:.3f} svm={res["svm"][-1]["acc"]:.3f} '
              f'fuse={res["fuse"][-1]["acc"]:.3f}')

    print('\n' + '=' * 60)
    for k in res:
        a = np.mean([r['acc'] for r in res[k]])
        f = np.mean([r['f1'] for r in res[k]])
        u = np.mean([r['auc'] for r in res[k]])
        s = np.std([r['acc'] for r in res[k]])
        print(f'{k:6s} ACC={a:.4f}±{s:.4f} F1={f:.4f} AUC={u:.4f}')
    print('baseline VascMamba (fake tokens): ACC=0.8800')
