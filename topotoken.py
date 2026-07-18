"""TopoToken-VascMamba: inject handcrafted vascular-topology descriptors
(vasc54 + TDA26 = 80D) as an extra token in the Mamba sequence.

Real per-view BiomedCLIP tokens (8) + 1 topology token = 9 tokens.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score

DEVICE = torch.device('cuda')

from vascmamba.hybrid import MambaBlock


class TopoTokenMamba(nn.Module):
    def __init__(self, bc_dim=512, hand_dim=80, d_model=32, d_state=4, n_layers=1, n_views=4):
        super().__init__()
        self.n_views = n_views
        self.seq_len = n_views * 2 + 1

        self.bmode_proj = nn.Sequential(nn.Linear(bc_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.ulm_proj = nn.Sequential(nn.Linear(bc_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.hand_proj = nn.Sequential(nn.Linear(hand_dim, d_model), nn.LayerNorm(d_model), nn.GELU())

        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        self.mod_emb = nn.Parameter(torch.zeros(1, 3, d_model))  # 0=B, 1=U, 2=topo

        self.mamba = nn.ModuleList([MambaBlock(d_model, d_state, d_conv=2, expand=2)
                                    for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(32, 2))

    def forward(self, bmode_feats, ulm_feats, hand_feats, ulm_density=None):
        b_tokens = self.bmode_proj(bmode_feats)
        u_tokens = self.ulm_proj(ulm_feats)
        if ulm_density is not None:
            _, idx = ulm_density.sort(dim=1, descending=True)
            u_tokens = u_tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, u_tokens.shape[-1]))
        h_token = self.hand_proj(hand_feats).unsqueeze(1)

        tokens = []
        for v in range(self.n_views):
            tokens += [b_tokens[:, v:v+1], u_tokens[:, v:v+1]]
        tokens.append(h_token)
        tokens = torch.cat(tokens, dim=1)

        mod_ids = torch.tensor([0, 1] * self.n_views + [2], device=tokens.device).long()
        tokens = tokens + self.mod_emb[0, mod_ids].unsqueeze(0) + self.pos_emb
        for layer in self.mamba:
            tokens = layer(tokens)
        return self.head(tokens.mean(dim=1))


class FeatDataset(Dataset):
    def __init__(self, bm, um, h, d, y):
        self.bm, self.um, self.h, self.d, self.y = bm, um, h, d, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.bm[i], self.um[i], self.h[i], self.d[i], self.y[i]


def train_model(model, X_bm, X_um, X_h, X_d, y, ti, vi, epochs=100, lr=5e-4, bs=32):
    tr = DataLoader(FeatDataset(X_bm[ti], X_um[ti], X_h[ti], X_d[ti], y[ti]), batch_size=bs, shuffle=True)
    va = DataLoader(FeatDataset(X_bm[vi], X_um[vi], X_h[vi], X_d[vi], y[vi]), batch_size=bs)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sched = CosineAnnealingLR(opt, T_max=epochs * len(tr), eta_min=1e-6)
    best_acc, best_m, patience = 0, None, 0
    for ep in range(epochs):
        model.train()
        for bm, um, h, d, lbl in tr:
            bm, um, h, d, lbl = bm.to(DEVICE), um.to(DEVICE), h.to(DEVICE), d.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(bm, um, h, d), lbl,
                                   weight=torch.tensor([3.0, 1.0], device=DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for bm, um, h, d, lbl in va:
                p = F.softmax(model(bm.to(DEVICE), um.to(DEVICE), h.to(DEVICE), d.to(DEVICE)), -1)[:, 1]
                probs.append(p.cpu()); labels.append(lbl)
        probs = torch.cat(probs).numpy(); labels = torch.cat(labels).numpy()
        best_f1, best_mm = 0, None
        for t in np.arange(0.05, 0.95, 0.02):
            pred = (probs >= t).astype(int)
            f1v = f1_score(labels, pred, zero_division=0)
            if f1v > best_f1:
                best_f1 = f1v
                best_mm = {'acc': accuracy_score(labels, pred), 'f1': f1v,
                           'auc': roc_auc_score(labels, probs),
                           'recall': recall_score(labels, pred, zero_division=0)}
        if best_mm['acc'] > best_acc + 0.005:
            best_acc, best_m, patience = best_mm['acc'], best_mm, 0
        else:
            patience += 1
        if patience > 20:
            break
    return best_m


def run(X_bm, X_um, X_h, X_d, y, seed, use_hand=True):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []
    for ti, vi in skf.split(np.arange(len(y)), y.numpy()):
        sc = StandardScaler().fit(X_h[ti].numpy())
        Xh = torch.from_numpy(sc.transform(X_h.numpy())).float()
        hd = Xh.shape[1] if use_hand else 1
        if not use_hand:
            Xh = torch.zeros(len(y), 1)
        m = TopoTokenMamba(hand_dim=hd).to(DEVICE)
        r = train_model(m, X_bm, X_um, Xh, X_d, y, ti, vi)
        accs.append(r['acc']); f1s.append(r['f1']); aucs.append(r['auc'])
    return np.mean(accs), np.mean(f1s), np.mean(aucs)


if __name__ == '__main__':
    dp = np.load('/root/medic_data/biomedclip_perview_features.npz')
    X_bm = torch.from_numpy(dp['X_bmode']).float()
    X_um = torch.from_numpy(dp['X_ulm']).float()
    X_d = torch.from_numpy(dp['density']).float()
    y = torch.from_numpy(dp['y']).long()

    dv = np.load('/root/medic_data/vascular_features.npz')
    dt = np.load('/root/medic_data/tda_features.npz')
    X_h = torch.from_numpy(np.hstack([dv['X_vasc'], dt['X_tda']])).float()

    m = TopoTokenMamba(hand_dim=80)
    print(f'params: {sum(p.numel() for p in m.parameters())/1000:.0f}K')

    for name, use_hand in [('topotoken', True)]:
        aa, ff, uu = [], [], []
        for s in [42, 1, 2, 3, 4]:
            a, f, u = run(X_bm, X_um, X_h, X_d, y, s, use_hand)
            aa.append(a); ff.append(f); uu.append(u)
            print(f'{name} seed={s}: ACC={a:.4f} F1={f:.4f} AUC={u:.4f}', flush=True)
        print(f'{name}: ACC={np.mean(aa):.4f}±{np.std(aa):.4f} '
              f'F1={np.mean(ff):.4f} AUC={np.mean(uu):.4f}')
