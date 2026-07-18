#!/usr/bin/env python3
"""VascMamba-Hybrid with DINOv3 intermediate patch features (no CLS).

Replaces BiomedCLIP CLS features with DINOv3 ViT-B/16 intermediate layer patch
tokens, mean-pooled per view, optionally averaged over layers.
The vessel-density ordering is preserved.
"""
import sys, os, numpy as np
sys.path.insert(0, '/root/medic_data')

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

OUT_DIR = '/root/medic_data/vascmamba'

# ═══════════════════════════════════════════════════════
# Lightweight Selective SSM (same as VascMamba-Hybrid)
# ═══════════════════════════════════════════════════════

class SelectiveSSM(nn.Module):
    def __init__(self, d_model=128, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(d_inner, 1)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))
        x_conv = x_conv[:, :, :L]
        x_conv = self.act(x_conv).transpose(1, 2)
        x_proj = self.x_proj(x_conv)
        B_ssm = x_proj[..., :self.d_state]
        C_ssm = x_proj[..., self.d_state:2*self.d_state]
        dt = F.softplus(self.dt_proj(x_conv))
        B_seq, L_seq, D_seq = x_conv.shape
        A = -torch.exp(self.A_log).view(1, 1, 1, -1)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)
        dt_exp = dt.unsqueeze(-1)
        B_exp = B_ssm.unsqueeze(2)
        x_exp = x_conv.unsqueeze(-1)
        B_bar_seq = dt_exp * B_exp * x_exp
        h = torch.zeros(B_seq, D_seq, self.d_state, device=x.device)
        outputs = []
        for t in range(L_seq):
            a_t = A_bar[:, t, 0]
            h = a_t.unsqueeze(1) * h + B_bar_seq[:, t]
            y = (h * C_ssm[:, t].unsqueeze(1)).sum(-1)
            outputs.append(y)
        y_scan = torch.stack(outputs, dim=1)
        y_out = y_scan * self.act(z)
        return self.out_proj(y_out)


class MambaBlock(nn.Module):
    def __init__(self, d_model=128, d_state=8, d_conv=2, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════
# VascMamba-DINOv3: DINOv3 patch features + Mamba head
# ═══════════════════════════════════════════════════════

class VascMambaDino3(nn.Module):
    def __init__(self, input_dim=768, d_model=64, d_state=4, n_layers=2, n_views=4):
        super().__init__()
        self.n_views = n_views
        self.seq_len = n_views * 2  # 4 B-mode + 4 ULM = 8 tokens

        self.bmode_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.ulm_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        self.mod_emb = nn.Parameter(torch.zeros(1, 2, d_model))

        self.mamba = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv=2, expand=2) for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, bmode_feats, ulm_feats, ulm_density=None):
        """
        bmode_feats: (B, 4, input_dim) — 4 B-mode views
        ulm_feats:   (B, 4, input_dim) — 4 ULM views
        ulm_density: (B, 4) — vessel density per ULM view (for ordering)
        Returns: logits (B, 2)
        """
        B = bmode_feats.shape[0]
        b_tokens = self.bmode_proj(bmode_feats)
        u_tokens = self.ulm_proj(ulm_feats)

        if ulm_density is not None:
            _, sort_idx = ulm_density.sort(dim=1, descending=True)
            u_tokens = u_tokens.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, u_tokens.shape[-1]))
            # Note: bmode is not reordered; only ULM ordering is vessel-guided.

        tokens = []
        for v in range(self.n_views):
            tokens.append(b_tokens[:, v:v+1])
            tokens.append(u_tokens[:, v:v+1])
        tokens = torch.cat(tokens, dim=1)

        mod_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=tokens.device).long()
        tokens = tokens + self.mod_emb[0, mod_ids].unsqueeze(0)
        tokens = tokens + self.pos_emb

        for layer in self.mamba:
            tokens = layer(tokens)

        pooled = tokens.mean(dim=1)
        return self.head(pooled)


# ═══════════════════════════════════════════════════════
# Feature loading and aggregation
# ═══════════════════════════════════════════════════════

def load_dinov3_patch_features(layer_pool='all'):
    """
    layer_pool: 'all' | 'late' | '11' | '9'
      - all:  mean over layers 3,6,9,11
      - late: mean over layers 9,11
      - 11:   use only layer 11
      - 9:    use only layer 9
    Returns: X_bmode (N, 4, 768), X_ulm (N, 4, 768), density (N, 4), y (N,)
    """
    data = np.load('/root/medic_data/dinov3_perview_patch_features.npz')
    X_b = data['X_bmode']   # (N, 4, 4, 196, 768)
    X_u = data['X_ulm']     # (N, 4, 4, 196, 768)
    density = data['density']
    y = data['y']
    layer_idxs = data['layer_idxs']  # [3, 6, 9, 11]

    # Mean-pool over spatial patches -> (N, 4, 4, 768)
    X_b = X_b.mean(axis=3)
    X_u = X_u.mean(axis=3)

    # Select / pool layers
    if layer_pool == 'all':
        X_b = X_b.mean(axis=2)
        X_u = X_u.mean(axis=2)
    elif layer_pool == 'late':
        # indices of layers 9 and 11 in layer_idxs: [3,6,9,11] -> 2,3
        X_b = X_b[:, :, [2, 3], :].mean(axis=2)
        X_u = X_u[:, :, [2, 3], :].mean(axis=2)
    elif layer_pool == '11':
        X_b = X_b[:, :, 3, :]
        X_u = X_u[:, :, 3, :]
    elif layer_pool == '9':
        X_b = X_b[:, :, 2, :]
        X_u = X_u[:, :, 2, :]
    else:
        raise ValueError(layer_pool)

    return (torch.from_numpy(X_b).float(), torch.from_numpy(X_u).float(),
            torch.from_numpy(density).float(), torch.from_numpy(y).long())


# ═══════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════

def train_hybrid(model, X_bmode, X_ulm, X_density, y, idx_train, idx_val,
                 epochs=100, lr=5e-4, batch_size=32):
    class FeatDataset(Dataset):
        def __init__(self, bm, um, d, y):
            self.bm, self.um, self.d, self.y = bm, um, d, y
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.bm[i], self.um[i], self.d[i], self.y[i]

    train_ds = FeatDataset(X_bmode[idx_train], X_ulm[idx_train], X_density[idx_train], y[idx_train])
    val_ds = FeatDataset(X_bmode[idx_val], X_ulm[idx_val], X_density[idx_val], y[idx_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sched = CosineAnnealingLR(opt, T_max=epochs * len(train_loader), eta_min=1e-6)

    best_acc, best_metrics, patience = 0, None, 0
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
            opt.step()
            sched.step()

        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for bm, um, d, lbl in val_loader:
                bm, um, d = bm.to(DEVICE), um.to(DEVICE), d.to(DEVICE)
                logits = model(bm, um, d)
                probs.append(F.softmax(logits, -1)[:, 1].cpu())
                labels.append(lbl)
        probs = torch.cat(probs).numpy(); labels = torch.cat(labels).numpy()

        best_t, best_f1 = 0.5, 0
        best_m = None
        for t in np.arange(0.05, 0.95, 0.02):
            pred = (probs >= t).astype(int)
            f1v = f1_score(labels, pred, zero_division=0)
            if f1v > best_f1:
                best_f1 = f1v; best_t = t
                best_m = {'acc': accuracy_score(labels, pred), 'auc': roc_auc_score(labels, probs),
                           'recall': recall_score(labels, pred, zero_division=0), 'f1': f1v}

        if best_m['acc'] > best_acc + 0.005:
            best_acc = best_m['acc']; best_metrics = best_m; patience = 0
        else:
            patience += 1
        if patience > 20:
            break

    return best_metrics


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print('VascMamba-DINOv3: DINOv3 intermediate patch features + Mamba head')
    print('=' * 70)

    for layer_pool in ['all', 'late', '11', '9']:
        print(f"\n--- Layer pooling: {layer_pool} ---")
        X_bmode, X_ulm, X_density, y_all = load_dinov3_patch_features(layer_pool)
        N = len(y_all)
        print(f'  Features: B-mode {X_bmode.shape}, ULM {X_ulm.shape}, density {X_density.shape}')

        model = VascMambaDino3(input_dim=768, d_model=32, d_state=4, n_layers=1)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Model: {n_params/1000:.0f}K params')

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_results = []

        for fi, (ti, vi) in enumerate(skf.split(np.arange(N), y_all.numpy())):
            m = VascMambaDino3(input_dim=768, d_model=32, d_state=4, n_layers=1).to(DEVICE)
            metrics = train_hybrid(m, X_bmode, X_ulm, X_density, y_all, ti, vi,
                                   epochs=100, lr=5e-4, batch_size=32)
            fold_results.append(metrics)
            print(f"  Fold {fi+1}: acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} f1={metrics['f1']:.4f} recall={metrics['recall']:.4f}")

        print(f"  Mean: acc={np.mean([r['acc'] for r in fold_results]):.4f} ± {np.std([r['acc'] for r in fold_results]):.4f} "
              f"auc={np.mean([r['auc'] for r in fold_results]):.4f} f1={np.mean([r['f1'] for r in fold_results]):.4f}")
        print(f"  Per-fold acc: {[r['acc'] for r in fold_results]}")

    print('\n' + '=' * 70)
    print('Baseline: VascMamba-Hybrid (BiomedCLIP CLS) = 0.8800 ACC')
    print('=' * 70)
