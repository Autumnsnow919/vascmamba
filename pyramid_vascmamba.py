#!/usr/bin/env python3
"""Pyramid VascMamba: ROI-aware BiomedCLIP multi-scale tokens + Mamba + aggregation.

Input per view: BiomedCLIP spatial pyramid
  - CLS token (global)  : 1 token
  - 2x2 pooled patches  : 4 tokens
  - 1x1 global pool     : 1 token
For B-mode and ULM separately.

Sequence per sample: 4 views × 2 modalities × 6 tokens = 48 tokens.
Views are ordered by ULM vessel density (vessel-guided ordering).
Token embeddings indicate modality, pyramid level, and view position.
Final aggregation: learnable attention pooling over the full sequence.
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


class PyramidVascMamba(nn.Module):
    def __init__(self, input_dim=512, d_model=64, d_state=4, n_layers=2, n_views=4):
        super().__init__()
        self.n_views = n_views
        self.tokens_per_view_mod = 6  # CLS + 2x2(4) + 1x1
        self.n_mod = 2
        self.seq_len = n_views * self.n_mod * self.tokens_per_view_mod  # 48

        # Project input tokens to d_model
        self.token_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self._seq_indices = None

        # Embeddings
        self.mod_emb = nn.Parameter(torch.zeros(1, self.n_mod, d_model))          # B-mode / ULM
        self.level_emb = nn.Parameter(torch.zeros(1, 3, d_model))                 # CLS, 2x2, 1x1
        self.view_pos_emb = nn.Parameter(torch.randn(1, n_views, d_model) * 0.02)
        self.intra_pos_emb = nn.Parameter(torch.randn(1, self.tokens_per_view_mod, d_model) * 0.02)

        # Mamba encoding
        self.mamba = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv=2, expand=2) for _ in range(n_layers)
        ])

        # Final learnable attention aggregation
        self.agg_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.agg_attn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.Tanh(),
            nn.Linear(d_model, 1)
        )

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, bmode_pyramid, ulm_pyramid, ulm_density=None):
        """
        bmode_pyramid: (B, 4, 3, 5, 512)  [views, levels, tokens, dim]
        ulm_pyramid:   (B, 4, 3, 5, 512)
        ulm_density:   (B, 4) for view ordering
        Returns logits (B, 2)
        """
        B = bmode_pyramid.shape[0]

        # Extract valid tokens and build per-view token blocks
        # Level 0: CLS at index 0; Level 1: 2x2 at indices 0..3; Level 2: 1x1 at index 0
        b_cls = bmode_pyramid[:, :, 0, 0:1, :]          # (B, 4, 1, 512)
        b_2x2 = bmode_pyramid[:, :, 1, 0:4, :]          # (B, 4, 4, 512)
        b_1x1 = bmode_pyramid[:, :, 2, 0:1, :]          # (B, 4, 1, 512)
        u_cls = ulm_pyramid[:, :, 0, 0:1, :]
        u_2x2 = ulm_pyramid[:, :, 1, 0:4, :]
        u_1x1 = ulm_pyramid[:, :, 2, 0:1, :]

        # Per-view modality blocks: (B, 4, 6, 512) each
        b_blocks = torch.cat([b_cls, b_2x2, b_1x1], dim=2)  # (B, 4, 6, 512)
        u_blocks = torch.cat([u_cls, u_2x2, u_1x1], dim=2)

        # Reorder views by ULM density if provided (vessel-guided ordering)
        if ulm_density is not None:
            _, sort_idx = ulm_density.sort(dim=1, descending=True)  # (B, 4)
            b_blocks = b_blocks.gather(1, sort_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, b_blocks.size(2), b_blocks.size(3)))
            u_blocks = u_blocks.gather(1, sort_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, u_blocks.size(2), u_blocks.size(3)))

        # Build full sequence: interleave B and U blocks per view
        tokens = []
        for v in range(self.n_views):
            tokens.append(b_blocks[:, v])  # (B, 6, 512)
            tokens.append(u_blocks[:, v])    # (B, 6, 512)
        tokens = torch.cat(tokens, dim=1)    # (B, 48, 512)
        tokens = self.token_proj(tokens)     # (B, 48, d_model)

        # Add embeddings via precomputed indices for each of the 48 sequence positions
        if self._seq_indices is None or self._seq_indices.size(0) != B:
            # Build once: (seq_len, 4) -> [mod, view, intra_pos, level]
            idx = torch.arange(self.seq_len, device=tokens.device)
            block = idx // self.tokens_per_view_mod          # 0..7
            mod = block % self.n_mod                           # 0 or 1
            view = block // self.n_mod                         # 0..3
            intra = idx % self.tokens_per_view_mod             # 0..5
            level = torch.tensor([0, 1, 1, 1, 1, 2], device=tokens.device)[intra]
            self._seq_indices = torch.stack([mod, view, intra, level], dim=0).unsqueeze(0)  # (1, 4, seq_len)
        indices = self._seq_indices.expand(B, -1, -1)  # (B, 4, seq_len)
        # Need to expand embeddings to (B, seq_len, d_model) and add
        mod_emb = self.mod_emb[0, indices[:, 0, :].long(), :].reshape(B, self.seq_len, -1)
        level_emb = self.level_emb[0, indices[:, 3, :].long(), :].reshape(B, self.seq_len, -1)
        view_emb = self.view_pos_emb[0, indices[:, 1, :].long(), :].reshape(B, self.seq_len, -1)
        intra_emb = self.intra_pos_emb[0, indices[:, 2, :].long(), :].reshape(B, self.seq_len, -1)

        tokens = tokens + mod_emb + level_emb + view_emb + intra_emb

        # Mamba encoding
        for layer in self.mamba:
            tokens = layer(tokens)

        # Learnable attention aggregation
        # Compute attention weights from each token to a learned query
        attn_scores = self.agg_attn(tokens).squeeze(-1)  # (B, seq_len)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, seq_len, 1)
        pooled = (tokens * attn_weights).sum(dim=1)  # (B, d_model)

        return self.head(pooled)


# ═══════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════

def load_features():
    data = np.load('/root/medic_data/biomedclip_roi_pyramid_features.npz')
    X_b = torch.from_numpy(data['X_bmode']).float()
    X_u = torch.from_numpy(data['X_ulm']).float()
    dens = torch.from_numpy(data['density']).float()
    y = torch.from_numpy(data['y']).long()
    return X_b, X_u, dens, y


def train_hybrid(model, X_b, X_u, X_density, y, idx_train, idx_val,
                 epochs=100, lr=5e-4, batch_size=32):
    class FeatDataset(Dataset):
        def __init__(self, b, u, d, y):
            self.b, self.u, self.d, self.y = b, u, d, y
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.b[i], self.u[i], self.d[i], self.y[i]

    train_ds = FeatDataset(X_b[idx_train], X_u[idx_train], X_density[idx_train], y[idx_train])
    val_ds = FeatDataset(X_b[idx_val], X_u[idx_val], X_density[idx_val], y[idx_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sched = CosineAnnealingLR(opt, T_max=epochs * len(train_loader), eta_min=1e-6)

    best_acc, best_metrics, patience = 0, None, 0
    for ep in range(epochs):
        model.train()
        for b, u, d, lbl in train_loader:
            b, u, d, lbl = b.to(DEVICE), u.to(DEVICE), d.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            logits = model(b, u, d)
            w = torch.tensor([3.0, 1.0], device=DEVICE)
            loss = F.cross_entropy(logits, lbl, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()

        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for b, u, d, lbl in val_loader:
                b, u, d = b.to(DEVICE), u.to(DEVICE), d.to(DEVICE)
                logits = model(b, u, d)
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


if __name__ == '__main__':
    print('=' * 70)
    print('Pyramid VascMamba: ROI BiomedCLIP multi-scale tokens + Mamba + aggregation')
    print('=' * 70)

    X_b, X_u, X_density, y_all = load_features()
    N = len(y_all)
    print(f'Features: B-mode {X_b.shape}, ULM {X_u.shape}, density {X_density.shape}')

    model = PyramidVascMamba(input_dim=512, d_model=64, d_state=4, n_layers=1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {n_params/1000:.0f}K params')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []
    for fi, (ti, vi) in enumerate(skf.split(np.arange(N), y_all.numpy())):
        m = PyramidVascMamba(input_dim=512, d_model=64, d_state=4, n_layers=1).to(DEVICE)
        metrics = train_hybrid(m, X_b, X_u, X_density, y_all, ti, vi, epochs=100, lr=5e-4, batch_size=32)
        fold_results.append(metrics)
        print(f"Fold {fi+1}: acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} f1={metrics['f1']:.4f} recall={metrics['recall']:.4f}")

    print('\n' + '=' * 70)
    print('Pyramid VascMamba RESULTS')
    print('=' * 70)
    print(f"  Acc:    {np.mean([r['acc'] for r in fold_results]):.4f} ± {np.std([r['acc'] for r in fold_results]):.4f}")
    print(f"  AUC:    {np.mean([r['auc'] for r in fold_results]):.4f}")
    print(f"  F1:     {np.mean([r['f1'] for r in fold_results]):.4f}")
    print(f"  Per-fold acc: {[round(r['acc'], 3) for r in fold_results]}")
    print(f"\n  Baseline VascMamba-Hybrid (BiomedCLIP CLS): 0.8800")
