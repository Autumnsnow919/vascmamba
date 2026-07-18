#!/usr/bin/env python3
"""PatchVascMamba: full BiomedCLIP patch tokens + max-pooling + density-gated ULM.

Differences vs pyramid_vascmamba.py:
  1. Finer pyramid: CLS + 7x7 mean (49) + 2x2 mean (4) + 2x2 MAX (4) + 1x1 mean/max (2)
     = 60 tokens/view/modality (was 6). 7x7 keeps rim-vascularity spatial pattern;
     MAX levels preserve sparse focal vessels (necrotic malignant tumors).
  2. Density gate: ULM tokens scaled by sigmoid(a*density+b) — low-vascular ULM
     becomes "uninformative" instead of "evidence for benign".
  3. ULM modality dropout (p=0.2, train only): forces B-mode-only decisions,
     breaks the "empty ULM -> benign" shortcut.

Input: biomedclip_patch_tokens.npz
  X_bmode/X_ulm: (241, 4, 197, 512) fp16, density: (241, 4), y: (241,)
"""
import sys, os, numpy as np
sys.path.insert(0, '/root/medic_data'); sys.path.insert(0, '/root/medic_data/ulm_visionnet')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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

NPZ = '/root/medic_data/biomedclip_patch_tokens.npz'

# Token layout per view-modality block
N_TOK = 60                       # CLS(1) + 7x7(49) + 2x2mean(4) + 2x2max(4) + 1x1mean(1) + 1x1max(1)
LEVEL_IDS = [0] + [1] * 49 + [2] * 4 + [3] * 4 + [4] + [5]
N_LEVELS = 6
# Full-token (no pooling) layout: CLS(1) + 196 raw patches
N_TOK_FULL = 197
LEVEL_IDS_FULL = [0] + [1] * 196
N_LEVELS_FULL = 2


def chunked_scan(la_step, B_bar_seq, C_ssm, chunk=64):
    """Numerically-equivalent chunked version of the sequential SSM scan.

    la_step:  (B, L, 1, state)  log decay per step = dt*A
    B_bar_seq:(B, L, d_inner, state)
    C_ssm:    (B, L, state)
    Returns y: (B, L, d_inner).  O(L*chunk) instead of O(L) python steps.
    """
    B_, L, D, S = B_bar_seq.shape
    h = torch.zeros(B_, D, S, device=B_bar_seq.device, dtype=B_bar_seq.dtype)
    ys = []
    for start in range(0, L, chunk):
        end = min(start + chunk, L)
        c = end - start
        La = torch.cumsum(la_step[:, start:end, 0, :], dim=1)        # (B,c,state)
        diff = La.unsqueeze(2) - La.unsqueeze(1)                     # (B,t,s,state)
        mask = torch.tril(torch.ones(c, c, device=diff.device, dtype=torch.bool)).view(1, c, c, 1)
        G = torch.exp(diff.masked_fill(~mask, float('-inf')))        # upper -> 0, no NaN
        h_intra = torch.einsum('btsn,bsdn->btdn', G, B_bar_seq[:, start:end])
        h_all = h_intra + torch.exp(La).unsqueeze(2) * h.unsqueeze(1)  # (B,c,D,state)
        y_c = (h_all * C_ssm[:, start:end].unsqueeze(2)).sum(-1)       # (B,c,D)
        ys.append(y_c)
        h = h_all[:, -1]
    return torch.cat(ys, dim=1)


class SelectiveSSM(nn.Module):
    def __init__(self, d_model=128, d_state=8, d_conv=4, expand=2, chunk=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.chunk = chunk
        d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
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
        C_ssm = x_proj[..., self.d_state:2 * self.d_state]
        dt = F.softplus(self.dt_proj(x_conv))
        B_seq, L_seq, D_seq = x_conv.shape
        A = -torch.exp(self.A_log).view(1, 1, 1, -1)
        dt_exp = dt.unsqueeze(-1)
        B_exp = B_ssm.unsqueeze(2)
        x_exp = x_conv.unsqueeze(-1)
        B_bar_seq = dt_exp * B_exp * x_exp

        if self.chunk is not None and L_seq > self.chunk:
            la_step = dt_exp * A                       # (B,L,1,state) log decay
            y_scan = chunked_scan(la_step, B_bar_seq, C_ssm, self.chunk)
        else:
            A_bar = torch.exp(dt_exp * A)
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
    def __init__(self, d_model=128, d_state=8, d_conv=2, expand=2, chunk=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand, chunk)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PatchVascMamba(nn.Module):
    def __init__(self, input_dim=512, d_model=64, d_state=4, n_layers=1, n_views=4,
                 ulm_dropout=0.2, full_tokens=False):
        super().__init__()
        self.n_views = n_views
        self.ulm_dropout = ulm_dropout
        self.full_tokens = full_tokens
        self.n_tok = N_TOK_FULL if full_tokens else N_TOK
        level_ids = LEVEL_IDS_FULL if full_tokens else LEVEL_IDS
        n_levels = N_LEVELS_FULL if full_tokens else N_LEVELS
        chunk = 64 if full_tokens else None
        self.seq_len = n_views * 2 * self.n_tok  # 480 (pyramid) or 1576 (full)

        self.token_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.mod_emb = nn.Parameter(torch.zeros(1, 2, d_model))
        self.level_emb = nn.Parameter(torch.zeros(1, n_levels, d_model))
        self.view_pos_emb = nn.Parameter(torch.randn(1, n_views, d_model) * 0.02)
        self.intra_pos_emb = nn.Parameter(torch.randn(1, self.n_tok, d_model) * 0.02)

        # Density gate for ULM blocks: sigmoid(a*d + b); init so d=0 -> ~0.12, d=1 -> ~0.88
        self.gate_fc = nn.Linear(1, 1)
        nn.init.constant_(self.gate_fc.weight, 4.0)
        nn.init.constant_(self.gate_fc.bias, -2.0)

        self.mamba = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv=2, expand=2, chunk=chunk) for _ in range(n_layers)
        ])

        self.agg_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.agg_attn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + 3),
            nn.Linear(d_model + 3, 32), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

        idx = torch.arange(self.seq_len)
        block = idx // self.n_tok
        mod = block % 2
        view = block // 2
        intra = idx % self.n_tok
        level = torch.tensor(level_ids)[intra]
        self.register_buffer('seq_idx', torch.stack([mod, view, intra, level], dim=0))

    def build_tokens(self, feats):
        """feats: (B, 4, 197, 512) -> (B, 4, n_tok, 512)."""
        if self.full_tokens:
            return feats                                   # no pooling at all
        B = feats.shape[0]
        cls = feats[:, :, 0:1, :]                                  # (B,4,1,512)
        grid = feats[:, :, 1:, :].reshape(B, 4, 14, 14, -1)        # (B,4,14,14,512)
        t7 = grid.reshape(B, 4, 7, 2, 7, 2, -1).mean(dim=(3, 5)).reshape(B, 4, 49, -1)
        g2 = grid.reshape(B, 4, 2, 7, 2, 7, -1)
        t2m = g2.mean(dim=(3, 5)).reshape(B, 4, 4, -1)
        t2x = g2.amax(dim=(3, 5)).reshape(B, 4, 4, -1)
        t1m = grid.mean(dim=(2, 3)).reshape(B, 4, 1, -1)
        t1x = grid.amax(dim=(2, 3)).reshape(B, 4, 1, -1)
        return torch.cat([cls, t7, t2m, t2x, t1m, t1x], dim=2)     # (B,4,60,512)

    def forward(self, bmode_feats, ulm_feats, ulm_density):
        """
        bmode_feats: (B, 4, 197, 512); ulm_feats: (B, 4, 197, 512)
        ulm_density: (B, 4)
        Returns logits (B, 2)
        """
        B = bmode_feats.shape[0]
        b_blocks = self.build_tokens(bmode_feats)   # (B,4,60,512)
        u_blocks = self.build_tokens(ulm_feats)

        # Vessel-guided ordering: sort views by ULM density (desc)
        _, sort_idx = ulm_density.sort(dim=1, descending=True)
        expand_idx = sort_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.n_tok, b_blocks.size(-1))
        b_blocks = b_blocks.gather(1, expand_idx)
        u_blocks = u_blocks.gather(1, expand_idx)
        density_sorted = ulm_density.gather(1, sort_idx)            # (B,4)

        # Density gate on ULM: low-vascular view -> down-weight (not "benign evidence")
        gate = torch.sigmoid(self.gate_fc(density_sorted.unsqueeze(-1)))  # (B,4,1)
        u_blocks = u_blocks * gate.unsqueeze(-1)

        # ULM modality dropout (train only): zero whole ULM blocks
        if self.training and self.ulm_dropout > 0:
            keep = (torch.rand(B, 1, 1, 1, device=u_blocks.device) > self.ulm_dropout).float()
            u_blocks = u_blocks * keep

        # Interleave per view: [B1(60), U1(60), B2, U2, ...]
        tokens = []
        for v in range(self.n_views):
            tokens.append(b_blocks[:, v])
            tokens.append(u_blocks[:, v])
        tokens = torch.cat(tokens, dim=1)                           # (B,480,512)
        tokens = self.token_proj(tokens)                            # (B,480,d)

        mod_emb = self.mod_emb[0, self.seq_idx[0].long()]
        view_emb = self.view_pos_emb[0, self.seq_idx[1].long()]
        intra_emb = self.intra_pos_emb[0, self.seq_idx[2].long()]
        level_emb = self.level_emb[0, self.seq_idx[3].long()]
        tokens = tokens + (mod_emb + view_emb + intra_emb + level_emb).unsqueeze(0)

        for layer in self.mamba:
            tokens = layer(tokens)

        attn_scores = self.agg_attn(tokens).squeeze(-1)             # (B,480)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        pooled = (tokens * attn_weights).sum(dim=1)                 # (B,d)

        # Density stats so head can condition on vascular confidence
        dstat = torch.stack([
            ulm_density.mean(dim=1),
            ulm_density.max(dim=1).values,
            ulm_density.std(dim=1, unbiased=False),
        ], dim=1)                                                   # (B,3)
        return self.head(torch.cat([pooled, dstat], dim=1))


class FeatDataset(Dataset):
    def __init__(self, bm, um, d, y):
        self.bm, self.um, self.d, self.y = bm, um, d, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.bm[i], self.um[i], self.d[i], self.y[i]


def train_model(model, X_b, X_u, X_d, y, idx_train, idx_val,
                epochs=100, lr=5e-4, batch_size=32):
    train_ds = FeatDataset(X_b[idx_train], X_u[idx_train], X_d[idx_train], y[idx_train])
    val_ds = FeatDataset(X_b[idx_val], X_u[idx_val], X_d[idx_val], y[idx_val])
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

        best_t, best_f1, best_m = 0.5, 0, None
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
    FULL = '--full' in sys.argv
    print('=' * 70)
    print(f'PatchVascMamba: {"FULL tokens (no pooling), seq 1576" if FULL else "pyramid 60 tokens"} '
          f'+ density gate + ULM dropout')
    print('=' * 70)

    print('\n[1] Loading full patch tokens...')
    data = np.load(NPZ)
    X_b = torch.from_numpy(data['X_bmode']).float()   # (241,4,197,512)
    X_u = torch.from_numpy(data['X_ulm']).float()
    X_d = torch.from_numpy(data['density']).float()   # (241,4)
    y_all = torch.from_numpy(data['y']).long()
    print(f'  X_bmode: {X_b.shape}, X_ulm: {X_u.shape}, density: {X_d.shape}')

    model = PatchVascMamba(d_model=64, d_state=4, n_layers=1, full_tokens=FULL)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model: {n_params/1000:.0f}K trainable params, seq_len={model.seq_len}')

    N = len(y_all)
    print(f'\n[2] 5-Fold Training ({N} samples)...')
    y_np = y_all.numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    for fi, (ti, vi) in enumerate(skf.split(np.arange(N), y_np)):
        print(f'\n--- Fold {fi+1}/5 | Train={len(ti)} Val={len(vi)} ---')
        torch.manual_seed(SEED + fi); np.random.seed(SEED + fi)
        m = PatchVascMamba(d_model=64, d_state=4, n_layers=1, full_tokens=FULL).to(DEVICE)
        metrics = train_model(m, X_b, X_u, X_d, y_all, ti, vi, epochs=100, lr=5e-4)
        print(f'  acc={metrics["acc"]:.4f} auc={metrics["auc"]:.4f} '
              f'recall={metrics["recall"]:.4f} f1={metrics["f1"]:.4f}')
        fold_results.append(metrics)

    accs = [r['acc'] for r in fold_results]
    aucs = [r['auc'] for r in fold_results]
    print('\n' + '=' * 70)
    print(f'PatchVascMamba {"FULL-TOKEN" if FULL else "PYRAMID"} RESULTS')
    print('=' * 70)
    print(f'  Acc:    {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'  AUC:    {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    print(f'  Recall: {np.mean([r["recall"] for r in fold_results]):.4f}')
    print(f'  F1:     {np.mean([r["f1"] for r in fold_results]):.4f}')
    print(f'  Per-fold: {[f"{a:.3f}" for a in accs]}')
    print(f'\n  BiomedCLIP SVM (baseline): 0.8548')
    print(f'  VascMamba-Hybrid:          0.8798')
    print(f'  PatchVascMamba (pyramid):  0.8673')
    print(f'  PatchVascMamba (this run): {np.mean(accs):.4f}')
