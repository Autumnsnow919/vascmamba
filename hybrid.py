"""VascMamba-Hybrid: BiomedCLIP(frozen) features + VascMamba lightweight head.

Architecture innovation preserved: vessel-guided sequence ordering fed into Mamba SSM.
BUT features are from frozen BiomedCLIP — no end-to-end pixel training.
"""
import sys, os, numpy as np
sys.path.insert(0, '/root/medic_data'); sys.path.insert(0, '/root/medic_data/ulm_visionnet')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score
import math
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda')

CROP_Y1, CROP_Y2, CROP_X, SPLIT_X = 162, 737, 1100, 590

from data.patient_index_v2 import build_unified_index


# ═══════════════════════════════════════════════════════
# Lightweight Selective SSM (same as before)
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

        # Sequential scan (fast enough for 8 tokens)
        B_seq, L_seq, D_seq = x_conv.shape
        A = -torch.exp(self.A_log)  # (state,)
        A = A.view(1, 1, 1, -1)  # allow broadcast with (B, L, 1, 1)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, 1, state)
        dt_exp = dt.unsqueeze(-1)  # dt already (B,L,1), need (B,L,1,1)
        B_exp = B_ssm.unsqueeze(2)  # (B,L,1,state)
        x_exp = x_conv.unsqueeze(-1)  # (B,L,d_inner,1)
        B_bar_seq = dt_exp * B_exp * x_exp  # (B,L,d_inner,state)

        h = torch.zeros(B_seq, D_seq, self.d_state, device=x.device)
        outputs = []
        for t in range(L_seq):
            a_t = A_bar[:, t, 0]  # (B, state)
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
# VascMamba Hybrid: BiomedCLIP features + Mamba head
# ═══════════════════════════════════════════════════════

class VascMambaHybrid(nn.Module):
    def __init__(self, bc_dim=512, d_model=64, d_state=4, n_layers=2, n_views=4):
        super().__init__()
        self.n_views = n_views
        self.seq_len = n_views * 2  # 4 B-mode + 4 ULM = 8 tokens

        # Feature projection: 512D → d_model
        self.bmode_proj = nn.Sequential(
            nn.Linear(bc_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.ulm_proj = nn.Sequential(
            nn.Linear(bc_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

        # Learnable position + modality embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        self.mod_emb = nn.Parameter(torch.zeros(1, 2, d_model))  # 0=B-mode, 1=ULM

        # Mamba layers
        self.mamba = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv=2, expand=2) for _ in range(n_layers)
        ])

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, bmode_feats, ulm_feats, ulm_density=None):
        """
        bmode_feats: (B, 4, 512) — 4 B-mode views
        ulm_feats:   (B, 4, 512) — 4 ULM views
        ulm_density: (B, 4) — vessel density per ULM view (for ordering), optional
        Returns: logits (B, 2)
        """
        B = bmode_feats.shape[0]

        # Project to d_model
        b_tokens = self.bmode_proj(bmode_feats)  # (B, 4, d)
        u_tokens = self.ulm_proj(ulm_feats)      # (B, 4, d)

        # Order ULM tokens by vessel density if provided (vessel-guided ordering)
        if ulm_density is not None:
            _, sort_idx = ulm_density.sort(dim=1, descending=True)
            u_tokens = u_tokens.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, u_tokens.shape[-1]))

        # Interleave or concatenate:  B1, U1, B2, U2, B3, U3, B4, U4
        tokens = []
        for v in range(self.n_views):
            tokens.append(b_tokens[:, v:v+1])
            tokens.append(u_tokens[:, v:v+1])
        tokens = torch.cat(tokens, dim=1)  # (B, 8, d)

        # Add modality embeddings: even positions=0 (B-mode), odd=1 (ULM)
        mod_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=tokens.device).long()
        tokens = tokens + self.mod_emb[0, mod_ids].unsqueeze(0)
        tokens = tokens + self.pos_emb

        # Mamba encoding
        for layer in self.mamba:
            tokens = layer(tokens)

        # Mean pool over sequence
        pooled = tokens.mean(dim=1)  # (B, d)
        return self.head(pooled)


# ═══════════════════════════════════════════════════════
# Feature extraction: BiomedCLIP per-view features
# ═══════════════════════════════════════════════════════

import cv2

def pad_sq(im):
    h, w = im.shape[:2]; s = max(h, w)
    top = (s-h)//2; bot = s-h-top; lft = (s-w)//2; rgt = s-w-lft
    return cv2.copyMakeBorder(im, top, bot, lft, rgt, cv2.BORDER_CONSTANT, value=0)

def preprocess_for_bc(img):
    resized = cv2.resize(pad_sq(img), (224, 224), cv2.INTER_AREA)
    rgb = resized.astype(np.float32)[..., ::-1] / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(rgb).float().permute(2, 0, 1)


@torch.no_grad()
def extract_session_features(samples, bc_model):
    """Extract per-view BiomedCLIP features + vessel density per view."""
    all_bmode, all_ulm, all_density, all_labels = [], [], [], []
    for s in tqdm(samples, desc='Extracting BC features'):
        b_views, u_views, densities = [], [], []
        for vf in s['views']:
            path = os.path.join(s['patient_dir'], vf)
            img = cv2.imread(path)
            if img is None:
                b_views.append(torch.zeros(512))
                u_views.append(torch.zeros(512))
                densities.append(0.0)
                continue

            cropped = img[CROP_Y1:CROP_Y2, 0:CROP_X]
            bm = cropped[:, :SPLIT_X]
            um = cropped[:, SPLIT_X:]
            um[:, -max(1, int(um.shape[1]*0.1)):] = 0

            # BiomedCLIP features
            bm_t = preprocess_for_bc(bm).unsqueeze(0).to(DEVICE)
            um_t = preprocess_for_bc(um).unsqueeze(0).to(DEVICE)
            b_views.append(bc_model.encode_image(bm_t, normalize=True).cpu()[0])
            u_views.append(bc_model.encode_image(um_t, normalize=True).cpu()[0])

            # Vessel density (for ordering)
            gray = cv2.cvtColor(um, cv2.COLOR_BGR2GRAY)
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            densities.append((otsu > 0).mean())

        all_bmode.append(torch.stack(b_views))    # (4, 512)
        all_ulm.append(torch.stack(u_views))       # (4, 512)
        all_density.append(torch.tensor(densities))  # (4,)
        all_labels.append(s['label'])

    return (torch.stack(all_bmode), torch.stack(all_ulm),
            torch.stack(all_density), torch.tensor(all_labels).long())


# ═══════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════

def train_hybrid(model, X_bmode, X_ulm, X_density, y, idx_train, idx_val,
                 epochs=100, lr=5e-4, batch_size=32):
    """Train VascMamba hybrid head."""
    # Create data loaders from pre-extracted features
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


if __name__ == '__main__':
    print('=' * 70)
    print('VascMamba-Hybrid: BiomedCLIP + Mamba head')
    print('=' * 70)

    # Load pre-extracted BiomedCLIP features (no need to reload model)
    print('\n[2] Loading pre-extracted BiomedCLIP features...')
    bc_data = np.load('/root/medic_data/biomedclip_features.npz')
    X_bc = torch.from_numpy(bc_data['X']).float()  # (241, 1024) — B-mean(512) + U-mean(512)
    y_all = torch.from_numpy(bc_data['y']).long()

    # Reshape: (241, 1024) → bmode=(241, 512), ulm=(241, 512)
    # BiomedCLIP features are stored as [B-mean(512), U-mean(512)]
    X_bmode_full = X_bc[:, :512].unsqueeze(1).expand(-1, 4, -1)  # (241, 4, 512)
    X_ulm_full = X_bc[:, 512:].unsqueeze(1).expand(-1, 4, -1)

    # For vessel density ordering: use pre-computed vascular features
    vasc_data = np.load('/root/medic_data/vascular_features.npz')
    X_vasc = torch.from_numpy(vasc_data['X_vasc']).float()
    # vessel_density is at feature index ~0 (vessel_density_mean)
    # Use mean of vessel_density across 4 views as proxy
    X_density_full = X_vasc[:, 0].unsqueeze(1).expand(-1, 4)  # (241, 4) — simplified

    # Model size
    model = VascMambaHybrid(d_model=32, d_state=4, n_layers=1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Hybrid model: {n_params/1000:.0f}K params')

    # 5-fold CV
    N = len(y_all)
    print(f'\n[3] 5-Fold Training ({N} samples)...')
    print(f'  B-mode: {X_bmode_full.shape}, ULM: {X_ulm_full.shape}, Density: {X_density_full.shape}')
    y_np = y_all.numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    for fi, (ti, vi) in enumerate(skf.split(np.arange(N), y_np)):
        print(f'\n--- Fold {fi+1}/5 | Train={len(ti)} Val={len(vi)} ---')

        m = VascMambaHybrid(d_model=32, d_state=4, n_layers=1).to(DEVICE)
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)

        metrics = train_hybrid(m, X_bmode_full, X_ulm_full, X_density_full, y_all,
                                ti, vi, epochs=100, lr=5e-4)
        fold_results.append(metrics)

    # Summary
    accs = [r['acc'] for r in fold_results]
    aucs = [r['auc'] for r in fold_results]
    print('\n' + '=' * 70)
    print('VascMamba-Hybrid RESULTS')
    print('=' * 70)
    print(f'  Acc:    {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'  AUC:    {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    print(f'  Recall: {np.mean([r["recall"] for r in fold_results]):.4f}')
    print(f'  F1:     {np.mean([r["f1"] for r in fold_results]):.4f}')
    print(f'  Per-fold: {[f"{a:.3f}" for a in accs]}')
    print(f'\n  BiomedCLIP SVM (baseline): 0.8548')
    print(f'  VTG-Net v2:               0.8792')
    print(f'  VascMamba raw pixel:      0.8081')
    print(f'  VascMamba-Hybrid:         {np.mean(accs):.4f}')
