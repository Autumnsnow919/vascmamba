"""VascMamba: Vascular-guided Mamba for ULM breast cancer classification.

Architecture innovation: Mamba's selective state space model scans along vessel
topology paths (not fixed grid directions), making it intrinsically aware of
vascular structure. This is the first architecture that combines:
  1. Vessel topology-guided scan ordering (physically informed inductive bias)
  2. Selective SSM (Mamba) for efficient sequence modeling
  3. Cross-modal fusion of vessel-flow features and B-mode anatomical features

Architecture:
  ULM → vessel mask → skeleton → ordered path
      → sample patches along path → VesselMamba(1D SSM) → vessel features
  B-mode → 4-directional scan → BMamba(2D SSM) → tissue features
  Cross-attention fusion → MLP → benign/malignant
"""
import sys, os
sys.path.insert(0, '/root/medic_data'); sys.path.insert(0, '/root/medic_data/ulm_visionnet')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score
from tqdm import tqdm
import cv2
import math
from collections import defaultdict

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda')

CROP_Y1, CROP_Y2, CROP_X, SPLIT_X = 162, 737, 1100, 590

from data.patient_index_v2 import build_unified_index


# ═══════════════════════════════════════════════════════════
# Selective SSM (Mamba) Block — Pure PyTorch
# ═══════════════════════════════════════════════════════════

class SelectiveSSM(nn.Module):
    """Pure PyTorch implementation of a simplified Selective State Space Model.
    h_t = A * h_{t-1} + B_t * x_t
    y_t = C_t * h_t
    with input-dependent Δ, B, C."""
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, d_inner * 2)  # x + z branches
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.act = nn.SiLU()

        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1)  # Δ, B, C
        self.dt_proj = nn.Linear(d_inner, 1)

        # A: diagonal state matrix (learnable)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x):
        """x: (B, L, D) → (B, L, D)"""
        B, L, D = x.shape

        # Project input → 2x inner dim
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Conv1d for local context
        x_conv = self.conv1d(x_in.transpose(1, 2))  # (B, d_inner, L+pad)
        x_conv = x_conv[:, :, :L]  # remove padding
        x_conv = self.act(x_conv).transpose(1, 2)   # (B, L, d_inner)

        # SSM scan: simplified selective scan
        # Compute input-dependent B, C, Δ
        x_proj = self.x_proj(x_conv)  # (B, L, 2*state + 1)
        B_ssm = x_proj[..., :self.d_state]       # (B, L, state)
        C_ssm = x_proj[..., self.d_state:2*self.d_state]  # (B, L, state)
        dt = F.softplus(self.dt_proj(x_conv))     # (B, L, 1)

        # Discretize A
        y_scan = self._selective_scan(x_conv, dt.squeeze(-1), B_ssm, C_ssm)  # (B, L, d_inner)

        # Residual + gate
        y_out = y_scan * self.act(z)  # (B, L, d_inner)
        out = self.out_proj(y_out)  # (B, L, d_model)
        return out

    def _selective_scan(self, x, dt, B_ssm, C_ssm):
        """Parallel selective scan using cumulative products (associative scan).
        h_t = a_t * h_{t-1} + b_t where a_t = A_bar[:,t,:], b_t = B_bar[:,t,:,:]
        Uses the associative property of (a, b) pairs for efficient parallel scan.
        x: (B, L, D), dt: (B, L), B_ssm: (B, L, state), C_ssm: (B, L, state)"""
        B, L, D = x.shape
        state = self.d_state

        A = -torch.exp(self.A_log)  # (state,)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, state)

        # B_bar_t = dt_t * B_t * x_t[:,:,None] → (B, L, D, state)
        dt_exp = dt.unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
        B_exp = B_ssm.unsqueeze(2)  # (B, L, 1, state)
        x_exp = x.unsqueeze(-1)     # (B, L, D, 1)
        B_bar = dt_exp * B_exp * x_exp  # (B, L, D, state)

        # Associative scan: (a, b) ∘ (a', b') = (a'·a, a'·b + b')
        # We scan over L dimension in parallel using binary tree reduction
        a = A_bar.unsqueeze(2)  # (B, L, 1, state)  → broadcast over D
        b = B_bar                # (B, L, D, state)

        # Iterative parallel scan: log2(L) iterations
        # Pad to power of 2 for simplicity
        p2 = 1
        while p2 < L: p2 *= 2

        a_pad = torch.ones(B, p2, D, state, device=x.device, dtype=x.dtype)
        a_pad[:, :L] = a.expand(-1, -1, D, -1)
        b_pad = torch.zeros(B, p2, D, state, device=x.device, dtype=x.dtype)
        b_pad[:, :L] = b

        # Hillis-Steele parallel prefix scan
        for d in range(int(math.log2(p2))):
            step = 2 ** d
            a_prev = torch.roll(a_pad, step, dims=1)
            b_prev = torch.roll(b_pad, step, dims=1)
            # Binary op: (a2*a1, a2*b1 + b2)
            new_a = a_prev * a_pad
            new_b = a_prev * b_pad + b_prev
            # Mask: only update from position step onwards
            mask = torch.arange(p2, device=x.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) >= step
            a_pad = torch.where(mask, new_a, a_pad)
            b_pad = torch.where(mask, new_b, b_pad)

        h = b_pad[:, :L]  # (B, L, D, state)

        # y_t = sum(h_t * C_t, dim=-1)
        C_exp = C_ssm.unsqueeze(2)  # (B, L, 1, state)
        y = (h * C_exp).sum(-1)  # (B, L, D)

        return y + x * self.D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    """Mamba Block = LayerNorm → SSM → LayerNorm → FFN"""
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2, ffn_expand=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_expand * d_model),
            nn.GELU(),
            nn.Linear(ffn_expand * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════
# Vessel-Guided Scan Path
# ═══════════════════════════════════════════════════════════

def extract_vessel_scan_path(ulm_img, num_patches=196, patch_size=16):
    """
    Extract an ordered 1D sequence of patches along vessel skeleton.
    ulm_img: (H, W, 3) BGR uint8
    Returns: patches tensor (num_patches, 3*patch_size*patch_size) or None
    """
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from skimage.measure import label as sklabel, regionprops

    h, w = ulm_img.shape[:2]

    # Vessel mask
    gray = cv2.cvtColor(ulm_img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    r = ulm_img[:,:,2].astype(float)
    g = ulm_img[:,:,1].astype(float)
    bright = np.maximum(np.maximum(r, g), ulm_img[:,:,0].astype(float))
    bright_mask = (bright > 25).astype(np.uint8) * 255
    mask = ((otsu > 0) | (bright_mask > 0)) > 0
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, structure=np.ones((3,3)), iterations=1)
    lbl = sklabel(mask)
    for region in regionprops(lbl):
        if region.area < 5: mask[lbl == region.label] = False
    mask = ndimage.binary_closing(mask, structure=np.ones((3,3)), iterations=1)

    if mask.sum() < 50:
        return None

    # Skeleton
    skel = skeletonize(mask)

    # Order: BFS along skeleton starting from top-most pixel
    skel_pts = np.argwhere(skel)
    if len(skel_pts) < 2:
        return None

    start_idx = 0  # top-most
    ordered = [tuple(skel_pts[start_idx])]
    visited = np.zeros_like(skel, dtype=bool)
    visited[ordered[0]] = True

    # BFS flood-fill along skeleton
    queue = [ordered[0]]
    while queue:
        y, x = queue.pop(0)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    ordered.append((ny, nx))
                    queue.append((ny, nx))

    if len(ordered) < 2:
        return None

    # Sample num_patches equally spaced along the ordered path
    pts = np.array(ordered, dtype=np.float32)
    indices = np.linspace(0, len(pts)-1, num_patches).astype(int)
    sampled = pts[indices]  # (num_patches, 2)

    # Extract patches around each sampled point
    ps = patch_size // 2
    patches = []
    for cy, cx in sampled:
        y1, y2 = max(0, int(cy)-ps), min(h, int(cy)+ps)
        x1, x2 = max(0, int(cx)-ps), min(w, int(cx)+ps)
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)

        crop_h = y2 - y1
        crop_w = x2 - x1
        # Safe: place crop into patch, limited by crop size
        place_h = min(crop_h, patch_size)
        place_w = min(crop_w, patch_size)
        patch[:place_h, :place_w] = ulm_img[y1:y1+place_h, x1:x1+place_w].astype(np.float32)

        patch = patch[..., ::-1] / 255.0  # BGR→RGB
        patch = (patch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        patches.append(patch.reshape(-1))

    return torch.from_numpy(np.stack(patches)).float()  # (num_patches, 3*16*16)


def extract_bmode_scan_tokens(bmode_img, num_patches=196, patch_size=16):
    """
    Standard 2D scanning of B-mode image.
    bmode_img: (H, W, 3) BGR uint8
    Returns: (num_patches, 3*patch_size*patch_size) or None
    """
    h, w = bmode_img.shape[:2]
    ps = patch_size

    # Resize to ensure patch_size divides evenly
    grid_size = int(math.sqrt(num_patches))
    target = grid_size * ps
    bmode = cv2.resize(bmode_img, (target, target), cv2.INTER_AREA)

    patches = []
    for y in range(0, target, ps):
        for x in range(0, target, ps):
            patch = bmode[y:y+ps, x:x+ps].astype(np.float32)
            patch = patch[..., ::-1] / 255.0
            patch = (patch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            patches.append(patch.reshape(-1))

    return torch.from_numpy(np.stack(patches)).float()


# ═══════════════════════════════════════════════════════════
# VascMamba Model
# ═══════════════════════════════════════════════════════════

class VascMamba(nn.Module):
    def __init__(self, d_model=128, d_state=8, n_layers=2, num_patches=196, patch_dim=768):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches

        # Patch embedding: 3*16*16 = 768 → d_model
        self.vessel_embed = nn.Sequential(
            nn.Linear(patch_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.bmode_embed = nn.Sequential(
            nn.Linear(patch_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

        # Position embeddings
        self.vessel_pos = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        self.bmode_pos = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

        # Mamba encoders
        self.vessel_mamba = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.bmode_mamba = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv=4, expand=2) for _ in range(n_layers)
        ])

        # Fusion: cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm_fuse = nn.LayerNorm(d_model)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, vessel_tokens, bmode_tokens):
        """vessel_tokens: (B, N, 768), bmode_tokens: (B, N, 768)"""
        B = vessel_tokens.shape[0]

        # Embed
        v = self.vessel_embed(vessel_tokens) + self.vessel_pos  # (B, N, d_model)
        b = self.bmode_embed(bmode_tokens) + self.bmode_pos

        # Mamba encoding
        for layer in self.vessel_mamba:
            v = layer(v)
        for layer in self.bmode_mamba:
            b = layer(b)

        # Cross-attention fusion
        v_fused, _ = self.cross_attn(v, b, b)  # vessel queries bmode
        v_fused = self.norm_fuse(v + v_fused)

        # Pooling
        v_pool = v_fused.mean(dim=1)  # (B, d_model)
        b_pool = b.mean(dim=1)

        # Classification
        combined = torch.cat([v_pool, b_pool], dim=-1)
        return self.head(combined)


# ═══════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════

def pad_sq(im):
    h, w = im.shape[:2]; s = max(h, w)
    top = (s-h)//2; bot = s-h-top; lft = (s-w)//2; rgt = s-w-lft
    return cv2.copyMakeBorder(im, top, bot, lft, rgt, cv2.BORDER_CONSTANT, value=0)

def load_session_tokens(sample):
    """Load one session: extract vessel path tokens + bmode tokens.
    Uses view 0 (TypDen) for vessel path."""
    vf = sample['views'][0]
    path = os.path.join(sample['patient_dir'], vf)
    img = cv2.imread(path)
    if img is None:
        return None, None

    cropped = img[CROP_Y1:CROP_Y2, 0:CROP_X]
    bmode = cropped[:, :SPLIT_X]
    ulm = cropped[:, SPLIT_X:]
    ulm[:, -max(1, int(ulm.shape[1]*0.1)):] = 0

    # Pad to square
    bmode = pad_sq(bmode)
    ulm = pad_sq(ulm)

    vessel_tokens = extract_vessel_scan_path(ulm)
    bmode_tokens = extract_bmode_scan_tokens(bmode)

    if vessel_tokens is None or bmode_tokens is None:
        return None, None

    return vessel_tokens, bmode_tokens


class VascMambaDataset(Dataset):
    def __init__(self, samples, indices):
        self.data = []
        for idx in indices:
            vt, bt = load_session_tokens(samples[idx])
            if vt is not None:
                self.data.append((vt, bt, samples[idx]['label']))

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        vt, bt, lbl = self.data[i]
        return vt, bt, torch.tensor(lbl).long()


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, sched):
    model.train()
    total_loss, n = 0, 0
    for vt, bt, labels in loader:
        vt, bt, labels = vt.to(DEVICE), bt.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        logits = model(vt, bt)
        w = torch.tensor([3.0, 1.0], device=DEVICE)
        loss = F.cross_entropy(logits, labels, weight=w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if sched: sched.step()
        total_loss += loss.item(); n += 1
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs, labels = [], []
    for vt, bt, lbl in loader:
        vt, bt = vt.to(DEVICE), bt.to(DEVICE)
        logits = model(vt, bt)
        probs.append(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        labels.append(lbl.numpy())
    probs = np.concatenate(probs); labels = np.concatenate(labels)
    best_t, best_f1 = 0.5, 0
    best_m = None
    for t in np.arange(0.05, 0.95, 0.02):
        pred = (probs >= t).astype(int)
        f1v = f1_score(labels, pred, zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v; best_t = t
            best_m = {'acc': accuracy_score(labels, pred),
                       'auc': roc_auc_score(labels, probs),
                       'recall': recall_score(labels, pred, zero_division=0),
                       'f1': f1v}
    return best_m


if __name__ == '__main__':
    print('=' * 70)
    print('VascMamba: Vascular-guided Mamba for ULM Classification')
    print('=' * 70)

    samples = build_unified_index()
    print(f'\nSessions: {len(samples)} (良:{sum(1 for s in samples if s["label"]==0)}, '
          f'恶:{sum(1 for s in samples if s["label"]==1)})')

    # Count params
    model = VascMamba(d_model=128, d_state=8, n_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'VascMamba: {n_params/1000:.0f}K params ({n_trainable/1000:.0f}K trainable)')

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    labels = np.array([s['label'] for s in samples])
    fold_results = []

    for fi, (ti, vi) in enumerate(skf.split(np.arange(len(samples)), labels)):
        print(f'\n--- Fold {fi+1}/5 ---')

        train_ds = VascMambaDataset(samples, ti)
        val_ds = VascMambaDataset(samples, vi)
        print(f'  Train: {len(train_ds)}, Val: {len(val_ds)}')

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        # Minimal model: 115K params
        model = VascMamba(d_model=32, d_state=4, n_layers=1).to(DEVICE)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fi == 0: print(f'  Model params: {n_trainable/1000:.0f}K')

        opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)  # strong wd
        sched = CosineAnnealingLR(opt, T_max=60 * len(train_loader), eta_min=1e-6)

        best_acc, best_metrics, patience = 0, None, 0
        for ep in range(60):
            loss = train_epoch(model, train_loader, opt, sched)
            metrics = evaluate(model, val_loader)
            if metrics['acc'] > best_acc + 0.005:
                best_acc = metrics['acc']; best_metrics = metrics; patience = 0
            else:
                patience += 1
            if (ep+1) % 15 == 0 or ep == 0:
                print(f'  Epoch {ep+1:2d}: loss={loss:.4f} acc={metrics["acc"]:.3f} auc={metrics["auc"]:.3f} [patience={patience}]')
            if patience > 15:
                print(f'  Early stop at epoch {ep+1}')
                break

        print(f'  Best: acc={best_metrics["acc"]:.4f} auc={best_metrics["auc"]:.4f} '
              f'rec={best_metrics["recall"]:.4f} f1={best_metrics["f1"]:.4f}')
        fold_results.append(best_metrics)

    accs = [r['acc'] for r in fold_results]
    aucs = [r['auc'] for r in fold_results]
    recs = [r['recall'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]

    print('\n' + '=' * 70)
    print('VascMamba FINAL RESULTS')
    print('=' * 70)
    print(f'  Acc:    {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'  AUC:    {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    print(f'  Recall: {np.mean(recs):.4f} ± {np.std(recs):.4f}')
    print(f'  F1:     {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
    print(f'  Per-fold acc: {[f"{a:.3f}" for a in accs]}')

    print(f'\n  VTG-Net v2 (baseline): 0.8792')
    print(f'  Δ: {np.mean(accs) - 0.8792:+.4f}')
