"""VascMamba-Hybrid with Domain Adversarial Neural Network (DANN).
Unsupervised domain adaptation: V2 (source, labeled) + external (target, unlabeled).
Test on V2 held-out and external labeled.
"""
import sys, os
sys.path.insert(0, '/root/medic_data')
sys.path.insert(0, '/root/medic_data/ulm_visionnet')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, f1_score,
    precision_score, confusion_matrix, balanced_accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.patient_index_v2 import build_unified_index


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D_MODEL = 32
D_STATE = 4
N_LAYERS = 1
EPOCHS = 100
LR = 5e-4
BATCH_SIZE = 32
PATIENCE = 20
CLASS_WEIGHT = torch.tensor([3.0, 1.0])

OUT_DIR = '/root/medic_data/vascmamba/dann_outputs'


# ═══════════════════════════════════════════════════════
# Model components (copied from vascmamba/hybrid.py to avoid modifying it)
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
        A = -torch.exp(self.A_log)
        A = A.view(1, 1, 1, -1)
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


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 0.0

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class VascMambaDANN(nn.Module):
    def __init__(self, bc_dim=512, d_model=64, d_state=4, n_layers=2, n_views=4):
        super().__init__()
        self.n_views = n_views
        self.seq_len = n_views * 2
        self.bmode_proj = nn.Sequential(
            nn.Linear(bc_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.ulm_proj = nn.Sequential(
            nn.Linear(bc_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
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
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, bmode_feats, ulm_feats, ulm_density=None):
        B = bmode_feats.shape[0]
        b_tokens = self.bmode_proj(bmode_feats)
        u_tokens = self.ulm_proj(ulm_feats)
        if ulm_density is not None:
            _, sort_idx = ulm_density.sort(dim=1, descending=True)
            u_tokens = u_tokens.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, u_tokens.shape[-1]))
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
        class_logits = self.head(pooled)
        domain_logits = self.domain_classifier(self.grl(pooled))
        return class_logits, domain_logits.squeeze(-1)


# ═══════════════════════════════════════════════════════
# Data and training
# ═══════════════════════════════════════════════════════

class FeatDataset(Dataset):
    def __init__(self, bm, um, d, y):
        self.bm, self.um, self.d, self.y = bm, um, d, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.bm[i], self.um[i], self.d[i], self.y[i]


def load_features():
    bc = np.load('/root/medic_data/biomedclip_features.npz')
    X_bc = torch.from_numpy(bc['X']).float()
    y = torch.from_numpy(bc['y']).long()
    vasc = np.load('/root/medic_data/vascular_features.npz')
    X_vasc = torch.from_numpy(vasc['X_vasc']).float()
    return X_bc, y, X_vasc


def dann_schedule(epoch, max_epochs):
    """Gradually increase lambda from 0 to 1."""
    p = float(epoch) / max_epochs
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


def train_dann(model, src_loader, tgt_loader, val_loader, epochs=100, lr=5e-4):
    opt = AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    total_steps = epochs * max(len(src_loader), len(tgt_loader))
    sched = CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)
    best_acc, best_state, patience_counter = 0, None, 0

    for ep in range(epochs):
        model.train()
        lambda_p = dann_schedule(ep, epochs)
        model.grl.set_lambda(lambda_p)

        # Iterate over both loaders
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        steps = max(len(src_loader), len(tgt_loader))
        for _ in range(steps):
            try:
                bm_s, um_s, d_s, y_s = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                bm_s, um_s, d_s, y_s = next(src_iter)
            try:
                bm_t, um_t, d_t, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                bm_t, um_t, d_t, _ = next(tgt_iter)

            bm_s, um_s, d_s, y_s = bm_s.to(DEVICE), um_s.to(DEVICE), d_s.to(DEVICE), y_s.to(DEVICE)
            bm_t, um_t, d_t = bm_t.to(DEVICE), um_t.to(DEVICE), d_t.to(DEVICE)

            opt.zero_grad()

            # Source: classification + domain
            cls_logits, dom_logits_s = model(bm_s, um_s, d_s)
            loss_cls = F.cross_entropy(cls_logits, y_s, weight=CLASS_WEIGHT.to(DEVICE))
            loss_dom_s = F.binary_cross_entropy_with_logits(
                dom_logits_s, torch.zeros_like(dom_logits_s)
            )

            # Target: only domain
            _, dom_logits_t = model(bm_t, um_t, d_t)
            loss_dom_t = F.binary_cross_entropy_with_logits(
                dom_logits_t, torch.ones_like(dom_logits_t)
            )

            loss = loss_cls + lambda_p * (loss_dom_s + loss_dom_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()

        # Evaluate on V2 validation
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for bm, um, d, lbl in val_loader:
                bm, um, d = bm.to(DEVICE), um.to(DEVICE), d.to(DEVICE)
                cls_logits, _ = model(bm, um, d)
                probs.append(F.softmax(cls_logits, -1)[:, 1].cpu())
                labels.append(lbl)
        probs = torch.cat(probs).numpy()
        labels = torch.cat(labels).numpy()

        best_t, best_f1 = 0.5, 0
        best_m = None
        for t in np.arange(0.05, 0.95, 0.02):
            pred = (probs >= t).astype(int)
            f1v = f1_score(labels, pred, zero_division=0)
            if f1v > best_f1:
                best_f1 = f1v
                best_t = t
                best_m = {
                    'acc': accuracy_score(labels, pred),
                    'auc': roc_auc_score(labels, probs),
                    'recall': recall_score(labels, pred, zero_division=0),
                    'f1': f1v,
                    'threshold': best_t,
                }

        if best_m['acc'] > best_acc + 0.005:
            best_acc = best_m['acc']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (ep + 1) % 10 == 0 or patience_counter == 0:
            print(f'    Epoch {ep+1:3d} | lambda={lambda_p:.3f} | val acc={best_m["acc"]:.4f} f1={best_m["f1"]:.4f} t={best_t:.2f}')

        if patience_counter > PATIENCE:
            print(f'    Early stop at epoch {ep+1}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


def evaluate(model, bm, um, d, y, threshold=0.5):
    model.eval()
    with torch.no_grad():
        cls_logits, _ = model(bm.to(DEVICE), um.to(DEVICE), d.to(DEVICE))
        probs = F.softmax(cls_logits, -1)[:, 1].cpu().numpy()
    y = y.numpy()
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'ACC': accuracy_score(y, pred),
        'AUC': roc_auc_score(y, probs),
        'Sensitivity': recall_score(y, pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'PPV': precision_score(y, pred, zero_division=0),
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        'F1': f1_score(y, pred, zero_division=0),
        'Balanced_ACC': balanced_accuracy_score(y, pred),
        'n': len(y),
        'threshold': threshold,
    }
    return metrics, probs, pred


def token_importance(model, bm, um, d):
    model.eval()
    bm_t = bm.to(DEVICE).requires_grad_(True)
    um_t = um.to(DEVICE).requires_grad_(True)
    d_t = d.to(DEVICE)
    cls_logits, _ = model(bm_t, um_t, d_t)
    probs = F.softmax(cls_logits, -1)[:, 1]
    probs.sum().backward()
    bm_grad = bm_t.grad.abs().mean(dim=(0, 2)).cpu().numpy()
    um_grad = um_t.grad.abs().mean(dim=(0, 2)).cpu().numpy()
    importance = np.zeros(8)
    for i in range(4):
        importance[i * 2] = bm_grad[i]
        importance[i * 2 + 1] = um_grad[i]
    importance = importance / (importance.sum() + 1e-12)
    return importance


def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Benign', 'Malignant'])
    plt.yticks(tick_marks, ['Benign', 'Malignant'])
    plt.ylabel('True')
    plt.xlabel('Pred')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc(y_true, probs, title, save_path):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC={auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr(y_true, probs, title, save_path):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'AP={ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    print('=' * 70)
    print('VascMamba-Hybrid + DANN')
    print('Source: V2 (labeled) | Target: external hospital (unlabeled)')
    print('=' * 70)
    os.makedirs(OUT_DIR, exist_ok=True)

    print('\n[1] Loading features and index...')
    X_bc, y_all, X_vasc = load_features()
    samples = build_unified_index()
    print(f'  Total sessions: {len(samples)}')

    is_v2 = np.array(['/output_ulm/' in s['patient_dir'] for s in samples])
    is_ext = np.array(['/数据分析/' in s['patient_dir'] for s in samples])
    v2_idx = np.where(is_v2)[0]
    ext_idx = np.where(is_ext)[0]
    print(f'  V2: {len(v2_idx)} (B:{int((y_all[v2_idx] == 0).sum())}, M:{int((y_all[v2_idx] == 1).sum())})')
    print(f'  External: {len(ext_idx)} (B:{int((y_all[ext_idx] == 0).sum())}, M:{int((y_all[ext_idx] == 1).sum())})')

    X_bmode = X_bc[:, :512].unsqueeze(1).expand(-1, 4, -1)
    X_ulm = X_bc[:, 512:].unsqueeze(1).expand(-1, 4, -1)
    X_density = X_vasc[:, 0].unsqueeze(1).expand(-1, 4)

    # Split V2 into train/val
    y_v2 = y_all[v2_idx].numpy()
    train_idx_v2, val_idx_v2 = train_test_split(
        v2_idx, test_size=0.2, stratify=y_v2, random_state=SEED
    )
    print(f'\n[2] V2 source split: train={len(train_idx_v2)}, val={len(val_idx_v2)}')
    print(f'  Train B/M: {int((y_all[train_idx_v2] == 0).sum())}/{int((y_all[train_idx_v2] == 1).sum())}')
    print(f'  Val   B/M: {int((y_all[val_idx_v2] == 0).sum())}/{int((y_all[val_idx_v2] == 1).sum())}')

    src_ds = FeatDataset(X_bmode[train_idx_v2], X_ulm[train_idx_v2],
                         X_density[train_idx_v2], y_all[train_idx_v2])
    val_ds = FeatDataset(X_bmode[val_idx_v2], X_ulm[val_idx_v2],
                         X_density[val_idx_v2], y_all[val_idx_v2])
    tgt_ds = FeatDataset(X_bmode[ext_idx], X_ulm[ext_idx],
                         X_density[ext_idx], y_all[ext_idx])  # labels not used in training

    src_loader = DataLoader(src_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    ext_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=False)

    print('\n[3] Training DANN...')
    model = VascMambaDANN(d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model: {n_params / 1000:.0f}K params')
    model, best_val_acc = train_dann(model, src_loader, tgt_loader, val_loader, epochs=EPOCHS, lr=LR)
    print(f'  Best val acc: {best_val_acc:.4f}')

    print('\n[4] Threshold search on V2 validation...')
    model.eval()
    v2_val_probs = []
    with torch.no_grad():
        for bm, um, d, _ in val_loader:
            cls_logits, _ = model(bm.to(DEVICE), um.to(DEVICE), d.to(DEVICE))
            v2_val_probs.append(F.softmax(cls_logits, -1)[:, 1].cpu())
    v2_val_probs = torch.cat(v2_val_probs).numpy()
    v2_val_labels = y_all[val_idx_v2].numpy()
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.02):
        f1v = f1_score(v2_val_labels, (v2_val_probs >= t).astype(int), zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_t = t
    print(f'  Best threshold: {best_t:.2f}, F1: {best_f1:.4f}')

    print('\n[5] Evaluating...')
    metrics_v2, v2_probs, v2_pred = evaluate(
        model, X_bmode[val_idx_v2], X_ulm[val_idx_v2],
        X_density[val_idx_v2], y_all[val_idx_v2], threshold=best_t
    )
    metrics_ext, ext_probs, ext_pred = evaluate(
        model, X_bmode[ext_idx], X_ulm[ext_idx],
        X_density[ext_idx], y_all[ext_idx], threshold=best_t
    )

    for name, m in [('V2 val', metrics_v2), ('External', metrics_ext)]:
        print(f'\n  {name}:')
        for k, v in m.items():
            if k == 'n':
                print(f'    {k}: {v}')
            else:
                print(f'    {k}: {v:.4f}')

    df = pd.DataFrame([metrics_v2, metrics_ext])
    df.insert(0, 'set', ['V2_val', 'External'])
    df.to_csv(os.path.join(OUT_DIR, 'metrics.csv'), index=False)
    print(f'\n  Saved metrics.csv')

    torch.save(model.state_dict(), os.path.join(OUT_DIR, 'model.pt'))
    print(f'  Saved model.pt')

    print('\n[6] Plotting...')
    plot_confusion(y_all[val_idx_v2].numpy(), v2_pred,
                   'V2 Val Confusion', os.path.join(OUT_DIR, 'confusion_v2.png'))
    plot_confusion(y_all[ext_idx].numpy(), ext_pred,
                   'External Confusion', os.path.join(OUT_DIR, 'confusion_external.png'))
    plot_roc(y_all[val_idx_v2].numpy(), v2_probs,
             'V2 Val ROC', os.path.join(OUT_DIR, 'roc_v2.png'))
    plot_roc(y_all[ext_idx].numpy(), ext_probs,
             'External ROC', os.path.join(OUT_DIR, 'roc_external.png'))
    plot_pr(y_all[val_idx_v2].numpy(), v2_probs,
            'V2 Val PR', os.path.join(OUT_DIR, 'pr_v2.png'))
    plot_pr(y_all[ext_idx].numpy(), ext_probs,
            'External PR', os.path.join(OUT_DIR, 'pr_external.png'))

    print('\n[7] Token importance heatmap...')
    importance = token_importance(model, X_bmode[val_idx_v2], X_ulm[val_idx_v2],
                                    X_density[val_idx_v2])
    print(f'  Importance: {importance}')
    plt.figure(figsize=(8, 3))
    labels = ['B1', 'U1', 'B2', 'U2', 'B3', 'U3', 'B4', 'U4']
    plt.bar(labels, importance)
    plt.title('Token Importance (Gradient Attribution, V2 Val)')
    plt.ylabel('Relative Importance')
    plt.ylim(0, importance.max() * 1.2 + 0.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'token_importance.png'), dpi=150)
    plt.close()

    print('\n' + '=' * 70)
    print('Done. Outputs in vascmamba/dann_outputs/')
    print('=' * 70)


if __name__ == '__main__':
    main()
