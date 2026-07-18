"""Generate a comparison table (JPG) of VascMamba experiments with fixed and dynamic thresholds."""
import sys, os
sys.path.insert(0, '/root/medic_data')
sys.path.insert(0, '/root/medic_data/ulm_visionnet')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, f1_score,
    precision_score, confusion_matrix, balanced_accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.patient_index_v2 import build_unified_index
from vascmamba.hybrid import VascMambaHybrid


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D_MODEL = 32
D_STATE = 4
N_LAYERS = 1
OUT_DIR = '/root/medic_data/vascmamba/comparison_outputs'


# ═══════════════════════════════════════════════════════
# DANN model definition (same as dann.py, copied here for loading)
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
        return class_logits


def load_features():
    bc = np.load('/root/medic_data/biomedclip_features.npz')
    X_bc = torch.from_numpy(bc['X']).float()
    y = torch.from_numpy(bc['y']).long()
    vasc = np.load('/root/medic_data/vascular_features.npz')
    X_vasc = torch.from_numpy(vasc['X_vasc']).float()
    return X_bc, y, X_vasc


def get_probs(model, bm, um, d):
    model.eval()
    with torch.no_grad():
        logits = model(bm.to(DEVICE), um.to(DEVICE), d.to(DEVICE))
        probs = F.softmax(logits, -1)[:, 1].cpu().numpy()
    return probs


def metrics_with_threshold(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'ACC': accuracy_score(y_true, pred),
        'Sensitivity': recall_score(y_true, pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'PPV': precision_score(y_true, pred, zero_division=0),
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        'F1': f1_score(y_true, pred, zero_division=0),
        'Balanced_ACC': balanced_accuracy_score(y_true, pred),
    }


def find_best_threshold(y_true, probs, criterion='acc'):
    best_t, best_v = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (probs >= t).astype(int)
        cm = confusion_matrix(y_true, pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        if criterion == 'acc':
            v = accuracy_score(y_true, pred)
        elif criterion == 'youden':
            v = sens + spec - 1
        elif criterion == 'f1':
            v = f1_score(y_true, pred, zero_division=0)
        else:
            raise ValueError(criterion)
        if v > best_v:
            best_v, best_t = v, t
    return best_t, best_v


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading data...')
    X_bc, y_all, X_vasc = load_features()
    samples = build_unified_index()
    is_v2 = np.array(['/output_ulm/' in s['patient_dir'] for s in samples])
    is_ext = np.array(['/数据分析/' in s['patient_dir'] for s in samples])
    v2_idx = np.where(is_v2)[0]
    ext_idx = np.where(is_ext)[0]

    X_bmode = X_bc[:, :512].unsqueeze(1).expand(-1, 4, -1)
    X_ulm = X_bc[:, 512:].unsqueeze(1).expand(-1, 4, -1)
    X_density = X_vasc[:, 0].unsqueeze(1).expand(-1, 4)

    # Reproduce splits
    y_v2 = y_all[v2_idx].numpy()
    y_ext = y_all[ext_idx].numpy()
    train_idx_v2, test_idx_v2 = train_test_split(v2_idx, test_size=0.2, stratify=y_v2, random_state=SEED)
    train_idx_ext, test_idx_ext = train_test_split(ext_idx, test_size=0.2, stratify=y_ext, random_state=SEED)
    train_idx_dann, val_idx_dann = train_test_split(v2_idx, test_size=0.2, stratify=y_v2, random_state=SEED)

    experiments = [
        ('V2→V2', '/root/medic_data/vascmamba/generalization_outputs/model.pt', VascMambaHybrid,
         test_idx_v2, y_all[test_idx_v2].numpy(), 0.55),
        ('V2→External', '/root/medic_data/vascmamba/generalization_outputs/model.pt', VascMambaHybrid,
         ext_idx, y_all[ext_idx].numpy(), 0.55),
        ('External→External', '/root/medic_data/vascmamba/generalization_outputs_reverse/model.pt', VascMambaHybrid,
         test_idx_ext, y_all[test_idx_ext].numpy(), 0.25),
        ('External→V2', '/root/medic_data/vascmamba/generalization_outputs_reverse/model.pt', VascMambaHybrid,
         v2_idx, y_all[v2_idx].numpy(), 0.25),
        ('DANN→V2', '/root/medic_data/vascmamba/dann_outputs/model.pt', VascMambaDANN,
         val_idx_dann, y_all[val_idx_dann].numpy(), 0.55),
        ('DANN→External', '/root/medic_data/vascmamba/dann_outputs/model.pt', VascMambaDANN,
         ext_idx, y_all[ext_idx].numpy(), 0.55),
    ]

    rows = []
    for name, model_path, model_cls, idx, y_true, fixed_t in experiments:
        print(f'Evaluating: {name}')
        model = model_cls(d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        probs = get_probs(model, X_bmode[idx], X_ulm[idx], X_density[idx])
        auc = roc_auc_score(y_true, probs)

        fixed_m = metrics_with_threshold(y_true, probs, fixed_t)
        dyn_t_acc, _ = find_best_threshold(y_true, probs, criterion='acc')
        dyn_m_acc = metrics_with_threshold(y_true, probs, dyn_t_acc)
        dyn_t_youden, _ = find_best_threshold(y_true, probs, criterion='youden')
        dyn_m_youden = metrics_with_threshold(y_true, probs, dyn_t_youden)

        rows.append({
            'Experiment': name,
            'N': len(y_true),
            'AUC': f'{auc:.4f}',
            'ACC_fixed': f'{fixed_m["ACC"]:.4f}',
            'ACC_dyn(ACC)': f'{dyn_m_acc["ACC"]:.4f}',
            'ACC_dyn(Youden)': f'{dyn_m_youden["ACC"]:.4f}',
            'Sens_fixed': f'{fixed_m["Sensitivity"]:.4f}',
            'Sens_dyn(ACC)': f'{dyn_m_acc["Sensitivity"]:.4f}',
            'Spec_fixed': f'{fixed_m["Specificity"]:.4f}',
            'Spec_dyn(ACC)': f'{dyn_m_acc["Specificity"]:.4f}',
            'F1_fixed': f'{fixed_m["F1"]:.4f}',
            'F1_dyn(ACC)': f'{dyn_m_acc["F1"]:.4f}',
            'Thr_fixed': f'{fixed_t:.2f}',
            'Thr_dyn(ACC)': f'{dyn_t_acc:.2f}',
            'Thr_dyn(Youden)': f'{dyn_t_youden:.2f}',
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'comparison_table.csv'), index=False)

    # Create JPG table
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.axis('tight')
    ax.axis('off')

    # Select a readable subset of columns for the main image
    display_cols = [
        'Experiment', 'N', 'AUC', 'ACC_fixed', 'ACC_dyn(ACC)',
        'Sens_fixed', 'Sens_dyn(ACC)', 'Spec_fixed', 'Spec_dyn(ACC)',
        'F1_fixed', 'Thr_fixed', 'Thr_dyn(ACC)'
    ]
    display_df = df[display_cols]
    table = ax.table(cellText=display_df.values,
                     colLabels=display_df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.12]*len(display_cols))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('VascMamba Generalization Comparison: Fixed vs Dynamic Threshold', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'comparison_table.jpg'), dpi=200, bbox_inches='tight')
    plt.close()

    # Also save a full table as JPG
    fig, ax = plt.subplots(figsize=(24, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.07]*len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    plt.title('VascMamba Generalization Full Comparison', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'comparison_table_full.jpg'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f'\nSaved comparison table and JPGs to {OUT_DIR}/')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
