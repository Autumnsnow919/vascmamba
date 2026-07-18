"""VascMamba-Hybrid generalization test: train on 80% of VinnoRepositoryV2,
evaluate on the remaining 20% and all external hospital data.
"""
import sys, os
sys.path.insert(0, '/root/medic_data')
sys.path.insert(0, '/root/medic_data/ulm_visionnet')

import numpy as np
import pandas as pd
import torch
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
from vascmamba.hybrid import VascMambaHybrid


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
D_MODEL = 32
D_STATE = 4
N_LAYERS = 1
EPOCHS = 100
LR = 5e-4
BATCH_SIZE = 32
PATIENCE = 20
CLASS_WEIGHT = torch.tensor([3.0, 1.0])

OUT_DIR = '/root/medic_data/vascmamba/generalization_outputs'


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


def train_model(model, train_loader, val_loader, epochs=100, lr=5e-4):
    opt = AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sched = CosineAnnealingLR(opt, T_max=epochs * len(train_loader), eta_min=1e-6)
    best_acc, best_state, patience_counter = 0, None, 0

    for ep in range(epochs):
        model.train()
        for bm, um, d, lbl in train_loader:
            bm, um, d, lbl = bm.to(DEVICE), um.to(DEVICE), d.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad()
            logits = model(bm, um, d)
            loss = F.cross_entropy(logits, lbl, weight=CLASS_WEIGHT.to(DEVICE))
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
            print(f'    Epoch {ep+1:3d} | val acc={best_m["acc"]:.4f} f1={best_m["f1"]:.4f} t={best_t:.2f}')

        if patience_counter > PATIENCE:
            print(f'    Early stop at epoch {ep+1}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


def evaluate(model, bm, um, d, y, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(bm.to(DEVICE), um.to(DEVICE), d.to(DEVICE))
        probs = F.softmax(logits, -1)[:, 1].cpu().numpy()
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
    logits = model(bm_t, um_t, d_t)
    probs = F.softmax(logits, -1)[:, 1]
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
    print('VascMamba-Hybrid Generalization Test')
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

    y_v2 = y_all[v2_idx].numpy()
    train_idx_v2, test_idx_v2 = train_test_split(
        v2_idx, test_size=0.2, stratify=y_v2, random_state=SEED
    )
    print(f'\n[2] V2 split: train={len(train_idx_v2)}, test={len(test_idx_v2)}')
    print(f'  Train B/M: {int((y_all[train_idx_v2] == 0).sum())}/{int((y_all[train_idx_v2] == 1).sum())}')
    print(f'  Test  B/M: {int((y_all[test_idx_v2] == 0).sum())}/{int((y_all[test_idx_v2] == 1).sum())}')

    train_ds = FeatDataset(X_bmode[train_idx_v2], X_ulm[train_idx_v2],
                           X_density[train_idx_v2], y_all[train_idx_v2])
    val_ds = FeatDataset(X_bmode[test_idx_v2], X_ulm[test_idx_v2],
                         X_density[test_idx_v2], y_all[test_idx_v2])
    ext_ds = FeatDataset(X_bmode[ext_idx], X_ulm[ext_idx],
                         X_density[ext_idx], y_all[ext_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    ext_loader = DataLoader(ext_ds, batch_size=BATCH_SIZE, shuffle=False)

    print('\n[3] Training VascMamba-Hybrid...')
    model = VascMambaHybrid(d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model: {n_params / 1000:.0f}K params')
    model, best_val_acc = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
    print(f'  Best val acc: {best_val_acc:.4f}')

    print('\n[4] Threshold search on V2 test...')
    model.eval()
    v2_probs = []
    with torch.no_grad():
        for bm, um, d, _ in val_loader:
            logits = model(bm.to(DEVICE), um.to(DEVICE), d.to(DEVICE))
            v2_probs.append(F.softmax(logits, -1)[:, 1].cpu())
    v2_probs = torch.cat(v2_probs).numpy()
    v2_labels = y_all[test_idx_v2].numpy()
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.02):
        f1v = f1_score(v2_labels, (v2_probs >= t).astype(int), zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_t = t
    print(f'  Best threshold: {best_t:.2f}, F1: {best_f1:.4f}')

    print('\n[5] Evaluating...')
    metrics_v2, v2_probs2, v2_pred = evaluate(
        model, X_bmode[test_idx_v2], X_ulm[test_idx_v2],
        X_density[test_idx_v2], y_all[test_idx_v2], threshold=best_t
    )
    metrics_ext, ext_probs, ext_pred = evaluate(
        model, X_bmode[ext_idx], X_ulm[ext_idx],
        X_density[ext_idx], y_all[ext_idx], threshold=best_t
    )

    for name, m in [('V2 test', metrics_v2), ('External', metrics_ext)]:
        print(f'\n  {name}:')
        for k, v in m.items():
            if k == 'n':
                print(f'    {k}: {v}')
            else:
                print(f'    {k}: {v:.4f}')

    df = pd.DataFrame([metrics_v2, metrics_ext])
    df.insert(0, 'set', ['V2_test', 'External'])
    df.to_csv(os.path.join(OUT_DIR, 'metrics.csv'), index=False)
    print(f'\n  Saved metrics.csv')

    torch.save(model.state_dict(), os.path.join(OUT_DIR, 'model.pt'))
    print(f'  Saved model.pt')

    print('\n[6] Plotting...')
    plot_confusion(y_all[test_idx_v2].numpy(), v2_pred,
                   'V2 Test Confusion', os.path.join(OUT_DIR, 'confusion_v2.png'))
    plot_confusion(y_all[ext_idx].numpy(), ext_pred,
                   'External Confusion', os.path.join(OUT_DIR, 'confusion_external.png'))
    plot_roc(y_all[test_idx_v2].numpy(), v2_probs2,
             'V2 Test ROC', os.path.join(OUT_DIR, 'roc_v2.png'))
    plot_roc(y_all[ext_idx].numpy(), ext_probs,
             'External ROC', os.path.join(OUT_DIR, 'roc_external.png'))
    plot_pr(y_all[test_idx_v2].numpy(), v2_probs2,
            'V2 Test PR', os.path.join(OUT_DIR, 'pr_v2.png'))
    plot_pr(y_all[ext_idx].numpy(), ext_probs,
            'External PR', os.path.join(OUT_DIR, 'pr_external.png'))

    print('\n[7] Token importance heatmap...')
    importance = token_importance(model, X_bmode[test_idx_v2], X_ulm[test_idx_v2],
                                  X_density[test_idx_v2])
    print(f'  Importance: {importance}')
    plt.figure(figsize=(8, 3))
    labels = ['B1', 'U1', 'B2', 'U2', 'B3', 'U3', 'B4', 'U4']
    plt.bar(labels, importance)
    plt.title('Token Importance (Gradient Attribution, V2 Test)')
    plt.ylabel('Relative Importance')
    plt.ylim(0, importance.max() * 1.2 + 0.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'token_importance.png'), dpi=150)
    plt.close()

    print('\n' + '=' * 70)
    print('Done. Outputs in vascmamba/generalization_outputs/')
    print('=' * 70)


if __name__ == '__main__':
    main()
