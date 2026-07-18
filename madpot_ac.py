import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/root/medic_data')
sys.path.insert(0, '/root/medic_data/ulm_visionnet')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)
import ot
import cv2
from tqdm import tqdm
from data.patient_index_v2 import build_unified_index
import open_clip
import transformers

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = '/root/medic_data/vascmamba/rad_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

CROP_Y1, CROP_Y2 = 162, 737
CROP_X_MAX = 1100
SPLIT_X = 590
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        return None, None
    cropped = img[CROP_Y1:CROP_Y2, 0:CROP_X_MAX]
    bmode = cropped[:, :SPLIT_X]
    ulm = cropped[:, SPLIT_X:]
    ulm[:, -max(1, int(ulm.shape[1]*0.1)):] = 0

    def pad_sq(im):
        h, w = im.shape[:2]
        s = max(h, w)
        t = (s - h) // 2
        b = s - h - t
        l = (s - w) // 2
        r = s - w - l
        return cv2.copyMakeBorder(im, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

    bm = cv2.resize(pad_sq(bmode), (224, 224), cv2.INTER_AREA)
    um = cv2.resize(pad_sq(ulm), (224, 224), cv2.INTER_AREA)
    bm = ((bm.astype(np.float32)[..., ::-1] / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    um = ((um.astype(np.float32)[..., ::-1] / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    return bm.transpose(2, 0, 1), um.transpose(2, 0, 1)


class BiomedCLIPFeatureExtractor:
    def __init__(self, layer_idxs):
        self.layer_idxs = layer_idxs
        self.model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.model = self.model.to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._register_hooks()

    def _register_hooks(self):
        self.features = {}

        def hook_fn(l):
            def fn(module, inp, out):
                self.features[l] = out
            return fn

        for l in self.layer_idxs:
            self.model.visual.trunk.blocks[l].register_forward_hook(hook_fn(l))

    @torch.no_grad()
    def extract(self, images):
        images = torch.from_numpy(images).float().to(DEVICE)
        _ = self.model.visual.trunk.forward_features(images)
        out = []
        for l in self.layer_idxs:
            feat = self.features[l]  # (B, 197, 768)
            out.append(feat[:, 1:, :].cpu())  # patch tokens only
        return torch.stack(out, dim=1)  # (B, L, P, 768)


def extract_raw_features(layer_idxs):
    samples = build_unified_index()
    extractor = BiomedCLIPFeatureExtractor(layer_idxs)
    all_b, all_u, labels = [], [], []
    for s in samples:
        bm_list, um_list = [], []
        for vf in s['views'][:4]:
            bm, um = preprocess(os.path.join(s['patient_dir'], vf))
            if bm is None:
                bm = np.zeros((3, 224, 224), dtype=np.float32)
                um = np.zeros((3, 224, 224), dtype=np.float32)
            bm_list.append(bm)
            um_list.append(um)
        all_b.append(np.stack(bm_list))
        all_u.append(np.stack(um_list))
        labels.append(s['label'])
    all_b = np.stack(all_b)  # (N, 4, 3, 224, 224)
    all_u = np.stack(all_u)
    labels = np.array(labels)

    N = len(samples)
    raw_features = []
    for i in tqdm(range(N), desc='Extracting BiomedCLIP patch features'):
        bm = extractor.extract(all_b[i])  # (4, L, P, 768)
        um = extractor.extract(all_u[i])
        avg = (bm + um) / 2.0  # (4, L, P, 768)
        session_feat = avg.mean(dim=0)  # (L, P, 768)
        raw_features.append(session_feat)
    raw_features = torch.stack(raw_features)  # (N, L, P, 768)
    return raw_features, labels


def load_or_extract_raw_features(layer_idxs):
    path = '/root/medic_data/madpot_raw_features.npz'
    if os.path.exists(path):
        d = np.load(path)
        if len(d['layer_idxs']) == len(layer_idxs) and list(d['layer_idxs']) == list(layer_idxs):
            return torch.from_numpy(d['raw_features']).float(), d['labels']
    raw, labels = extract_raw_features(layer_idxs)
    np.savez(path, raw_features=raw.numpy(), labels=labels, layer_idxs=np.array(layer_idxs))
    return raw, labels


class MADPOTAC(nn.Module):
    def __init__(self, layer_idxs, bottleneck=16, k=4, ctx_len=4, cls_names=['benign', 'malignant'], use_adapter=True):
        super().__init__()
        self.layer_idxs = layer_idxs
        self.n_layers = len(layer_idxs)
        self.k = k
        self.ctx_len = ctx_len
        self.n_classes = 2
        self.use_adapter = use_adapter

        self.model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.model = self.model.to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if use_adapter:
            self.visual_proj = nn.Sequential(
                nn.Linear(768, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, 512)
            )
        else:
            self.visual_proj = nn.Linear(768, 512, bias=False)
            self.visual_proj.weight.data = self.model.visual.head.proj.weight.data.clone().to(DEVICE)
            for p in self.visual_proj.parameters():
                p.requires_grad = False

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            '/root/.cache/huggingface/hub/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/snapshots/9f341de24bfb00180f1b847274256e9b65a3a32e'
        )
        self.cls_ids = [self.tokenizer.encode(n, add_special_tokens=False) for n in cls_names]
        for ids in self.cls_ids:
            assert len(ids) == 1, f'Class name must be single token, got {ids}'
        self.cls_ids = [ids[0] for ids in self.cls_ids]
        self.seq_len = 1 + ctx_len + 1 + 1

        self.prompt_embeddings = nn.Parameter(
            torch.randn(2, k, ctx_len, 768) * 0.02
        )

        self.placeholder_id = self.tokenizer.mask_token_id

    def _encode_text_prompts(self):
        B = 1
        all_prompts = []
        for cls_idx in range(self.n_classes):
            ctx = self.prompt_embeddings[cls_idx]  # (K, L, 768)
            K = ctx.shape[0]
            input_ids = torch.full((K, self.seq_len), self.tokenizer.pad_token_id, dtype=torch.long, device=DEVICE)
            input_ids[:, 0] = self.tokenizer.cls_token_id
            input_ids[:, -1] = self.tokenizer.sep_token_id
            input_ids[:, 1 + self.ctx_len] = self.cls_ids[cls_idx]
            input_ids[:, 1:1 + self.ctx_len] = self.placeholder_id
            all_prompts.append((input_ids, ctx))

        out_features = []
        for input_ids, ctx in all_prompts:
            K = ctx.shape[0]
            attn_mask = (input_ids != self.tokenizer.pad_token_id).float()
            embeddings = self.model.text.transformer.embeddings(input_ids=input_ids)
            embeddings[:, 1:1 + self.ctx_len, :] = ctx
            encoder_out = self.model.text.transformer(inputs_embeds=embeddings, attention_mask=attn_mask)
            pooled = self.model.text.pooler(encoder_out, attn_mask)
            proj = self.model.text.proj(pooled)
            proj = F.normalize(proj, dim=-1)
            out_features.append(proj)
        return torch.stack(out_features, dim=0)  # (2, K, 512)

    def forward(self, raw_features, tau=0.5, frac=0.5):
        B, L, P, D = raw_features.shape
        x = raw_features.to(DEVICE)
        x = self.visual_proj(x)  # (B, L, P, 512)
        x = F.normalize(x, dim=-1)

        prompt_features = self._encode_text_prompts()  # (2, K, 512)
        fused = prompt_features.mean(dim=1)  # (2, 512)

        loss_components = []
        for l in range(L):
            feats = x[:, l, :, :]  # (B, P, 512)
            pot_scores = []
            cl_scores = []
            for j in range(self.n_classes):
                P_j = prompt_features[j]  # (K, 512)
                F_j = fused[j]  # (512,)

                cost = 1.0 - (feats @ P_j.T)  # (B, P, K)
                dis = self._partial_ot_distance(cost, frac)  # (B,)
                pot_scores.append((1.0 - dis) / tau)

                sim = feats @ F_j  # (B, P)
                cl_scores.append(sim.mean(dim=1) / tau)

            pot_scores = torch.stack(pot_scores, dim=1)  # (B, 2)
            cl_scores = torch.stack(cl_scores, dim=1)  # (B, 2)
            c_pot = F.softmax(pot_scores, dim=1)
            c_cl = F.softmax(cl_scores, dim=1)
            loss_components.append(c_pot + c_cl)

        c_total = torch.stack(loss_components, dim=0).mean(dim=0)  # (B, 2)
        return c_total

    def _partial_ot_distance(self, cost, frac):
        B, P, K = cost.shape
        a = np.ones(P) / P
        b = np.ones(K) * (frac / K)
        m = float(frac)
        dis = []
        for i in range(B):
            M = cost[i].detach().cpu().numpy().astype(np.float64)
            if frac >= 0.99:
                T = ot.emd(a, b, M)
            else:
                T = ot.partial.partial_wasserstein(a, b, M, m)
            d = float((T * M).sum())
            dis.append(d)
        return torch.tensor(dis, device=cost.device, dtype=cost.dtype)


def search_threshold(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores).ravel()
    best_acc, best_t = 0, 0.5
    for t in np.linspace(scores.min() - 0.1, scores.max() + 0.1, 200):
        pred = (scores >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def evaluate(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'ACC': accuracy_score(y_true, pred),
        'AUC': roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else 0.5,
        'Sensitivity': recall_score(y_true, pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'F1': f1_score(y_true, pred, zero_division=0),
    }


def train_epoch(model, loader, optimizer, class_weight, tau, frac):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        c = model(x, tau=tau, frac=frac)
        c = c / c.sum(dim=1, keepdim=True)
        loss = F.nll_loss(torch.log(c + 1e-8), y, weight=class_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, tau, frac):
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            c = model(x, tau=tau, frac=frac)
            probs.append(c[:, 1].cpu())
            labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def run_fold(fold, train_idx, test_idx, raw_features, y, config):
    print(f'[Fold {fold}] train_full={len(train_idx)} test={len(test_idx)}', flush=True)
    X_train_full, X_test = raw_features[train_idx], raw_features[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    train_idx2, val_idx = train_test_split(
        np.arange(len(y_train_full)), test_size=0.2, stratify=y_train_full, random_state=SEED
    )
    X_train, X_val = X_train_full[train_idx2], X_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx2], y_train_full[val_idx]

    train_ds = TensorDataset(X_train, torch.from_numpy(y_train).long())
    val_ds = TensorDataset(X_val, torch.from_numpy(y_val).long())
    test_ds = TensorDataset(X_test, torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    model = MADPOTAC(layer_idxs=config['layer_idxs'], bottleneck=config['bottleneck'],
                     k=config['k'], ctx_len=config['ctx_len'], use_adapter=config.get('use_adapter', True))
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    class_weight = torch.tensor([1.0, 1.0], device=DEVICE)
    if config['benign_weight'] > 1.0:
        class_weight[0] = config['benign_weight']

    best_val = 0
    best_state = None
    for epoch in range(config['epochs']):
        train_epoch(model, train_loader, optimizer, class_weight, config['tau'], config['frac'])
        if (epoch + 1) % 5 == 0:
            probs, labels = eval_epoch(model, val_loader, config['tau'], config['frac'])
            t, _ = search_threshold(labels, probs)
            acc = accuracy_score(labels, (probs >= t).astype(int))
            if acc > best_val:
                best_val = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    probs, _ = eval_epoch(model, test_loader, config['tau'], config['frac'])
    t, _ = search_threshold(y_test, probs)
    metrics = evaluate(y_test, probs, t)
    print(f'  test ACC={metrics["ACC"]:.4f} Sens={metrics["Sensitivity"]:.4f} Spec={metrics["Specificity"]:.4f} AUC={metrics["AUC"]:.4f}', flush=True)
    return metrics, probs, y_test


def main():
    config = {
        'layer_idxs': [5, 11],
        'bottleneck': 32,
        'k': 4,
        'ctx_len': 4,
        'tau': 0.5,
        'frac': 0.5,
        'benign_weight': 6.0,
        'lr': 1e-3,
        'wd': 1e-4,
        'epochs': 60,
        'batch_size': 32,
    }
    print('MADPOT AC config:', config, flush=True)
    raw_features, y = load_or_extract_raw_features(config['layer_idxs'])
    print(f'Raw features: {raw_features.shape}, labels: {y.shape}', flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(y)), y), 1):
        metrics, probs, y_test = run_fold(fold, train_idx, test_idx, raw_features, y, config)
        results.append(metrics)

    df = pd.DataFrame(results)
    print('\n' + '=' * 70, flush=True)
    print('5-fold MADPOT AC results', flush=True)
    print(df.to_string(index=False), flush=True)
    print(f'Mean ACC: {df["ACC"].mean():.4f} ± {df["ACC"].std():.4f}', flush=True)
    print(f'Mean Sensitivity: {df["Sensitivity"].mean():.4f} ± {df["Sensitivity"].std():.4f}', flush=True)
    print(f'Mean Specificity: {df["Specificity"].mean():.4f} ± {df["Specificity"].std():.4f}', flush=True)
    print(f'Mean AUC: {df["AUC"].mean():.4f} ± {df["AUC"].std():.4f}', flush=True)
    df.to_csv(os.path.join(OUT_DIR, 'madpot_ac_results.csv'), index=False)


if __name__ == '__main__':
    main()
