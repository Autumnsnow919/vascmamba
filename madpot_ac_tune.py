import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.insert(0, '/root/medic_data')
sys.path.insert(0, '/root/medic_data/ulm_visionnet')
sys.path.insert(0, '/root/medic_data/vascmamba')
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from madpot_ac import load_or_extract_raw_features, run_fold, SEED

raw_features, y = load_or_extract_raw_features([5, 11])
print(f'Loaded features: {raw_features.shape}, labels: {y.shape}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(np.arange(len(y)), y))
train_idx, test_idx = folds[0]

base = {
    'layer_idxs': [5, 11],
    'bottleneck': 16,
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

configs = [
    {'name': 'baseline', **base},
    {'name': 'layer_last', 'layer_idxs': [11], **{k: v for k, v in base.items() if k != 'layer_idxs'}},
    {'name': 'frac0.2', **{**base, 'frac': 0.2}},
    {'name': 'frac1.0', **{**base, 'frac': 1.0}},
    {'name': 'tau0.1', **{**base, 'tau': 0.1}},
    {'name': 'tau1.0', **{**base, 'tau': 1.0}},
    {'name': 'bw10', **{**base, 'benign_weight': 10.0}},
    {'name': 'bottleneck8', **{**base, 'bottleneck': 8}},
    {'name': 'no_adapter', **{**base, 'use_adapter': False}},]

results = []
for cfg in configs:
    print(f'\n=== Config {cfg["name"]} ===', flush=True)
    metrics, probs, y_test = run_fold(cfg['name'], train_idx, test_idx, raw_features, y, cfg)
    metrics['config'] = cfg['name']
    results.append(metrics)

df = pd.DataFrame(results)
print('\n' + '=' * 70)
print('Tuning results on fold 1')
print(df.to_string(index=False))
print('=' * 70)
