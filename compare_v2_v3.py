import pandas as pd
import os

OUT_DIR = '/root/medic_data/vascmamba/rad_outputs'

v2 = pd.read_csv(os.path.join(OUT_DIR, 'rad_cls_patch_fast_best_acc.csv'))
v3 = pd.read_csv(os.path.join(OUT_DIR, 'rad_cls_patch_fast_v3_best_acc.csv'))

v2 = v2[['experiment', 'ACC', 'AUC', 'Sensitivity', 'Specificity', 'F1']].copy()
v3 = v3[['experiment', 'ACC', 'AUC', 'Sensitivity', 'Specificity', 'F1']].copy()

v2.columns = ['experiment'] + [f'{c}_v2' for c in v2.columns[1:]]
v3.columns = ['experiment'] + [f'{c}_v3' for c in v3.columns[1:]]

cmp = v2.merge(v3, on='experiment')
print(cmp.to_string(index=False))

cmp.to_csv(os.path.join(OUT_DIR, 'rad_v2_vs_v3_comparison.csv'), index=False)
print(f'\nSaved comparison to {os.path.join(OUT_DIR, "rad_v2_vs_v3_comparison.csv")}')

# Calculate average improvement
for col in ['ACC', 'AUC', 'Sensitivity', 'Specificity', 'F1']:
    diff = cmp[f'{col}_v3'] - cmp[f'{col}_v2']
    print(f'{col}: mean Δ = {diff.mean():.4f}, median Δ = {diff.median():.4f}')
