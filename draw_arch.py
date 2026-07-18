"""VascMamba-Hybrid architecture — clean schematic."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow
import os

for fp in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
           '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']:
    if os.path.exists(fp):
        import matplotlib.font_manager as fm
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(24, 16))
ax.set_xlim(0, 24); ax.set_ylim(0, 16); ax.axis('off')

# ──── color palette
C = {'bc':'#1565c0','proj':'#0d47a1','bmode':'#e3f2fd','ulm':'#e0f2f1',
     'sort':'#ef6c00','mamba':'#00695c','ssm':'#26a69a','head':'#c62828',
     'bg':  '#eceff1','arrow':'#78909c','text':'#263238','gold':'#f9a825',
     'pos':'#c6282815','mod':'#ef6c0015','out':'#1b5e20'}

def box(x, y, w, h, txt, c, fs=9, tc='white', bw=1):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.1',
                        fc=c, ec='#263238', lw=bw, alpha=0.95)
    ax.add_patch(b)
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs, color=tc, fontweight='bold' if bw>0.5 else 'normal')

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.5))

def label(x, y, txt, fs=9, c=C['text'], ha='center'):
    ax.text(x, y, txt, fontsize=fs, color=c, ha=ha, va='center')

def vline(x, y1, y2):
    ax.plot([x,x], [y1,y2], color=C['arrow'], lw=1, linestyle=':')

# ═══════ TITLE ═══════
label(12, 15.2, 'VascMamba-Hybrid', fs=18, c=C['bc'])
label(12, 14.6, 'BiomedCLIP (Frozen) + Vessel-Guided Mamba Head  |  50K Trainable', fs=11, c=C['arrow'])

# ═══════ ROW 1: INPUT IMAGES ═══════
y1 = 13.5
label(12, y1+0.4, 'Input: 4 ULM Composite Views per Patient', fs=10, c=C['text'])
# B-mode images
for i, yi in enumerate([y1+0.1,y1-0.3]):
    box(4.5+i*1.8, yi, 3.4, 0.45, f'B-mode View {i+1}', C['bmode'], 8, C['bc'], 0.5)
    box(14.5+i*1.8, yi, 3.4, 0.45, f'ULM View {i+1}', C['ulm'], 8, C['mamba'], 0.5)
label(12, y1-1.1, '... × 4 views ...', fs=8, c=C['arrow'])

# ═══════ ROW 2: BIOMEDCLIP ═══════
y2 = 12.0
label(12, y2+0.5, 'BiomedCLIP ViT-B/16  (Frozen, 86M params, pretrained on PMC-15M medical images)', fs=11, c=C['bc'])
box(6, y2-0.1, 7, 0.6, 'encode_image(B-mode)  →  512D', C['bmode'], 9, C['bc'], 0.5)
box(18, y2-0.1, 7, 0.6, 'encode_image(ULM)     →  512D', C['ulm'], 9, C['bc'], 0.5)
arrow(4.5, y1-1.2, 6, y2+0.2)
arrow(5, y1-1.2, 6, y2+0.2)
arrow(14.5, y1-1.2, 18, y2+0.2)
arrow(15, y1-1.2, 18, y2+0.2)

# ═══════ ROW 2b: FEATURE VECTORS ═══════
y2b = 11.0
for i in range(4):
    box(3+i*2.2, y2b, 1.9, 0.5, f'B{i+1} 512D', C['bmode'], 7.5, C['bc'], 0.5)
    box(15.2+i*2.2, y2b, 1.9, 0.5, f'U{i+1} 512D', C['ulm'], 7.5, C['bc'], 0.5)
arrow(6, y2-0.4, 3+1*2.2, y2b+0.25)
arrow(18, y2-0.4, 15.2+1*2.2, y2b+0.25)

# ═══════ ROW 3: VESSEL-GUIDED SORTING ═══════
y3 = 9.8
label(18.5, y3+0.4, 'Vessel-Guided Ordering  (Δ=+1.25%)', fs=10, c=C['sort'])
box(18.5, y3-0.1, 8.5, 0.55,
    'Sort U1..U4 by vessel_density descending', C['sort'], 9)

# ═══════ ROW 4: TOKEN CONSTRUCTION ═══════
y4 = 8.5
label(12, y4+0.4, 'Token Sequence: 8 tokens, 32D each, B1-U1-B2-U2-B3-U3-B4-U4  (interleaved)', fs=10, c=C['text'])

token_c = [C['bmode'], C['ulm']]*4
token_t = ['B1','U1','B2','U2','B3','U3','B4','U4']
for i in range(8):
    box(3.2+i*2.2, y4-0.1, 1.9, 0.55, f'{token_t[i]} 32D', token_c[i], 8,
        C['bc'] if i%2==0 else C['mamba'], 0.5)

# Drop attributes onto first token as example
ax.annotate('+ pos_emb[i]', xy=(3.2, y4-0.7), fontsize=7, color='#c62828', ha='center')
ax.annotate('+ mod_emb[B/U]', xy=(3.2, y4-1.0), fontsize=7, color='#ef6c00', ha='center')

# Arrow from sorting to tokens
arrow(18.5, y3-0.4, 15.2+(1.5)*2.2, y4+0.2)  # ULM tokens
for i in range(4):
    arrow(3+i*2.2, y2b-0.25, 3.2+i*2*2.2, y4+0.2)  # B-mode

# ═══════ ROW 5: MAMBA BLOCK ═══════
y5 = 6.5
# Background
bg = FancyBboxPatch((0.8, 5.3), 22.4, 2.2, boxstyle='round,pad=0.2',
                     fc=C['mamba'], ec='none', alpha=0.06)
ax.add_patch(bg)
label(12, y5+0.5, 'MambaBlock  (1 layer, d=32, state=4, 16K params)', fs=11, c=C['mamba'])

# Flow inside Mamba block
x_m = 2
for name, c, w in [('LayerNorm', C['mamba'], 2.5), ('SelectiveSSM', C['ssm'], 3.5),
                    ('LayerNorm', C['mamba'], 2.5), ('FFN(GELU)', C['mamba'], 3)]:
    box(x_m+w/2, y5-0.3, w, 0.65, name, c, 9)
    arrow(x_m+w+0.05, y5-0.3, x_m+w+0.35, y5-0.3)
    x_m += w + 0.4

# Residual connections
ax.annotate('', xy=(4.5+2.5+0.2, y5-0.5), xytext=(4.5+2.5+0.2, y5-1.3),
            arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1, ls=':'))
ax.annotate('', xy=(4.5+2.5+0.4+3.5+0.4+2.5+0.2, y5-0.5),
            xytext=(4.5+2.5+0.4+3.5+0.4+2.5+0.2, y5-1.3),
            arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1, ls=':'))
ax.text(7.4, y5-1.2, '+', fontsize=11, color=C['arrow'], ha='center')
ax.text(13.8, y5-1.2, '+', fontsize=11, color=C['arrow'], ha='center')

# SSM equation
label(12, y5-1.8, 'Selective SSM:   h = exp(Δ·A)·h  +  Δ·B·x ,   y = h·C',
      fs=9, c=C['ssm'])

# Arrow from tokens to Mamba
arrow(12, y4-0.5, 2.5, y5+.5)

# ═══════ ROW 6: CLASSIFICATION HEAD ═══════
y6 = 4.0
box(6, y6, 3, 0.55, 'Mean Pool (8×32→32)', C['mamba'], 9)
box(10.5, y6, 3, 0.55, 'LayerNorm', C['mamba'], 9)
box(14.5, y6, 3.5, 0.55, 'Linear(32→2)', C['head'], 9)
arrow(12, y5-2.3, 6, y6+0.3)
arrow(6+1.5, y6, 10.5-1.5, y6)
arrow(10.5+1.5, y6, 14.5-1.75, y6)

# ═══════ OUTPUT ═══════
y7 = 2.8
box(14.5, y7, 3.5, 0.6, 'P(Benign | Malignant)', C['out'], 10)

# ═══════ INNOVATION CALLOUTS ═══════
# Innovation 1
ax.annotate('1  Vessel-Guided Scan Order', xy=(18.5, 9.5), fontsize=8,
            color=C['sort'], fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', fc='white', ec=C['sort'], alpha=0.9))
ax.annotate('Mamba scans ULM tokens by\ndescending vessel density —\nmimics clinical scan path',
            xy=(18.5, 8.9), fontsize=7, color=C['text'], ha='center')

# Innovation 2
ax.annotate('2  Selective SSM Head', xy=(2.5, 5.8), fontsize=8,
            color=C['ssm'], fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', fc='white', ec=C['ssm'], alpha=0.9))
ax.annotate('Content-aware state updates:\nΔ_t, B_t, C_t adapt to each\ntoken\'s modality and position',
            xy=(2.5, 5.2), fontsize=7, color=C['text'], ha='center')

# Innovation 3
ax.annotate('3  50K Params, Single Model', xy=(14.5, 3.5), fontsize=8,
            color=C['head'], fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', fc='white', ec=C['head'], alpha=0.9))
ax.annotate('Matches 5-stream VTG-Net\n(0.8798 vs 0.8792)\nwith 1/5 the streams',
            xy=(14.5, 2.9), fontsize=7, color=C['text'], ha='center')

# ═══════ PARAM TABLE ═══════
tx, ty = 1.5, 13.5
label(tx, ty+0.3, 'Params', fs=9, c=C['text'], ha='left')
rows = [('BiomedCLIP ViT-B', '86M', 'Frozen'),
        ('Proj 512→32 (x8)', '33K', 'Train'),
        ('Pos Emb 8×32', '256', 'Train'),
        ('Mod Emb 2×32',  '64',  'Train'),
        ('Mamba SSM+FFN', '16K', 'Train'),
        ('Head 32→2',    '70',  'Train'),
        ('Total',         '~50K','')]
for i,(n,p,s) in enumerate(rows):
    yi = ty - i*0.42
    ax.text(tx, yi, n, fontsize=7.5, color=C['text'], fontweight='bold' if 'Total' in n else 'normal')
    ax.text(tx+5.5, yi, p, fontsize=7.5, color=C['out'] if 'Total' in n else C['bc'])
    ax.text(tx+7.5, yi, s, fontsize=7, color=C['arrow'])

# Save
out = '/root/medic_data/training_records/live/vascmamba_architecture.png'
plt.tight_layout(pad=0.3)
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')
