# VascMamba

乳腺 ULM（超声定位显微）/ B-mode 双模态超声良恶性二分类：冻结的
BiomedCLIP **真实逐视图特征** + 轻量 selective SSM 分类头。

## 当前推荐流程

当前 `hybrid.py` 不再把患者级平均特征复制成四个伪视图。默认使用
`fusion_v2`双路径网络：

```text
4个配对视图 (B1,U1) ... (B4,U4)
  → BiomedCLIP checkpoint自带的预处理
  → 每张图一个L2归一化的512维全局embedding
  → X_bmode/X_ulm = (N,4,512)
  ├─ 全局路径：masked mean(B/U原始特征) → 线性分类直连
  └─ 逐视图路径：B、U、|B-U|、B×U、density → 4个配对token
       → 配对视图按density同步排序
       → 共享权重的正向/反向Mamba
       → attention pooling + mean pooling
       → 与B/U全局投影、density统计拼接
  → 两条路径logits相加 → 2类输出
```

`fusion_v2`默认按density同步排列B-mode、ULM、density和valid mask，使Mamba
看到确定且可复现的序列。`--preserve-view-order`用于原始顺序消融。历史8-token
结构仍可通过`--architecture mamba_v1 --preserve-view-order`复现。

训练采用nested 5-fold cross-validation：

- 内层3-fold OOF：每名外层训练患者恰好参与一次epoch/分类阈值选择；
- 根据内层fold的最佳epoch取中位数，再用**完整外层训练集**重训，不再丢掉
  20%的外层训练样本；
- 每个外层fold默认训练3个不同随机种子的模型并平均概率，降低小样本方差；
- 外层测试折：只评价一次，不参与模型或阈值选择；
- 类别权重：仅根据当前训练集动态计算；主结果保留与旧实验相同的完全平衡
  权重，`--class-weight-power 0.5`仅作为预先声明的软化权重消融；
- 默认在内层OOF上选择F1最优阈值，并以Accuracy、Precision和Recall依次打破
  F1并列；如有预先定义的临床敏感度要求，可使用`--threshold-objective clinical
  --recall-floor 0.90`；
- 每折保存模型、阈值、划分索引和完整配置；
- 保存每名患者恰好一次的外层OOF概率，便于paired bootstrap和统计检验；
- 报告Accuracy、Balanced Accuracy、ROC-AUC、PR-AUC、Sensitivity、
  Specificity、F1和MCC。

## 数据格式

`hybrid.py` 接受一个NPZ：

| 数组 | shape | 说明 |
|---|---:|---|
| `X_bmode` | `(N,4,512)` | 4个真实B-mode视图的BiomedCLIP全局特征 |
| `X_ulm` | `(N,4,512)` | 与B-mode一一配对的4个ULM视图特征 |
| `density` | `(N,4)` | 每个ULM视图的血管密度，仅用于可选排序消融 |
| `valid` | `(N,4)` | 可选；视图是否有效。缺省时均视为有效 |
| `y` | `(N,)` | `0=良性，1=恶性` |

仓库提供 `extract_biomedclip_perview_features.py`。它使用BiomedCLIP checkpoint
返回的官方 `preprocess_val`，而不是手工套用ImageNet mean/std，并额外保存提取
元数据。私有数据需提供
`ulm_visionnet/data/patient_index_v2.build_unified_index()`。

## 运行

```bash
# 1. 离线提取真实逐视图特征
python extract_biomedclip_perview_features.py \
  --private-root /root/medic_data \
  --output /root/medic_data/biomedclip_perview_features.npz

# 2. 严格nested CV；默认不做人为density排序
python hybrid.py \
  --features /root/medic_data/biomedclip_perview_features.npz \
  --output-dir hybrid_nested_outputs

# 结构消融：上一版8-token Mamba
python hybrid.py \
  --features /root/medic_data/biomedclip_perview_features.npz \
  --output-dir hybrid_mamba_v1_outputs \
  --architecture mamba_v1 \
  --preserve-view-order

# 可选：预先规定恶性Recall至少0.90，再在可行阈值中最大化Accuracy
python hybrid.py \
  --features /root/medic_data/biomedclip_perview_features.npz \
  --output-dir hybrid_clinical_outputs \
  --threshold-objective clinical \
  --recall-floor 0.90

# 3. fusion_v2的原始视图顺序消融；必须与默认流程独立比较
python hybrid.py \
  --features /root/medic_data/biomedclip_perview_features.npz \
  --output-dir hybrid_original_order_outputs \
  --preserve-view-order
```

依赖：`torch`、`open_clip_torch`、`timm`、`transformers`、`scikit-learn`、
`opencv-python`、`Pillow`、`numpy`、`tqdm`。

## 历史结果与解释边界

仓库早期报告了以下内部5-fold结果。它们保留用于追溯，**不应与当前nested CV
结果直接比较，也不应表述为外部benchmark SOTA**：

| 历史方法 | Acc | AUC | Recall | F1 |
|---|---:|---:|---:|---:|
| BC-SVM（1024D session特征） | 0.8507 ± .042 | 0.8021 | 0.9778 | 0.9084 |
| Legacy VascMamba-Hybrid | 0.8798 ± .033 | 0.8002 | 0.9228 | 0.9193 |
| Pyramid VascMamba | 0.8715 ± .044 | 0.8139 | 0.9614 | 0.9186 |
| PatchVascMamba | 0.8673 ± .038 | 0.7995 | 0.9395 | 0.9138 |
| PatchVascMamba `--full` | 0.8591 ± .046 | 0.7963 | 0.9670 | 0.9120 |

历史 `Legacy VascMamba-Hybrid` 使用 `(N,1024)` 的患者级
`[B-mean(512), U-mean(512)]`，随后把两个平均向量各复制4次。其8-token序列只有
两个独立内容向量；平均density也被复制4次，因此所谓vessel-guided ordering对
该输入没有实际影响。当前入口不会再运行这条流程。

241例（历史数据为60良性/181恶性）的折间标准差较大。仅预测恶性即可得到
181/241=0.7510的Accuracy，因此Accuracy必须与Precision、Recall、Specificity、
F1和AUC联合解释。方法间小幅差异需要使用
相同外层OOF样本做paired bootstrap/统计检验，并最终在锁定阈值的外部医院数据
上验证。

## 其他实验文件

| 文件 | 说明 |
|---|---|
| `hybrid.py` | 当前推荐：真实逐视图特征、配对安全、nested CV |
| `extract_biomedclip_perview_features.py` | 可复现的逐视图BiomedCLIP特征提取 |
| `hybrid_perview.py` | 旧逐视图实验入口；评价协议仍为历史协议 |
| `patch_vascmamba.py` | CLS + 多尺度patch token实验 |
| `pyramid_vascmamba.py` | 早期多尺度mean pooling实验 |
| `compare_patch_vs_baselines.py` | 历史OOF对比脚本，不是nested CV |
| `dann.py` / `generalization*.py` | 域适应与外部泛化实验 |
| `madpot_ac.py` | 文本prompt + partial OT实验 |

## 实现说明

`hybrid.py` 中的MambaBlock是小样本场景下的纯PyTorch selective SSM，不等同于
官方 `mamba_ssm` CUDA kernel。本次修订修复了旧实现中未使用的 `D` 参数和
`x_proj` 额外输出，同时保持参数shape与历史Hybrid checkpoint兼容。若研究结论
要归因于Mamba，
仍需与参数量匹配的MLP、DeepSets和attention pooling基线进行消融。
