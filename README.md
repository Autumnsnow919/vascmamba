# VascMamba

乳腺 ULM（超声定位显微）/ B-mode 双模态超声良恶性二分类：冻结的
BiomedCLIP **真实逐视图特征** + 轻量 selective SSM 分类头。

## 当前推荐流程

当前 `hybrid.py` 不再把患者级平均特征复制成四个伪视图。推荐流程为：

```text
4个配对视图 (B1,U1) ... (B4,U4)
  → BiomedCLIP checkpoint自带的预处理
  → 每张图一个L2归一化的512维全局embedding
  → X_bmode/X_ulm = (N,4,512)
  → 模态独立投影 512→32
  → [B1,U1,B2,U2,B3,U3,B4,U4]
  → 1×轻量MambaBlock
  → masked mean pooling
  → 2类输出
```

默认保留数据中的原始视图顺序。`--order-by-density` 仅作为消融选项；启用
后会同步排列B-mode、ULM、density和valid mask，不会破坏模态配对。

训练采用nested 5-fold cross-validation：

- 内层验证集：early stopping和分类阈值选择；
- 外层测试折：只评价一次，不参与模型或阈值选择；
- 类别权重：仅根据当前内层训练集动态计算；
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

# 3. 可选的density排序消融；必须与默认流程独立比较
python hybrid.py \
  --features /root/medic_data/biomedclip_perview_features.npz \
  --output-dir hybrid_density_order_outputs \
  --order-by-density
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

241例（历史数据为60良性/181恶性）的折间标准差较大。方法间小幅差异需要使用
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
