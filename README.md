# VascMamba

乳腺 ULM（超声定位显微）/ B-mode 双模态超声 **良恶性二分类**：冻结 BiomedCLIP 特征 + 轻量 Mamba SSM 分类头。

**核心结论**：241 例样本无法支撑端到端深度训练（CNN/ViT/Mamba 端到端全部坍缩为单类输出），唯一有效范式是 **冻结基础模型特征 + 轻量可训练头（~50–110K 参数）**。

## 结果（5-fold StratifiedKFold，seed=42，逐 fold F1 阈值搜索，同协议实跑）

| 方法 | Acc | AUC | Recall | F1 |
|---|---|---|---|---|
| BC-SVM（1024D session 特征） | 0.8507 ± .042 | 0.8021 | 0.9778 | 0.9084 |
| **VascMamba-Hybrid（SOTA）** | **0.8798 ± .033** | 0.8002 | 0.9228 | **0.9193** |
| Pyramid VascMamba（60 tok，纯 mean） | 0.8715 ± .044 | **0.8139** | 0.9614 | 0.9186 |
| PatchVascMamba（60 tok，mean+max 金字塔） | 0.8673 ± .038 | 0.7995 | 0.9395 | 0.9138 |
| PatchVascMamba `--full`（197 tok 无池化） | 0.8591 ± .046 | 0.7963 | **0.9670** | 0.9120 |

低密度恶性子集（ULM 血管密度低于恶性中位数，n=90，对应"坏死性恶性"风险组）召回：BC-SVM 0.989 / Hybrid 0.944 / Patch 0.944。

**观察**：召回随 token 粒度单调上升（0.923→0.940→0.967），但 AUC 三者均 ≈0.80——池化丢信息不是瓶颈，~88% 准确率是当前 241 样本量下的实际上限。

## 模型文件

| 文件 | 说明 |
|---|---|
| `hybrid.py` | **SOTA**。BiomedCLIP session 特征 (241,1024) → 展开为 8 token（4 B-mode + 4 ULM 交错）→ 1×MambaBlock(d=32) → 2 类头，~50K 参数 |
| `patch_vascmamba.py` | PatchVascMamba：全 patch token 输入，默认 60 tok/view/mod 金字塔（CLS + 7×7mean + 2×2mean/max + 1×1mean/max）；`--full` 切换 197 tok 无池化（seq 1576，分块扫描加速） |
| `pyramid_vascmamba.py` | 早期金字塔版（CLS + 2×2 + 1×1，纯 mean 池化），48 token |
| `hybrid_perview.py` / `multiseed_perview.py` | 逐 view 特征 (241,8,512) 变体 / 多种子评估 |
| `compare_patch_vs_baselines.py` | OOF 对比脚本：BC-SVM / Hybrid / Patch 同协议实跑 + 低密度恶性子集召回，输出 `compare_oof_probs.npz` |

## 关键设计

- **Vessel-guided ordering**：按 ULM 血管密度对 4 个 view 降序排列，为 Mamba 提供临床先验的序列顺序（`hybrid.py:152-155`）
- **Density 门控**（patch_vascmamba）：`sigmoid(4d−2)` 对 ULM token 降权——低血流 ULM 视为"无信息"而非"良性证据"
- **ULM dropout**（p=0.2，仅训练）：随机整支清零 ULM，强迫模型仅凭 B-mode 形态学判断，打破"ULM 空→良性"捷径
- **mean+max 双路池化**：max 层保留稀疏局灶血管信号（坏死恶性的边缘环形血流）
- **分块扫描 `chunked_scan`**：log 空间 cumsum + 先 mask 后 exp，与逐 token 循环数值差 1.5e-8，使 seq=1576 训练仅 ~58ms/step

## 特征提取脚本（位于上级目录，不在本压缩包）

| 脚本 | 输出 |
|---|---|
| `extract_biomedclip_features.py` | `biomedclip_features.npz` (241,1024)，session 级 |
| `extract_biomedclip_patch_tokens.py` | `biomedclip_patch_tokens.npz` (241,4,197,512) fp16，无空间压缩 |
| `extract_biomedclip_roi_pyramid_features.py` | `biomedclip_roi_pyramid_features.npz` (241,4,3,5,512) |
| `extract_vascular_features.py` | `vascular_features.npz` (241,54) 血管形态学 |

## 快速开始

```bash
conda activate <py3.12 + torch/cuda 环境>
export HF_ENDPOINT=https://hf-mirror.com   # BiomedCLIP 权重镜像

# 1. 提取特征（一次性，需原始影像数据，见"数据"节）
python3 ../extract_biomedclip_features.py
python3 ../extract_biomedclip_patch_tokens.py

# 2. 训练 / 对比
python3 hybrid.py                        # SOTA 87.98%
python3 patch_vascmamba.py               # 金字塔 60 token
python3 patch_vascmamba.py --full        # 全 197 token 无池化
python3 compare_patch_vs_baselines.py    # OOF 三模型对比 + 子集分析
```

依赖：`torch`(CUDA)、`open_clip_torch`、`timm`、`transformers`、`scikit-learn`、`opencv-python`、`numpy`、`tqdm`。

## 数据

241 例（60 良性 / 181 恶性，标签来自病理，Benign-High-risk 与交界性并入恶性）。**私有数据，不包含在本仓库中**。代码假设存在 `ulm_visionnet/data/patient_index_v2.build_unified_index()` 返回 `[{patient_dir, patient_name, label, views[4], video}]`。

预处理（硬编码，勿改）：crop `[162:737, 0:1100]` → 以 `x=590` 分 B-mode/ULM → ULM 右侧 10% 置零 → 补方 resize 224² → ImageNet 归一化。

## 其他实验（参考）

`dann.py`（域对抗）、`madpot_ac.py`（文本 prompt + partial-OT，含 BiomedCLIP tokenizer 用法）、`mahalanobis_patchcore.py`（异常检测）、`fusion_tda.py`（拓扑特征融合）、`topotoken.py`（拓扑 token）、`generalization*.py`（外部泛化）、`draw_arch.py` / `*.excalidraw`（架构图）。

## 限制

- 241 样本；所有深度学习结果 ±3~4pt 标准差内互为噪声
- ~88% 为当前特征与数据量下的实际上限；进一步提升需要更多标注数据或视频信息利用
