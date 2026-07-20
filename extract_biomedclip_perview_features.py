"""Extract reproducible per-view BiomedCLIP embeddings for VascMamba-Hybrid.

The private patient index module is intentionally imported only inside ``main``
so the public model code remains importable without the private dataset.  Image
normalization is delegated to the transform shipped with the selected
BiomedCLIP checkpoint instead of hard-coding ImageNet statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm


MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CROP_Y1, CROP_Y2, CROP_X, SPLIT_X = 162, 737, 1100, 590


def pad_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)
    top = (side - height) // 2
    bottom = side - height - top
    left = (side - width) // 2
    right = side - width - left
    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )


def split_modalities(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cropped = image[CROP_Y1:CROP_Y2, :CROP_X]
    if cropped.shape[:2] != (CROP_Y2 - CROP_Y1, CROP_X):
        raise ValueError(
            f"image is too small for configured crop: got {image.shape}, "
            f"need at least ({CROP_Y2},{CROP_X},3)"
        )
    bmode = cropped[:, :SPLIT_X].copy()
    ulm = cropped[:, SPLIT_X:].copy()
    # The mask removes a known device overlay. Keep it configurable because a
    # fixed percentage is not portable across acquisition systems.
    return bmode, ulm


def mask_right_fraction(image: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("mask-right-fraction must be in [0,1)")
    result = image.copy()
    if fraction > 0:
        width = max(1, round(result.shape[1] * fraction))
        result[:, -width:] = 0
    return result


def to_pil_rgb(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(pad_square(image_bgr), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def vessel_density(ulm_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(ulm_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((binary > 0).mean())


@torch.no_grad()
def encode(model, preprocess, image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = preprocess(to_pil_rgb(image)).unsqueeze(0).to(device)
    return model.encode_image(tensor, normalize=True).cpu()[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("biomedclip_perview_features.npz"))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mask-right-fraction", type=float, default=0.10,
        help="Fraction of the ULM right edge occupied by a known overlay; use 0 to disable",
    )
    parser.add_argument(
        "--private-root", default="/root/medic_data",
        help="Directory containing the private ulm_visionnet package",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.private_root)
    sys.path.insert(0, os.path.join(args.private_root, "ulm_visionnet"))
    from data.patient_index_v2 import build_unified_index

    device = torch.device(args.device)
    model, _, preprocess_val = open_clip.create_model_and_transforms(args.model)
    model = model.to(device).eval()
    model.requires_grad_(False)
    embedding_dim = int(getattr(model, "embed_dim", 512))
    samples = build_unified_index()

    all_bmode, all_ulm, all_density, all_valid, all_labels = [], [], [], [], []
    for sample in tqdm(samples, desc="BiomedCLIP per-view features"):
        bmode_views, ulm_views, densities, valid = [], [], [], []
        views = list(sample["views"][:4])
        if len(views) != 4:
            raise ValueError(f"{sample.get('patient_name')} has {len(views)} views, expected 4")
        for filename in views:
            image = cv2.imread(os.path.join(sample["patient_dir"], filename))
            if image is None:
                bmode_views.append(torch.zeros(embedding_dim))
                ulm_views.append(torch.zeros(embedding_dim))
                densities.append(0.0)
                valid.append(False)
                continue
            bmode, ulm = split_modalities(image)
            ulm = mask_right_fraction(ulm, args.mask_right_fraction)
            bmode_views.append(encode(model, preprocess_val, bmode, device))
            ulm_views.append(encode(model, preprocess_val, ulm, device))
            densities.append(vessel_density(ulm))
            valid.append(True)

        all_bmode.append(torch.stack(bmode_views))
        all_ulm.append(torch.stack(ulm_views))
        all_density.append(torch.tensor(densities, dtype=torch.float32))
        all_valid.append(torch.tensor(valid, dtype=torch.bool))
        all_labels.append(int(sample["label"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X_bmode=torch.stack(all_bmode).numpy(),
        X_ulm=torch.stack(all_ulm).numpy(),
        density=torch.stack(all_density).numpy(),
        valid=torch.stack(all_valid).numpy(),
        y=np.asarray(all_labels, dtype=np.int64),
    )
    metadata = {
        "model": args.model,
        "feature_normalization": "L2 via encode_image(normalize=True)",
        "preprocess": "checkpoint-provided preprocess_val after crop/pad/BGR-to-RGB",
        "crop": [CROP_Y1, CROP_Y2, 0, CROP_X],
        "split_x": SPLIT_X,
        "mask_right_fraction": args.mask_right_fraction,
        "samples": len(samples),
    }
    with args.output.with_suffix(".metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"saved {args.output} and metadata for {len(samples)} samples")


if __name__ == "__main__":
    main()
