# =============================================================================
# Endoscopy Region Prediction AI - Medical Safer Single File Implementation
# =============================================================================
#
# IMPORTANT:
# - This is clinical decision support software, not autonomous diagnosis.
# - Final interpretation must be made by a qualified clinician.
#
# Example usage:
# python endoscopy_ai_medical.py --mode pretrain --config config.yaml
# python endoscopy_ai_medical.py --mode train --config config.yaml
# python endoscopy_ai_medical.py --mode continue-train --weights checkpoints/model_best.pt
# python endoscopy_ai_medical.py --mode calibrate --weights checkpoints/model_best.pt
# python endoscopy_ai_medical.py --mode predict --weights checkpoints/model_best.pt --image img.jpg --study "Colonoscopy" --cam
# python endoscopy_ai_medical.py --mode predict --weights checkpoints/model_best.pt --server

import os
import io
import re
import json
import base64
import random
import logging
import argparse
import traceback
import inspect
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from uuid import uuid4

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2

import timm
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn


# =============================================================================
# CONSTANTS
# =============================================================================

ORGAN_NAME = "ORGAN_NAME"
STUDY_NAME = "STUDY_NAME"
FILE_NAME = "FILE_NAME"
IMAGE_PATH = "image_path"
ORGAN_UID = "ORGAN_UID"
LABEL_ID = "label_id"
STUDY_ID = "study_id"

DEFAULT_MAPPING_FILENAME = "mappings.json"
DEFAULT_TEMPERATURE_FILENAME = "temperature.json"
DEFAULT_PROTOTYPE_FILENAME = "prototypes.pt"


# =============================================================================
# CONFIG & UTILITIES
# =============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(log_file="training.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_ddp():
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _safe_torch_load(path):
    kwargs = {"map_location": "cpu"}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
    except Exception:
        pass
    return torch.load(path, **kwargs)


def save_checkpoint(model, optimizer, epoch, metrics, path, is_best=False):
    if not is_main_process():
        return

    ensure_parent_dir(path)
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    state = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics
    }
    torch.save(state, path)

    if is_best:
        best_path = os.path.join(os.path.dirname(path), "model_best.pt")
        torch.save(state, best_path)


def load_checkpoint(model, optimizer=None, path="", strict=True, load_optimizer=True, verbose=True):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    state = _safe_torch_load(path)

    if isinstance(state, dict) and "model_state_dict" in state:
        model_state = state["model_state_dict"]
        ckpt_epoch = state.get("epoch", 0)
        ckpt_metrics = state.get("metrics", {})
        optimizer_state = state.get("optimizer_state_dict", None)
    else:
        model_state = state
        ckpt_epoch = 0
        ckpt_metrics = {}
        optimizer_state = None

    if not model_state:
        raise ValueError(f"Checkpoint at {path} contains an empty state_dict.")

    if list(model_state.keys())[0].startswith("module."):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}

    target_model = model.module if hasattr(model, "module") else model
    current_state = target_model.state_dict()

    if strict:
        target_model.load_state_dict(model_state, strict=True)
        if optimizer is not None and load_optimizer and optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
            except Exception as e:
                if verbose:
                    print(f"Warning: optimizer state not loaded: {e}")
        return ckpt_epoch, ckpt_metrics

    compatible_state = {}
    skipped_keys = []
    for k, v in model_state.items():
        if k in current_state and current_state[k].shape == v.shape:
            compatible_state[k] = v
        else:
            skipped_keys.append(k)

    load_result = target_model.load_state_dict(compatible_state, strict=False)

    if verbose:
        print(f"Loaded {len(compatible_state)} compatible tensor(s) from {path}")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} mismatched tensor(s)")
            for k in skipped_keys[:20]:
                ckpt_shape = tuple(model_state[k].shape) if hasattr(model_state[k], "shape") else "?"
                cur_shape = tuple(current_state[k].shape) if k in current_state and hasattr(current_state[k], "shape") else "missing"
                print(f" - {k}: checkpoint {ckpt_shape} != model {cur_shape}")
            if len(skipped_keys) > 20:
                print(f" ... and {len(skipped_keys) - 20} more")
        if load_result.missing_keys:
            print(f"Missing model tensor(s) after partial load: {len(load_result.missing_keys)}")

    if optimizer is not None and load_optimizer and optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as e:
            if verbose:
                print(f"Warning: optimizer state not loaded after partial load: {e}")

    return ckpt_epoch, ckpt_metrics


def load_pretrained_for_finetuning(model, path, verbose=True):
    if verbose:
        print(f"Loading pretrained weights for fine-tuning from {path}")
    return load_checkpoint(
        model=model,
        optimizer=None,
        path=path,
        strict=False,
        load_optimizer=False,
        verbose=verbose
    )


def normalize_text(value, lower=False):
    if pd.isna(value):
        return "unknown" if lower else "Unknown"
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    if lower:
        value = value.lower()
    return value


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if ORGAN_NAME in df.columns:
        df[ORGAN_NAME] = df[ORGAN_NAME].apply(lambda x: normalize_text(x, lower=True))
    if STUDY_NAME in df.columns:
        df[STUDY_NAME] = df[STUDY_NAME].apply(lambda x: normalize_text(x, lower=False))
    if FILE_NAME in df.columns:
        df[FILE_NAME] = df[FILE_NAME].apply(lambda x: normalize_text(x, lower=False))
    if IMAGE_PATH in df.columns:
        df[IMAGE_PATH] = df[IMAGE_PATH].apply(lambda x: normalize_text(x, lower=False))
    if ORGAN_UID in df.columns:
        df[ORGAN_UID] = df[ORGAN_UID].apply(lambda x: normalize_text(x, lower=False))

    return df


def validate_batch_labels(labels, num_classes, prefix="train"):
    if labels.dtype != torch.long:
        labels = labels.long()

    bad_mask = (labels < 0) | (labels >= num_classes)
    if bad_mask.any():
        bad_values = labels[bad_mask].detach().cpu().tolist()
        raise ValueError(
            f"{prefix}: invalid target label(s) found: {bad_values} with num_classes={num_classes}"
        )


def validate_training_dataframe(df: pd.DataFrame, organ_to_id: dict, study_to_id: dict, name="train"):
    if LABEL_ID not in df.columns:
        raise ValueError(f"{name} dataframe missing {LABEL_ID} column")
    if STUDY_ID not in df.columns:
        raise ValueError(f"{name} dataframe missing {STUDY_ID} column")

    bad_label = df[(df[LABEL_ID] < 0) | (df[LABEL_ID] >= len(organ_to_id))]
    if not bad_label.empty:
        cols = [c for c in [ORGAN_NAME, ORGAN_UID, LABEL_ID] if c in bad_label.columns]
        raise ValueError(
            f"{name} dataframe contains invalid {LABEL_ID} values.\n"
            f"{bad_label[cols].head(20).to_string(index=False)}"
        )

    bad_study = df[(df[STUDY_ID] < 0) | (df[STUDY_ID] >= len(study_to_id))]
    if not bad_study.empty:
        cols = [c for c in [STUDY_NAME, STUDY_ID] if c in bad_study.columns]
        raise ValueError(
            f"{name} dataframe contains invalid {STUDY_ID} values.\n"
            f"{bad_study[cols].head(20).to_string(index=False)}"
        )

    print(f"{name} validation passed: {len(df)} rows")


def create_train_val_split(df: pd.DataFrame, test_size=0.15, random_state=42):
    counts = df[ORGAN_NAME].value_counts()
    rare_classes = counts[counts < 2].index.tolist()

    if rare_classes:
        print(f"Warning: {len(rare_classes)} rare class(es) moved to train only.")
        rare_df = df[df[ORGAN_NAME].isin(rare_classes)].copy()
        split_df = df[~df[ORGAN_NAME].isin(rare_classes)].copy()
    else:
        rare_df = pd.DataFrame(columns=df.columns)
        split_df = df.copy()

    if split_df.empty:
        raise ValueError("No samples available for train/validation split after removing rare classes.")

    split_counts = split_df[ORGAN_NAME].value_counts()
    still_rare = split_counts[split_counts < 2].index.tolist()
    if still_rare:
        raise ValueError(f"Some classes still have < 2 samples after normalization: {still_rare}")

    if split_df[ORGAN_NAME].nunique() < 2:
        print("Warning: fewer than 2 eligible classes remain; using non-stratified split.")
        train_df, val_df = train_test_split(
            split_df, test_size=test_size, random_state=random_state, shuffle=True
        )
    else:
        train_df, val_df = train_test_split(
            split_df, test_size=test_size, random_state=random_state, stratify=split_df[ORGAN_NAME]
        )

    if not rare_df.empty:
        train_df = pd.concat([train_df, rare_df], ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train_df, val_df


def get_mapping_path(config: dict) -> str:
    checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints")
    return config.get("training", {}).get("mapping_path", os.path.join(checkpoint_dir, DEFAULT_MAPPING_FILENAME))


def get_temperature_path(config: dict) -> str:
    checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints")
    return config.get("training", {}).get("temperature_path", os.path.join(checkpoint_dir, DEFAULT_TEMPERATURE_FILENAME))


def get_prototype_path(config: dict) -> str:
    checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints")
    return config.get("medical", {}).get("prototype_path", os.path.join(checkpoint_dir, DEFAULT_PROTOTYPE_FILENAME))


def save_mappings(save_path: str, study_to_id: dict, organ_to_id: dict, id_to_organ: dict):
    ensure_parent_dir(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "study_to_id": study_to_id,
            "organ_to_id": organ_to_id,
            "id_to_organ": {str(k): v for k, v in id_to_organ.items()}
        }, f, ensure_ascii=False, indent=2)


def load_mappings(load_path: str):
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Mapping file not found: {load_path}")
    with open(load_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return (
        obj["study_to_id"],
        obj["organ_to_id"],
        {int(k): v for k, v in obj["id_to_organ"].items()}
    )


def save_temperature(save_path: str, temperature: float):
    ensure_parent_dir(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"temperature": float(temperature)}, f, indent=2)


def load_temperature(load_path: str) -> Optional[float]:
    if not os.path.isfile(load_path):
        return None
    with open(load_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return float(obj.get("temperature", 1.0))


def save_prototypes(path: str, prototypes: dict):
    ensure_parent_dir(path)
    torch.save({int(k): v.cpu() for k, v in prototypes.items()}, path)


def load_prototypes(path: str):
    if not os.path.exists(path):
        return None
    obj = torch.load(path, map_location="cpu")
    return {int(k): v for k, v in obj.items()}


def get_num_workers(config: dict) -> int:
    return int(config["data"].get("num_workers", 0))


def get_pin_memory(config: dict) -> bool:
    return bool(config["data"].get("pin_memory", False))


def get_label_smoothing(config: dict) -> float:
    return float(config.get("training", {}).get("label_smoothing", 0.0))


def get_tta_views(config: dict) -> int:
    return int(config.get("inference", {}).get("tta_views", 4))


def get_top_k(config: dict) -> int:
    return int(config.get("inference", {}).get("top_k", 3))


def get_medical_cfg(config: dict) -> dict:
    defaults = {
        "enable_abstention": True,
        "accept_threshold": 0.80,
        "review_threshold": 0.55,
        "ood_distance_threshold": 1.25,
        "min_blur_score": 80.0,
        "min_brightness": 35.0,
        "max_brightness": 220.0,
        "min_contrast": 18.0,
        "save_audit_logs": True,
        "audit_log_path": "logs/medical_audit.jsonl",
        "require_known_study": False
    }
    merged = defaults.copy()
    merged.update(config.get("medical", {}))
    return merged


def write_audit_log(log_path: str, payload: dict):
    ensure_parent_dir(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def assess_image_quality(image_np: np.ndarray, medical_cfg: dict) -> dict:
    if image_np is None or image_np.size == 0:
        return {
            "valid": False,
            "quality_ok": False,
            "reasons": ["empty_image"],
            "blur_score": 0.0,
            "brightness": 0.0,
            "contrast": 0.0
        }

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    reasons = []
    if blur_score < medical_cfg["min_blur_score"]:
        reasons.append("blurry_image")
    if brightness < medical_cfg["min_brightness"]:
        reasons.append("too_dark")
    if brightness > medical_cfg["max_brightness"]:
        reasons.append("too_bright")
    if contrast < medical_cfg["min_contrast"]:
        reasons.append("low_contrast")

    return {
        "valid": True,
        "quality_ok": len(reasons) == 0,
        "reasons": reasons,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2)
    }


# =============================================================================
# AUGMENTATIONS & DATASET
# =============================================================================

class MedicalAugmentations:
    @staticmethod
    def get_ssl_transforms(image_size: int):
        color_jitter = v2.ColorJitter(0.4, 0.4, 0.4, 0.1)
        return v2.Compose([
            v2.RandomResizedCrop(size=(image_size, image_size), scale=(0.2, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([color_jitter], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_supervised_train_transforms(image_size: int):
        return v2.Compose([
            v2.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_eval_transforms(image_size: int):
        return v2.Compose([
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class EndoscopyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, mode: str, image_size: int = 224, organ_to_id=None, study_to_id=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.mode = mode
        self.organ_to_id = organ_to_id
        self.study_to_id = study_to_id

        if self.mode == "ssl":
            self.transform = MedicalAugmentations.get_ssl_transforms(image_size)
        elif self.mode == "train":
            self.transform = MedicalAugmentations.get_supervised_train_transforms(image_size)
        else:
            self.transform = MedicalAugmentations.get_eval_transforms(image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if FILE_NAME in row.index:
            img_path = os.path.join(self.image_dir, row[FILE_NAME])
        elif IMAGE_PATH in row.index:
            img_path = row[IMAGE_PATH]
        else:
            raise KeyError(f"Neither {FILE_NAME} nor {IMAGE_PATH} found in row")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.mode == "ssl":
            return self.transform(image), self.transform(image)

        if STUDY_ID in row.index:
            study_idx = int(row[STUDY_ID])
        else:
            study_name = normalize_text(row.get(STUDY_NAME, "Unknown"), lower=False)
            unknown_idx = self.study_to_id.get("Unknown", 0) if self.study_to_id else 0
            study_idx = self.study_to_id.get(study_name, unknown_idx) if self.study_to_id else 0

        img_tensor = self.transform(image)

        if self.mode == "predict":
            return img_tensor, torch.tensor(study_idx, dtype=torch.long)

        if LABEL_ID in row.index:
            label = int(row[LABEL_ID])
        else:
            organ_name = normalize_text(row.get(ORGAN_NAME, "unknown"), lower=True)
            label = self.organ_to_id.get(organ_name, -1) if self.organ_to_id else -1

        return img_tensor, torch.tensor(study_idx, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# =============================================================================
# FUSION MODULES
# =============================================================================

class ConcatMLPFusion(nn.Module):
    def __init__(self, vis_dim: int, text_dim: int, hidden_dim: int):
        super().__init__()
        self.out_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(vis_dim + text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, vis_embeds, text_embeds):
        fused = torch.cat([vis_embeds, text_embeds], dim=-1)
        return self.mlp(fused)


class CrossAttentionFusion(nn.Module):
    def __init__(self, vis_dim: int, text_dim: int, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.vis_proj = nn.Linear(vis_dim, embed_dim) if vis_dim != embed_dim else nn.Identity()
        self.text_proj = nn.Linear(text_dim, embed_dim) if text_dim != embed_dim else nn.Identity()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.out_dim = embed_dim

    def forward(self, vis_embeds, text_embeds):
        if vis_embeds.dim() == 2:
            vis_embeds = vis_embeds.unsqueeze(1)
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(1)

        v = self.vis_proj(vis_embeds)
        q = self.text_proj(text_embeds)

        attn_out, _ = self.attention(query=q, key=v, value=v)
        x = self.norm1(q + attn_out)
        out = self.norm2(x + self.mlp(x))
        return out.squeeze(1)


# =============================================================================
# LOSSES
# =============================================================================

class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        sim.fill_diagonal_(-float("inf"))
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z_i.device)
        return F.cross_entropy(sim, labels)


class FocalLabelSmoothingLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        if targets.dtype != torch.long:
            targets = targets.long()

        num_classes = inputs.size(1)
        bad_mask = (targets < 0) | (targets >= num_classes)
        if bad_mask.any():
            bad_vals = targets[bad_mask].detach().cpu().tolist()
            raise ValueError(f"Invalid targets for loss: {bad_vals}, num_classes={num_classes}")

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.label_smoothing / max(num_classes - 1, 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        ce = -(true_dist * log_probs).sum(dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal = self.alpha * ((1 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


# =============================================================================
# METRICS
# =============================================================================

class MetricsTracker:
    def __init__(self, log_dir: str = "logs", num_classes: int = None):
        self.log_dir = log_dir
        self.num_classes = num_classes
        os.makedirs(self.log_dir, exist_ok=True)
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.all_losses = []

    def update(self, preds, targets, loss: float):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        pred_classes = np.argmax(preds, axis=1)
        self.all_preds.extend(pred_classes.tolist())
        self.all_targets.extend(targets.tolist())
        self.all_losses.append(float(loss))

    def compute(self, phase: str = "val", epoch: int = 0) -> dict:
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)
        avg_loss = float(np.mean(self.all_losses)) if len(self.all_losses) > 0 else 0.0
        acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        ) if len(y_true) > 0 else (0.0, 0.0, 0.0, None)

        metrics = {
            f"{phase}_loss": avg_loss,
            f"{phase}_acc": acc,
            f"{phase}_precision": precision,
            f"{phase}_recall": recall,
            f"{phase}_f1": f1,
        }

        if phase == "val" and len(np.unique(y_true)) > 1:
            self._plot_confusion_matrix(y_true, y_pred, epoch)
        return metrics

    def _plot_confusion_matrix(self, y_true, y_pred, epoch: int):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - Epoch {epoch}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"cm_epoch_{epoch}.png"))
        plt.close()


# =============================================================================
# MODEL
# =============================================================================

class MultimodalEndoscopyModel(nn.Module):
    def __init__(
        self,
        backbone_type: str = "vit_base_patch16_224",
        num_classes: int = 10,
        num_study_types: int = 20,
        embed_dim: int = 256,
        pretrained: bool = True,
        fusion_method: str = "cross_attention",
        ssl_dim: int = 128
    ):
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")
        if num_study_types <= 0:
            raise ValueError(f"num_study_types must be > 0, got {num_study_types}")

        self.backbone = timm.create_model(backbone_type, pretrained=pretrained, num_classes=0)

        with torch.no_grad():
            vis_dim = self.backbone(torch.randn(1, 3, 224, 224)).shape[-1]

        self.study_emb = nn.Embedding(num_study_types, embed_dim)

        if fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(vis_dim, embed_dim, embed_dim)
            fused_dim = self.fusion.out_dim
        elif fusion_method == "concat":
            self.fusion = ConcatMLPFusion(vis_dim, embed_dim, embed_dim)
            fused_dim = self.fusion.out_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.vision_proj = nn.Sequential(
            nn.Linear(vis_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.fused_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fused_dim, num_classes)
        )

        self.vision_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes)
        )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        self.ssl_projector = nn.Sequential(
            nn.Linear(vis_dim, vis_dim),
            nn.ReLU(),
            nn.Linear(vis_dim, ssl_dim)
        )

    def forward(self, x, study_idx=None, mode: str = "supervised", return_aux: bool = False):
        v_features = self.backbone(x)

        if mode == "ssl":
            return self.ssl_projector(v_features)

        if mode == "vision_only":
            return v_features

        assert study_idx is not None, "study_idx required for supervised forward pass"
        t_features = self.study_emb(study_idx)
        fused = self.fusion(v_features, t_features)
        v_embed = self.vision_proj(v_features)

        fused_logits = self.fused_head(fused)
        vision_logits = self.vision_head(v_embed)
        gate = self.gate(torch.cat([v_embed, t_features], dim=-1))
        logits = gate * fused_logits + (1.0 - gate) * vision_logits

        if return_aux:
            return {
                "logits": logits,
                "fused_logits": fused_logits,
                "vision_logits": vision_logits,
                "gate": gate,
                "vision_embedding": v_features
            }
        return logits


# =============================================================================
# GRAD-CAM
# =============================================================================

class GradCAM:
    def __init__(self, model: MultimodalEndoscopyModel, target_layer: nn.Module = None):
        self.model = model
        self.gradients = None
        self.activations = None

        if target_layer is None:
            target_layer = self._find_target_layer()

        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _find_target_layer(self):
        last_conv = None
        for m in self.model.backbone.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            return last_conv

        last_ln = None
        for m in self.model.backbone.modules():
            if isinstance(m, nn.LayerNorm):
                last_ln = m
        if last_ln is not None:
            return last_ln

        raise ValueError("Cannot auto-detect Grad-CAM target layer. Pass target_layer explicitly.")

    def _save_activation(self, module, inp, output):
        self.activations = output

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, img_tensor: torch.Tensor, study_tensor: torch.Tensor, class_idx: int = None, image_size: int = 224) -> np.ndarray:
        self.model.eval()
        logits = self.model(img_tensor, study_idx=study_tensor, mode="supervised")

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not fire. Check target_layer.")

        grads = self.gradients.detach()
        acts = self.activations.detach()

        if grads.dim() == 4:
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1).squeeze(0)
        elif grads.dim() == 3:
            weights = grads.mean(dim=2, keepdim=True)
            cam = (weights * acts).sum(dim=2).squeeze(0)
            n_patches = cam.shape[0]
            side = int((n_patches - 1) ** 0.5)
            if side * side == n_patches - 1:
                cam = cam[1:].reshape(side, side)
            else:
                side = int(n_patches ** 0.5)
                cam = cam[: side * side].reshape(side, side)
        else:
            cam = grads.squeeze(0)

        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cv2.resize(cam.astype(np.float32), (image_size, image_size))

    def overlay_on_image(self, original_image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()


# =============================================================================
# CALIBRATION & PROTOTYPES
# =============================================================================

class TemperatureScaler(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits):
        temp = torch.clamp(self.temperature, min=1e-3)
        return logits / temp


def collect_logits_and_labels(model, loader, device, amp_enabled=False):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, study_idxs, labels in loader:
            images = images.to(device)
            study_idxs = study_idxs.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images, study_idx=study_idxs, mode="supervised")

            all_logits.append(logits.detach())
            all_labels.append(labels.detach())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_temperature(model, val_loader, device, amp_enabled=False, max_iter=100):
    logits, labels = collect_logits_and_labels(model, val_loader, device, amp_enabled=amp_enabled)
    logits = logits.to(device)
    labels = labels.to(device)

    scaler = TemperatureScaler(1.0).to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(scaler.temperature.detach(), min=1e-3).item())


def compute_class_prototypes(model, loader, device, amp_enabled=False):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, study_idxs, batch_labels in loader:
            images = images.to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                feats = model(images, mode="vision_only")
            embeddings.append(feats.detach().cpu())
            labels.append(batch_labels.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    prototypes = {}
    for cls in labels.unique().tolist():
        cls_mask = labels == cls
        prototypes[int(cls)] = embeddings[cls_mask].mean(dim=0)

    return prototypes


# =============================================================================
# PREDICTOR
# =============================================================================

class EndoscopyPredictor:
    def __init__(
        self,
        config: dict,
        weights_path: str,
        id_to_organ: dict,
        study_to_id: dict,
        device: str = "cpu",
        temperature: Optional[float] = None
    ):
        self.device = torch.device(device)
        self.id_to_organ = id_to_organ
        self.study_to_id = study_to_id
        self.config = config
        self.temperature = float(temperature) if temperature is not None else 1.0
        self.medical_cfg = get_medical_cfg(config)
        self.prototypes = load_prototypes(get_prototype_path(config))

        if not self.id_to_organ:
            raise ValueError("id_to_organ is empty. Prediction requires saved mappings.")
        if not self.study_to_id:
            raise ValueError("study_to_id is empty. Prediction requires saved mappings.")
        if "Unknown" not in self.study_to_id:
            raise ValueError('study_to_id must contain "Unknown".')

        self.model = MultimodalEndoscopyModel(
            backbone_type=config["model"]["backbone_type"],
            pretrained=False,
            num_classes=len(id_to_organ),
            num_study_types=len(study_to_id),
            embed_dim=config["model"]["embedding_dim"],
            fusion_method=config["model"]["fusion_method"]
        )

        load_checkpoint(
            model=self.model,
            optimizer=None,
            path=weights_path,
            strict=True,
            load_optimizer=False,
            verbose=False
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = MedicalAugmentations.get_eval_transforms(
            image_size=config["data"]["image_size"]
        )

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        temp = max(float(self.temperature), 1e-3)
        return logits / temp

    def _build_tta_images(self, image: Image.Image, tta: int) -> List[Image.Image]:
        views = [image]
        if tta <= 1:
            return views

        candidates = [
            image.transpose(Image.FLIP_LEFT_RIGHT),
            image.transpose(Image.FLIP_TOP_BOTTOM),
            image.rotate(5, resample=Image.BILINEAR),
            image.rotate(-5, resample=Image.BILINEAR),
            image.rotate(10, resample=Image.BILINEAR),
            image.rotate(-10, resample=Image.BILINEAR),
        ]
        views.extend(candidates[: max(0, tta - 1)])
        return views[:tta]

    def _predict_logits_tta(self, image: Image.Image, study_tensor: torch.Tensor, tta: int):
        views = self._build_tta_images(image, tta)
        logits_list = []

        with torch.no_grad():
            for view in views:
                x = self.transform(view).unsqueeze(0).to(self.device)
                logits = self.model(x, study_idx=study_tensor, mode="supervised")
                logits_list.append(logits)

        return torch.stack(logits_list, dim=0).mean(dim=0)

    def _extract_embedding(self, image: Image.Image) -> torch.Tensor:
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(x, mode="vision_only")
        return emb.squeeze(0).detach().cpu()

    def _compute_ood_score(self, embedding: torch.Tensor):
        if not self.prototypes:
            return None, None

        distances = {}
        for cls_id, proto in self.prototypes.items():
            d = torch.norm(embedding - proto.cpu(), p=2).item()
            distances[int(cls_id)] = float(d)

        best_cls = min(distances, key=distances.get)
        return best_cls, distances[best_cls]

    def predict_image(self, image_path: str, study_name: str, return_cam: bool = False, tta: Optional[int] = None, top_k: Optional[int] = None) -> dict:
        case_id = str(uuid4())
        started_at = datetime.utcnow().isoformat() + "Z"

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            result = {
                "case_id": case_id,
                "timestamp_utc": started_at,
                "status": "REJECTED_INPUT",
                "error": str(e),
                "decision": "manual_review",
                "disclaimer": "AI decision support only. Final interpretation requires a qualified clinician."
            }
            if self.medical_cfg["save_audit_logs"]:
                write_audit_log(self.medical_cfg["audit_log_path"], result)
            return result

        image_np = np.array(image)
        quality = assess_image_quality(image_np, self.medical_cfg)

        if not quality["valid"]:
            result = {
                "case_id": case_id,
                "timestamp_utc": started_at,
                "status": "REJECTED_INPUT",
                "error": "invalid_image_content",
                "image_quality": quality,
                "decision": "manual_review",
                "disclaimer": "AI decision support only. Final interpretation requires a qualified clinician."
            }
            if self.medical_cfg["save_audit_logs"]:
                write_audit_log(self.medical_cfg["audit_log_path"], result)
            return result

        study_name_norm = normalize_text(study_name, lower=False)
        unknown_idx = self.study_to_id.get("Unknown", 0)
        study_found = study_name_norm in self.study_to_id
        study_idx = self.study_to_id.get(study_name_norm, unknown_idx)
        study_tensor = torch.tensor([study_idx], dtype=torch.long, device=self.device)

        tta = int(tta if tta is not None else get_tta_views(self.config))
        top_k = int(top_k if top_k is not None else get_top_k(self.config))

        logits = self._predict_logits_tta(image, study_tensor, tta)
        logits = self._apply_temperature(logits)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

        top_k = min(top_k, len(probs))
        topk_idx = np.argsort(probs)[::-1][:top_k]
        pred_id = int(topk_idx[0])
        pred_conf = float(probs[pred_id])

        top_predictions = []
        for i in topk_idx:
            organ_info = self.id_to_organ.get(int(i), {})
            top_predictions.append({
                "class_id": int(i),
                "organ_uid": organ_info.get("UID", "Unknown"),
                "organ_name": organ_info.get("Name", "Unknown"),
                "confidence": round(float(probs[i]), 4)
            })

        embedding = self._extract_embedding(image)
        _, ood_dist = self._compute_ood_score(embedding)

        status = "ACCEPTED"
        review_reasons = []

        if not quality["quality_ok"]:
            status = "REVIEW_REQUIRED"
            review_reasons.extend(quality["reasons"])

        if not study_found and self.medical_cfg["require_known_study"]:
            status = "REVIEW_REQUIRED"
            review_reasons.append("unknown_study")

        if self.medical_cfg["enable_abstention"]:
            if pred_conf < self.medical_cfg["review_threshold"]:
                status = "REVIEW_REQUIRED"
                review_reasons.append("very_low_confidence")
            elif pred_conf < self.medical_cfg["accept_threshold"]:
                status = "REVIEW_REQUIRED"
                review_reasons.append("low_confidence")

        if ood_dist is not None and ood_dist > self.medical_cfg["ood_distance_threshold"]:
            status = "REVIEW_REQUIRED"
            review_reasons.append("possible_out_of_distribution")

        predicted_organ = self.id_to_organ.get(pred_id, {})
        result = {
            "case_id": case_id,
            "timestamp_utc": started_at,
            "status": status,
            "decision": "accept_prediction" if status == "ACCEPTED" else "manual_review",
            "predicted_organ_uid": predicted_organ.get("UID", "Unknown"),
            "predicted_organ_name": predicted_organ.get("Name", "Unknown"),
            "confidence": round(pred_conf, 4),
            "confidence_band": (
                "high" if pred_conf >= self.medical_cfg["accept_threshold"]
                else "medium" if pred_conf >= self.medical_cfg["review_threshold"]
                else "low"
            ),
            "study_input": study_name,
            "study_used": study_name_norm if study_found else "Unknown",
            "study_found_in_mapping": bool(study_found),
            "study_index_used": int(study_idx),
            "tta_views_used": int(tta),
            "temperature_used": round(float(self.temperature), 4),
            "top_predictions": top_predictions,
            "image_quality": quality,
            "ood_distance": round(float(ood_dist), 4) if ood_dist is not None else None,
            "review_reasons": sorted(list(set(review_reasons))),
            "disclaimer": "AI decision support only. Final interpretation requires a qualified clinician."
        }

        if return_cam:
            try:
                grad_cam = GradCAM(self.model)
                img_grad = self.transform(image).unsqueeze(0).to(self.device)
                img_size = self.config["data"]["image_size"]
                cam_map = grad_cam.generate(img_grad, study_tensor, class_idx=pred_id, image_size=img_size)
                img_resized = cv2.resize(image_np, (img_size, img_size))
                overlay = grad_cam.overlay_on_image(img_resized, cam_map)
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                if ok:
                    result["cam_heatmap_b64"] = base64.b64encode(buf).decode("utf-8")
                grad_cam.remove_hooks()
            except Exception as e:
                result["cam_error"] = str(e)

        if self.medical_cfg["save_audit_logs"]:
            write_audit_log(self.medical_cfg["audit_log_path"], result)

        return result

    def predict_batch(self, image_paths: list, study_names: list, tta: Optional[int] = None, top_k: Optional[int] = None) -> list:
        return [self.predict_image(p, s, tta=tta, top_k=top_k) for p, s in zip(image_paths, study_names)]

    def export_onnx(self, save_path: str = "checkpoints/model.onnx"):
        class OnnxWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x, s):
                return self.m(x, study_idx=s, mode="supervised")

        wrapped = OnnxWrapper(self.model).to(self.device).eval()
        ensure_parent_dir(save_path)
        dummy_img = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_study = torch.tensor([0], dtype=torch.long).to(self.device)

        torch.onnx.export(
            wrapped, (dummy_img, dummy_study), save_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=["image", "study_idx"], output_names=["logits"],
            dynamic_axes={"image": {0: "batch_size"}, "study_idx": {0: "batch_size"}, "logits": {0: "batch_size"}}
        )
        print(f"Exported ONNX -> {save_path}")

    def export_torchscript(self, save_path: str = "checkpoints/model.pt"):
        class TracedWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x, s):
                return self.m(x, study_idx=s, mode="supervised")

        wrapped = TracedWrapper(self.model).to(self.device).eval()
        ensure_parent_dir(save_path)
        dummy_img = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_study = torch.tensor([0], dtype=torch.long).to(self.device)
        traced = torch.jit.trace(wrapped, (dummy_img, dummy_study))
        traced.save(save_path)
        print(f"Exported TorchScript -> {save_path}")


# =============================================================================
# FASTAPI
# =============================================================================

app = FastAPI(title="Endoscopy Region Prediction AI - Medical Safer")
_predictor_global = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def api_predict(
    file: UploadFile = File(...),
    study_name: str = Form(...),
    return_cam: bool = Form(False),
    tta: int = Form(4),
    top_k: int = Form(3)
):
    if _predictor_global is None:
        return {"error": "Predictor not initialized."}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tmp_path = "temp_inference.jpg"
    image.save(tmp_path)

    try:
        result = _predictor_global.predict_image(tmp_path, study_name, return_cam=return_cam, tta=tta, top_k=top_k)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result


def run_server(predictor_obj, host: str = "0.0.0.0", port: int = 8000):
    global _predictor_global
    _predictor_global = predictor_obj
    uvicorn.run(app, host=host, port=port)


# =============================================================================
# FACTORY & HELPERS
# =============================================================================

def create_model_and_optimizer(config, num_classes, num_study_types, device, local_rank):
    is_ddp = dist.is_initialized()

    model = MultimodalEndoscopyModel(
        backbone_type=config["model"]["backbone_type"],
        pretrained=config["model"]["pretrained"],
        num_classes=num_classes,
        num_study_types=num_study_types,
        embed_dim=config["model"]["embedding_dim"],
        fusion_method=config["model"]["fusion_method"]
    ).to(device)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )
    return model, optimizer


def flush_grad_accum_if_needed(step_count, grad_accum, scaler, optimizer):
    if step_count > 0 and (step_count % grad_accum) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


def build_train_sampler(train_df: pd.DataFrame, is_ddp: bool):
    if is_ddp:
        return None
    counts = train_df[LABEL_ID].value_counts().to_dict()
    sample_weights = train_df[LABEL_ID].map(lambda x: 1.0 / counts[x]).astype(np.float32).values
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )


# =============================================================================
# PRETRAINING
# =============================================================================

def run_pretraining(config, df, device, local_rank: int = 0):
    is_ddp = dist.is_initialized()
    amp_enabled = (device.type == "cuda" and config["training"]["mixed_precision"])

    dataset = EndoscopyDataset(
        df=df,
        image_dir=config["data"]["image_dir"],
        mode="ssl",
        image_size=config["data"]["image_size"]
    )

    sampler = DistributedSampler(dataset) if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=config["ssl"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=get_num_workers(config),
        pin_memory=get_pin_memory(config),
        drop_last=False
    )

    model = MultimodalEndoscopyModel(
        backbone_type=config["model"]["backbone_type"],
        pretrained=config["model"]["pretrained"],
        num_classes=1,
        num_study_types=1,
        embed_dim=config["model"]["embedding_dim"],
        fusion_method=config["model"]["fusion_method"],
        ssl_dim=config["ssl"]["projection_dim"]
    ).to(device)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["ssl"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = NTXentLoss(temperature=config["ssl"]["temperature"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["ssl"]["epochs"])

    epochs = config["ssl"]["epochs"]
    chkpt_dir = config["training"]["checkpoint_dir"]
    grad_accum = int(config["training"]["gradient_accumulation_steps"])

    if local_rank == 0:
        print(f"Starting SSL Pre-training for {epochs} epoch(s)...")

    for epoch in range(1, epochs + 1):
        if is_ddp and sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        steps_seen = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"SSL Epoch {epoch}/{epochs}", disable=(local_rank != 0))
        for step, (view1, view2) in enumerate(pbar):
            view1, view2 = view1.to(device), view2.to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                z1 = model(view1, mode="ssl")
                z2 = model(view2, mode="ssl")
                loss = criterion(z1, z2)
                loss_scaled = loss / grad_accum

            scaler.scale(loss_scaled).backward()
            steps_seen += 1

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())

            if local_rank == 0:
                pbar.set_postfix({"ssl_loss": f"{loss.item():.4f}"})

        flush_grad_accum_if_needed(steps_seen, grad_accum, scaler, optimizer)

        avg_loss = total_loss / max(len(loader), 1)
        scheduler.step()

        if local_rank == 0:
            print(f"Epoch {epoch} | Avg SSL Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % 10 == 0 or epoch == epochs:
            path = os.path.join(chkpt_dir, f"ssl_checkpoint_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, {"ssl_loss": avg_loss}, path)
            save_checkpoint(model, optimizer, epoch, {"ssl_loss": avg_loss}, os.path.join(chkpt_dir, "ssl_latest.pt"))

    if local_rank == 0:
        print("SSL Pre-training complete!")


# =============================================================================
# TRAINING
# =============================================================================

def run_training(config, train_df, val_df, organ_to_id, study_to_id, device, local_rank: int = 0, resume_path=None, continuous: bool = False):
    is_ddp = dist.is_initialized()
    amp_enabled = (device.type == "cuda" and config["training"]["mixed_precision"])

    train_dataset = EndoscopyDataset(
        df=train_df,
        image_dir=config["data"]["image_dir"],
        mode="train",
        image_size=config["data"]["image_size"],
        organ_to_id=organ_to_id,
        study_to_id=study_to_id
    )
    val_dataset = EndoscopyDataset(
        df=val_df,
        image_dir=config["data"]["image_dir"],
        mode="val",
        image_size=config["data"]["image_size"],
        organ_to_id=organ_to_id,
        study_to_id=study_to_id
    )

    train_sampler = DistributedSampler(train_dataset) if is_ddp else None
    weighted_sampler = build_train_sampler(train_df, is_ddp=is_ddp)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(train_sampler is None and weighted_sampler is None),
        sampler=train_sampler if train_sampler is not None else weighted_sampler,
        num_workers=get_num_workers(config),
        pin_memory=get_pin_memory(config)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        sampler=val_sampler,
        num_workers=get_num_workers(config),
        pin_memory=get_pin_memory(config)
    )

    num_classes = len(organ_to_id)
    num_study_types = len(study_to_id)
    model, optimizer = create_model_and_optimizer(config, num_classes, num_study_types, device, local_rank)

    start_epoch = 1
    best_loss = float("inf")
    early_stop_counter = 0

    if resume_path and os.path.exists(resume_path):
        if local_rank == 0:
            print(f"Loading weights from {resume_path} (continuous={continuous})")

        if continuous:
            ckpt_epoch, ckpt_metrics = load_checkpoint(
                model=model,
                optimizer=optimizer,
                path=resume_path,
                strict=True,
                load_optimizer=True,
                verbose=(local_rank == 0)
            )
            start_epoch = ckpt_epoch + 1
            best_loss = ckpt_metrics.get("val_loss", float("inf"))
            if local_rank == 0:
                print(f"Resuming from epoch {start_epoch} | best_loss={best_loss:.4f}")
        else:
            load_pretrained_for_finetuning(model=model, path=resume_path, verbose=(local_rank == 0))

    teacher_model = None
    if continuous and resume_path and os.path.exists(resume_path):
        teacher_model, _ = create_model_and_optimizer(config, num_classes, num_study_types, device, local_rank)
        load_checkpoint(
            model=teacher_model,
            optimizer=None,
            path=resume_path,
            strict=True,
            load_optimizer=False,
            verbose=(local_rank == 0)
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        if local_rank == 0:
            print("Teacher snapshot loaded for Knowledge Distillation.")

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = FocalLabelSmoothingLoss(
        alpha=float(config["training"].get("focal_alpha", 0.25)),
        gamma=float(config["training"].get("focal_gamma", 2.0)),
        label_smoothing=get_label_smoothing(config)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    metrics_tracker = MetricsTracker(log_dir=config["training"]["log_dir"], num_classes=num_classes)
    epochs = config["training"]["epochs"]
    chkpt_dir = config["training"]["checkpoint_dir"]
    grad_accum = int(config["training"]["gradient_accumulation_steps"])

    if local_rank == 0:
        print(f"Starting training (epochs {start_epoch}-{epochs})...")

    for epoch in range(start_epoch, epochs + 1):
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        metrics_tracker.reset()
        optimizer.zero_grad(set_to_none=True)
        steps_seen = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", disable=(local_rank != 0))
        for step, (images, study_idxs, labels) in enumerate(pbar):
            images = images.to(device)
            study_idxs = study_idxs.to(device)
            labels = labels.to(device)

            validate_batch_labels(labels, num_classes, prefix="train")

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(images, study_idx=study_idxs, mode="supervised", return_aux=True)
                logits = out["logits"]
                fused_logits = out["fused_logits"]
                vision_logits = out["vision_logits"]

                main_loss = criterion(logits, labels)
                fused_loss = criterion(fused_logits, labels)
                vision_loss = criterion(vision_logits, labels)
                loss = 0.6 * main_loss + 0.2 * fused_loss + 0.2 * vision_loss

                if continuous and teacher_model is not None:
                    with torch.no_grad():
                        t_logits = teacher_model(images, study_idx=study_idxs, mode="supervised")
                    T = 2.0
                    kd_loss = F.kl_div(
                        F.log_softmax(logits / T, dim=1),
                        F.softmax(t_logits / T, dim=1),
                        reduction="batchmean"
                    ) * (T ** 2)
                    loss = 0.7 * loss + 0.3 * kd_loss

                loss_scaled = loss / grad_accum

            scaler.scale(loss_scaled).backward()
            steps_seen += 1

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            metrics_tracker.update(logits.detach(), labels.detach(), float(loss.item()))

            if local_rank == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        flush_grad_accum_if_needed(steps_seen, grad_accum, scaler, optimizer)

        train_metrics = metrics_tracker.compute(phase="train", epoch=epoch)
        scheduler.step()

        model.eval()
        metrics_tracker.reset()

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Val {epoch}/{epochs}", disable=(local_rank != 0))
            for images, study_idxs, labels in vbar:
                images = images.to(device)
                study_idxs = study_idxs.to(device)
                labels = labels.to(device)

                validate_batch_labels(labels, num_classes, prefix="val")

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(images, study_idx=study_idxs, mode="supervised")
                    loss = criterion(logits, labels)

                metrics_tracker.update(logits, labels, float(loss.item()))

        val_metrics = metrics_tracker.compute(phase="val", epoch=epoch)

        if local_rank == 0:
            print(
                f"Epoch {epoch:>4} | "
                f"Train loss={train_metrics['train_loss']:.4f} f1={train_metrics['train_f1']:.4f} | "
                f"Val loss={val_metrics['val_loss']:.4f} f1={val_metrics['val_f1']:.4f} "
                f"acc={val_metrics['val_acc']:.4f}"
            )

            all_metrics = {**train_metrics, **val_metrics}
            is_best = val_metrics["val_loss"] < best_loss
            chkpt_path = os.path.join(chkpt_dir, f"checkpoint_epoch_{epoch}.pt")

            if is_best:
                best_loss = val_metrics["val_loss"]
                early_stop_counter = 0
                print("✓ New best - model saved.")
            else:
                early_stop_counter += 1

            save_checkpoint(model, optimizer, epoch, all_metrics, chkpt_path, is_best=is_best)

            if early_stop_counter >= config["training"]["early_stopping_patience"]:
                print(f"Early stopping after {epoch} epochs (patience={config['training']['early_stopping_patience']}).")
                break

    return model, train_loader, val_loader


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_and_mappings(csv_path: str):
    df = pd.read_csv(csv_path)
    df = normalize_dataframe(df)

    required_cols = [ORGAN_NAME, STUDY_NAME]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if FILE_NAME not in df.columns and IMAGE_PATH not in df.columns:
        raise ValueError(f"CSV must contain either {FILE_NAME} or {IMAGE_PATH}")

    if ORGAN_UID not in df.columns:
        df[ORGAN_UID] = df[ORGAN_NAME]

    df[ORGAN_NAME] = df[ORGAN_NAME].fillna("unknown").astype(str).str.strip().str.lower()
    df[STUDY_NAME] = df[STUDY_NAME].fillna("Unknown").astype(str).str.strip()

    if FILE_NAME in df.columns:
        df = df[df[FILE_NAME].notna()].copy()
        df[FILE_NAME] = df[FILE_NAME].astype(str).str.strip()

    if IMAGE_PATH in df.columns:
        df = df[df[IMAGE_PATH].notna()].copy()
        df[IMAGE_PATH] = df[IMAGE_PATH].astype(str).str.strip()

    df = df[df[ORGAN_NAME] != ""].copy()
    df = df[df[STUDY_NAME] != ""].copy()
    df.reset_index(drop=True, inplace=True)

    unique_studies = sorted(df[STUDY_NAME].dropna().unique().tolist())
    if "Unknown" not in unique_studies:
        unique_studies = ["Unknown"] + unique_studies
    study_to_id = {name: idx for idx, name in enumerate(unique_studies)}

    unique_organs = (
        df[[ORGAN_UID, ORGAN_NAME]]
        .dropna()
        .drop_duplicates(subset=[ORGAN_NAME])
        .reset_index(drop=True)
    )

    organ_to_id = {row[ORGAN_NAME]: idx for idx, row in unique_organs.iterrows()}
    id_to_organ = {
        idx: {"UID": row[ORGAN_UID], "Name": row[ORGAN_NAME]}
        for idx, row in unique_organs.iterrows()
    }

    df[LABEL_ID] = df[ORGAN_NAME].map(organ_to_id)
    bad_label_rows = df[LABEL_ID].isna()
    if bad_label_rows.any():
        bad_samples = df.loc[bad_label_rows, [ORGAN_NAME, ORGAN_UID]].head(10)
        raise ValueError(
            f"Found rows with unmapped {ORGAN_NAME} values after mapping:\n"
            f"{bad_samples.to_string(index=False)}"
        )

    df[LABEL_ID] = df[LABEL_ID].astype(int)
    df[STUDY_ID] = df[STUDY_NAME].map(lambda x: study_to_id.get(x, study_to_id["Unknown"])).astype(int)

    print(f"Loaded {len(df)} rows")
    print(f"Num classes: {len(organ_to_id)}")
    print(f"Num study types: {len(study_to_id)}")
    print(f"Label range: {df[LABEL_ID].min()} -> {df[LABEL_ID].max()}")

    return df, study_to_id, organ_to_id, id_to_organ


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Endoscopy Region Prediction AI - Medical Safer")
    parser.add_argument("--mode", type=str, required=True, choices=["pretrain", "train", "continue-train", "predict", "calibrate"])
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image", type=str, help="Image path for single prediction")
    parser.add_argument("--study", type=str, help="Study name for prediction")
    parser.add_argument("--weights", type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--server", action="store_true", help="Launch FastAPI server")
    parser.add_argument("--cam", action="store_true", help="Return Grad-CAM in prediction")
    parser.add_argument("--tta", type=int, default=None, help="Number of TTA views")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k predictions")
    parser.add_argument("--export-onnx", type=str, default=None, metavar="PATH")
    parser.add_argument("--export-torchscript", type=str, default=None, metavar="PATH")
    args = parser.parse_args()

    local_rank = setup_ddp()
    set_seed(42 + local_rank)

    if is_main_process():
        setup_logging()

    config = load_config(args.config)
    device = torch.device(args.device)

    mapping_path = get_mapping_path(config)
    temperature_path = get_temperature_path(config)
    prototype_path = get_prototype_path(config)

    csv_path = config["data"]["csv_path"]

    if args.mode != "predict" and not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data CSV not found: {csv_path}")

    df = None
    study_to_id = {}
    organ_to_id = {}
    id_to_organ = {}

    if os.path.exists(csv_path):
        df, study_to_id, organ_to_id, id_to_organ = load_data_and_mappings(csv_path)

        if is_main_process():
            save_mappings(mapping_path, study_to_id, organ_to_id, id_to_organ)
            print(f"Saved mappings -> {mapping_path}")
    else:
        if os.path.exists(mapping_path):
            study_to_id, organ_to_id, id_to_organ = load_mappings(mapping_path)
            if is_main_process():
                print(f"Loaded mappings from {mapping_path}")
        elif args.mode == "predict":
            raise FileNotFoundError(
                f"Prediction requires mappings, but neither CSV nor mapping file was found.\n"
                f"CSV path: {csv_path}\n"
                f"Mapping path: {mapping_path}"
            )

    if args.mode == "pretrain":
        if df is None:
            raise ValueError("Pretraining requires CSV data.")
        run_pretraining(config, df, device, local_rank)

    elif args.mode in ["train", "continue-train"]:
        if df is None:
            raise ValueError("Training requires CSV data.")

        train_df, val_df = create_train_val_split(df, test_size=0.15, random_state=42)
        validate_training_dataframe(train_df, organ_to_id, study_to_id, name="train")
        validate_training_dataframe(val_df, organ_to_id, study_to_id, name="val")

        continuous = (args.mode == "continue-train")
        resume_path = args.weights if continuous else None

        if not continuous:
            ssl_latest = os.path.join(config["training"]["checkpoint_dir"], "ssl_latest.pt")
            if os.path.exists(ssl_latest):
                resume_path = ssl_latest
                if local_rank == 0:
                    print(f"Auto-loading SSL pre-trained weights from {ssl_latest}")

        trained_model, train_loader, val_loader = run_training(
            config,
            train_df,
            val_df,
            organ_to_id=organ_to_id,
            study_to_id=study_to_id,
            device=device,
            local_rank=local_rank,
            resume_path=resume_path,
            continuous=continuous
        )

        if is_main_process():
            best_model_path = os.path.join(config["training"]["checkpoint_dir"], "model_best.pt")

            if os.path.exists(best_model_path):
                proto_model = MultimodalEndoscopyModel(
                    backbone_type=config["model"]["backbone_type"],
                    pretrained=False,
                    num_classes=len(organ_to_id),
                    num_study_types=len(study_to_id),
                    embed_dim=config["model"]["embedding_dim"],
                    fusion_method=config["model"]["fusion_method"]
                ).to(device)

                load_checkpoint(
                    model=proto_model,
                    optimizer=None,
                    path=best_model_path,
                    strict=True,
                    load_optimizer=False,
                    verbose=True
                )

                prototypes = compute_class_prototypes(
                    proto_model,
                    train_loader,
                    device=device,
                    amp_enabled=(device.type == "cuda" and config["training"]["mixed_precision"])
                )
                save_prototypes(prototype_path, prototypes)
                print(f"Saved prototypes -> {prototype_path}")

            if bool(config.get("training", {}).get("auto_calibrate_after_train", True)):
                print("Fitting temperature scaling on validation set...")

                model_for_cal = MultimodalEndoscopyModel(
                    backbone_type=config["model"]["backbone_type"],
                    pretrained=False,
                    num_classes=len(organ_to_id),
                    num_study_types=len(study_to_id),
                    embed_dim=config["model"]["embedding_dim"],
                    fusion_method=config["model"]["fusion_method"]
                ).to(device)

                if os.path.exists(best_model_path):
                    load_checkpoint(
                        model=model_for_cal,
                        optimizer=None,
                        path=best_model_path,
                        strict=True,
                        load_optimizer=False,
                        verbose=True
                    )
                    temperature = fit_temperature(
                        model_for_cal,
                        val_loader,
                        device=device,
                        amp_enabled=(device.type == "cuda" and config["training"]["mixed_precision"])
                    )
                    save_temperature(temperature_path, temperature)
                    print(f"Saved temperature -> {temperature_path} | value={temperature:.4f}")
                else:
                    print(f"Skipping calibration because best model was not found: {best_model_path}")

    elif args.mode == "calibrate":
        if df is None:
            raise ValueError("Calibration requires CSV data.")
        if not os.path.exists(args.weights):
            raise FileNotFoundError(f"Weights not found: {args.weights}")

        train_df, val_df = create_train_val_split(df, test_size=0.15, random_state=42)
        validate_training_dataframe(train_df, organ_to_id, study_to_id, name="train")
        validate_training_dataframe(val_df, organ_to_id, study_to_id, name="val")

        val_dataset = EndoscopyDataset(
            df=val_df,
            image_dir=config["data"]["image_dir"],
            mode="val",
            image_size=config["data"]["image_size"],
            organ_to_id=organ_to_id,
            study_to_id=study_to_id
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"] * 2,
            shuffle=False,
            num_workers=get_num_workers(config),
            pin_memory=get_pin_memory(config)
        )

        model = MultimodalEndoscopyModel(
            backbone_type=config["model"]["backbone_type"],
            pretrained=False,
            num_classes=len(organ_to_id),
            num_study_types=len(study_to_id),
            embed_dim=config["model"]["embedding_dim"],
            fusion_method=config["model"]["fusion_method"]
        ).to(device)

        load_checkpoint(
            model=model,
            optimizer=None,
            path=args.weights,
            strict=True,
            load_optimizer=False,
            verbose=True
        )

        temperature = fit_temperature(
            model,
            val_loader,
            device=device,
            amp_enabled=(device.type == "cuda" and config["training"]["mixed_precision"])
        )
        save_temperature(temperature_path, temperature)
        print(f"Saved temperature -> {temperature_path} | value={temperature:.4f}")

    elif args.mode == "predict":
        if not os.path.exists(args.weights):
            raise FileNotFoundError(f"Weights not found: {args.weights}")

        if not study_to_id or not id_to_organ:
            if os.path.exists(mapping_path):
                study_to_id, organ_to_id, id_to_organ = load_mappings(mapping_path)
            else:
                raise FileNotFoundError(
                    f"Prediction requires mappings, but mapping file was not found: {mapping_path}"
                )

        temperature = load_temperature(temperature_path)
        if temperature is not None:
            print(f"Loaded temperature scaling value: {temperature:.4f}")
        else:
            print("Temperature file not found. Using default temperature=1.0")

        predictor = EndoscopyPredictor(
            config=config,
            weights_path=args.weights,
            id_to_organ=id_to_organ,
            study_to_id=study_to_id,
            device=args.device,
            temperature=temperature
        )

        if args.export_onnx:
            predictor.export_onnx(args.export_onnx)

        if args.export_torchscript:
            predictor.export_torchscript(args.export_torchscript)

        if args.server:
            print("Starting Endoscopy AI Server -> http://0.0.0.0:8000")
            run_server(predictor)
        elif args.image and args.study:
            res = predictor.predict_image(
                args.image,
                args.study,
                return_cam=args.cam,
                tta=args.tta,
                top_k=args.top_k
            )
            print(json.dumps(res, indent=4, ensure_ascii=False))
        else:
            raise ValueError(
                "Provide --image and --study for CLI prediction, or use --server to start the API."
            )

    cleanup_ddp()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        print(traceback.format_exc())
        raise


