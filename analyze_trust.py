#!/usr/bin/env python3
"""
Trust analysis for PAT-Mamba.

1. TTA-based restoration confidence / uncertainty estimation
2. Prototype-based degradation attribution from encoder features
"""

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics import ImageMetrics
from net.model import PATMamba


ATTRIBUTION_TEMPERATURE = 20.0
DEGRADATION_ORDER: List[str] = [
    "gaussian_noise",
    "narrow_band",
    "under_angle",
    "under_sampling",
]


@dataclass(frozen=True)
class TTATransform:
    name: str
    forward: callable
    inverse: callable


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def vflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])


def rot90(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=1, dims=(-2, -1))


def rot270(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=3, dims=(-2, -1))


TTA_TRANSFORMS = [
    TTATransform("identity", identity, identity),
    TTATransform("hflip", hflip, hflip),
    TTATransform("vflip", vflip, vflip),
    TTATransform("rot90", rot90, rot270),
]


def infer_degradation_type(filename: str) -> str:
    lower_name = os.path.basename(filename).lower()
    for degradation_type in sorted(DEGRADATION_ORDER, key=len, reverse=True):
        if lower_name.startswith(degradation_type.lower()):
            return degradation_type
    raise ValueError(f"Cannot infer degradation type from filename: {filename}")


class MixedDegradationDataset(Dataset):
    """Dataset loader for mixed multi-degradation splits."""

    def __init__(self, data_root: str, split: str) -> None:
        self.data_root = data_root
        self.split = split
        self.lq_dir = os.path.join(data_root, split, "LQ")
        self.gt_dir = os.path.join(data_root, split, "GT")
        self.to_tensor = ToTensor()
        self.image_names = self._collect_image_names()

    def _collect_image_names(self) -> List[str]:
        if not os.path.isdir(self.lq_dir) or not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(
                f"Dataset split not found: {self.lq_dir} or {self.gt_dir}"
            )

        lq_files = set(os.listdir(self.lq_dir))
        gt_files = set(os.listdir(self.gt_dir))
        common_files = sorted(
            file_name
            for file_name in lq_files.intersection(gt_files)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not common_files:
            raise RuntimeError(f"No image pairs found under {self.data_root}/{self.split}")
        return common_files

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        image = Image.open(path).convert("L")
        image = np.asarray(image, dtype=np.float32) / 255.0
        return np.stack([image, image, image], axis=2)

    def __getitem__(self, index: int) -> Tuple[str, str, torch.Tensor, torch.Tensor]:
        file_name = self.image_names[index]
        label = infer_degradation_type(file_name)
        lq_img = self._load_image(os.path.join(self.lq_dir, file_name))
        gt_img = self._load_image(os.path.join(self.gt_dir, file_name))
        return (
            os.path.splitext(file_name)[0],
            label,
            self.to_tensor(lq_img),
            self.to_tensor(gt_img),
        )

    def __len__(self) -> int:
        return len(self.image_names)


class EncoderFeatureHook:
    """Hook that captures second-stage encoder features."""

    def __init__(self, model: nn.Module) -> None:
        self._features = None
        self._hook = model.encoder_level2.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output) -> None:
        self._features = torch.mean(output.detach(), dim=(2, 3))

    def clear(self) -> None:
        self._features = None

    def pop(self) -> torch.Tensor:
        if self._features is None:
            raise RuntimeError("Encoder features were not captured in the last forward pass")
        features = self._features
        self._features = None
        return features

    def close(self) -> None:
        self._hook.remove()


def load_model(checkpoint_path: str, device: torch.device) -> PATMamba:
    model = PATMamba(decoder=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def forward_with_features(
    model: nn.Module, hook: EncoderFeatureHook, inputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    hook.clear()
    outputs = model(inputs)
    features = hook.pop()
    return outputs, features


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, ord=2)
    if norm == 0:
        return array
    return array / norm


def softmax(values: np.ndarray) -> np.ndarray:
    stable_values = values - np.max(values)
    exp_values = np.exp(stable_values)
    return exp_values / np.sum(exp_values)


def tensor_to_gray_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = np.mean(array, axis=0)
    return np.clip(array, 0.0, 1.0)


def save_grayscale_png(image: np.ndarray, path: Path) -> None:
    array = np.clip(image, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def save_heatmap_png(
    image: np.ndarray,
    path: Path,
    cmap: str,
    vmin: float = None,
    vmax: float = None,
) -> None:
    plt.imsave(path, image, cmap=cmap, vmin=vmin, vmax=vmax)


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def infer_dataset_label(data_root: str) -> str:
    lower_root = data_root.lower()
    if "artery" in lower_root:
        return "Artery"
    if "mice" in lower_root:
        return "Mice"
    if "peitai" in lower_root or "embryo" in lower_root:
        return "Embryo"
    return os.path.basename(data_root)


def default_degradation_order(data_root: str) -> List[str]:
    lower_root = data_root.lower()
    if "artery" in lower_root:
        return ["32_channel", "SNR_5dB", "90_degree", "narrow_1e5_1e6_Hz"]
    if "mice" in lower_root:
        return ["64_channel", "SNR_5dB", "mid_0p60_2e6", "π_angle"]
    return ["gaussian_noise", "narrow_band", "under_angle", "under_sampling"]


def pretty_degradation_label(label: str) -> str:
    mapping = {
        "gaussian_noise": "Gaussian",
        "narrow_band": "Narrow",
        "under_angle": "Angle",
        "under_sampling": "Sampling",
        "32_channel": "32-ch",
        "64_channel": "64-ch",
        "SNR_5dB": "Noise",
        "90_degree": "90 deg",
        "narrow_1e5_1e6_Hz": "Narrow",
        "mid_0p60_2e6": "Mid-band",
        "π_angle": "Pi angle",
    }
    return mapping.get(label, label)


def build_prototypes(
    model: nn.Module,
    data_root: str,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, np.ndarray]:
    dataset = MixedDegradationDataset(data_root, split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    hook = EncoderFeatureHook(model)
    features_by_class: Dict[str, List[np.ndarray]] = {key: [] for key in DEGRADATION_ORDER}

    with torch.inference_mode():
        progress_bar = tqdm(dataloader, desc=f"Building prototypes ({split})", leave=False)
        for _, labels, lq_imgs, _ in progress_bar:
            lq_imgs = lq_imgs.to(device)
            _, features = forward_with_features(model, hook, lq_imgs)
            features_np = features.detach().cpu().numpy()
            for index, label in enumerate(labels):
                features_by_class[label].append(features_np[index])

    hook.close()

    prototypes = {}
    for label in DEGRADATION_ORDER:
        class_features = np.stack(features_by_class[label], axis=0)
        prototypes[label] = np.mean(class_features, axis=0)
    return prototypes


def compute_attribution(
    feature_vector: np.ndarray, prototypes: Dict[str, np.ndarray]
) -> Tuple[str, float, Dict[str, float], Dict[str, float]]:
    normalized_feature = l2_normalize(feature_vector)
    similarities = {}
    for label, prototype in prototypes.items():
        normalized_prototype = l2_normalize(prototype)
        similarities[label] = float(np.dot(normalized_feature, normalized_prototype))

    similarity_vector = np.array([similarities[label] for label in DEGRADATION_ORDER], dtype=np.float64)
    similarity_vector = similarity_vector * ATTRIBUTION_TEMPERATURE
    probabilities_vector = softmax(similarity_vector)
    probabilities = {
        label: float(probabilities_vector[index]) for index, label in enumerate(DEGRADATION_ORDER)
    }
    predicted_label = DEGRADATION_ORDER[int(np.argmax(probabilities_vector))]
    confidence = probabilities[predicted_label]
    return predicted_label, confidence, similarities, probabilities


def analyze_test_samples(
    model: nn.Module,
    data_root: str,
    split: str,
    device: torch.device,
    prototypes: Dict[str, np.ndarray],
) -> pd.DataFrame:
    dataset = MixedDegradationDataset(data_root, split)
    image_metrics = ImageMetrics(device=device)
    hook = EncoderFeatureHook(model)
    records: List[Dict[str, float]] = []

    with torch.inference_mode():
        progress_bar = tqdm(dataset, desc=f"Analyzing {split} split", leave=False)
        for sample_name, label, lq_tensor, gt_tensor in progress_bar:
            lq_tensor = lq_tensor.unsqueeze(0).to(device)
            gt_tensor = gt_tensor.unsqueeze(0).to(device)

            tta_outputs = []
            feature_vector = None
            for transform in TTA_TRANSFORMS:
                transformed_input = transform.forward(lq_tensor)
                prediction, features = forward_with_features(model, hook, transformed_input)
                restored_prediction = transform.inverse(prediction)
                tta_outputs.append(restored_prediction.detach().cpu())
                if transform.name == "identity":
                    feature_vector = features[0].detach().cpu().numpy()

            stacked_outputs = torch.stack(tta_outputs, dim=0)
            mean_prediction = stacked_outputs.mean(dim=0).to(device)
            uncertainty_map = stacked_outputs.std(dim=0, unbiased=False).mean(dim=1, keepdim=True)
            uncertainty_score = float(uncertainty_map.mean().item())
            high_uncertainty_ratio = float(
                (uncertainty_map > (uncertainty_map.mean() + uncertainty_map.std(unbiased=False)))
                .float()
                .mean()
                .item()
            )

            metrics = image_metrics.calculate_metrics(mean_prediction, gt_tensor)
            predicted_label, attribution_confidence, similarities, probabilities = compute_attribution(
                feature_vector, prototypes
            )

            record = {
                "sample_name": sample_name,
                "ground_truth_degradation": label,
                "predicted_degradation": predicted_label,
                "attribution_confidence": attribution_confidence,
                "attribution_correct": int(predicted_label == label),
                "psnr": float(metrics["psnr"]),
                "ssim": float(metrics["ssim"]),
                "lpips": float(metrics["lpips"]),
                "uncertainty_score": uncertainty_score,
                "high_uncertainty_ratio": high_uncertainty_ratio,
            }

            for degradation_type in DEGRADATION_ORDER:
                record[f"similarity_{degradation_type}"] = similarities[degradation_type]
                record[f"probability_{degradation_type}"] = probabilities[degradation_type]

            records.append(record)
            progress_bar.set_postfix(
                psnr=f"{record['psnr']:.2f}",
                unc=f"{record['uncertainty_score']:.4f}",
                attr=predicted_label,
            )

    hook.close()

    results_df = pd.DataFrame.from_records(records)
    min_uncertainty = results_df["uncertainty_score"].min()
    max_uncertainty = results_df["uncertainty_score"].max()
    denominator = max(max_uncertainty - min_uncertainty, 1e-8)
    results_df["confidence_score"] = 1.0 - (
        (results_df["uncertainty_score"] - min_uncertainty) / denominator
    )
    return results_df


def compute_summary(results_df: pd.DataFrame) -> Dict[str, float]:
    high_threshold = results_df["confidence_score"].quantile(0.75)
    low_threshold = results_df["confidence_score"].quantile(0.25)

    high_group = results_df[results_df["confidence_score"] >= high_threshold].copy()
    low_group = results_df[results_df["confidence_score"] <= low_threshold].copy()

    confidence_series = results_df["confidence_score"]
    summary = {
        "sample_count": int(len(results_df)),
        "attribution_accuracy": float(
            accuracy_score(
                results_df["ground_truth_degradation"],
                results_df["predicted_degradation"],
            )
        ),
        "attribution_macro_f1": float(
            f1_score(
                results_df["ground_truth_degradation"],
                results_df["predicted_degradation"],
                labels=DEGRADATION_ORDER,
                average="macro",
            )
        ),
        "mean_attribution_confidence": float(results_df["attribution_confidence"].mean()),
        "confidence_psnr_spearman": float(confidence_series.corr(results_df["psnr"], method="spearman")),
        "confidence_ssim_spearman": float(confidence_series.corr(results_df["ssim"], method="spearman")),
        "confidence_lpips_spearman": float(confidence_series.corr(results_df["lpips"], method="spearman")),
        "high_confidence_count": int(len(high_group)),
        "low_confidence_count": int(len(low_group)),
        "high_confidence_mean_psnr": float(high_group["psnr"].mean()),
        "low_confidence_mean_psnr": float(low_group["psnr"].mean()),
        "high_confidence_mean_ssim": float(high_group["ssim"].mean()),
        "low_confidence_mean_ssim": float(low_group["ssim"].mean()),
        "high_confidence_mean_lpips": float(high_group["lpips"].mean()),
        "low_confidence_mean_lpips": float(low_group["lpips"].mean()),
        "high_confidence_mean_uncertainty": float(high_group["uncertainty_score"].mean()),
        "low_confidence_mean_uncertainty": float(low_group["uncertainty_score"].mean()),
        "confidence_score_min": float(results_df["confidence_score"].min()),
        "confidence_score_max": float(results_df["confidence_score"].max()),
    }
    return summary


def save_confusion_artifacts(
    results_df: pd.DataFrame, output_dir: str, dataset_label: str
) -> Tuple[pd.DataFrame, str]:
    matrix = confusion_matrix(
        results_df["ground_truth_degradation"],
        results_df["predicted_degradation"],
        labels=DEGRADATION_ORDER,
    )
    matrix_df = pd.DataFrame(matrix, index=DEGRADATION_ORDER, columns=DEGRADATION_ORDER)
    matrix_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    matrix_df.to_csv(matrix_csv_path)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(DEGRADATION_ORDER)))
    ax.set_yticks(range(len(DEGRADATION_ORDER)))
    ax.set_xticklabels(DEGRADATION_ORDER, rotation=25, ha="right")
    ax.set_yticklabels(DEGRADATION_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"{dataset_label} Degradation Attribution Confusion Matrix")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                str(matrix[row_index, col_index]),
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    matrix_figure_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(matrix_figure_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "confusion_matrix.pdf"), bbox_inches="tight")
    plt.close(fig)
    return matrix_df, matrix_figure_path


def create_main_table(summary: Dict[str, float], output_dir: str) -> pd.DataFrame:
    table_df = pd.DataFrame(
        [
            {
                "Attribution Accuracy": summary["attribution_accuracy"],
                "Attribution Macro-F1": summary["attribution_macro_f1"],
                "Mean Attribution Confidence": summary["mean_attribution_confidence"],
                "Confidence-PSNR Spearman": summary["confidence_psnr_spearman"],
                "Confidence-SSIM Spearman": summary["confidence_ssim_spearman"],
                "Confidence-LPIPS Spearman": summary["confidence_lpips_spearman"],
                "High-Confidence PSNR": summary["high_confidence_mean_psnr"],
                "Low-Confidence PSNR": summary["low_confidence_mean_psnr"],
                "High-Confidence SSIM": summary["high_confidence_mean_ssim"],
                "Low-Confidence SSIM": summary["low_confidence_mean_ssim"],
                "High-Confidence LPIPS": summary["high_confidence_mean_lpips"],
                "Low-Confidence LPIPS": summary["low_confidence_mean_lpips"],
            }
        ]
    )
    csv_path = os.path.join(output_dir, "trust_main_table.csv")
    markdown_path = os.path.join(output_dir, "trust_main_table.md")
    table_df.to_csv(csv_path, index=False)
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(table_df.to_markdown(index=False))
        handle.write("\n")
    return table_df


def select_cases(results_df: pd.DataFrame) -> Tuple[str, str]:
    high_case_name = results_df.loc[results_df["confidence_score"].idxmax(), "sample_name"]
    low_case_name = results_df.loc[results_df["confidence_score"].idxmin(), "sample_name"]
    return str(high_case_name), str(low_case_name)


def analyze_single_case(
    model: nn.Module,
    data_root: str,
    split: str,
    sample_name: str,
    device: torch.device,
    prototypes: Dict[str, np.ndarray],
) -> Dict[str, object]:
    dataset = MixedDegradationDataset(data_root, split)
    matched_index = None
    for index, current_name in enumerate(dataset.image_names):
        if os.path.splitext(current_name)[0] == sample_name:
            matched_index = index
            break
    if matched_index is None:
        raise ValueError(f"Sample not found in split '{split}': {sample_name}")

    sample_name, label, lq_tensor, gt_tensor = dataset[matched_index]
    lq_tensor = lq_tensor.unsqueeze(0).to(device)
    gt_tensor = gt_tensor.unsqueeze(0).to(device)

    hook = EncoderFeatureHook(model)
    tta_outputs = []
    feature_vector = None
    with torch.inference_mode():
        for transform in TTA_TRANSFORMS:
            transformed_input = transform.forward(lq_tensor)
            prediction, features = forward_with_features(model, hook, transformed_input)
            restored_prediction = transform.inverse(prediction)
            tta_outputs.append(restored_prediction.detach().cpu())
            if transform.name == "identity":
                feature_vector = features[0].detach().cpu().numpy()
    hook.close()

    stacked_outputs = torch.stack(tta_outputs, dim=0)
    mean_prediction = stacked_outputs.mean(dim=0)
    uncertainty_map = stacked_outputs.std(dim=0, unbiased=False).mean(dim=1, keepdim=True)
    predicted_label, attribution_confidence, _, probabilities = compute_attribution(feature_vector, prototypes)

    return {
        "sample_name": sample_name,
        "ground_truth_degradation": label,
        "predicted_degradation": predicted_label,
        "attribution_confidence": attribution_confidence,
        "probabilities": probabilities,
        "degraded": tensor_to_gray_numpy(lq_tensor),
        "restored": tensor_to_gray_numpy(mean_prediction),
        "ground_truth": tensor_to_gray_numpy(gt_tensor),
        "uncertainty_map": tensor_to_gray_numpy(uncertainty_map),
    }


def create_main_figure(
    case_payloads: Sequence[Dict[str, object]],
    results_df: pd.DataFrame,
    output_dir: str,
    dataset_label: str,
) -> str:
    fig, axes = plt.subplots(
        len(case_payloads),
        6,
        figsize=(18, 4.8 * len(case_payloads)),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.0, 1.0, 1.2]},
    )
    if len(case_payloads) == 1:
        axes = np.expand_dims(axes, axis=0)

    uncertainty_max = max(float(np.max(case["uncertainty_map"])) for case in case_payloads)
    uncertainty_max = max(uncertainty_max, 1e-8)

    for row_index, case in enumerate(case_payloads):
        sample_row = results_df.loc[results_df["sample_name"] == case["sample_name"]].iloc[0]
        error_map = np.abs(case["restored"] - case["ground_truth"])
        panels = [
            (case["degraded"], "Degraded Input", "gray", None),
            (case["restored"], "Restored Output", "gray", None),
            (case["ground_truth"], "Ground Truth", "gray", None),
            (error_map, "Error Map", "magma", (0.0, max(float(error_map.max()), 1e-8))),
            (
                case["uncertainty_map"],
                "Uncertainty Map",
                "viridis",
                (0.0, uncertainty_max),
            ),
        ]

        for col_index, (image, title, cmap, limits) in enumerate(panels):
            axis = axes[row_index, col_index]
            if limits is None:
                axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
            else:
                axis.imshow(image, cmap=cmap, vmin=limits[0], vmax=limits[1])
            axis.set_title(title, fontsize=11)
            axis.axis("off")

        bar_axis = axes[row_index, 5]
        probabilities = [case["probabilities"][label] for label in DEGRADATION_ORDER]
        bar_axis.bar(range(len(DEGRADATION_ORDER)), probabilities, color=["#c44e52", "#55a868", "#8172b2", "#4c72b0"])
        bar_axis.set_ylim(0.0, 1.0)
        bar_axis.set_xticks(range(len(DEGRADATION_ORDER)))
        bar_axis.set_xticklabels(
            [pretty_degradation_label(label) for label in DEGRADATION_ORDER],
            rotation=25,
            ha="right",
        )
        bar_axis.set_ylabel("Probability")
        bar_axis.set_title("Degradation Attribution", fontsize=11)

        group_label = "High-confidence case" if row_index == 0 else "Low-confidence case"
        row_text = (
            f"{group_label}\n"
            f"Sample: {case['sample_name']}\n"
            f"GT: {pretty_degradation_label(case['ground_truth_degradation'])}\n"
            f"Pred: {pretty_degradation_label(case['predicted_degradation'])}\n"
            f"Attr. conf.: {sample_row['attribution_confidence']:.3f}\n"
            f"Conf.: {sample_row['confidence_score']:.3f}\n"
            f"PSNR: {sample_row['psnr']:.2f} dB\n"
            f"SSIM: {sample_row['ssim']:.4f}"
        )
        axes[row_index, 0].text(
            -0.35,
            0.5,
            row_text,
            transform=axes[row_index, 0].transAxes,
            fontsize=10,
            va="center",
            ha="left",
        )

    fig.suptitle(
        f"{dataset_label} trust analysis: restoration uncertainty and degradation attribution",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.02, 0.01, 1.0, 0.98])
    output_png = os.path.join(output_dir, "trust_main_figure.png")
    output_pdf = os.path.join(output_dir, "trust_main_figure.pdf")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return output_pdf


def export_case_assets(
    case_payload: Dict[str, object],
    sample_row: pd.Series,
    case_dir: Path,
    group_label: str,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)

    degraded = case_payload["degraded"]
    restored = case_payload["restored"]
    ground_truth = case_payload["ground_truth"]
    uncertainty_map = case_payload["uncertainty_map"]
    error_map = np.abs(restored - ground_truth)

    save_grayscale_png(degraded, case_dir / "degraded.png")
    save_grayscale_png(restored, case_dir / "restored.png")
    save_grayscale_png(ground_truth, case_dir / "ground_truth.png")
    save_heatmap_png(
        error_map,
        case_dir / "error_map.png",
        cmap="magma",
        vmin=0.0,
        vmax=max(float(error_map.max()), 1e-8),
    )
    save_heatmap_png(
        uncertainty_map,
        case_dir / "uncertainty_map.png",
        cmap="viridis",
        vmin=0.0,
        vmax=max(float(uncertainty_map.max()), 1e-8),
    )

    bar_fig, bar_ax = plt.subplots(figsize=(4.0, 3.2))
    probabilities = [case_payload["probabilities"][label] for label in DEGRADATION_ORDER]
    bar_ax.bar(
        range(len(DEGRADATION_ORDER)),
        probabilities,
        color=["#c44e52", "#55a868", "#8172b2", "#4c72b0"],
    )
    bar_ax.set_ylim(0.0, 1.0)
    bar_ax.set_xticks(range(len(DEGRADATION_ORDER)))
    bar_ax.set_xticklabels(
        [pretty_degradation_label(label) for label in DEGRADATION_ORDER],
        rotation=25,
        ha="right",
    )
    bar_ax.set_ylabel("Probability")
    bar_ax.set_title("Degradation Attribution")
    bar_fig.tight_layout()
    bar_fig.savefig(case_dir / "degradation_attribution.png", dpi=300, bbox_inches="tight")
    plt.close(bar_fig)

    np.save(case_dir / "degraded.npy", degraded)
    np.save(case_dir / "restored.npy", restored)
    np.save(case_dir / "ground_truth.npy", ground_truth)
    np.save(case_dir / "error_map.npy", error_map)
    np.save(case_dir / "uncertainty_map.npy", uncertainty_map)

    metadata = {
        "group": group_label,
        "sample_name": case_payload["sample_name"],
        "ground_truth_degradation": case_payload["ground_truth_degradation"],
        "predicted_degradation": case_payload["predicted_degradation"],
        "attribution_confidence": float(sample_row["attribution_confidence"]),
        "confidence_score": float(sample_row["confidence_score"]),
        "uncertainty_score": float(sample_row["uncertainty_score"]),
        "psnr": float(sample_row["psnr"]),
        "ssim": float(sample_row["ssim"]),
        "lpips": float(sample_row["lpips"]),
        "high_uncertainty_ratio": float(sample_row["high_uncertainty_ratio"]),
        "probabilities": {
            label: float(case_payload["probabilities"][label]) for label in DEGRADATION_ORDER
        },
    }
    save_json(case_dir / "metadata.json", metadata)


def export_ranked_case_assets(
    model: nn.Module,
    results_df: pd.DataFrame,
    data_root: str,
    split: str,
    device: torch.device,
    prototypes: Dict[str, np.ndarray],
    output_dir: str,
    top_k: int,
) -> None:
    cases_root = Path(output_dir) / "cases"
    if cases_root.exists():
        shutil.rmtree(cases_root)
    for group_name, ranked_df in (
        ("high", results_df.nlargest(top_k, "confidence_score")),
        ("low", results_df.nsmallest(top_k, "confidence_score")),
    ):
        for _, sample_row in ranked_df.iterrows():
            case_payload = analyze_single_case(
                model=model,
                data_root=data_root,
                split=split,
                sample_name=str(sample_row["sample_name"]),
                device=device,
                prototypes=prototypes,
            )
            case_dir = cases_root / group_name / str(sample_row["sample_name"])
            export_case_assets(case_payload, sample_row, case_dir, group_name)


def write_summary_markdown(
    summary: Dict[str, float],
    high_case_name: str,
    low_case_name: str,
    output_dir: str,
    dataset_label: str,
) -> None:
    markdown_path = os.path.join(output_dir, "trust_summary.md")
    lines = [
        f"# {dataset_label} Trust Analysis Summary",
        "",
        f"- Attribution accuracy: {summary['attribution_accuracy']:.4f}",
        f"- Attribution macro-F1: {summary['attribution_macro_f1']:.4f}",
        f"- Mean attribution confidence: {summary['mean_attribution_confidence']:.4f}",
        f"- Confidence-PSNR Spearman: {summary['confidence_psnr_spearman']:.4f}",
        f"- Confidence-SSIM Spearman: {summary['confidence_ssim_spearman']:.4f}",
        f"- Confidence-LPIPS Spearman: {summary['confidence_lpips_spearman']:.4f}",
        f"- High-confidence mean PSNR: {summary['high_confidence_mean_psnr']:.2f} dB",
        f"- Low-confidence mean PSNR: {summary['low_confidence_mean_psnr']:.2f} dB",
        f"- High-confidence mean SSIM: {summary['high_confidence_mean_ssim']:.4f}",
        f"- Low-confidence mean SSIM: {summary['low_confidence_mean_ssim']:.4f}",
        f"- High-confidence mean LPIPS: {summary['high_confidence_mean_lpips']:.4f}",
        f"- Low-confidence mean LPIPS: {summary['low_confidence_mean_lpips']:.4f}",
        f"- Selected high-confidence case: {high_case_name}",
        f"- Selected low-confidence case: {low_case_name}",
        "",
    ]
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trust analysis for PAT-Mamba")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "/root/autodl-tmp/PAT-Mamba/checkpoints/"
            "PAT-Mamba_mixed_multi_degradation_peitai_high_20251105_163440/"
            "best_model_psnr36.97_mixed_multi_degradation_peitai_high_20251105_220425.pth"
        ),
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/mixed_multi_degradation_peitai_high",
        help="Dataset root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split used for confidence analysis",
    )
    parser.add_argument(
        "--prototype_split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split used to build degradation prototypes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Execution device, for example cuda:0 or cpu",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size used for prototype extraction",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--degradation_types",
        type=str,
        nargs="+",
        default=None,
        help="Override degradation prefixes used for label inference",
    )
    parser.add_argument(
        "--case_export_count",
        type=int,
        default=10,
        help="Number of high-confidence and low-confidence cases to export as individual assets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_label = infer_dataset_label(args.data_root)

    global DEGRADATION_ORDER
    DEGRADATION_ORDER = (
        args.degradation_types if args.degradation_types is not None else default_degradation_order(args.data_root)
    )
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "/root/demo_review/PAT-Mamba/trust_results",
            dataset_label.lower(),
        )
    os.makedirs(args.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Dataset label: {dataset_label}")
    print(f"Degradation types: {DEGRADATION_ORDER}")

    model = load_model(args.checkpoint, device)

    print("\nBuilding degradation prototypes...")
    prototypes = build_prototypes(
        model=model,
        data_root=args.data_root,
        split=args.prototype_split,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    np.save(
        os.path.join(args.output_dir, "degradation_prototypes.npy"),
        prototypes,
        allow_pickle=True,
    )
    print("Prototype extraction finished.")

    print(f"\nRunning {dataset_label} trust analysis on the {args.split} split...")
    results_df = analyze_test_samples(
        model=model,
        data_root=args.data_root,
        split=args.split,
        device=device,
        prototypes=prototypes,
    )
    results_csv_path = os.path.join(args.output_dir, "per_sample_metrics.csv")
    results_df.to_csv(results_csv_path, index=False)
    print("Per-sample analysis finished.")

    summary = compute_summary(results_df)
    summary["checkpoint"] = args.checkpoint
    summary["data_root"] = args.data_root
    summary["analysis_split"] = args.split
    summary["prototype_split"] = args.prototype_split

    high_case_name, low_case_name = select_cases(results_df)
    summary["high_confidence_case"] = high_case_name
    summary["low_confidence_case"] = low_case_name

    save_json(os.path.join(args.output_dir, "summary_metrics.json"), summary)

    matrix_df, matrix_figure_path = save_confusion_artifacts(results_df, args.output_dir, dataset_label)
    create_main_table(summary, args.output_dir)

    high_case = analyze_single_case(
        model=model,
        data_root=args.data_root,
        split=args.split,
        sample_name=high_case_name,
        device=device,
        prototypes=prototypes,
    )
    low_case = analyze_single_case(
        model=model,
        data_root=args.data_root,
        split=args.split,
        sample_name=low_case_name,
        device=device,
        prototypes=prototypes,
    )
    main_figure_path = create_main_figure([high_case, low_case], results_df, args.output_dir, dataset_label)

    high_confidence_path = os.path.join(args.output_dir, "high_confidence_cases.csv")
    low_confidence_path = os.path.join(args.output_dir, "low_confidence_cases.csv")
    results_df.nlargest(args.case_export_count, "confidence_score").to_csv(high_confidence_path, index=False)
    results_df.nsmallest(args.case_export_count, "confidence_score").to_csv(low_confidence_path, index=False)
    export_ranked_case_assets(
        model=model,
        results_df=results_df,
        data_root=args.data_root,
        split=args.split,
        device=device,
        prototypes=prototypes,
        output_dir=args.output_dir,
        top_k=args.case_export_count,
    )
    write_summary_markdown(summary, high_case_name, low_case_name, args.output_dir, dataset_label)

    print("\nAnalysis completed successfully.")
    print(f"Per-sample results: {results_csv_path}")
    print(f"Summary metrics: {os.path.join(args.output_dir, 'summary_metrics.json')}")
    print(f"Confusion matrix CSV: {os.path.join(args.output_dir, 'confusion_matrix.csv')}")
    print(f"Confusion matrix figure: {matrix_figure_path}")
    print(f"Main figure: {main_figure_path}")
    print(f"Main table: {os.path.join(args.output_dir, 'trust_main_table.csv')}")
    print(f"Case assets root: {os.path.join(args.output_dir, 'cases')}")


if __name__ == "__main__":
    main()
