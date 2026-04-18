from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/PRO6000_disk/swd/ele_servo_gpt5/experiments_smoke_20260316")
MODELS = ROOT / "models"
FIG_DIR = ROOT / "figures" / "four_modality"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_bar(labels, values, title, ylabel, out_path: Path, color="#2C6E49"):
    plt.figure(figsize=(10, 4.8))
    x = np.arange(len(labels))
    bars = plt.bar(x, values, color=color)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, min(1.02, max(values) + 0.1))
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_line(series_map, title, ylabel, out_path: Path):
    plt.figure(figsize=(10, 5))
    for label, values in series_map.items():
        plt.plot(range(1, len(values) + 1), values, marker="o", linewidth=1.8, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confusion(ax, cm, labels, title):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return im


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Main staged results
    stage_paths = {
        "Stage1": MODELS / "exp1_decoupled_v3" / "stage1_encoder_cls" / "report.json",
        "Stage2": MODELS / "exp1_decoupled_v3" / "stage2_signal_fusion" / "report.json",
        "Stage3": MODELS / "exp1_decoupled_v3" / "stage3_signal_text_align" / "report.json",
        "Stage4": MODELS / "exp1_decoupled_v3" / "stage4_signal_text_llm" / "report.json",
        "Stage3+Qwen": MODELS / "exp1_decoupled_v3_qwen_dialog_qa_lora" / "stage3_signal_text_align_qwen" / "report.json",
        "Stage4+Qwen": MODELS / "exp1_decoupled_v3_qwen_dialog_qa_lora" / "stage4_signal_text_llm_qwen_cls" / "report.json",
    }
    labels = []
    vals = []
    for name, path in stage_paths.items():
        d = load_json(path)
        labels.append(name)
        vals.append(float(d["test_scenario_accuracy"]))
    plot_bar(labels, vals, "Four-Modality Staged Framework Test Accuracy", "Test Accuracy", FIG_DIR / "staged_framework_test_accuracy.png")

    # Learning curves
    curve_paths = {
        "Stage1": MODELS / "exp1_decoupled_v3" / "stage1_encoder_cls" / "report.json",
        "Stage2": MODELS / "exp1_decoupled_v3" / "stage2_signal_fusion" / "report.json",
        "Stage3": MODELS / "exp1_decoupled_v3" / "stage3_signal_text_align" / "report.json",
        "Stage4": MODELS / "exp1_decoupled_v3" / "stage4_signal_text_llm" / "report.json",
    }
    series_map = {}
    for name, path in curve_paths.items():
        d = load_json(path)
        series_map[name] = [float(row["val_scenario_accuracy"]) for row in d["history"]]
    plot_line(series_map, "Validation Accuracy Across Epochs", "Validation Accuracy", FIG_DIR / "staged_framework_val_curves.png")

    # Traditional baselines
    trad_paths = {
        "CNN-TCN": MODELS / "traditional_compare_current_modalities" / "cnn_tcn" / "cnn_tcn_current_modalities_report.json",
        "BiLSTM": MODELS / "traditional_compare_current_modalities" / "bilstm" / "bilstm_current_modalities_report.json",
        "ResNet-FCN": MODELS / "traditional_compare_current_modalities" / "resnet_fcn" / "resnet_fcn_current_modalities_report.json",
        "Transformer": MODELS / "traditional_compare_current_modalities" / "transformer_encoder" / "transformer_encoder_current_modalities_report.json",
        "Dual-branch XAttn": MODELS / "traditional_compare_current_modalities" / "dual_branch_xattn" / "dual_branch_xattn_current_modalities_report.json",
    }
    labels = []
    vals = []
    for name, path in trad_paths.items():
        d = load_json(path)
        labels.append(name)
        vals.append(float(d["test_acc"]))
    plot_bar(labels, vals, "Traditional Baselines Under Current Four-Modality Input", "Test Accuracy", FIG_DIR / "traditional_baselines_test_accuracy.png", color="#5B8E7D")

    # Main ablation
    ablation_paths = {
        "Full": MODELS / "exp1_decoupled_v3_ablation_async" / "A0_full" / "stage4_signal_text_llm" / "report.json",
        "No Position": MODELS / "exp1_decoupled_v3_ablation_async" / "A1_no_position" / "stage4_signal_text_llm" / "report.json",
        "No Electrical": MODELS / "exp1_decoupled_v3_ablation_async" / "A2_no_electrical" / "stage4_signal_text_llm" / "report.json",
        "No Thermal": MODELS / "exp1_decoupled_v3_ablation_async" / "A3_no_thermal" / "stage4_signal_text_llm" / "report.json",
        "No Vibration": MODELS / "exp1_decoupled_v3_ablation_async" / "A4_no_vibration" / "stage4_signal_text_llm" / "report.json",
    }
    labels = []
    vals = []
    for name, path in ablation_paths.items():
        d = load_json(path)
        labels.append(name)
        vals.append(float(d["test_scenario_accuracy"]))
    plot_bar(labels, vals, "Main Model Leave-One-Out Ablation", "Test Accuracy", FIG_DIR / "main_model_ablation.png", color="#BC6C25")

    # CNN-TCN ablation
    cnn_paths = {
        "Full": MODELS / "cnn_tcn_modality_ablation_current_modalities_async" / "cnn_tcn_full" / "cnn_tcn_full_report.json",
        "No Position": MODELS / "cnn_tcn_modality_ablation_current_modalities_async" / "cnn_tcn_no_position" / "cnn_tcn_no_position_report.json",
        "No Electrical": MODELS / "cnn_tcn_modality_ablation_current_modalities_async" / "cnn_tcn_no_electrical" / "cnn_tcn_no_electrical_report.json",
        "No Thermal": MODELS / "cnn_tcn_modality_ablation_current_modalities_async" / "cnn_tcn_no_thermal" / "cnn_tcn_no_thermal_report.json",
        "No Vibration": MODELS / "cnn_tcn_modality_ablation_current_modalities_async" / "cnn_tcn_no_vibration" / "cnn_tcn_no_vibration_report.json",
    }
    labels = []
    vals = []
    for name, path in cnn_paths.items():
        d = load_json(path)
        labels.append(name)
        vals.append(float(d["test_acc"]))
    plot_bar(labels, vals, "CNN-TCN Leave-One-Out Ablation", "Test Accuracy", FIG_DIR / "cnn_tcn_ablation.png", color="#A44A3F")

    # Single modality
    single_paths = {
        "Only Electrical": MODELS / "exp1_decoupled_v3_single_modality_async" / "only_electrical" / "stage2_signal_fusion" / "report.json",
        "Only Position": MODELS / "exp1_decoupled_v3_single_modality_async" / "only_position" / "stage2_signal_fusion" / "report.json",
        "Only Vibration": MODELS / "exp1_decoupled_v3_single_modality_async" / "only_vibration" / "stage2_signal_fusion" / "report.json",
        "Only Thermal": MODELS / "exp1_decoupled_v3_single_modality_async" / "only_thermal" / "stage3_signal_text_align" / "report.json",
    }
    labels = []
    vals = []
    for name, path in single_paths.items():
        d = load_json(path)
        labels.append(name)
        vals.append(float(d["test_scenario_accuracy"]))
    plot_bar(labels, vals, "Single-Modality Best Results", "Test Accuracy", FIG_DIR / "single_modality_results.png", color="#6C91C2")

    # QA result
    qa_report = load_json(MODELS / "exp1_decoupled_v3_qwen_dialog_qa_lora" / "stage4_fault_dialog_qa_lora" / "report.json")
    plot_bar(
        ["Dialog QA"],
        [float(qa_report["test_scenario_accuracy"])],
        "Qwen Dialog QA Scenario Accuracy",
        "Scenario Accuracy",
        FIG_DIR / "qwen_dialog_qa_accuracy.png",
        color="#7C4D8B",
    )

    # Confusion matrices
    stage2_cm = load_json(MODELS / "exp1_decoupled_v3" / "stage2_signal_fusion" / "confusion_matrix.json")
    stage3q_cm = load_json(MODELS / "exp1_decoupled_v3_qwen_dialog_qa_lora" / "stage3_signal_text_align_qwen" / "confusion_matrix.json")
    labels = [stage2_cm["labels"][str(i)] for i in range(len(stage2_cm["labels"]))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im1 = plot_confusion(axes[0], np.array(stage2_cm["test_confusion_matrix"]), labels, "Stage 2 Test Confusion Matrix")
    im2 = plot_confusion(axes[1], np.array(stage3q_cm["test_confusion_matrix"]), labels, "Stage 3 + Qwen Test Confusion Matrix")
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrices_stage2_vs_stage3qwen.png", dpi=180)
    plt.close()

    print(FIG_DIR)


if __name__ == "__main__":
    main()
