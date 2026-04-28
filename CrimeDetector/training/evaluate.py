"""
Evaluate a trained MIL Anomaly Detection model.

Usage:
    python -m training.evaluate --dataset_root Dataset --checkpoint model_weights/best_model.pt

Computes:
- Video-level AUC (ROC)
- Per-category detection rates
- Confusion matrix at optimal threshold
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset import scan_dataset, load_c3d_features
from training.model import AnomalyMILModel


def evaluate_model(checkpoint_path: str, dataset_root: str):
    """Full evaluation of a trained model."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")
    
    # ── Load checkpoint ──
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_args = checkpoint.get("args", {})
    model = AnomalyMILModel(
        feature_dim=4096,
        hidden_dim=model_args.get("hidden_dim", 512),
        bottleneck_dim=model_args.get("bottleneck_dim", 32),
        dropout=model_args.get("dropout", 0.6),
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    train_auc = checkpoint.get("auc", "N/A")
    train_epoch = checkpoint.get("epoch", "N/A")
    print(f"   Trained for {train_epoch} epochs, best AUC: {train_auc}")
    
    # ── Load test data ──
    splits = scan_dataset(dataset_root)
    
    test_anomaly_paths = splits["anomaly_test"]
    test_normal_paths = splits["normal_test"]
    
    print(f"\n📊 Test set: {len(test_anomaly_paths)} anomaly, "
          f"{len(test_normal_paths)} normal videos")
    
    # ── Score all test videos ──
    all_scores = []
    all_labels = []
    all_names = []
    
    print("\nScoring test videos...")
    
    with torch.no_grad():
        # Anomaly videos
        for path in test_anomaly_paths:
            try:
                features = load_c3d_features(path)
                features_t = torch.from_numpy(features).unsqueeze(0).to(device)
                scores = model(features_t).squeeze()  # (32,)
                max_score = scores.max().item()
                all_scores.append(max_score)
                all_labels.append(1)
                all_names.append(os.path.basename(path))
            except Exception as e:
                print(f"  [WARN] Skipping {path}: {e}")
        
        # Normal videos
        for path in test_normal_paths:
            try:
                features = load_c3d_features(path)
                features_t = torch.from_numpy(features).unsqueeze(0).to(device)
                scores = model(features_t).squeeze()
                max_score = scores.max().item()
                all_scores.append(max_score)
                all_labels.append(0)
                all_names.append(os.path.basename(path))
            except Exception as e:
                print(f"  [WARN] Skipping {path}: {e}")
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # ── AUC ──
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    
    print(f"\n{'='*50}")
    print(f"  Video-Level ROC-AUC: {auc:.4f}")
    print(f"  Average Precision:   {ap:.4f}")
    print(f"{'='*50}")
    
    # ── Find optimal threshold ──
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[min(best_f1_idx, len(thresholds) - 1)]
    best_f1 = f1_scores[best_f1_idx]
    
    print(f"\n  Optimal threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
    
    # ── Confusion matrix at optimal threshold ──
    predictions = (all_scores >= best_threshold).astype(int)
    cm = confusion_matrix(all_labels, predictions)
    
    print(f"\n  Confusion Matrix (threshold={best_threshold:.4f}):")
    print(f"  {'':>12} Pred Normal  Pred Anomaly")
    print(f"  {'True Normal':>12}  {cm[0,0]:>8}     {cm[0,1]:>8}")
    print(f"  {'True Anomaly':>12}  {cm[1,0]:>8}     {cm[1,1]:>8}")
    
    print(f"\n  Classification Report:")
    print(classification_report(
        all_labels, predictions,
        target_names=["Normal", "Anomaly"],
        digits=4,
    ))
    
    # ── Per-category analysis ──
    print(f"\n{'='*50}")
    print("  Per-Category Detection (Anomaly videos):")
    print(f"{'='*50}")
    
    import re
    category_scores = {}
    for name, score, label in zip(all_names, all_scores, all_labels):
        if label == 0:
            continue
        match = re.match(r"([A-Za-z]+)\d+", name)
        if match:
            cat = match.group(1)
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(score)
    
    for cat in sorted(category_scores.keys()):
        scores = category_scores[cat]
        detected = sum(1 for s in scores if s >= best_threshold)
        avg_score = np.mean(scores)
        print(f"  {cat:>20}: {detected}/{len(scores)} detected, "
              f"avg_score={avg_score:.4f}")
    
    print(f"\n✅ Evaluation complete!")
    return auc


def main():
    parser = argparse.ArgumentParser(description="Evaluate MIL Anomaly Model")
    parser.add_argument(
        "--checkpoint", type=str, default="model_weights/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset_root", type=str, default="Dataset",
        help="Path to Dataset/ directory"
    )
    args = parser.parse_args()
    evaluate_model(args.checkpoint, args.dataset_root)


if __name__ == "__main__":
    main()
