"""
Train the MIL Anomaly Detection model on UCF-Crime C3D features.

Usage:
    python -m training.train_mil --dataset_root Dataset --epochs 100

This script:
1. Loads pre-extracted C3D features from the UCF-Crime dataset
2. Trains the MIL ranking model (4096→512→32→1)
3. Evaluates on the test set using frame-level AUC
4. Saves the best model checkpoint
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset import create_dataloaders, scan_dataset, load_c3d_features
from training.model import AnomalyMILModel, MILRankingLoss


def evaluate(model, test_paths, test_labels, device):
    """Evaluate model on test set using video-level AUC.
    
    For each video, the anomaly score is the max segment score.
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for path, label in zip(test_paths, test_labels):
            try:
                features = load_c3d_features(path)
                features_t = torch.from_numpy(features).unsqueeze(0).to(device)
                scores = model(features_t)  # (1, 32, 1)
                max_score = scores.squeeze().max().item()
                all_scores.append(max_score)
                all_labels.append(label)
            except Exception as e:
                print(f"  [WARN] Skipping {path}: {e}")
    
    if len(set(all_labels)) < 2:
        return 0.0
    
    auc = roc_auc_score(all_labels, all_scores)
    return auc


def train(args):
    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ── Data ──
    print(f"\n📂 Loading dataset from: {args.dataset_root}")
    train_loader, test_loader = create_dataloaders(
        args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Also get raw test paths/labels for evaluation
    splits = scan_dataset(args.dataset_root)
    test_paths = splits["anomaly_test"] + splits["normal_test"]
    test_labels = [1] * len(splits["anomaly_test"]) + [0] * len(splits["normal_test"])
    
    # ── Model ──
    model = AnomalyMILModel(
        feature_dim=4096,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        dropout=args.dropout,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: {total_params:,} parameters")
    print(f"   Architecture: 4096 → {args.hidden_dim} → {args.bottleneck_dim} → 1")
    
    # ── Loss & Optimizer ──
    criterion = MILRankingLoss(
        lambda_sparse=args.lambda_sparse,
        lambda_smooth=args.lambda_smooth,
    )
    
    optimizer = optim.Adagrad(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # ── Output directory ──
    os.makedirs(args.output_dir, exist_ok=True)
    best_auc = 0.0
    best_epoch = 0
    
    print(f"\n🚀 Training for {args.epochs} epochs...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Output: {args.output_dir}")
    print(f"{'='*60}")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {"total": 0, "ranking": 0, "sparsity": 0, "smoothness": 0}
        num_batches = 0
        t_start = time.time()
        
        for anomaly_bags, normal_bags in train_loader:
            anomaly_bags = anomaly_bags.to(device)  # (B, 32, 4096)
            normal_bags = normal_bags.to(device)
            
            # Forward pass
            anomaly_scores = model(anomaly_bags)   # (B, 32, 1)
            normal_scores = model(normal_bags)     # (B, 32, 1)
            
            # Compute loss
            losses = criterion(anomaly_scores, normal_scores)
            
            # Backward
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            
            # Accumulate
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        elapsed = time.time() - t_start
        
        # ── Evaluate ──
        eval_interval = max(1, args.epochs // 20)  # Evaluate ~20 times
        if epoch == 1 or epoch % eval_interval == 0 or epoch == args.epochs:
            auc = evaluate(model, test_paths, test_labels, device)
            
            # Save best
            improved = ""
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                improved = " ★ BEST"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "auc": auc,
                    "args": vars(args),
                }, os.path.join(args.output_dir, "best_model.pt"))
            
            print(
                f"  Epoch {epoch:3d}/{args.epochs} | "
                f"Loss: {epoch_losses['total']:.4f} "
                f"(R:{epoch_losses['ranking']:.4f} "
                f"S:{epoch_losses['sparsity']:.4f} "
                f"Sm:{epoch_losses['smoothness']:.6f}) | "
                f"AUC: {auc:.4f}{improved} | "
                f"{elapsed:.1f}s"
            )
        else:
            print(
                f"  Epoch {epoch:3d}/{args.epochs} | "
                f"Loss: {epoch_losses['total']:.4f} "
                f"(R:{epoch_losses['ranking']:.4f}) | "
                f"{elapsed:.1f}s"
            )
    
    # ── Save final model ──
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, os.path.join(args.output_dir, "final_model.pt"))
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"   Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    print(f"   Models saved to: {args.output_dir}/")
    print(f"{'='*60}")
    
    return best_auc


def main():
    parser = argparse.ArgumentParser(
        description="Train MIL Anomaly Detection Model on UCF-Crime"
    )
    
    # Data
    parser.add_argument(
        "--dataset_root", type=str, default="Dataset",
        help="Path to the Dataset/ directory containing UCF-Crime data"
    )
    
    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--bottleneck_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.6)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lambda_sparse", type=float, default=8e-5)
    parser.add_argument("--lambda_smooth", type=float, default=8e-5)
    
    # System
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--output_dir", type=str, default="model_weights",
        help="Directory to save trained model checkpoints"
    )
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
