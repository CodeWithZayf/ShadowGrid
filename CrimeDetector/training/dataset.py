"""
UCF-Crime C3D Feature Dataset for MIL Training.

Loads pre-extracted C3D features (32 segments × 4096 dims) from text files
and pairs anomaly/normal bags for MIL ranking loss training.

References:
    Sultani et al. "Real-World Anomaly Detection in Surveillance Videos"
    CVPR 2018. https://arxiv.org/abs/1801.04264
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────
FEATURE_DIM = 4096
NUM_SEGMENTS = 32

# Map from video path pattern → C3D feature directory
ANOMALY_FEATURE_DIRS = [
    "Anomaly-Vidoes-Part-1-txt",  # Abuse, Arrest, Arson, Assault
    "Anomaly-Vidoes-Part-2-txt",  # Burglary, Explosion, Fighting
    "Anomaly-Vidoes-Part-3-txt",  # RoadAccidents, Robbery, Shooting
    "Anomaly-Vidoes-Part-4-txt",  # Shoplifting, Stealing, Vandalism
]

NORMAL_FEATURE_DIR = "Training-Normal-Videos-Part-1-txt"
TEST_NORMAL_DIR = "Testing_Normal_Videos_Anomaly_txt"


# ── Feature Loading ───────────────────────────────────────────────────
def load_c3d_features(filepath: str) -> np.ndarray:
    """Load C3D features from a text file.
    
    Each file contains 32 lines, each with 4096 space-separated floats.
    Returns shape (32, 4096).
    """
    # Open in binary mode to avoid Windows cp1252 encoding issues
    with open(filepath, "rb") as f:
        features = np.loadtxt(f, dtype=np.float32, max_rows=NUM_SEGMENTS)
    if features.shape != (NUM_SEGMENTS, FEATURE_DIM):
        raise ValueError(
            f"Unexpected shape {features.shape} for {filepath}. "
            f"Expected ({NUM_SEGMENTS}, {FEATURE_DIM})"
        )
    return features


def _resolve_anomaly_feature_path(video_name: str, all_data_root: str) -> str | None:
    """Convert an anomaly video name to its C3D feature file path.
    
    Anomaly_Train.txt has entries like: Abuse/Abuse001_x264.mp4
    C3D features are at: All_Data/Anomaly-Vidoes-Part-X-txt/Abuse/Abuse001_x264_C.txt
    """
    # Extract category and base name
    parts = video_name.strip().replace("\\", "/").split("/")
    if len(parts) < 2:
        return None
    
    category = parts[-2]
    base_name = Path(parts[-1]).stem  # Remove .mp4
    feature_filename = f"{base_name}_C.txt"
    
    # Search across all anomaly feature directories
    for feat_dir in ANOMALY_FEATURE_DIRS:
        candidate = os.path.join(all_data_root, feat_dir, category, feature_filename)
        if os.path.exists(candidate):
            return candidate
    
    return None


def _resolve_normal_feature_path(entry: str, all_data_root: str) -> str | None:
    """Convert a Normal_Train.txt entry to its full file path.
    
    Normal_Train.txt has entries like:
        Training-Normal-Videos-Part-1-txt/Normal_Videos001_x264_C.txt
    """
    candidate = os.path.join(all_data_root, entry.strip())
    if os.path.exists(candidate):
        return candidate
    return None


# ── Dataset Scanner ───────────────────────────────────────────────────
def scan_dataset(dataset_root: str) -> dict:
    """Scan the UCF-Crime dataset and return paths for train/test splits.
    
    Args:
        dataset_root: Path to the Dataset/ directory
        
    Returns:
        Dict with keys: 'anomaly_train', 'normal_train', 'anomaly_test', 'normal_test'
        Each value is a list of absolute paths to C3D feature files.
    """
    all_data_root = os.path.join(dataset_root, "All_Data", "All_Data")
    
    result = {
        "anomaly_train": [],
        "normal_train": [],
        "anomaly_test": [],
        "normal_test": [],
    }
    
    # ── Train anomaly files ──
    anomaly_train_file = os.path.join(dataset_root, "Anomaly_Train.txt")
    if os.path.exists(anomaly_train_file):
        with open(anomaly_train_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path = _resolve_anomaly_feature_path(line, all_data_root)
                if path:
                    result["anomaly_train"].append(path)
    
    # Also check the inner All_Data/All_Data/Anomaly_Train.txt
    inner_anomaly_train = os.path.join(all_data_root, "Anomaly_Train.txt")
    if os.path.exists(inner_anomaly_train) and not result["anomaly_train"]:
        with open(inner_anomaly_train, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path = _resolve_anomaly_feature_path(line, all_data_root)
                if path:
                    result["anomaly_train"].append(path)
    
    # ── Train normal files ──
    normal_train_file = os.path.join(dataset_root, "Normal_Train.txt")
    if os.path.exists(normal_train_file):
        with open(normal_train_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path = _resolve_normal_feature_path(line, all_data_root)
                if path:
                    result["normal_train"].append(path)
    
    # ── Test normal files ──
    test_normal_dir = os.path.join(all_data_root, TEST_NORMAL_DIR)
    if os.path.exists(test_normal_dir):
        for fname in sorted(os.listdir(test_normal_dir)):
            if fname.endswith("_C.txt"):
                result["normal_test"].append(os.path.join(test_normal_dir, fname))
    
    # ── Test anomaly files (from Anomaly_Test.txt) ──
    anomaly_test_file = os.path.join(dataset_root, "Anomaly_Test.txt")
    if os.path.exists(anomaly_test_file):
        with open(anomaly_test_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path = _resolve_anomaly_feature_path(line, all_data_root)
                if path:
                    result["anomaly_test"].append(path)
    
    return result


# ── MIL Training Dataset ─────────────────────────────────────────────
class MILAnomalyDataset(Dataset):
    """Dataset that yields (anomaly_bag, normal_bag) pairs for MIL training.
    
    Each bag is a tensor of shape (32, 4096) — 32 segments of C3D features.
    The MIL ranking loss will be applied to the max-scoring segment in each bag.
    """
    
    def __init__(self, anomaly_paths: list[str], normal_paths: list[str]):
        """
        Args:
            anomaly_paths: List of paths to anomaly C3D feature files
            normal_paths: List of paths to normal C3D feature files
        """
        self.anomaly_paths = anomaly_paths
        self.normal_paths = normal_paths
        
        if not anomaly_paths:
            raise ValueError("No anomaly training files found!")
        if not normal_paths:
            raise ValueError("No normal training files found!")
        
        print(f"[Dataset] Loaded {len(anomaly_paths)} anomaly, "
              f"{len(normal_paths)} normal videos")
    
    def __len__(self):
        # One epoch = one pass through all anomaly videos
        return len(self.anomaly_paths)
    
    def __getitem__(self, idx):
        # Load anomaly bag
        anomaly_features = load_c3d_features(self.anomaly_paths[idx])
        
        # Randomly pair with a normal bag
        normal_idx = np.random.randint(0, len(self.normal_paths))
        normal_features = load_c3d_features(self.normal_paths[normal_idx])
        
        return (
            torch.from_numpy(anomaly_features),   # (32, 4096)
            torch.from_numpy(normal_features),     # (32, 4096)
        )


# ── Test Dataset ──────────────────────────────────────────────────────
class TestVideoDataset(Dataset):
    """Dataset for evaluation — loads individual videos with labels."""
    
    def __init__(self, paths: list[str], labels: list[int]):
        """
        Args:
            paths: List of C3D feature file paths
            labels: 1 for anomaly, 0 for normal
        """
        self.paths = paths
        self.labels = labels
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        features = load_c3d_features(self.paths[idx])
        return torch.from_numpy(features), self.labels[idx]


# ── Convenience ───────────────────────────────────────────────────────
def create_dataloaders(
    dataset_root: str,
    batch_size: int = 30,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train and test data loaders.
    
    Returns:
        (train_loader, test_loader)
    """
    splits = scan_dataset(dataset_root)
    
    print(f"\n{'='*50}")
    print(f"Dataset Summary:")
    print(f"  Train anomaly: {len(splits['anomaly_train'])} videos")
    print(f"  Train normal:  {len(splits['normal_train'])} videos")
    print(f"  Test anomaly:  {len(splits['anomaly_test'])} videos")
    print(f"  Test normal:   {len(splits['normal_test'])} videos")
    print(f"{'='*50}\n")
    
    use_pin_memory = torch.cuda.is_available()
    
    # Training dataset (MIL pairs)
    train_dataset = MILAnomalyDataset(
        splits["anomaly_train"],
        splits["normal_train"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    
    # Test dataset (individual videos)
    test_paths = splits["anomaly_test"] + splits["normal_test"]
    test_labels = [1] * len(splits["anomaly_test"]) + [0] * len(splits["normal_test"])
    
    test_dataset = TestVideoDataset(test_paths, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    
    return train_loader, test_loader

