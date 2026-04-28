"""
MIL Anomaly Detection Model and Loss.

Implements the architecture and MIL ranking loss from:
    Sultani et al. "Real-World Anomaly Detection in Surveillance Videos"
    CVPR 2018. https://arxiv.org/abs/1801.04264

Architecture: FC(4096→512, ReLU) → Dropout(0.6) → FC(512→32) → Dropout(0.6) → FC(32→1, Sigmoid)
Loss: MIL ranking loss with sparsity and smoothness constraints.
"""

import torch
import torch.nn as nn


class AnomalyMILModel(nn.Module):
    """MIL-based anomaly scorer.
    
    Takes a bag of 32 C3D segment features (4096-dim each) and produces
    an anomaly score (0-1) for each segment:
    
        Input:  (batch, 32, 4096)
        Output: (batch, 32, 1)  — anomaly score per segment
    """
    
    def __init__(
        self,
        feature_dim: int = 4096,
        hidden_dim: int = 512,
        bottleneck_dim: int = 32,
        dropout: float = 0.6,
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_segments, feature_dim) — bag of segment features
            
        Returns:
            (batch, num_segments, 1) — anomaly score per segment
        """
        return self.classifier(x)


class MILRankingLoss(nn.Module):
    """MIL ranking loss from Sultani et al.
    
    Given anomaly and normal bags, the loss encourages:
    1. max(anomaly_scores) > max(normal_scores) + margin  (ranking)
    2. Anomaly scores should be sparse (temporal sparsity)
    3. Anomaly scores should be smooth (temporal smoothness)
    
    Total loss = ranking_loss + λ_sparse * sparsity + λ_smooth * smoothness
    """
    
    def __init__(
        self,
        lambda_sparse: float = 8e-5,
        lambda_smooth: float = 8e-5,
        margin: float = 1.0,
    ):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_smooth = lambda_smooth
        self.margin = margin
    
    def forward(
        self,
        anomaly_scores: torch.Tensor,
        normal_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            anomaly_scores: (batch, 32, 1) — segment scores for anomaly bags
            normal_scores:  (batch, 32, 1) — segment scores for normal bags
            
        Returns:
            Dict with 'total', 'ranking', 'sparsity', 'smoothness' losses
        """
        # Squeeze last dim: (batch, 32)
        a_scores = anomaly_scores.squeeze(-1)  # (B, 32)
        n_scores = normal_scores.squeeze(-1)   # (B, 32)
        
        # ── 1. MIL Ranking Loss ──
        # max score from each bag
        a_max = a_scores.max(dim=1).values  # (B,)
        n_max = n_scores.max(dim=1).values  # (B,)
        
        # Hinge ranking loss: we want a_max > n_max by at least margin
        ranking_loss = torch.relu(self.margin - a_max + n_max).mean()
        
        # ── 2. Sparsity Loss ──
        # Encourage anomaly scores to be sparse (few high-scoring segments)
        sparsity_loss = a_scores.mean()
        
        # ── 3. Temporal Smoothness Loss ──
        # Penalize large differences between consecutive segments
        diff = (a_scores[:, 1:] - a_scores[:, :-1]) ** 2
        smoothness_loss = diff.mean()
        
        # ── Total Loss ──
        total_loss = (
            ranking_loss
            + self.lambda_sparse * sparsity_loss
            + self.lambda_smooth * smoothness_loss
        )
        
        return {
            "total": total_loss,
            "ranking": ranking_loss,
            "sparsity": sparsity_loss,
            "smoothness": smoothness_loss,
        }
