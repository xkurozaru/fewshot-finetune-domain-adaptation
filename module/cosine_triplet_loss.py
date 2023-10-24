"""Cosine Triplet Loss module."""

import torch
import torch.nn as nn


# loss = max(0, 1 - cos(x, p) + cos(x, n)) + max(0, 1 - cos(x, p))
class CosineTripletLoss(nn.Module):
    """Cosine Triplet loss module."""

    def __init__(self, alpha: float = 0.0, beta: float = 0.1) -> None:
        """Initialize Cosine Triplet loss module."""
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Forward pass of Cosine Triplet loss module."""
        pos_sim = torch.cosine_similarity(x, p).mean()
        neg_sim = torch.cosine_similarity(x, n).mean()
        loss = torch.clamp(1.0 - pos_sim + neg_sim - self.alpha, min=0.0)
        loss += torch.clamp(1.0 - pos_sim - self.beta, min=0.0)
        return loss
