"""Triplet loss module."""

import torch
import torch.nn as nn


# loss = max(0, ||x - p|| - ||x - n|| + alpha) + max(0, ||x - p|| - beta)
class TripletLoss(nn.Module):
    """Triplet loss module."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.1) -> None:
        """Initialize Triplet loss module."""
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Forward pass of Triplet loss module."""
        pos_dist = torch.norm(x - p, p=2, dim=1).mean()
        neg_dist = torch.norm(x - n, p=2, dim=1).mean()
        loss = torch.clamp(pos_dist - neg_dist + self.alpha, min=0.0)
        loss += torch.clamp(pos_dist - self.beta, min=0.0)
        return loss
