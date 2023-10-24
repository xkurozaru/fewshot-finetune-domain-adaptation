"""L2 loss module."""

import torch
import torch.nn as nn


# loss = ||x - y||_2
class L2Loss(nn.Module):
    """L2 loss module."""

    def __init__(self) -> None:
        """Initialize L2 loss module."""
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of L2 loss module."""
        return torch.norm(x - y, p=2, dim=1).mean()
