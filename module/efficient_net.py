"""EfficientNet model for Metric Domain Adaptation."""

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class EfficientNetEncoder(nn.Module):
    """EfficientNet encoder."""

    def __init__(self) -> None:
        """Initialize EfficientNet encoder."""
        super(EfficientNetEncoder, self).__init__()
        self.encoder = nn.Sequential(
            efficientnet_v2_s(EfficientNet_V2_S_Weights.IMAGENET1K_V1).features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of EfficientNet encoder."""
        x = self.encoder(x)
        return x


class EfficientNetClassifier(nn.Module):
    """EfficientNet classifier."""

    def __init__(self, num_classes: int) -> None:
        """Initialize EfficientNet classifier."""
        super(EfficientNetClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of EfficientNet classifier."""
        x = self.classifier(x)
        return x


class BaselinePP(nn.Module):
    """Baseline++ model."""

    def __init__(self, num_classes: int) -> None:
        """Initialize Baseline++ model."""
        super(BaselinePP, self).__init__()
        self.protos_layer = nn.Linear(1280, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Baseline++ model."""
        x = x / torch.norm(x, dim=1, keepdim=True)
        protos = self.protos_layer.weight / torch.norm(self.protos_layer.weight, dim=1, keepdim=True)
        cos_sim = torch.mm(x, protos.t())

        return cos_sim
