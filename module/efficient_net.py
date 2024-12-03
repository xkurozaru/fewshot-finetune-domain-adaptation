"""EfficientNet model for Metric Domain Adaptation."""

import torch
import torch.nn as nn
from torch.autograd import Function
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


class GradientReversalLayer(Function):
    @staticmethod
    def forward(context, x, constant):
        context.constant = constant
        return x.view_as(x) * constant

    @staticmethod
    def backward(context, grad):
        return grad.neg() * context.constant, None


class DANN(nn.Module):
    """Domain Adversarial Neural Network (DANN)."""

    def __init__(self, num_classes: int) -> None:
        """Initialize DANN."""
        super(DANN, self).__init__()
        self.encoder = EfficientNetEncoder()
        self.classifier = EfficientNetClassifier(num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass of DANN."""
        x = self.encoder(x)
        y = self.classifier(x)
        d = self.domain_classifier(GradientReversalLayer.apply(x, 1.0))
        return y, d
