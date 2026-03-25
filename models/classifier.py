"""
Steel Surface Defect Classifier
--------------------------------
Fine-tuned EfficientNet-B0 on NEU Steel Surface Defect Dataset.
6 defect classes: Crazing, Inclusion, Patches, Pitted Surface,
                   Rolled-in Scale, Scratches
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from pathlib import Path


DEFECT_CLASSES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted Surface",
    "Rolled-in Scale",
    "Scratches",
]

CLASS_COLORS_BGR = {
    "Crazing":          (50,  205, 50),
    "Inclusion":        (0,   165, 255),
    "Patches":          (0,   255, 255),
    "Pitted Surface":   (0,   0,   255),
    "Rolled-in Scale":  (255, 0,   0),
    "Scratches":        (255, 0,   255),
}


class SteelDefectClassifier(nn.Module):
    """
    EfficientNet-B0 backbone with a custom classification head.

    Architecture
    ------------
    Backbone  : EfficientNet-B0 (ImageNet pre-trained, partially frozen)
    Head      : Dropout(0.4) → Linear(1280 → 256) → ReLU → Dropout(0.3)
                → Linear(256 → num_classes)

    Training notes
    --------------
    - Unfreeze last 3 blocks of backbone after 5 warm-up epochs
    - Use label smoothing (0.1) + mixup augmentation
    - LR schedule: cosine annealing with warm restarts
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Freeze all backbone layers initially (fine-tuning strategy)
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 feature blocks for fine-tuning
        for block in list(backbone.features.children())[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        in_features = backbone.classifier[1].in_features  # 1280

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def unfreeze_all(self):
        """Call after warm-up epochs to allow full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> SteelDefectClassifier:
    """Load a trained model from a .pt checkpoint."""
    model = SteelDefectClassifier(num_classes=len(DEFECT_CLASSES), pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and {'model_state_dict': ...} formats
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)
