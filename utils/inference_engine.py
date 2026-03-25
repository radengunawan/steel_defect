"""
Inference Engine
----------------
Combines:
  1. SteelPreprocessor  — OpenCV pipeline (edges, blobs, ROI extraction)
  2. SteelDefectClassifier — EfficientNet-B0 CNN
  3. GradCAM            — visual explanation of what the model looks at

Usage
-----
  engine = InferenceEngine("checkpoints/best_model.pt")
  result = engine.predict("path/to/image.jpg")
  cv2.imwrite("out.jpg", result.annotated_bgr)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.classifier import SteelDefectClassifier, DEFECT_CLASSES, CLASS_COLORS_BGR
from utils.preprocessor import SteelPreprocessor, DefectRegion


# ── ImageNet normalisation (must match training) ───────────────────────────
_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    _NORM,
])


# ── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    label: str
    confidence: float                    # 0-1
    all_probs: dict[str, float]          # {class_name: probability}
    regions: List[DefectRegion]          # OpenCV-detected blobs
    annotated_bgr: np.ndarray            # Final overlay image
    gradcam_heatmap: Optional[np.ndarray]  # Grad-CAM heatmap (or None)
    processing_steps: dict[str, np.ndarray]  # Intermediate OpenCV stages


# ── Grad-CAM ────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Hooks into the last convolutional block of EfficientNet-B0.
    """
    def __init__(self, model: SteelDefectClassifier):
        self.model = model
        self._fmaps: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        target_layer = list(model.features.children())[-1]  # last feature block
        self._fh = target_layer.register_forward_hook(self._save_fmaps)
        self._bh = target_layer.register_full_backward_hook(self._save_grads)

    def _save_fmaps(self, _, __, output):
        self._fmaps = output.detach()

    def _save_grads(self, _, __, grad_output):
        self._grads = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a (H, W) heatmap in [0, 1]."""
        self.model.zero_grad()
        out = self.model(input_tensor)
        out[0, class_idx].backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
        cam = (weights * self._fmaps).sum(dim=1).squeeze(0)
        cam = F.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self._fh.remove()
        self._bh.remove()


# ── Inference Engine ────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Full inference pipeline: load image → preprocess → CNN → annotate.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to a .pt checkpoint produced by train.py.
        Pass None to run with a randomly-initialised model (demo mode).
    device : str
        "cuda", "cpu", or "mps"
    edge_threshold : int
        Passed to SteelPreprocessor. Controls blob detection sensitivity.
    use_gradcam : bool
        Whether to compute Grad-CAM heatmaps (slightly slower).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: str = "cpu",
        edge_threshold: int = 40,
        use_gradcam: bool = True,
    ):
        self.device = torch.device(device)
        self.model  = self._load_model(checkpoint_path)
        self.prep   = SteelPreprocessor(edge_threshold=edge_threshold)
        self.gradcam = GradCAM(self.model) if use_gradcam else None
        print(f"InferenceEngine ready | device={device} | gradcam={use_gradcam}")

    # ── Public API ────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray | str | Path) -> PredictionResult:
        """
        Run full pipeline on an image.

        Parameters
        ----------
        image : BGR numpy array, or path to an image file.
        """
        bgr = self._load_image(image)
        steps = self.prep.run(bgr)
        regions = steps["regions"]

        # CNN inference on the full frame (224×224 resize)
        tensor = self._preprocess_for_cnn(bgr)          # (1, 3, 224, 224)
        probs  = self._run_cnn(tensor)                   # (num_classes,)
        label  = DEFECT_CLASSES[probs.argmax()]
        conf   = float(probs.max())
        all_probs = {c: round(float(p), 4) for c, p in zip(DEFECT_CLASSES, probs)}

        # Grad-CAM heatmap
        heatmap = None
        if self.gradcam is not None:
            t = tensor.clone().requires_grad_(True)
            cam = self.gradcam.generate(t.to(self.device), probs.argmax())
            heatmap = cv2.resize(cam, (bgr.shape[1], bgr.shape[0]))

        annotated = self._build_annotation(bgr, steps, label, conf, all_probs, heatmap)

        return PredictionResult(
            label=label,
            confidence=conf,
            all_probs=all_probs,
            regions=regions,
            annotated_bgr=annotated,
            gradcam_heatmap=heatmap,
            processing_steps={k: v for k, v in steps.items() if k != "regions"},
        )

    # ── Internals ─────────────────────────────────────────────────────────

    def _load_model(self, ckpt: Optional[str | Path]) -> SteelDefectClassifier:
        model = SteelDefectClassifier(num_classes=len(DEFECT_CLASSES), pretrained=False)
        if ckpt and Path(ckpt).exists():
            state = torch.load(ckpt, map_location=self.device)
            sd = state.get("model_state_dict", state)
            model.load_state_dict(sd)
            print(f"Loaded checkpoint: {ckpt}")
        else:
            print("WARNING: No checkpoint — using untrained model (demo mode)")
        model.eval().to(self.device)
        return model

    def _load_image(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
            return bgr
        return image.copy()

    def _preprocess_for_cnn(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))
        tensor = _TO_TENSOR(rgb).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def _run_cnn(self, tensor: torch.Tensor) -> np.ndarray:
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs

    def _build_annotation(
        self,
        bgr: np.ndarray,
        steps: dict,
        label: str,
        conf: float,
        all_probs: dict,
        heatmap: Optional[np.ndarray],
    ) -> np.ndarray:
        out = bgr.copy()

        # Blend Grad-CAM heatmap if available
        if heatmap is not None:
            hm_uint8 = (heatmap * 255).astype(np.uint8)
            colored  = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
            out = cv2.addWeighted(out, 0.6, colored, 0.4, 0)

        # Draw blob bounding boxes from OpenCV stage
        for i, r in enumerate(steps["regions"]):
            color = CLASS_COLORS_BGR.get(label, (0, 220, 100))
            cv2.rectangle(out, (r.x, r.y), (r.x + r.w, r.y + r.h), color, 2)
            cv2.putText(out, f"#{i+1}", (r.x, r.y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Top-left HUD: prediction summary
        h_img = out.shape[0]
        self._draw_hud(out, label, conf, all_probs)

        return out

    @staticmethod
    def _draw_hud(
        img: np.ndarray, label: str, conf: float, all_probs: dict
    ) -> None:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (240, 30 + len(all_probs) * 18 + 12), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        cv2.putText(img, f"{label} {conf*100:.1f}%", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        for i, (cls, prob) in enumerate(sorted(all_probs.items(), key=lambda x: -x[1])):
            bar_w = int(prob * 120)
            y = 36 + i * 18
            cv2.rectangle(img, (8, y), (8 + bar_w, y + 10),
                          CLASS_COLORS_BGR.get(cls, (180, 180, 180)), -1)
            cv2.putText(img, f"{cls[:14]:14s} {prob*100:4.1f}%", (140, y + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1, cv2.LINE_AA)
