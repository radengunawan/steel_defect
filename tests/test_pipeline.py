"""
Unit Tests — Steel Defect Detection Pipeline
Tests run without a trained checkpoint or real dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch


# ── Model tests ──────────────────────────────────────────────────────────────

class TestSteelDefectClassifier:
    def setup_method(self):
        from models.classifier import SteelDefectClassifier, DEFECT_CLASSES
        self.model  = SteelDefectClassifier(num_classes=6, pretrained=False)
        self.n_cls  = len(DEFECT_CLASSES)

    def test_output_shape(self):
        x = torch.randn(4, 3, 224, 224)
        out = self.model(x)
        assert out.shape == (4, self.n_cls)

    def test_predict_proba_sums_to_one(self):
        x = torch.randn(2, 3, 224, 224)
        probs = self.model.predict_proba(x)
        assert probs.shape == (2, self.n_cls)
        np.testing.assert_allclose(probs.sum(dim=1).numpy(), [1.0, 1.0], atol=1e-5)

    def test_unfreeze_all(self):
        self.model.unfreeze_all()
        for p in self.model.parameters():
            assert p.requires_grad

    def test_frozen_backbone_initially(self):
        from models.classifier import SteelDefectClassifier
        model = SteelDefectClassifier(num_classes=6, pretrained=False)
        # First backbone layers should be frozen
        first_block = list(model.features.children())[0]
        for p in first_block.parameters():
            assert not p.requires_grad


# ── Preprocessor tests ────────────────────────────────────────────────────────

class TestSteelPreprocessor:
    def setup_method(self):
        from utils.preprocessor import SteelPreprocessor
        self.prep = SteelPreprocessor(edge_threshold=30, min_defect_area=20)

    def _make_bgr(self, h=200, w=200):
        rng = np.random.default_rng(0)
        gray = (130 + rng.normal(0, 15, (h, w))).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    def test_run_returns_all_keys(self):
        bgr = self._make_bgr()
        result = self.prep.run(bgr)
        for key in ("gray","equalized","blurred","edges","binary","cleaned","regions","annotated"):
            assert key in result, f"Missing key: {key}"

    def test_gray_output_shape(self):
        bgr = self._make_bgr(150, 180)
        result = self.prep.run(bgr)
        assert result["gray"].shape == (150, 180)

    def test_annotated_is_bgr(self):
        bgr = self._make_bgr()
        result = self.prep.run(bgr)
        assert result["annotated"].ndim == 3
        assert result["annotated"].shape[2] == 3

    def test_regions_are_sorted_by_area(self):
        bgr = self._make_bgr()
        result = self.prep.run(bgr)
        areas = [r.area for r in result["regions"]]
        assert areas == sorted(areas, reverse=True)

    def test_defect_region_crop(self):
        from utils.preprocessor import DefectRegion
        bgr = self._make_bgr(100, 100)
        r = DefectRegion(x=10, y=10, w=30, h=20, area=600,
                         aspect_ratio=1.5, mean_intensity=120.0, edge_density=0.1)
        crop = r.crop(bgr, padding=5)
        assert crop.ndim == 3


# ── Inference engine tests ─────────────────────────────────────────────────────

class TestInferenceEngine:
    def setup_method(self):
        from utils.inference_engine import InferenceEngine
        # No checkpoint — demo mode
        self.engine = InferenceEngine(checkpoint_path=None, device="cpu", use_gradcam=False)

    def _make_bgr(self):
        rng = np.random.default_rng(1)
        gray = (130 + rng.normal(0, 15, (200, 200))).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    def test_predict_returns_valid_label(self):
        from models.classifier import DEFECT_CLASSES
        result = self.engine.predict(self._make_bgr())
        assert result.label in DEFECT_CLASSES

    def test_confidence_in_range(self):
        result = self.engine.predict(self._make_bgr())
        assert 0.0 <= result.confidence <= 1.0

    def test_all_probs_sum_to_one(self):
        result = self.engine.predict(self._make_bgr())
        total = sum(result.all_probs.values())
        assert abs(total - 1.0) < 5e-4

    def test_annotated_shape_matches_input(self):
        bgr = self._make_bgr()
        result = self.engine.predict(bgr)
        assert result.annotated_bgr.shape == bgr.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
