"""
OpenCV Preprocessing Pipeline
-------------------------------
All image-processing steps that run before the CNN:
  1. Grayscale conversion & histogram equalization (CLAHE)
  2. Gaussian denoising
  3. Sobel edge detection + binary thresholding
  4. Morphological operations for blob cleanup
  5. Defect Region-of-Interest (ROI) extraction
  6. Visualization overlay helpers
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


# ── Types ──────────────────────────────────────────────────────────────────

@dataclass
class DefectRegion:
    """Bounding box + metadata for one detected anomaly region."""
    x: int
    y: int
    w: int
    h: int
    area: int
    aspect_ratio: float
    mean_intensity: float
    edge_density: float
    contour: np.ndarray = field(repr=False, default=None)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def crop(self, image: np.ndarray, padding: int = 8) -> np.ndarray:
        """Return padded crop of this region from the source image."""
        h, w = image.shape[:2]
        x1 = max(0, self.x - padding)
        y1 = max(0, self.y - padding)
        x2 = min(w, self.x + self.w + padding)
        y2 = min(h, self.y + self.h + padding)
        return image[y1:y2, x1:x2]


# ── Pipeline ───────────────────────────────────────────────────────────────

class SteelPreprocessor:
    """
    Full OpenCV preprocessing pipeline for steel surface images.

    Usage
    -----
    >>> prep = SteelPreprocessor(edge_threshold=40, min_defect_area=50)
    >>> result = prep.run(bgr_image)
    >>> regions = result["regions"]
    >>> annotated = result["annotated"]
    """

    def __init__(
        self,
        edge_threshold: int   = 40,
        min_defect_area: int  = 50,
        max_defect_area: int  = 50_000,
        clahe_clip: float     = 2.0,
        clahe_grid: int       = 8,
        blur_kernel: int      = 5,
        morph_kernel: int     = 3,
    ):
        self.edge_threshold   = edge_threshold
        self.min_defect_area  = min_defect_area
        self.max_defect_area  = max_defect_area
        self.clahe            = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )
        self.blur_kernel      = blur_kernel
        self.morph_kernel     = morph_kernel

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, bgr: np.ndarray) -> dict:
        """
        Full pipeline. Returns dict with keys:
            gray, equalized, blurred, edges, binary,
            cleaned, regions, annotated
        """
        gray      = self._to_gray(bgr)
        equalized = self._equalize(gray)
        blurred   = self._denoise(equalized)
        edges     = self._detect_edges(blurred)
        binary    = self._threshold(edges)
        cleaned   = self._morphology(binary)
        regions   = self._find_regions(cleaned, gray)
        annotated = self._annotate(bgr.copy(), regions)

        return {
            "gray":      gray,
            "equalized": equalized,
            "blurred":   blurred,
            "edges":     edges,
            "binary":    binary,
            "cleaned":   cleaned,
            "regions":   regions,
            "annotated": annotated,
        }

    def extract_roi_crops(
        self,
        bgr: np.ndarray,
        regions: List[DefectRegion],
        size: Tuple[int, int] = (224, 224),
    ) -> List[np.ndarray]:
        """Return list of resized BGR crops for CNN inference."""
        crops = []
        for region in regions:
            crop = region.crop(bgr, padding=12)
            if crop.size == 0:
                continue
            crops.append(cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR))
        return crops

    # ── Pipeline Steps ─────────────────────────────────────────────────────

    def _to_gray(self, bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def _equalize(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE — improves contrast on low-contrast steel textures."""
        return self.clahe.apply(gray)

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        k = self.blur_kernel | 1  # ensure odd
        return cv2.GaussianBlur(gray, (k, k), sigmaX=0)

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Sobel in X and Y, combined as gradient magnitude."""
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _threshold(self, edges: np.ndarray) -> np.ndarray:
        """Otsu-guided adaptive threshold over Sobel map."""
        _, binary = cv2.threshold(
            edges, self.edge_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    def _morphology(self, binary: np.ndarray) -> np.ndarray:
        """Close small gaps, remove isolated noise specks."""
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        )
        closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
        opened  = cv2.morphologyEx(closed,  cv2.MORPH_OPEN,  k, iterations=1)
        return opened

    def _find_regions(
        self, binary: np.ndarray, gray: np.ndarray
    ) -> List[DefectRegion]:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_defect_area <= area <= self.max_defect_area):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio   = w / max(h, 1)
            roi_gray       = gray[y:y+h, x:x+w]
            mean_intensity = float(roi_gray.mean()) if roi_gray.size else 0.0
            # Edge density = fraction of edge pixels inside bounding box
            edge_density   = float(binary[y:y+h, x:x+w].mean()) / 255.0
            regions.append(DefectRegion(
                x=x, y=y, w=w, h=h,
                area=int(area),
                aspect_ratio=round(aspect_ratio, 2),
                mean_intensity=round(mean_intensity, 1),
                edge_density=round(edge_density, 3),
                contour=cnt,
            ))
        # Sort largest first
        return sorted(regions, key=lambda r: r.area, reverse=True)

    def _annotate(self, bgr: np.ndarray, regions: List[DefectRegion]) -> np.ndarray:
        overlay = bgr.copy()
        for i, r in enumerate(regions):
            color = (0, 220, 100)
            cv2.rectangle(overlay, (r.x, r.y), (r.x+r.w, r.y+r.h), color, 2)
            if r.contour is not None:
                cv2.drawContours(overlay, [r.contour], -1, (0, 180, 255), 1)
            label = f"#{i+1} area={r.area}"
            cv2.putText(
                overlay, label,
                (r.x, max(r.y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )
        return overlay
