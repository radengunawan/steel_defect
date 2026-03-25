"""
demo.py — Run the full pipeline on a single image (no server needed)
----------------------------------------------------------------------
Generates a synthetic steel surface texture if no real image is given,
runs the full OpenCV + CNN pipeline, and saves result images.

Usage
-----
  # Demo with synthetic image (no dataset needed):
  python demo.py

  # Run on a real image:
  python demo.py --image path/to/steel.jpg

  # Use a trained checkpoint:
  python demo.py --image path/to/steel.jpg --checkpoint checkpoints/best_model.pt

Output
------
  output/annotated.jpg     — annotated frame with bboxes + Grad-CAM
  output/pipeline_grid.jpg — all OpenCV intermediate stages in one image
"""

import utils.session
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ── Synthetic steel surface generator (no dataset needed for demo) ──────────

def generate_synthetic_steel(
    width: int = 320,
    height: int = 320,
    defect_type: str = "crack",
    noise_std: float = 18.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Render a plausible steel texture + one injected defect.

    defect_type : "crack" | "pit" | "scale" | "scratch" | "clean"
    """
    rng = np.random.default_rng(seed)
    # Base steel texture — gaussian noise + slow rolling gradient
    base = 130 + rng.normal(0, noise_std, (height, width)).astype(np.float32)
    xx, yy = np.meshgrid(np.linspace(0, 6, width), np.linspace(0, 6, height))
    base += 12 * np.sin(xx) * np.cos(yy * 0.7)
    base = np.clip(base, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    if defect_type == "crack":
        # Thin dark line with slight jitter
        x0, y0 = rng.integers(30, 80), rng.integers(40, 80)
        for k in range(180):
            px = int(x0 + k * 1.1 + rng.normal(0, 2))
            py = int(y0 + k * 0.35 + rng.normal(0, 1.2))
            if 0 <= px < width and 0 <= py < height:
                for dw in range(-1, 2):
                    if 0 <= py + dw < height:
                        bgr[py + dw, px] = (35, 35, 35)

    elif defect_type == "pit":
        for _ in range(rng.integers(3, 7)):
            cx, cy = rng.integers(20, width - 20), rng.integers(20, height - 20)
            r = rng.integers(5, 14)
            cv2.circle(bgr, (cx, cy), r, (45, 45, 45), -1)

    elif defect_type == "scale":
        sy = rng.integers(height // 4, 3 * height // 4)
        for x in range(width):
            for dy in range(-8, 9):
                y = int(sy + dy * 0.4 + rng.normal(0, 1.5))
                if 0 <= y < height:
                    v = int(70 + rng.uniform(0, 40))
                    bgr[y, x] = (v, v, v)

    elif defect_type == "scratch":
        x0, y0 = rng.integers(10, 50), rng.integers(10, 50)
        for k in range(200):
            px = int(x0 + k + rng.normal(0, 1))
            py = int(y0 + k * 0.05 + rng.normal(0, 0.8))
            if 0 <= px < width and 0 <= py < height:
                bgr[py, px] = (20, 20, 20)

    return bgr


# ── Pipeline visualisation grid ────────────────────────────────────────────

def make_pipeline_grid(steps: dict) -> np.ndarray:
    """Tile the OpenCV intermediate stages into a labelled grid image."""
    stage_names = ["gray", "equalized", "blurred", "edges", "binary", "cleaned"]
    cols = []
    for name in stage_names:
        img = steps.get(name)
        if img is None:
            continue
        # Convert to BGR for display
        if img.ndim == 2:
            disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            disp = img.copy()
        disp = cv2.resize(disp, (200, 160))
        # Label banner
        cv2.rectangle(disp, (0, 0), (200, 22), (30, 30, 30), -1)
        cv2.putText(disp, name.upper(), (6, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cols.append(disp)

    row1 = np.hstack(cols[:3])
    row2 = np.hstack(cols[3:])
    return np.vstack([row1, row2])


# ── Main ────────────────────────────────────────────────────────────────────

def main(args):
    # ── Step 1: get image ───────────────────────────────────────────────────
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"ERROR: Cannot read {args.image}"); sys.exit(1)
        print(f"Loaded image: {args.image}  {bgr.shape}")
    else:
        defect = args.defect
        print(f"No image provided — generating synthetic steel surface ({defect})")
        bgr = generate_synthetic_steel(defect_type=defect)

    # ── Step 2: import engine (after optional sys.path fix) ─────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.preprocessor import SteelPreprocessor
    from utils.inference_engine import InferenceEngine

    # ── Step 3: run full pipeline ────────────────────────────────────────────
    print("Running inference pipeline...")
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        device="cpu",
        edge_threshold=args.edge_threshold,
        use_gradcam=True,
    )
    result = engine.predict(bgr)

    # ── Step 4: print results ────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Prediction  : {result.label}")
    print(f"  Confidence  : {result.confidence*100:.1f}%")
    print(f"  Regions     : {len(result.regions)} anomaly blob(s) detected")
    print(f"\n  Per-class probabilities:")
    for cls, p in sorted(result.all_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 30)
        print(f"    {cls:<20s} {p*100:5.1f}%  {bar}")
    print(f"{'='*50}\n")

    # ── Step 5: save outputs ─────────────────────────────────────────────────
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    annotated_path = out_dir / "annotated.jpg"
    cv2.imwrite(str(annotated_path), result.annotated_bgr)
    print(f"Saved: {annotated_path}")

    grid = make_pipeline_grid(result.processing_steps)
    grid_path = out_dir / "pipeline_grid.jpg"
    cv2.imwrite(str(grid_path), grid)
    print(f"Saved: {grid_path}")

    # Show with cv2 if display available
    try:
        cv2.imshow("Annotated", result.annotated_bgr)
        cv2.imshow("Pipeline stages", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass  # headless environment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steel defect detection demo")
    parser.add_argument("--image",          default=None,   help="Path to input image")
    parser.add_argument("--defect",         default="crack",
                        choices=["crack","pit","scale","scratch","clean"],
                        help="Defect type for synthetic generation")
    parser.add_argument("--checkpoint",     default=None,   help="Path to .pt checkpoint")
    parser.add_argument("--edge_threshold", type=int, default=40)
    main(parser.parse_args())
