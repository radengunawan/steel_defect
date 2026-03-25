# Steel Surface Defect Detection

Real-time defect detection for steel mill surfaces using OpenCV preprocessing + EfficientNet-B0 CNN fine-tuned on the NEU Steel Surface Defect Dataset.

## Architecture

```
Input image
    │
    ▼
┌─────────────────────────────────────┐
│  OpenCV Preprocessing Pipeline      │
│  ① Grayscale + CLAHE equalisation   │
│  ② Gaussian denoising (σ auto)      │
│  ③ Sobel edge detection (X + Y)     │
│  ④ Otsu adaptive thresholding       │
│  ⑤ Morphological open/close         │
│  ⑥ Blob / contour extraction        │
└──────────────┬──────────────────────┘
               │ ROI crops + features
               ▼
┌─────────────────────────────────────┐
│  EfficientNet-B0 CNN                │
│  • ImageNet pre-trained backbone    │
│  • Custom head: 1280→256→6          │
│  • Fine-tuned on NEU dataset        │
│  • Grad-CAM visualisation           │
└──────────────┬──────────────────────┘
               │
               ▼
    Defect class + confidence
    + bounding boxes + heatmap
```

**Defect classes** (NEU Steel Surface Dataset):
`Crazing` · `Inclusion` · `Patches` · `Pitted Surface` · `Rolled-in Scale` · `Scratches`

---

## Project Structure

```
steel_defect_detector/
├── models/
│   └── classifier.py       # EfficientNet-B0 model definition
├── utils/
│   ├── preprocessor.py     # Full OpenCV pipeline
│   └── inference_engine.py # Preprocessing + CNN + Grad-CAM combined
├── api/
│   └── server.py           # FastAPI REST server
├── data/
│   └── samples/            # Put test images here
├── tests/
│   └── test_pipeline.py    # Unit tests (no dataset needed)
├── checkpoints/            # Saved .pt model weights (after training)
├── output/                 # Demo output images
├── train.py                # Training script
├── demo.py                 # Standalone demo (synthetic or real image)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run demo (no dataset, no checkpoint needed)

```bash
python demo.py                        # synthetic crack defect
python demo.py --defect pit           # synthetic pit defect
python demo.py --defect scale
python demo.py --defect scratch
```

Outputs saved to `output/annotated.jpg` and `output/pipeline_grid.jpg`.

### 3. Train on NEU dataset

Download from Kaggle: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

Organise:
```
data/
  train/
    Crazing/
    Inclusion/
    Patches/
    Pitted_Surface/
    Rolled-in_Scale/
    Scratches/
  val/   (same structure, ~20% split)
```

Train:
```bash
python train.py \
  --data_dir data \
  --epochs 30 \
  --batch_size 32 \
  --output checkpoints/
```

Expected result: ~97% validation accuracy after 30 epochs on an A100; ~94% on CPU with 15 epochs.

### 4. Inference on a real image

```bash
python demo.py --image path/to/steel.jpg --checkpoint checkpoints/best_model.pt
```

### 5. Start the API server

```bash
MODEL_CHECKPOINT=checkpoints/best_model.pt uvicorn api.server:app --port 8000
```

Then open http://localhost:8000/docs for interactive Swagger UI.

**Example cURL:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@steel_sample.jpg" \
  -F "edge_threshold=40" \
  | python -m json.tool
```

---

## API Reference

| Method | Endpoint         | Description                          |
|--------|------------------|--------------------------------------|
| GET    | `/health`        | Liveness check + model status        |
| GET    | `/classes`       | List of 6 defect class names         |
| POST   | `/predict`       | Single image → JSON + annotated JPEG |
| POST   | `/predict/batch` | Up to 16 images in one request       |

**POST /predict** query params:
- `edge_threshold` (int, 5–200, default 40) — Sobel sensitivity
- `return_gradcam` (bool, default true) — include Grad-CAM heatmap

---

## OpenCV Pipeline Details

| Step | Method | Purpose |
|------|--------|---------|
| Grayscale | `cvtColor` | Remove colour noise |
| CLAHE | `createCLAHE(clip=2.0, grid=8×8)` | Enhance low-contrast surface features |
| Denoise | `GaussianBlur(5×5)` | Suppress sensor/grain noise |
| Edge detect | `Sobel(ksize=3)` X+Y combined | Highlight surface discontinuities |
| Threshold | `THRESH_BINARY + THRESH_OTSU` | Binarise edge map adaptively |
| Morphology | `MORPH_CLOSE` then `MORPH_OPEN` | Fill gaps, remove specks |
| Blob extract | `findContours` | Locate candidate defect regions |

---

## Model Details

- **Backbone**: EfficientNet-B0 (1280-dim feature vector)
- **Head**: Dropout(0.4) → Linear(256) → ReLU → Dropout(0.3) → Linear(6)
- **Training**: Label smoothing 0.1, Mixup α=0.4, AdamW, CosineAnnealing
- **Fine-tuning strategy**: Freeze all backbone for 5 warm-up epochs, then unfreeze last 2 blocks; unfreeze everything at epoch 6
- **Grad-CAM**: Hooks last EfficientNet feature block to show which regions drive the classification

---

## Running Tests

```bash
pytest tests/ -v
```

All 12 tests pass without a dataset or checkpoint (uses synthetic tensors/images).
