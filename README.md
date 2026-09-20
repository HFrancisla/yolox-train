# yolox-train

> **A simple repo for self YOLOX training**

`yolox-train` is a lightweight, reproducible repository for self-use [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) training, custom dataset fine-tuning, and modern inference deployment.

It modernizes the official YOLOX codebase for **Python 3.12** environments using [**uv**](https://github.com/astral-sh/uv), includes custom dataset adapters, fixes PyTorch 2.x ONNX export compatibility, and provides end-to-end CPU deployment with OpenVINO.

---

## Key Highlights

- ⚡ **Fast & Deterministic Environment (`uv`)**: Managed via PEP 621 [`pyproject.toml`](./pyproject.toml) and [`uv.lock`](./uv.lock). Automatically handles PyTorch C++ extension builds (`FastCOCOEvalOp`) with `no-build-isolation-package = ["yolox"]`.
- 🎯 **Custom Dataset Support**: Custom VOC-format dataset adapter ([`TriangleVOCDetection`](./yolox/data/datasets/triangle_voc.py)) that removes Pascal VOC's 20-class limitation, paired with automated train/val split tools ([`tools/split_and_copy_dataset.py`](./tools/split_and_copy_dataset.py)).
- 🚀 **Modern ONNX & OpenVINO Export**: Updated [`tools/export_onnx.py`](./tools/export_onnx.py) compatible with PyTorch 2.x (`torch.onnx.export(..., dynamo=False)`), and direct conversion to OpenVINO IR (`convert_model` / `save_model`).
- 📖 **Self-Contained Training Guides**: Practical step-by-step reproduction documentation in [`TRAINING.md`](./TRAINING.md) and [`docs/human-read/`](./docs/human-read/).

---

## Repository Structure

```text
.
├── exps/
│   ├── default/                 # Upstream default experiments (yolox_s, yolox_tiny, etc.)
│   └── triangle/                # Custom training experiments (e.g., yolox_triangle_tiny.py)
├── tools/
│   ├── train.py                 # Training entrypoint
│   ├── eval.py                  # Evaluation entrypoint
│   ├── export_onnx.py           # ONNX export with PyTorch 2.x support
│   └── split_and_copy_dataset.py # Automated VOC dataset split script
├── yolox/
│   └── data/datasets/
│       └── triangle_voc.py      # Custom VOC dataset adapter for arbitrary classes
├── docs/human-read/             # Detailed deployment & reproduction guides
├── pyproject.toml               # uv / PEP 621 project configuration
├── uv.lock                      # Locked dependency graph (Python 3.12)
└── TRAINING.md                  # Detailed training walkthrough
```

---

## Quick Start

### 1. Environment Setup

Install [uv](https://docs.astral.sh/uv/) if not already installed, then clone and sync the virtual environment:

```bash
git clone https://github.com/HFrancisla/yolox-train.git
cd yolox-train

# Create .venv and install all dependencies (PyTorch 2.14+, torchvision, ONNX, OpenVINO, etc.)
uv sync
```

Alternatively, manually create the virtual environment using `uv pip`:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements-train.txt
uv pip install --no-build-isolation -e . --no-deps
uv pip install -U onnx onnxsim openvino onnxruntime
```

Verify the environment:

```bash
uv run python -c "import torch, yolox; print('PyTorch:', torch.__version__, '| YOLOX:', yolox.__version__, '| CUDA:', torch.cuda.is_available())"
```

---

### 2. Dataset Preparation

Organize your annotations and images into VOC format, then split into train/val subsets:

```bash
uv run python tools/split_and_copy_dataset.py \
  --data_dir data/YOUR_DATASET \
  --train_ratio 0.8 \
  --seed 42
```

This generates `Annotations/`, `JPEGImages/`, and `ImageSets/Main/{train.txt, val.txt}` under `data/YOUR_DATASET/VOC2007/`.

---

### 3. Training

Launch training with an experiment configuration:

```bash
TRIANGLE_DATA_DIR="$PWD/data/YOUR_DATASET" \
CUDA_VISIBLE_DEVICES=0 \
uv run python tools/train.py \
  -f exps/triangle/yolox_triangle_tiny.py \
  -expn your_experiment \
  -d 1 \
  -b 8 \
  --fp16 \
  -o \
  -c weights/yolox_tiny.pth
```

> **Note**: If training from scratch without pre-trained weights, omit `-c`. For multi-GPU training, adjust `-d <num_gpus>` and `-b <total_batch_size>`.

---

### 4. Evaluation

Evaluate your best checkpoint on the validation split:

```bash
TRIANGLE_DATA_DIR="$PWD/data/YOUR_DATASET" \
CUDA_VISIBLE_DEVICES=0 \
uv run python tools/eval.py \
  -f exps/triangle/yolox_triangle_tiny.py \
  -c YOLOX_outputs/your_experiment/best_ckpt.pth \
  -b 8 \
  -d 1 \
  --conf 0.1 \
  --nms 0.5 \
  --fp16 \
  --fuse
```

---

### 5. Export to ONNX & OpenVINO

Export the trained model to ONNX:

```bash
mkdir -p artifacts

uv run python tools/export_onnx.py \
  -f exps/triangle/yolox_triangle_tiny.py \
  -c YOLOX_outputs/your_experiment/best_ckpt.pth \
  --output-name artifacts/model.onnx \
  --opset 11 \
  --batch-size 1
```

Convert ONNX to OpenVINO IR for CPU inference:

```bash
uv run python - <<'PY'
import openvino as ov

model = ov.convert_model("artifacts/model.onnx")
ov.save_model(model, "artifacts/model_openvino.xml", compress_to_fp16=True)
print("OpenVINO IR saved to artifacts/model_openvino.xml and .bin")
PY
```

For complete benchmarking and post-processing alignment (letterbox, NMS, class decoding), see [OpenVINO 与 ONNX 模型部署指南](docs/human-read/Openvino与onnx模型.md).

---

## Upstream & Acknowledgements

This repository is built upon [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). We express our gratitude to the authors for their open-source contributions.

### Citation

```bibtex
@article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

### In Memory of Dr. Jian Sun

Without the guidance of [Dr. Jian Sun](https://scholar.google.com/citations?user=ALVSZAYAAAAJ), YOLOX would not have been released to the community. We honor and remember Dr. Sun's extraordinary contributions to computer vision and artificial intelligence.
