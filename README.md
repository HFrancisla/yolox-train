# yolox-train

Private training repo for YOLOX on custom datasets.

> Tested on NVIDIA RTX 50-series GPUs (Blackwell / CUDA 13+) with PyTorch 2.14+ and Python 3.12 managed by `uv`.

## Repository Structure

```text
.
├── datasets/
│   └── triangle-2851/           # Six-class triangle transfer dataset (2851 images, VOC format)
├── exps/
│   ├── default/                 # Upstream default experiments (yolox_s, yolox_tiny, etc.)
│   └── triangle/
│       └── yolox_triangle_tiny.py  # Custom experiment config for triangle-2851
├── tools/
│   ├── train.py                 # Training entrypoint
│   ├── eval.py                  # Evaluation entrypoint
│   ├── export_onnx.py           # ONNX export (PyTorch 2.x compatible)
│   ├── split_and_copy_dataset.py   # VOC dataset split helper
│   └── demo.py                  # Inference demo
├── yolox/
│   └── data/datasets/
│       └── triangle_voc.py      # Custom VOC dataset adapter for arbitrary class sets
├── docs/human-read/             # Deployment & reproduction guides
│   ├── YOLOX训练准备与复现指南.md
│   └── Openvino与onnx模型.md
├── _record/                     # Training run records
├── weights/                     # Pretrained weights (not tracked by git)
├── pyproject.toml               # uv / PEP 621 project config
├── uv.lock                      # Locked dependency graph (Python 3.12)
└── TRAINING.md                  # Step-by-step training walkthrough
```

## Quick Start

```bash
# Install dependencies
uv sync

# Train
python tools/train.py \
    -n triangle_yolox_16 \
    -f exps/triangle/yolox_triangle_tiny.py \
    -d 1 -b 16 --fp16 \
    -c weights/yolox_tiny.pth
```

See [TRAINING.md](./TRAINING.md) for the full walkthrough including dataset preparation, evaluation, and ONNX export.

## Dataset

| Split | Images | Annotations |
|-------|--------|-------------|
| train | 2,280  | —           |
| val   | 571    | —           |
| **total** | **2,851** | **21,800** |

Six classes: `0011c1rod`, `001jcu5ox`, `001ob4y7z`, `001py6kcr`, `001rq6x2k`, `001xlffir`

Dataset path: `datasets/triangle-2851/` (ignored by git, set `TRIANGLE_DATA_DIR` env var to override).

## Latest Results (2026-09-20, epoch 300)

| Metric | Value |
|--------|-------|
| mAP@0.50 | **89.81%** |
| mAP@0.50:0.95 | **75.74%** |
| Inference | 2.50 ms / image |

See [`_record/20260920-yolox-triangle/训练记录.md`](./_record/20260920-yolox-triangle/训练记录.md) for full details.
