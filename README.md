# yolox-train

A simple repo for self YOLOX training

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
