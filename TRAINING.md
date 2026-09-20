# Triangle-transfer YOLOX training

This checkout uses the official YOLOX project at commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`.
The original training server and its custom experiment file are unavailable, so
the project-local experiment keeps the historical YOLOX-tiny settings that are
known from the training record and uses the available `JQ-M-001-0049` dataset.

This is a pipeline reproduction, not a bit-for-bit reproduction of the old
`JQ-M-001-0047` run. The old run used 439 images; the available dataset has
2,851 images.

## Environment

This machine has Python 3.12, so use a project-local virtual environment:

```bash
cd /home/hzf/workspace/projects-dev/yolox-train
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -U pip
uv pip install -r requirements-train.txt
uv pip install --no-build-isolation -e . --no-deps
```

The `requirements-train.txt` install includes PyTorch. Verify the CUDA runtime
with:

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)'
```

The upstream `requirements.txt` also pins an old `onnx-simplifier` release;
that dependency is not needed for training and currently fails to build on
Python 3.12. Install an export toolchain separately if ONNX export is needed.

## Prepare the available dataset

```bash
cd /home/hzf/workspace/projects-dev/yolox-train
mkdir -p data
unzip -q \
  '/home/hzf/workspace/personal/career/Practice_record/20251112_20250206传递三角/训练数据2851张-JQ-M-001-0049.zip' \
  -d data

python tools/split_and_copy_dataset.py \
  --data_dir data/JQ-M-01-0049 \
  --train_ratio 0.8 \
  --seed 42
```

Expected split: 2,280 train images and 571 validation images. The experiment
expects the generated `VOC2007/` directory under `data/JQ-M-01-0049/`.

## Train

The old server used batch size 32 on a 24 GB GPU. This workstation has an
8 GB GPU, so start with batch size 8 and increase only if it fits:

```bash
cd /home/hzf/workspace/projects-dev/yolox-train
source .venv/bin/activate

TRIANGLE_DATA_DIR="$PWD/data/JQ-M-01-0049" \
CUDA_VISIBLE_DEVICES=0 \
python tools/train.py \
  -f exps/triangle/yolox_triangle_tiny.py \
  -expn triangle_0049 \
  -d 1 \
  -b 8 \
  --fp16 \
  -o \
  -c weights/yolox_tiny.pth
```

If no pretrained checkpoint is available, omit `-c` to train from a random
initialization. That is valid for a pipeline smoke test but is not comparable
to the historical run.

## Evaluate

```bash
TRIANGLE_DATA_DIR="$PWD/data/JQ-M-01-0049" \
CUDA_VISIBLE_DEVICES=0 \
python tools/eval.py \
  -f exps/triangle/yolox_triangle_tiny.py \
  -c YOLOX_outputs/triangle_0049/best_ckpt.pth \
  -b 8 \
  -d 1 \
  --conf 0.1 \
  --nms 0.5 \
  --fp16 \
  --fuse
```

The custom dataset adapter is needed because upstream YOLOX's built-in VOC
loader is hard-coded to Pascal VOC's 20 class names.
