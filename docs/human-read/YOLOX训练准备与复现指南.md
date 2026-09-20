# 传递三角 YOLOX 训练准备与复现指南

本文记录在原训练服务器和原 YOLOX 私有环境不可用的情况下，如何在
/home/hzf/workspace/projects-dev/yolox-train 使用官方 YOLOX 项目重新准备
并启动传递三角目标检测训练。

## 1. 复现范围

本次准备复现的是“训练流程”，不是旧模型的逐位完全复现。

旧训练记录已知的参数：

- YOLOX-tiny；
- 输入尺寸 416 x 416；
- 6 个目标类别；
- 使用 seed=42 划分 VOC 数据集；
- 原服务器单卡 batch size 约为 32；
- FP16 混合精度；
- 预训练权重 yolox_tiny.pth。

原服务器环境已不可得，旧数据集 JQ-M-001-0047（439 张图像）也不在当前
工作区。当前使用项目中现有的：

~~~text
训练数据2851张-JQ-M-001-0049.zip
~~~

该数据集包含 2,851 张图像、2,851 个 XML 标注，类别仍然是旧训练记录中的
6 个类别。因此当前方案是“旧训练参数 + 当前 0049 数据集 + 官方 YOLOX”。

## 2. 当前源码版本

训练目录使用官方 YOLOX 仓库：

~~~text
https://github.com/Megvii-BaseDetection/YOLOX.git
~~~

当前固定 commit：

~~~text
6ddff4824372906469a7fae2dc3206c7aa4bbaee
~~~

如果从空目录重新准备：

~~~bash
mkdir -p /home/hzf/workspace/projects-dev/yolox-train
cd /home/hzf/workspace/projects-dev/yolox-train
git clone https://github.com/Megvii-BaseDetection/YOLOX.git .
git checkout 6ddff4824372906469a7fae2dc3206c7aa4bbaee
~~~

本次新增的适配文件：

~~~text
TRAINING.md
requirements-train.txt
tools/split_and_copy_dataset.py
exps/triangle/yolox_triangle_tiny.py
yolox/data/datasets/triangle_voc.py
~~~

## 3. 创建训练环境

当前机器没有 conda，但有 Python 3.12 和 uv，因此使用项目内虚拟环境。

~~~bash
cd /home/hzf/workspace/projects-dev/yolox-train

uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install -U pip
uv pip install -r requirements-train.txt
uv pip install --no-build-isolation -e . --no-deps
~~~

验证 PyTorch 和 CUDA：

~~~bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
~~~

本次准备阶段的验证结果：

~~~text
torch: 2.14.0+cu130
cuda available: True
gpu: NVIDIA GeForce RTX 5060 Laptop GPU
~~~

### 为什么不直接安装上游 requirements.txt

上游 requirements.txt 固定了较老的 onnx-simplifier==0.4.10。该依赖主要
用于 ONNX 导出，并非训练所必需；在当前 Python 3.12 环境中会因为旧版构建
脚本和 CMake/Git 元数据问题构建失败。因此本项目增加了
requirements-train.txt，只安装训练所需依赖。

## 4. 准备数据集

将现有数据压缩包解压到 YOLOX 项目的 data 目录：

~~~bash
cd /home/hzf/workspace/projects-dev/yolox-train
mkdir -p data

unzip -q \
  '/home/hzf/workspace/personal/career/Practice_record/20251112_20250206传递三角/训练数据2851张-JQ-M-001-0049.zip' \
  -d data
~~~

解压后的原始目录：

~~~text
data/JQ-M-01-0049/
├── Annotations/
├── JPEGImages/
└── classes.txt
~~~

生成 YOLOX VOC 目录：

~~~bash
python tools/split_and_copy_dataset.py \
  --data_dir data/JQ-M-01-0049 \
  --train_ratio 0.8 \
  --seed 42
~~~

生成的目录：

~~~text
data/JQ-M-01-0049/
└── VOC2007/
    ├── JPEGImages/
    ├── Annotations/
    └── ImageSets/
        └── Main/
            ├── train.txt
            ├── val.txt
            └── trainval.txt
~~~

预期划分结果：

~~~text
总数：2851
训练集：2280
验证集：571
~~~

classes.txt 的类别顺序必须保持为：

~~~text
0011c1rod
001jcu5ox
001ob4y7z
001py6kcr
001rq6x2k
001xlffir
~~~

该顺序同时决定 XML 类别到模型类别 ID 的映射。自定义数据集加载器会校验
classes.txt，顺序不一致时会直接报错，避免训练标签错位。

## 5. 为什么需要自定义 VOC 适配器

官方 YOLOX 的内置 VOCDetection 使用 Pascal VOC 的固定 20 类名称。传递三角
数据使用 6 个业务类别，不能只把实验配置里的 num_classes 从 20 改成 6，
否则 XML 标签映射和 VOC 评估仍会使用错误的类别表。

新增的 yolox/data/datasets/triangle_voc.py 复用官方 VOC 图像读取、增强和
标注解析逻辑，但替换了：

- 6 类类别映射；
- 训练/验证数据的类别元数据；
- 6 类 VOC 结果写出逻辑；
- 6 类 mAP 评估逻辑。

## 6. 训练实验配置

训练配置位于：

~~~text
exps/triangle/yolox_triangle_tiny.py
~~~

主要参数：

~~~text
num_classes = 6
depth = 0.33
width = 0.375
input_size = (416, 416)
test_size = (416, 416)
warmup_epochs = 1
enable_mixup = False
test_conf = 0.1
nmsthre = 0.5
~~~

数据目录默认读取 data/JQ-M-01-0049，也可以通过环境变量覆盖：

~~~bash
export TRIANGLE_DATA_DIR=/path/to/another/dataset
~~~

### 6.1 参数含义

这些参数分别控制类别数、模型规模、输入尺寸、训练初期的学习率、数据增强和
推理后处理：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `num_classes` | `6` | 数据集类别数。不包含背景类；YOLOX 每个位置输出 `4` 个框回归值、`1` 个目标置信度和 `6` 个类别置信度，因此输出通道数为 `11`。 |
| `depth` | `0.33` | 深度倍率，控制网络中重复模块的数量；与 `width=0.375` 一起对应 YOLOX-tiny 级别的模型规模。 |
| `width` | `0.375` | 宽度倍率，控制各层通道数；数值越小，模型参数量、显存占用和计算量通常越低。 |
| `input_size` | `(416, 416)` | 训练时将图像缩放、填充到的尺寸。 |
| `test_size` | `(416, 416)` | 验证和推理时使用的输入尺寸。与训练尺寸一致有利于保持实验设置一致。 |
| `warmup_epochs` | `1` | 学习率预热轮数。前 1 个 epoch 逐步把学习率调整到正式训练范围，不是只训练 1 个 epoch。 |
| `enable_mixup` | `False` | 关闭 MixUp 增强，即不把两张图像及其标签按比例混合。当前配置仍继承并启用 Mosaic 增强。 |
| `test_conf` | `0.1` | 验证/推理时的置信度阈值，通常依据目标置信度与类别置信度的组合结果过滤候选框；不是训练损失阈值。 |
| `nmsthre` | `0.5` | NMS 的 IoU 阈值。重叠 IoU 达到该阈值的重复框会被抑制；评估命令中的 `--conf`、`--nms` 可以覆盖配置值。 |

其中 `warmup_epochs=1` 和 `enable_mixup=False` 是根据现有训练记录及当前复现条件
作出的重建配置；由于原服务器上的 `cdsj_voc_exp.py` 已不可获得，不能宣称它们与
旧实验逐项完全相同。

## 7. 准备预训练权重

下载官方 YOLOX-tiny 预训练权重：

~~~bash
mkdir -p weights
curl -L --fail \
  'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth' \
  -o weights/yolox_tiny.pth
~~~

本次准备阶段得到的 SHA256：

~~~text
9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0
~~~

该权重原本是 80 类 COCO 模型。加载到 6 类模型时，YOLOX 的 `load_ckpt` 只会
加载“参数名相同且张量形状相同”的权重；形状不匹配的参数会跳过，因此出现
80 类和 6 类检测头形状不匹配的警告是正常现象，不代表整个模型加载失败。

具体加载结果可以理解为：

| 模块 | 80 类权重与 6 类模型的关系 | 处理方式 |
| --- | --- | --- |
| `backbone`、`neck`、`head.cls_convs` | 结构和形状兼容 | 复用 COCO 预训练权重 |
| `head.reg_preds.*` | 都输出 4 个框回归值 | 复用 COCO 预训练权重 |
| `head.obj_preds.*` | 都输出 1 个目标置信度 | 复用 COCO 预训练权重 |
| `head.cls_preds.*` | COCO 输出 80 类，当前输出 6 类 | 形状不匹配，跳过加载，使用 6 类模型初始化的分类预测层 |

因此，这不是“把 COCO 中任意 6 个类别的参数拿出来复用”，也不是“整个检测头
全部从零训练”。YOLOX 不会根据类别名称自动建立 COCO 类别与当前 6 个业务类别的
语义映射，也不会对 80 个分类通道做切片；当前类别名称本身也不是 COCO 的 80 类。
实际过程是：复用 backbone/neck 和检测头中通用的特征、回归、目标性参数，只有
最终的 6 类分类预测层重新学习。

这会影响训练初期的分类收敛，因为 6 类分类预测层没有可直接使用的 COCO 分类
参数，但不会导致训练失效；相比整个模型随机初始化，复用通用视觉特征通常仍然
更有利。加载时若只看到 `head.cls_preds` 的形状不匹配，而 backbone/neck 等
兼容参数正常加载，就符合预期。

旧训练记录中的命令包含：

~~~bash
python tools/train.py -f exps/cdsj_voc_exp.py -expn cdsj_0115 -d 1 -b 32 --fp16 -o -c yolox_tiny.pth
~~~

并没有使用 `--resume`。这说明旧训练也是从一个已有的 `yolox_tiny.pth` 检查点
开始的；按当前 YOLOX 训练器的语义，`-c` 是加载检查点进行 fine-tuning，训练
epoch 从 0 开始，而不是恢复旧的 epoch、优化器和学习率状态。原检查点文件已经
不可获得，所以无法仅凭现有材料百分之百确认它的内部类别数；如果它就是官方
YOLOX-tiny 的 80 类 COCO 权重，那么旧训练也会经历上述“分类预测层跳过、其余
兼容层复用”的加载过程。

## 8. 启动训练

原服务器使用 24 GB 显存、batch size 32；当前机器显存约 8 GB，因此先使用
batch size 8：

~~~bash
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
~~~

训练输出默认位于：

~~~text
YOLOX_outputs/triangle_0049/
├── best_ckpt.pth
├── latest_ckpt.pth
└── train_log.txt
~~~

如果显存不足，将 batch size 从 8 依次改为 4 或 2。也可以去掉 -o，因为
该选项会提前占用 GPU 显存。

## 9. 评估

训练产生 best_ckpt.pth 后执行：

~~~bash
cd /home/hzf/workspace/projects-dev/yolox-train
source .venv/bin/activate

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
~~~

当前使用的是 0049 数据集，评估结果应作为新实验结果记录，不能直接与旧
0047 训练记录比较。旧记录的参考值为：

~~~text
mAP50 ≈ 0.9006
mAP50:95 ≈ 0.7647
~~~

## 10. 准备阶段验证结果

本次已经完成以下检查：

1. CUDA 可用，识别到 RTX 5060 Laptop GPU；
2. 训练集加载数量为 2,280；
3. 验证集加载数量为 571；
4. 6 个 XML 类别均能正确映射；
5. YOLOX-tiny 6 类模型成功构建；
6. eval 前向输出形状为 (1, 3549, 11)；
7. 使用 FP16 完成一个真实训练 batch 的前向、loss、反向和 optimizer step；
8. tools/train.py --help 和 tools/eval.py --help 正常运行。

当前目录已经具备正式启动训练的条件，但尚未启动完整的长时间训练。

如需重复执行一次训练 batch 的冒烟验证：

~~~bash
source .venv/bin/activate
python - <<'PY'
import torch
from yolox.exp import get_exp
from yolox.utils import load_ckpt

exp = get_exp("exps/triangle/yolox_triangle_tiny.py", None)
exp.data_num_workers = 0
loader = exp.get_data_loader(batch_size=2, is_distributed=False, no_aug=False)
inputs, targets, _, _ = next(iter(loader))

model = exp.get_model().cuda()
checkpoint = torch.load("weights/yolox_tiny.pth", map_location="cpu")
state = checkpoint["model"] if "model" in checkpoint else checkpoint
load_ckpt(model, state)

optimizer = exp.get_optimizer(batch_size=2)
scaler = torch.amp.GradScaler("cuda", enabled=True)
model.train()
inputs = inputs.cuda().to(torch.float16)
targets = targets.cuda().to(torch.float16)

with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(inputs, targets)["total_loss"]

optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
print("one training step: OK", float(loss.detach().cpu()))
PY
~~~

## 11. 常见问题

### 找不到 train.txt

重新执行数据划分命令，并确认下面的文件存在：

~~~text
data/JQ-M-01-0049/VOC2007/ImageSets/Main/train.txt
~~~

### 类别数量或名称错误

检查 data/JQ-M-01-0049/classes.txt，必须与实验配置中的 6 类顺序一致。

### 显存不足

依次尝试 batch size 8、4、2；必要时去掉 -o。

### 预训练权重分类头形状不匹配

80 类权重加载到 6 类检测器时，分类头形状不匹配警告是预期行为；只要
backbone/neck 权重正常加载即可。

## 12. 复现限制

以下材料已经不可获得，因此当前结果不能承诺与旧模型完全一致：

- 原训练服务器上的 YOLOX commit/fork；
- 原 cdsj_voc_exp.py；
- 原 JQ-M-001-0047 数据集；
- 原 yolox_numpy1 conda 环境；
- 原训练过程中的随机种子、CUDA/PyTorch 版本和训练 checkpoint。

当前文档的目标是让后续人员能够在本目录中独立完成：

~~~text
源码准备 -> 环境安装 -> 数据划分 -> 类别适配 -> 预训练权重加载
-> YOLOX-tiny 训练 -> VOC mAP 评估
~~~
