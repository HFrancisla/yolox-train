# YOLOX 权重导出 ONNX 并转换 OpenVINO 完整指南

```text
YOLOX_outputs/triangle_0049/
│
├── best_ckpt.pth                # 原始最佳模型
│
├── triangle_0049.onnx           # 通用部署模型
│
└── openvino/
    ├── triangle_0049.xml        # OpenVINO IR
    └── triangle_0049.bin
```

本文说明当前项目从训练权重到 CPU 部署模型的完整流程：

~~~text
YOLOX 训练权重（.pth）
        │
        │ tools/export_onnx.py
        ▼
ONNX 模型（.onnx）
        ├── ONNX Runtime：直接使用 CPU 推理
        │
        └── OpenVINO convert_model / ovc
                ▼
        OpenVINO IR（.xml + .bin）
                │
                └── OpenVINO CPU 推理
~~~

当前项目目录：

~~~text
/home/hzf/workspace/projects-dev/yolox-train
~~~

当前实验是 6 类、YOLOX-tiny、输入尺寸 416×416 的模型。

## 1. 为什么需要 ONNX 和 OpenVINO

### 1.1 三种模型文件的职责

| 文件 | 主要用途 | 典型运行环境 |
| --- | --- | --- |
| .pth | PyTorch/YOLOX 训练、验证和继续训练 | PyTorch、YOLOX |
| .onnx | 通用模型交换格式，方便交给不同推理引擎 | ONNX Runtime、OpenVINO、TensorRT |
| .xml + .bin | OpenVINO IR 格式，保存网络结构和权重 | OpenVINO CPU/GPU/NPU |

.pth 更适合研究和训练，.onnx 更适合跨框架推理，OpenVINO IR 更适合
OpenVINO 部署。这个过程是模型格式转换和推理优化，不会重新训练模型。

### 1.2 ONNX 能不能直接在 CPU 上运行

可以，但 ONNX 文件本身不是运行时，需要安装并使用 ONNX Runtime：

~~~bash
uv pip install onnxruntime
~~~

然后指定 CPUExecutionProvider。如果只想在 CPU 上验证模型，直接使用 ONNX
Runtime 就不必再转换为 OpenVINO。

OpenVINO 也可以直接读取 ONNX 并编译到 CPU。将 ONNX 保存为 IR 的主要好处是
部署时模型格式固定、加载依赖更少，并且可以提前完成 OpenVINO 图转换和
FP16 压缩。

### 1.3 转换后的模型仍然需要 YOLOX 后处理

当前导出脚本默认导出网络前向结果，不会自动完成完整的业务检测流程。应用侧
仍然需要处理：

1. 按 YOLOX 的 `preproc` 逻辑完成图像预处理：保持长宽比 resize、使用 114 做
   padding、HWC→CHW、转换为 float32；
2. YOLOX 输出解码；
3. 目标置信度与类别置信度相乘；
4. NMS；
5. 6 个业务类别名称映射。

注意：标准 YOLOX 非 legacy 推理流程默认**不会额外执行 `/255` 或
mean/std normalization**。只有项目明确使用 legacy 预处理时，才应加入对应归一化。
部署端必须与 PyTorch/YOLOX 基准推理保持完全一致。

因此，模型转换成 OpenVINO 不等于已经有一个完整的图片检测应用。

## 2. 当前项目已有的相关代码

项目中已有以下文件：

- tools/export_onnx.py：YOLOX .pth 导出 .onnx；
- demo/ONNXRuntime/README.md：ONNX Runtime 示例说明；
- demo/ONNXRuntime/onnx_inference.py：ONNX Runtime 图片推理示例；
- demo/OpenVINO/python/README.md：OpenVINO 示例说明；
- demo/OpenVINO/python/openvino_inference.py：旧版 OpenVINO 图片推理示例。

这些文件来自官方 YOLOX 项目，但当前自定义模型使用 6 个业务类别，因此不能
完全不修改就直接用于最终推理：

- ONNX/OpenVINO 示例默认使用 COCO_CLASSES，需要替换为当前 6 类名称；
- demo/OpenVINO/python/openvino_inference.py 使用旧版
  openvino.inference_engine.IECore API；
- 旧 README 使用 OpenVINO 2021 的 mo.py，与当前 pip 安装的新版 OpenVINO
  Python API 不是同一套接口；
- 当前 PyTorch 2.14 中不存在旧脚本调用的 torch.onnx._export；本项目已经将其
  修改为兼容当前 PyTorch 的 torch.onnx.export。

## 3. 环境和目录规划

### 3.1 不需要单独创建新的项目目录

转换应该在当前 YOLOX 项目目录中进行，因为导出过程需要加载：

- 当前 YOLOX 源码；
- exps/triangle/yolox_triangle_tiny.py；
- 6 类模型结构；
- 训练产生的 best_ckpt.pth。

因此，导出和转换可以继续使用当前环境：

~~~bash
cd /home/hzf/workspace/projects-dev/yolox-train
source .venv/bin/activate
~~~

### 3.2 在当前训练环境中安装导出依赖

当前训练依赖文件有意没有固定旧版 onnx-simplifier。在 Python 3.12 环境中，
建议安装当前可用版本：

~~~bash
cd /home/hzf/workspace/projects-dev/yolox-train
source .venv/bin/activate

uv pip install -U onnx onnxsim openvino onnxruntime
~~~

各包职责：

| 包 | 是否必须 | 用途 |
| --- | --- | --- |
| onnx | 导出和校验需要 | ONNX 模型读写、结构检查 |
| onnxsim | 当前导出脚本默认需要 | 简化和校验 ONNX 图 |
| openvino | 转换为 IR、OpenVINO 推理需要 | 当前推荐的 OpenVINO Python API |
| onnxruntime | 直接运行 ONNX 或做对比验证需要 | CPU/GPU ONNX 推理 |

如果只做 OpenVINO 转换和推理，可以不安装 onnxruntime；如果希望比较
ONNX Runtime 和 OpenVINO 的 CPU 性能，建议全部安装。

### 3.3 是否要创建独立的推理环境

不是必须的。推荐分为两种情况：

- 训练、导出、转换：继续使用当前 .venv，因为需要 PyTorch 和 YOLOX；
- 生产推理：可以另建 .venv-infer，只安装 openvino 或 onnxruntime、numpy、
  opencv-python 和业务代码，以减少部署依赖。

独立的是 Python 环境，不是必须独立的项目目录。

## 4. 转换前检查训练权重

转换前必须先有当前 6 类模型的训练权重：

~~~text
YOLOX_outputs/triangle_0049/best_ckpt.pth
~~~

检查文件：

~~~bash
cd /home/hzf/workspace/projects-dev/yolox-train
test -f YOLOX_outputs/triangle_0049/best_ckpt.pth && echo "checkpoint: OK"
~~~

不要把官方 80 类 COCO 预训练权重 weights/yolox_tiny.pth 直接作为当前
6 类模型的导出权重。它用于训练开始时的部分预训练加载；而
tools/export_onnx.py 使用严格的 load_state_dict，直接加载 80 类权重到
6 类模型会再次遇到分类头形状不匹配。

正确关系：

~~~text
weights/yolox_tiny.pth
    └── 训练初始化时使用

YOLOX_outputs/triangle_0049/best_ckpt.pth
    └── 训练完成后用于导出部署模型
~~~

## 5. ONNX 导出脚本的 PyTorch 兼容性修正

当前 tools/export_onnx.py 调用了旧的私有接口：

~~~python
torch.onnx._export(
~~~

当前环境中的 PyTorch 2.14 没有这个属性。本项目已经将它改为公开接口。为了兼容
YOLOX 原有的 legacy ONNX 导出行为，这里显式使用 `dynamo=False`：

~~~python
torch.onnx.export(
    model,
    dummy_input,
    args.output_name,
    input_names=[args.input],
    output_names=[args.output],
    dynamic_axes={args.input: {0: "batch"},
                  args.output: {0: "batch"}} if args.dynamic else None,
    opset_version=args.opset,
    dynamo=False,
)
~~~

当前版本的 tools/export_onnx.py 已经包含上述修改，其他逻辑保持不变。若以后
重新拷贝了未修改的上游 YOLOX 脚本，需要再次应用这个改动。

`dynamo=False` 是为了兼容旧 YOLOX 导出链路的有意选择，并不表示这是当前
PyTorch 对所有新项目的默认推荐方式。后续如果切换到新版 ONNX exporter，应重新
验证 YOLOX 输出 shape、数值和 OpenVINO 转换结果。

如果导出时提示 onnxsim 不存在，可以先安装 onnxsim；如果暂时不需要简化，
也可以在导出命令中增加 --no-onnxsim。

## 6. 第一步：.pth 导出为 .onnx

### 6.1 推荐导出命令

使用固定 batch size 1 和固定 416×416 输入，适合当前部署场景。

**导出前必须确认 `exp.test_size`，因为 YOLOX 的 ONNX 导出脚本使用的是
`exp.test_size`，不是训练时的 `exp.input_size`。**

建议先检查：

~~~bash
grep -n "input_size\|test_size" exps/triangle/yolox_triangle_tiny.py
~~~

当前项目应确认至少满足：

~~~python
self.input_size = (416, 416)
self.test_size = (416, 416)
~~~

其中训练期间启用 multi-scale 时出现的 256、320、352、...、576 等尺寸，不代表
部署模型也会动态使用这些尺寸。当前部署按固定 `test_size=(416,416)` 导出。

~~~bash
cd /home/hzf/workspace/projects-dev/yolox-train
source .venv/bin/activate

mkdir -p artifacts

python tools/export_onnx.py \
  -f exps/triangle/yolox_triangle_tiny.py \
  -c YOLOX_outputs/triangle_0049/best_ckpt.pth \
  --output-name artifacts/triangle_0049.onnx \
  --opset 11 \
  --batch-size 1
~~~

参数说明：

| 参数 | 作用 |
| --- | --- |
| -f | 使用当前 6 类 YOLOX 实验配置 |
| -c | 指定训练完成的 6 类 checkpoint |
| --output-name | ONNX 输出路径 |
| --opset 11 | 当前现代 ONNX/OpenVINO 流程使用的算子集版本 |
| --batch-size 1 | 固定单张图片推理，减少部署时的输入约束 |
| --dynamic | 允许 batch 动态变化；当前单图部署通常不需要 |
| --decode_in_inference | 把部分 YOLOX 输出解码放入模型；默认关闭，保持现有后处理流程 |
| --no-onnxsim | 跳过 ONNX 图简化，用于排查简化失败问题 |

导出成功后预期得到：

~~~text
artifacts/triangle_0049.onnx
~~~

当同时满足以下条件时：

- `exp.test_size = (416, 416)`；
- YOLOX 检测头使用默认 strides `[8, 16, 32]`；
- `num_classes = 6`；
- batch size = 1；

模型原始输出通常是：

~~~text
(1, 3549, 11)
~~~

其中：

~~~text
3549 = 52×52 + 26×26 + 13×13
     = (416/8)² + (416/16)² + (416/32)²

11 = 4 个框回归值 + 1 个目标置信度 + 6 个类别置信度
~~~

因此 `3549` 不是固定常数；如果 `test_size` 或检测头 stride 配置变化，候选预测点
数量也会变化。

### 6.2 导出时为什么使用 opset 11

旧训练记录中的命令使用了：

~~~bash
python tools/export_onnx.py ... -o 10
~~~

这是因为旧流程使用 OpenVINO 2021 的 Model Optimizer mo.py，当时的转换链路
对 opset 10 更稳妥。当前推荐使用新版 OpenVINO Python API 或 ovc，通常可以
使用 opset 11。

只有在必须使用旧版 mo.py 时，才应优先尝试：

~~~bash
python tools/export_onnx.py \
  -f exps/triangle/yolox_triangle_tiny.py \
  -c YOLOX_outputs/triangle_0049/best_ckpt.pth \
  --output-name artifacts/triangle_0049.onnx \
  --opset 10 \
  --batch-size 1
~~~

## 7. 第二步：校验 ONNX 文件

### 7.1 使用 ONNX checker 校验结构

~~~bash
cd /home/hzf/workspace/projects-dev/yolox-train
source .venv/bin/activate

python - <<'PY'
import onnx

path = "artifacts/triangle_0049.onnx"
model = onnx.load(path)
onnx.checker.check_model(model)

print("ONNX check: OK")
for value in model.graph.input:
    print("input:", value.name)
for value in model.graph.output:
    print("output:", value.name)
PY
~~~

### 7.2 使用 ONNX Runtime 做 CPU 冒烟测试

该测试只验证 ONNX 能否在 CPU 上加载和前向，不包含图片预处理和 NMS：

~~~bash
python - <<'PY'
import numpy as np
import onnxruntime as ort

path = "artifacts/triangle_0049.onnx"
session = ort.InferenceSession(
    path,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
x = np.zeros((1, 3, 416, 416), dtype=np.float32)
outputs = session.run(None, {input_name: x})

print("providers:", session.get_providers())
print("output shapes:", [output.shape for output in outputs])
PY
~~~

如果输出 provider 包含 CPUExecutionProvider，且输出形状正常，则 ONNX 模型
可以在 CPU 上运行。

### 7.3 使用项目自带的 ONNX 图片 demo

~~~bash
python demo/ONNXRuntime/onnx_inference.py \
  -m artifacts/triangle_0049.onnx \
  -i /path/to/test.jpg \
  -o artifacts/onnx_demo \
  -s 0.1 \
  --input_shape 416,416
~~~

但该 demo 默认导入 COCO_CLASSES。当前模型推理时必须改为 6 个业务类别，
否则框可能能正确检测，但显示出来的类别名称会错误。

建议使用：

~~~python
from yolox.data.datasets import TRIANGLE_CLASSES
~~~

替换 demo 中的：

~~~python
from yolox.data.datasets import COCO_CLASSES
~~~

并将 class_names=COCO_CLASSES 改为：

~~~python
class_names=TRIANGLE_CLASSES
~~~

## 8. 第三步：ONNX 转 OpenVINO IR

### 8.1 当前推荐：OpenVINO Python API

使用当前 pip 包提供的 openvino.convert_model 和 openvino.save_model：

~~~bash
cd /home/hzf/workspace/projects-dev/yolox-train
source .venv/bin/activate

python - <<'PY'
import openvino as ov

onnx_path = "artifacts/triangle_0049.onnx"
xml_path = "artifacts/triangle_0049_openvino.xml"

ov_model = ov.convert_model(onnx_path)
ov.save_model(
    ov_model,
    xml_path,
    compress_to_fp16=True,
)

print("OpenVINO IR saved:", xml_path)
print("The corresponding .bin file is saved beside the .xml file.")
PY
~~~

预期文件：

~~~text
artifacts/
├── triangle_0049.onnx
├── triangle_0049_openvino.xml
└── triangle_0049_openvino.bin
~~~

如果需要保留 FP32 模型用于精度对比，可以关闭 FP16 压缩：

~~~python
ov.save_model(
    ov_model,
    "artifacts/triangle_0049_openvino_fp32.xml",
    compress_to_fp16=False,
)
~~~

这里的 `compress_to_fp16=True` 主要表示将 IR 中可压缩的浮点权重以 FP16
形式保存，通常可以明显减小模型文件体积。它**不等价于强制整个 OpenVINO CPU
网络全程使用 FP16 算术执行**。

实际执行精度和性能由 OpenVINO 插件、CPU 指令集、具体算子以及运行配置共同决定。
因此是否在目标 CPU 上更快，必须在目标设备上进行基准测试。FP16 权重压缩不是
重新训练，也不是 INT8 量化。

### 8.2 使用 ovc 命令转换

安装 openvino 后通常会提供 ovc 命令。可以在输出目录执行：

~~~bash
mkdir -p artifacts/openvino_ovc
cd artifacts/openvino_ovc

ovc ../triangle_0049.onnx \
  --compress_to_fp16=True
~~~

如果需要精确控制输出文件名，优先使用上一节的 Python API。

### 8.3 旧项目使用的 mo.py 流程

原训练记录中的转换命令是：

~~~bash
# 在 YOLOX 环境导出 ONNX
conda activate yolox_numpy1
python tools/export_onnx.py \
  -f exps/cdsj_voc_exp.py \
  -c YOLOX_outputs/cdsj_0115/best_ckpt.pth \
  --output-name yolox_cdsj_0115.onnx \
  -o 10

# 切换到 OpenVINO 环境转换
conda activate openvino
mo \
  --input_model yolox_cdsj_0115.onnx \
  --output_dir models_converted/cdsj_0115/openvino
~~~

这条命令依赖旧版 OpenVINO Toolkit 的 Model Optimizer，通常还需要旧版
openvino_2021 环境。当前不建议为了普通复现再安装旧版 mo.py；旧服务器和
原 conda 环境已经不可获得，使用新版 openvino pip 包的 convert_model 或
ovc 更容易维护。

如果目标是尽可能复现旧部署文件，而不是仅实现同样的推理功能，则需要同时恢复：

- 原 YOLOX commit 和导出脚本；
- 原 Python、PyTorch、ONNX 版本；
- OpenVINO 2021 Toolkit 和 Model Optimizer；
- 原 opset 10 配置；
- 原 FP16、输入形状和后处理配置。

当前流程能够得到功能等价的 OpenVINO 模型，但不保证 XML/BIN 文件二进制一致。

## 9. 第四步：OpenVINO CPU 加载和冒烟测试

### 9.1 查看 OpenVINO 设备

~~~bash
python - <<'PY'
import openvino as ov

core = ov.Core()
print("available devices:", core.available_devices)
PY
~~~

至少应能看到：

~~~text
CPU
~~~

### 9.2 加载 IR 并执行 CPU 前向

~~~bash
python - <<'PY'
import numpy as np
import openvino as ov

xml_path = "artifacts/triangle_0049_openvino.xml"

core = ov.Core()
model = core.read_model(xml_path)
compiled_model = core.compile_model(model, "CPU")

input_port = compiled_model.inputs[0]
input_name = input_port.any_name
x = np.zeros((1, 3, 416, 416), dtype=np.float32)

outputs = compiled_model({input_name: x})
print("OpenVINO CPU inference: OK")
print("input:", input_name)
print("output shapes:", [np.asarray(value).shape for value in outputs.values()])
PY
~~~

OpenVINO 也可以不保存 IR，直接编译 ONNX：

~~~python
import openvino as ov

core = ov.Core()
compiled_model = core.compile_model(
    "artifacts/triangle_0049.onnx",
    "CPU",
)
~~~

不过部署时建议提前保存 XML/BIN，以缩短启动时的转换和加载过程，并减少部署
环境对 ONNX 前端的依赖。

## 10. OpenVINO 图片推理时的适配要求

### 10.1 旧版 demo 不能直接用于当前新版 API

demo/OpenVINO/python/openvino_inference.py 使用：

~~~python
from openvino.inference_engine import IECore
~~~

这是旧版 OpenVINO Inference Engine API。现代 OpenVINO Python API 应使用：

~~~python
import openvino as ov

core = ov.Core()
model = core.read_model("model.xml")
compiled_model = core.compile_model(model, "CPU")
~~~

### 10.2 当前模型的类别名称

当前模型类别顺序必须保持为：

~~~python
TRIANGLE_CLASSES = (
    "0011c1rod",
    "001jcu5ox",
    "001ob4y7z",
    "001py6kcr",
    "001rq6x2k",
    "001xlffir",
)
~~~

不能继续使用官方 demo 中的 COCO_CLASSES，否则类别 ID 与显示名称不对应。

### 10.3 预处理、解码和 NMS 必须一致

ONNX Runtime 和 OpenVINO 应使用与 PyTorch 评估一致的：

- 输入尺寸：416 × 416（即当前 `exp.test_size`）；
- letterbox/缩放填充方式；
- HWC→CHW 和 float32 转换方式；
- 是否执行 `/255`、mean/std normalization（标准非 legacy YOLOX 默认不执行）；
- 输出解码方式；
- conf 阈值；
- NMS IoU 阈值；
- 类别顺序。

`conf` 和 `NMS IoU` 阈值不是 YOLOX 网络结构中的固定常数，应以当前项目在
PyTorch 基准推理/验证阶段实际使用的配置为准。做 PyTorch、ONNX Runtime、
OpenVINO 三方一致性对比时，必须使用相同的阈值。

否则即使模型本身转换正确，也可能出现框坐标偏移、置信度不一致或类别名称错误。

## 11. 推荐的完整执行顺序

从训练完成的项目状态开始，推荐按以下顺序执行：

~~~bash
cd /home/hzf/workspace/projects-dev/yolox-train
source .venv/bin/activate

# 1. 安装转换和推理依赖
uv pip install -U onnx onnxsim openvino onnxruntime

# 2. 确认 6 类训练权重存在
test -f YOLOX_outputs/triangle_0049/best_ckpt.pth

# 3. 导出 ONNX
mkdir -p artifacts
python tools/export_onnx.py \
  -f exps/triangle/yolox_triangle_tiny.py \
  -c YOLOX_outputs/triangle_0049/best_ckpt.pth \
  --output-name artifacts/triangle_0049.onnx \
  --opset 11 \
  --batch-size 1

# 4. 校验 ONNX
python -c 'import onnx; m=onnx.load("artifacts/triangle_0049.onnx"); onnx.checker.check_model(m); print("onnx: OK")'

# 5. 转换为 OpenVINO IR
python - <<'PY'
import openvino as ov

model = ov.convert_model("artifacts/triangle_0049.onnx")
ov.save_model(
    model,
    "artifacts/triangle_0049_openvino.xml",
    compress_to_fp16=True,
)
print("openvino IR: OK")
PY

# 6. 验证 OpenVINO CPU 加载
python - <<'PY'
import openvino as ov

core = ov.Core()
compiled = core.compile_model(
    "artifacts/triangle_0049_openvino.xml",
    "CPU",
)
print("devices:", core.available_devices)
print("compiled: OK", compiled.inputs[0].any_name)
PY
~~~

## 12. 转换结果的正确性验证

转换完成后，至少使用同一张图片分别执行：

1. PyTorch/YOLOX 推理；
2. ONNX Runtime CPU 推理；
3. OpenVINO CPU 推理。

比较以下内容：

- 输入图像经过预处理后的张量是否一致；
- 输出张量形状是否一致；
- 最高置信度类别是否一致；
- 框坐标是否只存在很小的浮点误差；
- NMS 后保留框数量是否大致一致。

如果 OpenVINO 使用 FP16，而 PyTorch 使用 FP32，输出存在微小数值差异是正常的。
如果类别完全错位、框整体偏移或输出形状不一致，应优先检查输入尺寸、类别顺序、
预处理和后处理，而不是立即判断转换失败。

## 13. 推荐的输出目录

建议将转换产物统一保存到 artifacts/，不要混入训练 checkpoint 目录：

~~~text
artifacts/
├── triangle_0049.onnx
├── triangle_0049_openvino.xml
├── triangle_0049_openvino.bin
├── onnx_demo/
└── openvino_demo/
~~~

训练目录和部署目录的职责：

~~~text
YOLOX_outputs/triangle_0049/
├── best_ckpt.pth       # 训练结果，供导出使用
└── latest_ckpt.pth

artifacts/
├── triangle_0049.onnx  # 通用推理模型
└── triangle_0049_openvino.xml/.bin  # OpenVINO 部署模型
~~~

## 14. 常见问题

### 14.1 其他 YOLOX 副本中出现 torch.onnx._export 不存在

当前 PyTorch 已移除旧私有接口。将：

~~~python
torch.onnx._export(
~~~

改为：

~~~python
torch.onnx.export(
    ...,
    dynamo=False,
)
~~~

### 14.2 ModuleNotFoundError: No module named onnxsim

安装：

~~~bash
uv pip install onnxsim
~~~

或者导出时增加：

~~~bash
--no-onnxsim
~~~

### 14.3 80 类和 6 类分类头形状不匹配

训练开始时加载 weights/yolox_tiny.pth 出现该警告是预期行为；导出时应使用
训练完成后的：

~~~text
YOLOX_outputs/triangle_0049/best_ckpt.pth
~~~

不要用 80 类 COCO 权重直接导出当前 6 类模型。

### 14.4 mo: command not found

mo 是旧版 OpenVINO Model Optimizer 命令，不属于当前推荐流程。使用：

- Python API：ov.convert_model + ov.save_model；
- 当前命令行：ovc。

只有在必须复现旧 OpenVINO 2021 环境时，才恢复旧版 mo.py。

### 14.5 openvino.inference_engine 导入失败

这是旧版 OpenVINO API 与新版 pip 包不兼容。使用现代 API：

~~~python
import openvino as ov

core = ov.Core()
model = core.read_model("model.xml")
compiled_model = core.compile_model(model, "CPU")
~~~

### 14.6 结果类别名称错误

检查推理代码是否仍然使用 COCO_CLASSES。当前模型必须使用 6 类
TRIANGLE_CLASSES，且顺序不能改变。

### 14.7 OpenVINO 转换成功但检测结果不对

依次检查：

1. 是否使用了 6 类 `best_ckpt.pth`；
2. `exp.test_size` 是否为 `(416,416)`，实际模型输入是否为 `1×3×416×416`；
3. 图像预处理是否与 YOLOX 一致，包括 resize、114 padding、HWC→CHW、float32，
   以及是否错误地额外执行了 `/255` 或 mean/std normalization；
4. 输出是否执行了与导出方式匹配的 YOLOX decode；
5. 是否计算了 `objectness × class_confidence`；
6. conf 和 NMS IoU 阈值是否与 PyTorch 基准推理配置一致；
7. 类别名称和类别 ID 顺序是否一致。

### 14.8 OpenVINO CPU 速度与历史记录不同

历史记录中的耗时依赖特定 CPU、OpenVINO 版本、线程配置、输入图像和服务并发。
当前转换流程只能复现模型格式链路，不能保证复现原服务器上的具体毫秒数。

## 15. 历史流程与当前推荐流程的对应关系

| 历史流程 | 当前推荐流程 |
| --- | --- |
| conda activate yolox_numpy1 | 当前项目 .venv |
| tools/export_onnx.py -o 10 | tools/export_onnx.py --opset 11 |
| 单独的 openvino conda 环境 | 当前 .venv 安装 openvino，或单独的推理 .venv-infer |
| mo --input_model ... | ov.convert_model(...) 或 ovc ... |
| IECore / openvino.runtime.Core | openvino.Core() |
| OpenVINO XML/BIN CPU 部署 | OpenVINO IR XML/BIN CPU 部署 |

当前推荐流程与历史流程的目标相同：将训练得到的 6 类 YOLOX 模型部署到 CPU；
差异主要来自 OpenVINO API、PyTorch 和 Python 版本变化。

## 16. 参考文件

- YOLOX 训练准备与复现指南：
  docs/human-read/YOLOX训练准备与复现指南.md
- YOLOX ONNX 导出脚本：
  tools/export_onnx.py
- YOLOX ONNX Runtime 示例：
  demo/ONNXRuntime/README.md
- YOLOX OpenVINO Python 示例：
  demo/OpenVINO/python/README.md
- 当前训练依赖：
  requirements-train.txt

官方当前 API 的核心调用：

~~~python
import openvino as ov

ov_model = ov.convert_model("model.onnx")
ov.save_model(ov_model, "model.xml")

core = ov.Core()
compiled_model = core.compile_model("model.xml", "CPU")
~~~
