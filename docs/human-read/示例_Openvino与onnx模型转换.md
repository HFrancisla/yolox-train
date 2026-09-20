# 示例：ONNX 与 OpenVINO 模型转换全流程

> 本文记录 `triangle_yolox_16` 实验从训练权重到 ONNX、再到 OpenVINO IR 的完整实际执行过程，
> 包含每一步的输出结果、指标和遇到的问题及修复方法。
>
> 项目目录：`/home/hzf/workspace/projects-dev/yolox-train`

---

## 一、模型信息

| 项目 | 值 |
|------|----|
| 实验名 | `triangle_yolox_16` |
| 模型结构 | YOLOX-Tiny（depth=0.33, width=0.375） |
| 参数量 | 5.03M |
| 类别数 | 6 |
| 输入尺寸 | 416×416（`test_size`） |
| 训练权重 | `YOLOX_outputs/triangle_yolox_16/best_ckpt.pth`（39 MB） |
| mAP@0.50 | 89.81% |
| mAP@0.50:0.95 | 75.74% |

---

## 二、环境与依赖

```bash
# 确认依赖已安装（项目 .venv 内）
python -c "import onnx, onnxsim, openvino, onnxruntime; \
  print('onnx:', onnx.__version__); \
  print('openvino:', openvino.__version__); \
  print('onnxruntime:', onnxruntime.__version__)"
```

实际输出：

```
onnx: 1.23.0
openvino: 2026.4.0-22959-99c81491cc3-releases/2026/4
onnxruntime: 1.30.0
```

若未安装，执行：

```bash
uv pip install -U onnx onnxsim openvino onnxruntime
```

---

## 三、输出目录规划

```text
YOLOX_outputs/triangle_yolox_16/
├── best_ckpt.pth          # 训练权重（输入，不修改）
├── onnx/
│   └── triangle_yolox_16.onnx
└── openvino/
    ├── triangle_yolox_16.xml
    └── triangle_yolox_16.bin
```

```bash
mkdir -p YOLOX_outputs/triangle_yolox_16/onnx
mkdir -p YOLOX_outputs/triangle_yolox_16/openvino
```

---

## 四、Step 1：导出 ONNX

### 4.1 执行命令

```bash
cd /home/hzf/workspace/projects-dev/yolox-train

python tools/export_onnx.py \
  -f exps/triangle/yolox_triangle_tiny.py \
  -c YOLOX_outputs/triangle_yolox_16/best_ckpt.pth \
  --output-name YOLOX_outputs/triangle_yolox_16/onnx/triangle_yolox_16.onnx \
  --opset 11 \
  --batch-size 1
```

参数说明：

| 参数 | 值 | 说明 |
|------|----|------|
| `-f` | `exps/triangle/yolox_triangle_tiny.py` | 6 类实验配置 |
| `-c` | `best_ckpt.pth` | 训练最优权重，**不能用 COCO 预训练权重** |
| `--output-name` | `.../onnx/triangle_yolox_16.onnx` | 输出路径 |
| `--opset` | `11` | 算子集版本，兼容当前 OpenVINO |
| `--batch-size` | `1` | 固定 batch，适合单图部署 |

### 4.2 遇到的问题：`torch.load` UnpicklingError

**错误信息**：

```
_pickle.UnpicklingError: Weights only load failed.
WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar
```

**原因**：PyTorch 2.6+ 将 `torch.load` 的 `weights_only` 参数默认值从 `False` 改为 `True`，
而 YOLOX checkpoint 中包含 `numpy._core.multiarray.scalar` 类型，不在白名单内。

**修复**（`tools/export_onnx.py` 第 79 行）：

```python
# 修改前
ckpt = torch.load(ckpt_file, map_location="cpu")

# 修改后
ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)  # noqa: S614
```

> 本项目 checkpoint 来自本地训练，为受信任来源，使用 `weights_only=False` 是安全的。

### 4.3 成功输出

```
INFO - loading checkpoint done.
INFO - generated onnx model named YOLOX_outputs/triangle_yolox_16/onnx/triangle_yolox_16.onnx
INFO - generated simplified onnx model named YOLOX_outputs/triangle_yolox_16/onnx/triangle_yolox_16.onnx
```

| 文件 | 大小 |
|------|------|
| `triangle_yolox_16.onnx` | **20 MB**（含 onnxsim 简化） |

---

## 五、Step 2：ONNX 校验

### 5.1 结构校验 + ONNX Runtime 冒烟测试

```python
import onnx
import numpy as np
import onnxruntime as ort

onnx_path = "YOLOX_outputs/triangle_yolox_16/onnx/triangle_yolox_16.onnx"

# 结构校验
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
print("ONNX checker: OK")
for v in model.graph.input:
    print("input :", v.name)
for v in model.graph.output:
    print("output:", v.name)

# CPU 冒烟测试
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
x = np.zeros((1, 3, 416, 416), dtype=np.float32)
outputs = session.run(None, {input_name: x})
print("providers   :", session.get_providers())
print("output shape:", [o.shape for o in outputs])
```

### 5.2 实际结果

```
=== ONNX checker: OK ===
  input : images
  output: output

=== ONNX Runtime CPU: OK ===
  providers   : ['CPUExecutionProvider']
  output shape: [(1, 3549, 11)]
  file size   : 19.3 MB
```

### 5.3 输出 Shape 解析

输出 `(1, 3549, 11)` 含义：

```
batch=1

3549 = 52×52 + 26×26 + 13×13
     = (416/8)²  +  (416/16)²  +  (416/32)²
     （三个检测头的候选预测点总数）

11 = 4（框回归：cx, cy, w, h）
   + 1（目标置信度 obj_conf）
   + 6（6 个类别置信度）
```

> ⚠️ `3549` 不是固定常数，若 `test_size` 或检测头 stride 配置变化，此数值也会改变。

---

## 六、Step 3：转换 OpenVINO IR

### 6.1 使用 OpenVINO Python API 转换

```python
import openvino as ov

onnx_path = "YOLOX_outputs/triangle_yolox_16/onnx/triangle_yolox_16.onnx"
xml_path  = "YOLOX_outputs/triangle_yolox_16/openvino/triangle_yolox_16.xml"

ov_model = ov.convert_model(onnx_path)
ov.save_model(ov_model, xml_path, compress_to_fp16=True)
```

`compress_to_fp16=True`：将 IR 中的浮点权重以 FP16 形式保存，体积约减半。
不等于强制整个网络用 FP16 算术执行，实际执行精度由 OpenVINO 插件和 CPU 指令集决定。

### 6.2 实际结果

```
=== OpenVINO IR (FP16): OK ===
  saved: YOLOX_outputs/triangle_yolox_16/openvino/triangle_yolox_16.xml
  xml :  339.8 KB
  bin :    9.6 MB
```

| 文件 | 大小 | 说明 |
|------|------|------|
| `triangle_yolox_16.xml` | 340 KB | 网络结构描述 |
| `triangle_yolox_16.bin` | **9.6 MB** | FP16 权重（原 ONNX 20MB → 约减半） |

---

## 七、Step 4：OpenVINO CPU 冒烟测试

### 7.1 测试代码

```python
import numpy as np
import openvino as ov

xml_path = "YOLOX_outputs/triangle_yolox_16/openvino/triangle_yolox_16.xml"

core = ov.Core()
print("available devices:", core.available_devices)

model = core.read_model(xml_path)
compiled = core.compile_model(model, "CPU")

x = np.zeros((1, 3, 416, 416), dtype=np.float32)
input_name = compiled.inputs[0].any_name
outputs = compiled({input_name: x})
print("input :", input_name)
print("output shapes:", [np.asarray(v).shape for v in outputs.values()])
```

### 7.2 实际结果

```
=== OpenVINO 设备 ===
  available: ['CPU']

=== OpenVINO CPU 推理: OK ===
  input : images
  output shapes: [(1, 3549, 11)]
```

---

## 八、最终输出文件汇总

```text
YOLOX_outputs/triangle_yolox_16/
├── best_ckpt.pth                    39 MB   训练最优权重（PyTorch）
├── onnx/
│   └── triangle_yolox_16.onnx       20 MB   通用 ONNX 模型（opset 11，onnxsim 简化）
└── openvino/
    ├── triangle_yolox_16.xml       340 KB   OpenVINO IR 网络结构
    └── triangle_yolox_16.bin         9.6 MB  OpenVINO IR 权重（FP16 压缩）
```

| 模型 | 大小 | 测试结果 | 输出 shape |
|------|------|----------|-----------|
| `.pth`（PyTorch） | 39 MB | mAP@0.5=89.81%，mAP@0.5:0.95=75.74% | — |
| `.onnx` | 20 MB | ONNX checker ✅，ONNX Runtime CPU ✅ | `(1, 3549, 11)` |
| `.xml/.bin`（OpenVINO FP16） | 9.6 MB | OpenVINO CPU 编译+推理 ✅ | `(1, 3549, 11)` |

---

## 九、注意事项

### 9.1 类别名称

部署时必须使用 6 类 `TRIANGLE_CLASSES`，不能用 COCO_CLASSES：

```python
TRIANGLE_CLASSES = (
    "0011c1rod",
    "001jcu5ox",
    "001ob4y7z",
    "001py6kcr",
    "001rq6x2k",
    "001xlffir",
)
```

### 9.2 后处理

ONNX/OpenVINO 只输出网络原始预测，应用侧仍需：

1. 预处理：保持长宽比 resize + 114 padding，HWC→CHW，float32（**不做** `/255` 归一化）
2. 解码 YOLOX 输出（cx, cy, w, h → x1, y1, x2, y2）
3. `obj_conf × cls_conf` 得到最终置信度
4. NMS（`conf_thre=0.1`，`nms_thre=0.5`）
5. 类别 ID 映射到 `TRIANGLE_CLASSES`

### 9.3 PyTorch 2.6+ 兼容性

`tools/export_onnx.py` 已修复 `torch.load` 的 `weights_only` 问题，其他脚本若有相同用法也需同步修改。

---

## 十、参考

- [Openvino与onnx模型.md](./Openvino与onnx模型.md)：完整原理与 FAQ
- [训练记录](../_record/20260920-yolox-triangle/训练记录.md)：本次训练权重的详细指标
