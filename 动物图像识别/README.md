# 动物图像识别与计数系统

这是一个基于 Ultralytics YOLO 的动物目标检测项目，支持模型训练、图片批量预测、检测结果预览和 JSON 计数结果导出。

## 功能

- 使用 YOLO 格式数据集训练动物检测模型
- 对图片文件夹批量识别并统计动物数量
- 通过 Tkinter 图形界面配置训练和预测参数
- 预测时逐张显示带检测框的图片
- 将结果保存为课程/比赛常用的 JSON 格式

## 项目结构

```text
.
├── app.py              # 图形界面入口
├── main.py             # 训练、推理、数据集处理逻辑
├── label_tool.py       # YOLO 数据集标注工具
├── data.yaml           # YOLO 数据集配置模板
├── requirements.txt    # Python 依赖
├── .gitignore          # Git 忽略规则
└── README.md           # 项目说明
```

大文件不会提交到 GitHub，包括训练数据、模型权重、运行结果、虚拟环境和缓存文件。请按需在本地自行准备。

## 环境安装

建议使用 Python 3.10 或更新版本。

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

如果需要使用 NVIDIA GPU 训练，请根据自己的 CUDA 版本安装对应的 PyTorch。

## 数据集格式

训练数据需要采用 YOLO 检测格式：

```text
train/
├── images/
│   ├── image_1.jpg
│   └── image_2.jpg
└── labels/
    ├── image_1.txt
    └── image_2.txt

val/
├── images/
└── labels/
```

每张图片对应一个同名 `.txt` 标注文件。标注格式为：

```text
class_id x_center y_center width height
```

坐标均为相对图片宽高归一化后的数值。

## 支持类别

当前配置包含 20 类动物：

```text
cat, dog, horse, cow, sheep, goat, pig, rabbit, chicken, duck,
goose, deer, monkey, fox, wolf, bear, tiger, lion, zebra, giraffe
```

`main.py` 中还会把部分同义类别统一映射，例如 `cattle -> cow`、`hen -> chicken`。

## 运行图形界面

```bash
python app.py
```

界面包含两个主要功能：

1. 预测与计数：选择权重文件、图片文件夹和输出 JSON 路径后开始预测。
2. 模型训练：选择 `data.yaml`、初始权重、训练轮数、图片尺寸和设备后开始训练。

训练完成后，Ultralytics 会在 `runs/detect/train*/weights/best.pt` 生成最优权重。

## 标注自己的数据集

可以使用内置标注工具给图片生成 YOLO 格式标签：

```bash
python label_tool.py --images path/to/images --labels path/to/labels
```

如果已经有可用模型，也可以让模型先自动生成初稿，再人工修正：

```bash
python label_tool.py --images path/to/images --labels path/to/labels --model path/to/best.pt --auto-predict
```

常用快捷键：`S` 保存，`Enter` 保存并下一张，`A` 上一张，`D` 下一张，`Delete` 删除选中框。

## 命令行预测

也可以直接使用命令行批量预测：

```bash
python main.py --images path/to/images --model path/to/best.pt --output predictions.json
```

可选参数：

```bash
python main.py --images path/to/images --model path/to/best.pt --output predictions.json --conf 0.25 --iou 0.45
```

## 输出格式

预测结果会保存为 JSON：

```json
{
  "image_1.jpg": {
    "cat": 1,
    "dog": 2
  },
  "image_2.jpg": {
    "duck": 3
  }
}
```

## 说明

- GitHub 仓库中不包含 `.pt` 权重文件和训练图片，避免仓库过大。
- `data.yaml` 是模板，请根据本地数据集实际位置修改 `path`、`train` 和 `val`。
- 如果显存不足，可以减小 `batch` 或 `imgsz`。
