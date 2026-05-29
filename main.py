"""
main.py
动物检测与计数：训练 + 推理主程序

功能：
1. 加载 YOLOv8 模型
2. 使用 YOLOv8 格式数据集训练模型
3. 对图片文件夹批量预测
4. 输出课程要求的 JSON 计数结果
5. 训练时支持 GPU / CPU / 自动选择设备

重要说明：
- 如果日志里显示 torch-xxx+cpu，说明你安装的是 CPU 版 PyTorch。
- 想用 NVIDIA 显卡训练，必须安装 CUDA 版 PyTorch。
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

import cv2

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError("未安装 ultralytics，请先执行：pip install ultralytics") from exc


# ============================================================
# 只输出这 20 类动物
# ============================================================
ANIMAL_WHITELIST: List[str] = [
    "cat",
    "dog",
    "horse",
    "cow",
    "sheep",
    "goat",
    "pig",
    "rabbit",
    "chicken",
    "duck",
    "goose",
    "deer",
    "monkey",
    "fox",
    "wolf",
    "bear",
    "tiger",
    "lion",
    "zebra",
    "giraffe",
]


# ============================================================
# 数据集类别名到目标类别名的映射
# 例如数据集中写 Cattle，但最终 JSON 要输出 cow
# ============================================================
ANIMAL_SYNONYMS: Dict[str, str] = {
    "cattle": "cow",
    "bull": "cow",
    "brown-bear": "bear",
    "brown_bear": "bear",
    "hen": "chicken",
    "rooster": "chicken",
}


def get_torch_device_info() -> str:
    """
    返回当前 PyTorch / CUDA / GPU 信息，用于在 GUI 日志中显示。
    """
    try:
        import torch
    except ImportError:
        return "未安装 PyTorch，无法检测 GPU。"

    info = [
        f"PyTorch版本: {torch.__version__}",
        f"CUDA是否可用: {torch.cuda.is_available()}",
        f"PyTorch CUDA版本: {torch.version.cuda}",
    ]

    if torch.cuda.is_available():
        info.append(f"GPU数量: {torch.cuda.device_count()}")
        info.append(f"当前GPU: {torch.cuda.get_device_name(0)}")
    else:
        info.append("当前没有检测到可用 CUDA GPU。")

    return " | ".join(info)


def resolve_train_device(device: str = "cuda") -> int | str:
    """
    将用户选择的训练设备转换为 Ultralytics 可接受的 device 参数。

    参数：
    - "cuda" / "gpu" / "0"：强制使用第 1 张 NVIDIA GPU
    - "auto"：有 GPU 就用 GPU，没有 GPU 就回退 CPU
    - "cpu"：强制使用 CPU
    """
    device = str(device).strip().lower()

    if device in {"cuda", "gpu", "cuda:0", "0"}:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("你选择了 GPU 训练，但当前环境没有安装 PyTorch。") from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "你选择了 GPU 训练，但当前 PyTorch 不能使用 CUDA。\n"
                "从你的日志看，当前环境很可能安装的是 CPU 版 PyTorch，例如 torch-xxx+cpu。\n"
                "请先安装 CUDA 版 PyTorch，然后再重新运行 app.py。"
            )

        print(f"使用 GPU 训练: {torch.cuda.get_device_name(0)}")
        return 0

    if device == "auto":
        try:
            import torch
        except ImportError:
            print("未安装 PyTorch，自动回退 CPU。")
            return "cpu"

        if torch.cuda.is_available():
            print(f"自动检测到 GPU: {torch.cuda.get_device_name(0)}")
            return 0

        print("未检测到 CUDA GPU，自动回退 CPU。")
        return "cpu"

    if device == "cpu":
        print("使用 CPU 训练。")
        return "cpu"

    # 允许高级用户输入 "1"、"0,1" 等 Ultralytics 支持的写法
    return device


def normalize_model_arch(model_arch: str) -> str:
    """
    将 yolov8m 自动修正为 yolov8m.pt，避免 Ultralytics 反复尝试下载错误文件。
    """
    model_arch = str(model_arch).strip()
    if not model_arch:
        return "yolov8m.pt"

    if model_arch.startswith("yolov8") and not model_arch.endswith((".pt", ".yaml", ".yml")):
        model_arch += ".pt"

    return model_arch


def load_model(model_path: str | Path) -> YOLO:
    """
    加载 YOLO 模型。
    """
    return YOLO(str(model_path))


def detect_animals(
    image_path: str | Path,
    model: YOLO,
    animal_whitelist: Iterable[str] = ANIMAL_WHITELIST,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict[str, int]:
    """
    对单张图像进行动物检测，并返回计数字典。
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
        save=False,
    )

    result = results[0]
    counts: Dict[str, int] = {}

    whitelist = {x.lower() for x in animal_whitelist}

    for cls_id, conf, bbox in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
        raw_name = model.names[int(cls_id)].lower()
        mapped_name = ANIMAL_SYNONYMS.get(raw_name, raw_name)

        if mapped_name not in whitelist:
            continue

        counts[mapped_name] = counts.get(mapped_name, 0) + 1

    return counts


def process_directory(
    image_dir: str | Path,
    model: YOLO,
    output_json_path: str | Path,
    animal_whitelist: Iterable[str] = ANIMAL_WHITELIST,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> None:
    """
    批量处理目录中的所有图像，并将预测结果写入 JSON 文件。
    """
    image_dir = Path(image_dir)
    output_json_path = Path(output_json_path)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [p for p in image_dir.iterdir() if p.suffix.lower() in supported_exts]

    predictions: Dict[str, Dict[str, int]] = {}

    for img_path in sorted(image_files):
        try:
            counts = detect_animals(
                image_path=img_path,
                model=model,
                animal_whitelist=animal_whitelist,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            predictions[img_path.name] = counts
        except Exception as e:
            predictions[img_path.name] = {}
            print(f"警告：处理 {img_path} 失败: {e}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"已为 {len(predictions)} 张图像写入预测结果到 {output_json_path}")


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_dataset_root(data_yaml_path: Path, cfg: dict) -> Path:
    """
    根据 YAML 中的 path 字段得到数据集根目录。

    如果 path 是相对路径，则按照 YAML 文件所在目录进行解析。
    """
    yaml_dir = data_yaml_path.parent
    root_value = cfg.get("path", ".")
    root = Path(str(root_value))
    if not root.is_absolute():
        root = (yaml_dir / root).resolve()
    return root


def _collect_train_images_from_item(item: str | Path, dataset_root: Path) -> List[Path]:
    """
    从 train 配置项中收集图片路径。

    train 既可以是图片文件夹，也可以是包含图片路径的 txt 文件。
    """
    item_path = Path(str(item))
    if not item_path.is_absolute():
        item_path = (dataset_root / item_path).resolve()

    image_files: List[Path] = []

    if item_path.is_file() and item_path.suffix.lower() == ".txt":
        for line in item_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            img_path = Path(line)
            if not img_path.is_absolute():
                img_path = (item_path.parent / img_path).resolve()
            if img_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                image_files.append(img_path)
        return image_files

    if item_path.is_dir():
        for img_path in item_path.rglob("*"):
            if img_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                image_files.append(img_path.resolve())
        return image_files

    raise FileNotFoundError(f"无法找到训练图片路径: {item_path}")


def create_limited_train_yaml(
    data_yaml: str | Path,
    max_train_images: Optional[int] = None,
    seed: int = 0,
) -> str:
    """
    根据原始 data.yaml 创建一个“只训练前 N 张/随机 N 张图片”的临时 YAML。

    YOLO 检测任务的 labels 不需要复制，因为 Ultralytics 会根据图片路径
    自动把 images 替换成 labels 去寻找对应的 .txt 标注文件。

    参数：
    - data_yaml：原始数据集 YAML
    - max_train_images：最多训练多少张图片。None 或 <=0 表示使用全部图片。
    - seed：随机抽样种子，保证每次抽样可复现。
    """
    if max_train_images is None or int(max_train_images) <= 0:
        return str(data_yaml)

    data_yaml_path = Path(data_yaml).resolve()
    cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML 格式不正确: {data_yaml_path}")
    if "train" not in cfg:
        raise ValueError("data.yaml 中缺少 train 字段")

    dataset_root = _resolve_dataset_root(data_yaml_path, cfg)
    train_cfg = cfg["train"]

    image_files: List[Path] = []
    if isinstance(train_cfg, list):
        for item in train_cfg:
            image_files.extend(_collect_train_images_from_item(item, dataset_root))
    else:
        image_files.extend(_collect_train_images_from_item(train_cfg, dataset_root))

    image_files = sorted(set(image_files))
    if not image_files:
        raise RuntimeError("没有在 train 配置路径下找到图片，请检查 data.yaml 的 path/train 设置。")

    max_train_images = int(max_train_images)
    selected_count = min(max_train_images, len(image_files))

    rng = random.Random(seed)
    rng.shuffle(image_files)
    selected_images = image_files[:selected_count]

    cache_dir = data_yaml_path.parent / "_yolo_subset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    subset_txt = cache_dir / f"train_subset_{selected_count}_seed{seed}.txt"
    subset_txt.write_text(
        "\n".join(p.as_posix() for p in selected_images) + "\n",
        encoding="utf-8",
    )

    subset_cfg = copy.deepcopy(cfg)
    # train 使用绝对 txt 路径，避免相对路径在 Windows 下解析出错。
    subset_cfg["train"] = subset_txt.as_posix()

    # 为了“快速试跑”，验证集也使用同一份子集。
    # 否则虽然训练只用了 300 张，但每个 epoch 结束后仍会在全部 7681 张图片上验证，
    # 速度会被验证阶段拖慢。
    subset_cfg["val"] = subset_txt.as_posix()

    subset_yaml = cache_dir / f"data_subset_{selected_count}_seed{seed}_fastval.yaml"
    subset_yaml.write_text(
        yaml.safe_dump(subset_cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"原训练集图片数: {len(image_files)}")
    print(f"本次只训练图片数: {selected_count}")
    print(f"临时训练列表: {subset_txt}")
    print(f"临时数据配置: {subset_yaml}")

    return str(subset_yaml)

def train_yolo(
    data_yaml: str | Path,
    model_arch: str = "yolov8m.pt",
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 8,
    lr: float = 0.01,
    device: str = "cuda",
    workers: int = 0,
    max_train_images: Optional[int] = None,
    sample_seed: int = 0,
) -> None:
    """
    使用自定义数据集训练 YOLO 模型。

    参数说明：
    - data_yaml：数据集配置文件路径
    - model_arch：模型权重，例如 yolov8n.pt / yolov8s.pt / yolov8m.pt
    - epochs：训练轮数
    - img_size：输入图片尺寸
    - batch_size：批大小；显存不足时改小，例如 4 或 2
    - lr：初始学习率
    - device：训练设备，默认 cuda，强制使用 GPU
    - workers：Windows 下建议为 0，避免多进程加载数据导致异常
    - max_train_images：最多训练多少张图片；None、0 或负数表示使用全部图片
    - sample_seed：随机抽样种子
    """
    print(get_torch_device_info())

    model_arch = normalize_model_arch(model_arch)
    selected_device = resolve_train_device(device)
    train_data_yaml = create_limited_train_yaml(
        data_yaml=data_yaml,
        max_train_images=max_train_images,
        seed=sample_seed,
    )

    model = YOLO(model_arch)

    results = model.train(
        data=str(train_data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=lr,
        device=selected_device,
        workers=workers,
    )

    print("训练完成。最优权重保存在:", results.save_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animal detection and counting using YOLO.")
    parser.add_argument("--images", type=str, required=True, help="待预测图片文件夹")
    parser.add_argument("--output", type=str, default="predictions.json", help="输出 JSON 文件路径")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU 阈值")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    process_directory(
        image_dir=args.images,
        model=model,
        output_json_path=args.output,
        animal_whitelist=ANIMAL_WHITELIST,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )


if __name__ == "__main__":
    main()
