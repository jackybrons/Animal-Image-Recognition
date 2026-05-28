"""
label_tool.py

YOLO检测数据集的动物数据标注与校正工具。

Examples:
    python label_tool.py --images path/to/images --labels path/to/labels
    python label_tool.py --images path/to/images --labels path/to/labels --model path/to/best.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


CLASSES = [
    "cat", "dog", "horse", "cow",
    "sheep", "goat", "pig", "rabbit",
    "chicken", "duck", "goose", "deer",
    "monkey", "fox", "wolf", "bear",
    "tiger", "lion", "zebra", "giraffe",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class LabelToolConfig:
    image_dir: Path
    label_dir: Path
    model_name: str | None = None
    auto_predict_when_no_label: bool = False
    use_yolo_world: str | bool = "AUTO"
    conf: float = 0.25
    iou: float = 0.5


@dataclass
class Box:
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def xyxy_to_yolo(box: Box, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x_center = ((box.x1 + box.x2) / 2) / img_w
    y_center = ((box.y1 + box.y2) / 2) / img_h
    w = (box.x2 - box.x1) / img_w
    h = (box.y2 - box.y1) / img_h

    x_center = min(max(x_center, 0), 1)
    y_center = min(max(y_center, 0), 1)
    w = min(max(w, 0), 1)
    h = min(max(h, 0), 1)
    return x_center, y_center, w, h


def yolo_to_xyxy(cls_id: int, x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Box:
    box_w = w * img_w
    box_h = h * img_h
    cx = x * img_w
    cy = y * img_h
    x1 = cx - box_w / 2
    y1 = cy - box_h / 2
    x2 = cx + box_w / 2
    y2 = cy + box_h / 2

    x1 = min(max(x1, 0), img_w - 1)
    y1 = min(max(y1, 0), img_h - 1)
    x2 = min(max(x2, 0), img_w - 1)
    y2 = min(max(y2, 0), img_h - 1)
    return Box(cls_id, x1, y1, x2, y2)


class AnimalLabelUI:
    def __init__(self, root: tk.Tk, config: LabelToolConfig):
        self.root = root
        self.config = config
        self.root.title("动物数据集标注工具 - YOLO格式")
        self.root.geometry("1450x900")

        self.config.label_dir.mkdir(parents=True, exist_ok=True)
        self.images = sorted([p for p in self.config.image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
        self.index = 0

        self.model = None
        self.current_image_path: Path | None = None
        self.current_label_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.tk_image = None

        self.img_w = 1
        self.img_h = 1
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.boxes: list[Box] = []
        self.selected_idx: int | None = None

        self.start_x = None
        self.start_y = None
        self.preview_rect = None
        self.is_drawing = False

        self.build_ui()
        self.bind_shortcuts()

        if not self.images:
            messagebox.showerror("没有找到图片", f"图片文件夹中没有找到图片：\n{self.config.image_dir}")
        else:
            self.load_image(0)

    def build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(main, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)

        self.info_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.info_var).pack(anchor="w", pady=(0, 6))

        self.canvas = tk.Canvas(left, bg="#222222", width=1100, height=780)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        ttk.Label(right, text="当前标注类别").pack(anchor="w")
        self.class_var = tk.StringVar(value=CLASSES[0])
        self.class_combo = ttk.Combobox(right, textvariable=self.class_var, values=CLASSES, state="readonly", height=20)
        self.class_combo.pack(fill=tk.X, pady=(2, 10))
        self.class_combo.bind("<<ComboboxSelected>>", self.on_class_changed)

        ttk.Button(right, text="删除选中框", command=self.delete_selected).pack(fill=tk.X, pady=3)
        ttk.Button(right, text="清空当前图片所有框", command=self.clear_boxes).pack(fill=tk.X, pady=3)
        ttk.Button(right, text="自动识别当前图片作为初稿", command=self.predict_current).pack(fill=tk.X, pady=(12, 3))

        ttk.Separator(right).pack(fill=tk.X, pady=10)

        ttk.Label(right, text="当前图片中的目标框").pack(anchor="w")
        self.listbox = tk.Listbox(right, height=22)
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=(3, 8))
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

        nav = ttk.Frame(right)
        nav.pack(fill=tk.X, pady=5)
        ttk.Button(nav, text="上一张 A", command=self.prev_image).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(nav, text="下一张 D", command=self.next_image).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))

        ttk.Button(right, text="保存标签 S", command=self.save_labels).pack(fill=tk.X, pady=(8, 3))
        ttk.Button(right, text="保存并下一张 Enter", command=self.save_and_next).pack(fill=tk.X, pady=3)

        help_text = (
            "鼠标左键拖拽画框。\n"
            "右侧选择动物类别。\n"
            "选中列表里的框后可修改类别或删除。\n"
            "保存后会生成同名 .txt YOLO 标签。\n\n"
            "快捷键：\n"
            "S 保存；Enter 保存并下一张；\n"
            "A 上一张；D 下一张；Delete 删除选中框。"
        )
        ttk.Label(right, text=help_text, justify=tk.LEFT, wraplength=300).pack(anchor="w", pady=(10, 0))

    def bind_shortcuts(self):
        self.root.bind("<s>", lambda e: self.save_labels())
        self.root.bind("<S>", lambda e: self.save_labels())
        self.root.bind("<Return>", lambda e: self.save_and_next())
        self.root.bind("<a>", lambda e: self.prev_image())
        self.root.bind("<A>", lambda e: self.prev_image())
        self.root.bind("<d>", lambda e: self.next_image())
        self.root.bind("<D>", lambda e: self.next_image())
        self.root.bind("<Delete>", lambda e: self.delete_selected())

    def load_model(self):
        if not self.config.model_name:
            messagebox.showwarning("未设置模型", "未通过 --model 指定模型，只能手动标注。")
            return None
        if YOLO is None:
            messagebox.showerror("缺少 ultralytics", "没有安装 ultralytics，无法自动识别。")
            return None

        if self.model is None:
            self.model = YOLO(self.config.model_name)
            if self.config.use_yolo_world == "AUTO":
                should_set_classes = "world" in str(self.config.model_name).lower() and hasattr(self.model, "set_classes")
            else:
                should_set_classes = bool(self.config.use_yolo_world)
            if should_set_classes and hasattr(self.model, "set_classes"):
                self.model.set_classes(CLASSES)

        return self.model

    def model_class_to_target_class(self, raw_cls_id: int) -> int | None:
        if self.model is None:
            return None

        names = getattr(self.model, "names", None)
        cls_name = None
        if isinstance(names, dict):
            cls_name = names.get(raw_cls_id, names.get(str(raw_cls_id)))
        elif isinstance(names, (list, tuple)) and 0 <= raw_cls_id < len(names):
            cls_name = names[raw_cls_id]

        if isinstance(cls_name, str) and cls_name in CLASSES:
            return CLASSES.index(cls_name)
        if 0 <= raw_cls_id < len(CLASSES):
            return raw_cls_id
        return None

    def label_path_for(self, image_path: Path) -> Path:
        return self.config.label_dir / f"{image_path.stem}.txt"

    def load_image(self, idx: int):
        if idx < 0 or idx >= len(self.images):
            return

        self.index = idx
        self.current_image_path = self.images[self.index]
        self.current_label_path = self.label_path_for(self.current_image_path)

        self.original_image = Image.open(self.current_image_path).convert("RGB")
        self.img_w, self.img_h = self.original_image.size

        self.boxes = []
        self.selected_idx = None

        if self.current_label_path.exists():
            self.load_labels_from_file()
        elif self.config.auto_predict_when_no_label:
            self.auto_predict_silent()

        self.render_image()
        self.refresh_list()
        self.update_info()

    def update_info(self):
        label_status = "已有标签" if self.current_label_path and self.current_label_path.exists() else "未保存标签"
        self.info_var.set(
            f"[{self.index + 1}/{len(self.images)}]  {self.current_image_path.name}  "
            f"尺寸: {self.img_w}x{self.img_h}  目标数: {len(self.boxes)}  状态: {label_status}"
        )

    def load_labels_from_file(self):
        assert self.current_label_path is not None
        text = self.current_label_path.read_text(encoding="utf-8").strip()
        if not text:
            return

        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            if 0 <= cls_id < len(CLASSES):
                self.boxes.append(yolo_to_xyxy(cls_id, x, y, w, h, self.img_w, self.img_h))

    def auto_predict_silent(self):
        model = self.load_model()
        if model is None or self.current_image_path is None:
            return

        results = model.predict(source=str(self.current_image_path), conf=self.config.conf, iou=self.config.iou, verbose=False)
        self.boxes = []
        result = results[0]

        for b in result.boxes:
            raw_cls_id = int(b.cls[0])
            cls_id = self.model_class_to_target_class(raw_cls_id)
            if cls_id is None:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1 = min(max(x1, 0), self.img_w - 1)
            y1 = min(max(y1, 0), self.img_h - 1)
            x2 = min(max(x2, 0), self.img_w - 1)
            y2 = min(max(y2, 0), self.img_h - 1)
            if x2 > x1 and y2 > y1:
                self.boxes.append(Box(cls_id, x1, y1, x2, y2))

    def predict_current(self):
        self.auto_predict_silent()
        self.selected_idx = None
        self.render_image()
        self.refresh_list()
        self.update_info()

    def render_image(self):
        if self.original_image is None:
            return

        self.root.update_idletasks()
        canvas_w = max(self.canvas.winfo_width(), 800)
        canvas_h = max(self.canvas.winfo_height(), 600)

        self.scale = min(canvas_w / self.img_w, canvas_h / self.img_h, 1.0)
        disp_w = int(self.img_w * self.scale)
        disp_h = int(self.img_h * self.scale)

        resized = self.original_image.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.offset_x = (canvas_w - disp_w) // 2
        self.offset_y = (canvas_h - disp_h) // 2
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)
        self.draw_boxes()

    def draw_boxes(self):
        for i, box in enumerate(self.boxes):
            x1 = self.offset_x + box.x1 * self.scale
            y1 = self.offset_y + box.y1 * self.scale
            x2 = self.offset_x + box.x2 * self.scale
            y2 = self.offset_y + box.y2 * self.scale

            color = "#ff3333" if i == self.selected_idx else "#00ff66"
            width = 3 if i == self.selected_idx else 2

            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
            label = f"{i + 1}: {CLASSES[box.cls_id]}"
            self.canvas.create_text(
                x1 + 4,
                max(y1 - 16, self.offset_y + 10),
                anchor=tk.NW,
                text=label,
                fill=color,
                font=("Arial", 12, "bold"),
            )

    def canvas_to_image_xy(self, cx: float, cy: float) -> tuple[float, float]:
        x = (cx - self.offset_x) / self.scale
        y = (cy - self.offset_y) / self.scale
        x = min(max(x, 0), self.img_w - 1)
        y = min(max(y, 0), self.img_h - 1)
        return x, y

    def is_inside_image_area(self, cx: float, cy: float) -> bool:
        x, y = self.canvas_to_image_xy(cx, cy)
        return 0 <= x <= self.img_w - 1 and 0 <= y <= self.img_h - 1

    def hit_test_box(self, cx: float, cy: float) -> int | None:
        if self.original_image is None or self.scale <= 0:
            return None

        x, y = self.canvas_to_image_xy(cx, cy)
        tolerance = max(3 / self.scale, 2)
        candidates: list[tuple[float, int]] = []

        for i, box in enumerate(self.boxes):
            inside = box.x1 - tolerance <= x <= box.x2 + tolerance and box.y1 - tolerance <= y <= box.y2 + tolerance
            if inside:
                area = max((box.x2 - box.x1) * (box.y2 - box.y1), 1)
                candidates.append((area, i))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def select_box(self, idx: int | None):
        if idx is None or not (0 <= idx < len(self.boxes)):
            self.selected_idx = None
            self.listbox.selection_clear(0, tk.END)
        else:
            self.selected_idx = idx
            self.class_var.set(CLASSES[self.boxes[idx].cls_id])
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)
        self.render_image()

    def on_mouse_down(self, event):
        if self.original_image is None or not self.is_inside_image_area(event.x, event.y):
            return

        clicked_idx = self.hit_test_box(event.x, event.y)
        if clicked_idx is not None:
            self.select_box(clicked_idx)
            self.start_x = self.start_y = None
            self.preview_rect = None
            self.is_drawing = False
            return

        self.selected_idx = None
        self.listbox.selection_clear(0, tk.END)
        self.start_x, self.start_y = event.x, event.y
        self.is_drawing = True
        self.preview_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y, outline="#ffff00", width=2, dash=(4, 2)
        )

    def on_mouse_move(self, event):
        if not self.is_drawing or self.preview_rect is None or self.start_x is None or self.start_y is None:
            return

        x = min(max(event.x, self.offset_x), self.offset_x + self.img_w * self.scale)
        y = min(max(event.y, self.offset_y), self.offset_y + self.img_h * self.scale)
        self.canvas.coords(self.preview_rect, self.start_x, self.start_y, x, y)

    def on_mouse_up(self, event):
        if not self.is_drawing or self.preview_rect is None or self.start_x is None or self.start_y is None:
            return

        end_x = min(max(event.x, self.offset_x), self.offset_x + self.img_w * self.scale)
        end_y = min(max(event.y, self.offset_y), self.offset_y + self.img_h * self.scale)

        x1c, x2c = sorted([self.start_x, end_x])
        y1c, y2c = sorted([self.start_y, end_y])

        self.canvas.delete(self.preview_rect)
        self.preview_rect = None
        self.is_drawing = False

        if abs(x2c - x1c) < 8 or abs(y2c - y1c) < 8:
            self.start_x = self.start_y = None
            return

        x1, y1 = self.canvas_to_image_xy(x1c, y1c)
        x2, y2 = self.canvas_to_image_xy(x2c, y2c)

        cls_id = CLASSES.index(self.class_var.get())
        self.boxes.append(Box(cls_id, x1, y1, x2, y2))
        self.selected_idx = len(self.boxes) - 1

        self.start_x = self.start_y = None
        self.render_image()
        self.refresh_list()
        self.update_info()

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for i, box in enumerate(self.boxes):
            cls_name = CLASSES[box.cls_id]
            self.listbox.insert(tk.END, f"{i + 1}. {cls_name}  [{box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f}]")

        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.boxes):
            self.listbox.selection_set(self.selected_idx)
            self.listbox.see(self.selected_idx)
            self.class_var.set(CLASSES[self.boxes[self.selected_idx].cls_id])

    def on_list_select(self, event):
        sel = self.listbox.curselection()
        if sel:
            self.select_box(int(sel[0]))

    def on_class_changed(self, event=None):
        if self.selected_idx is None or not (0 <= self.selected_idx < len(self.boxes)):
            return

        cls_id = CLASSES.index(self.class_var.get())
        if self.boxes[self.selected_idx].cls_id == cls_id:
            return

        self.boxes[self.selected_idx].cls_id = cls_id
        self.render_image()
        self.refresh_list()

    def delete_selected(self):
        if self.selected_idx is None or not (0 <= self.selected_idx < len(self.boxes)):
            return

        del self.boxes[self.selected_idx]
        self.selected_idx = None
        self.render_image()
        self.refresh_list()
        self.update_info()

    def clear_boxes(self):
        if self.boxes and messagebox.askyesno("确认清空", "确定要清空当前图片的所有目标框吗？"):
            self.boxes = []
            self.selected_idx = None
            self.render_image()
            self.refresh_list()
            self.update_info()

    def save_labels(self):
        if self.current_label_path is None:
            return

        lines = []
        for box in self.boxes:
            if box.x2 <= box.x1 or box.y2 <= box.y1:
                continue
            x, y, w, h = xyxy_to_yolo(box, self.img_w, self.img_h)
            if w > 0 and h > 0:
                lines.append(f"{box.cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        self.current_label_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_label_path.write_text("\n".join(lines), encoding="utf-8")
        self.update_info()

    def save_and_next(self):
        self.save_labels()
        self.next_image()

    def prev_image(self):
        if self.index > 0:
            self.load_image(self.index - 1)

    def next_image(self):
        if self.index < len(self.images) - 1:
            self.load_image(self.index + 1)
        else:
            messagebox.showinfo("完成", "已经是最后一张图片。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO animal dataset labeling tool.")
    parser.add_argument("--images", required=True, help="图片文件夹路径")
    parser.add_argument("--labels", required=True, help="YOLO 标签输出文件夹路径")
    parser.add_argument("--model", default=None, help="可选：用于自动预标注的 YOLO 权重")
    parser.add_argument("--auto-predict", action="store_true", help="没有标签时自动使用模型预标注")
    parser.add_argument("--conf", type=float, default=0.25, help="自动预标注置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="自动预标注 IoU 阈值")
    return parser.parse_args()


def main():
    args = parse_args()
    config = LabelToolConfig(
        image_dir=Path(args.images),
        label_dir=Path(args.labels),
        model_name=args.model,
        auto_predict_when_no_label=args.auto_predict,
        conf=args.conf,
        iou=args.iou,
    )
    root = tk.Tk()
    AnimalLabelUI(root, config)
    root.mainloop()


if __name__ == "__main__":
    main()
