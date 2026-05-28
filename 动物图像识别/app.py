"""
app_yolo26m_step_viewer_clean.py
动物检测与计数系统 GUI - YOLO 单张确认预览版

本版改动：
1. 主界面重新整理：只放训练、预测参数和日志，不再把大图片塞进主界面。
2. 预测时自动打开一个独立的“预测结果查看窗口”。
3. 每预测完一张图，就在新窗口中显示带框图片，并暂停等待用户点击“下一张”。
4. 点击“下一张”后才继续预测下一张图片。
5. 保留 agnostic_nms=True 和二次手动去重，减少重复框计数。
6. JSON 输出格式仍然是：图片文件名 -> 各动物类别数量。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageTk = None

from main import (
    get_torch_device_info,
    load_model,
    train_yolo,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class AnimalDetectApp:
    """动物检测与计数 GUI 程序 - YOLO 单张确认预览版。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("动物检测与计数系统 - YOLO 单张预览版")
        self.root.geometry("1050x760")
        self.root.minsize(980, 680)

        self.predict_thread: Optional[threading.Thread] = None
        self.train_thread: Optional[threading.Thread] = None
        self.next_event = threading.Event()
        self.stop_predict_flag = threading.Event()

        # 独立预览窗口相关变量
        self.viewer_window: Optional[tk.Toplevel] = None
        self.viewer_canvas: Optional[tk.Canvas] = None
        self.viewer_info_var = tk.StringVar(value="尚未开始预测")
        self.viewer_count_var = tk.StringVar(value="")
        self.viewer_status_var = tk.StringVar(value="")
        self.viewer_photo = None
        self.viewer_pil_image = None
        self.viewer_max_width = 1100
        self.viewer_max_height = 760
        self.next_button: Optional[tk.Button] = None
        self.stop_button: Optional[tk.Button] = None

        self.build_ui()
        self.append_log("程序已启动：YOLO 单张确认预览版。")
        self.append_log("预测时会打开独立图片窗口，每张预测完后必须点击“下一张”才会继续。")
        self.append_log("默认预测参数：conf=0.40, iou=0.4, imgsz=960, max_det=30, 二次去重IoU=0.4。")
        self.append_log(get_torch_device_info())

    # ============================================================
    # UI
    # ============================================================
    def build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        title = tk.Label(
            self.root,
            text="动物检测与计数系统 - YOLO 单张预览版",
            font=("Microsoft YaHei", 16, "bold"),
            anchor="w",
        )
        title.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)

        self.train_tab = ttk.Frame(notebook)
        self.predict_tab = ttk.Frame(notebook)
        notebook.add(self.predict_tab, text="预测与计数")
        notebook.add(self.train_tab, text="模型训练")

        self.build_predict_tab(self.predict_tab)
        self.build_train_tab(self.train_tab)

        # 日志固定放下面，不挤占图片区域
        log_frame = ttk.LabelFrame(self.root, text="运行日志")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=(4, 12))
        self.root.grid_rowconfigure(2, weight=0)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    def build_predict_tab(self, parent: ttk.Frame) -> None:
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(3, weight=1)

        # 文件区域
        file_frame = ttk.LabelFrame(parent, text="1. 选择预测输入输出")
        file_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=10, pady=10)
        file_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="预测模型：").grid(row=0, column=0, sticky="w", padx=8, pady=7)
        self.pretrained_var = tk.StringVar(value="自定义")
        self.pretrained_options = [
            "自定义", "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
            "yolov8l.pt", "yolov8x.pt",
        ]
        self.pretrained_menu = ttk.OptionMenu(file_frame, self.pretrained_var, self.pretrained_var.get(), *self.pretrained_options)
        self.pretrained_menu.grid(row=0, column=1, sticky="w", padx=8, pady=7)

        self.model_entry = ttk.Entry(file_frame)
        self.model_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=7)
        ttk.Label(file_frame, text="自定义权重：").grid(row=1, column=0, sticky="w", padx=8, pady=7)
        ttk.Button(file_frame, text="选择权重文件", command=self.choose_model_file).grid(row=1, column=2, padx=8, pady=7)

        ttk.Label(file_frame, text="图片文件夹：").grid(row=2, column=0, sticky="w", padx=8, pady=7)
        self.images_entry = ttk.Entry(file_frame)
        self.images_entry.grid(row=2, column=1, sticky="ew", padx=8, pady=7)
        ttk.Button(file_frame, text="选择图片文件夹", command=self.choose_images_dir).grid(row=2, column=2, padx=8, pady=7)

        ttk.Label(file_frame, text="输出 JSON：").grid(row=3, column=0, sticky="w", padx=8, pady=7)
        self.output_entry = ttk.Entry(file_frame)
        self.output_entry.grid(row=3, column=1, sticky="ew", padx=8, pady=7)
        ttk.Button(file_frame, text="选择保存路径", command=self.choose_output_file).grid(row=3, column=2, padx=8, pady=7)

        # 参数区域
        param_frame = ttk.LabelFrame(parent, text="2. 预测参数")
        param_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=10, pady=8)
        for col in range(6):
            param_frame.grid_columnconfigure(col, weight=1)

        ttk.Label(param_frame, text="conf：").grid(row=0, column=0, sticky="e", padx=6, pady=8)
        self.conf_entry = ttk.Entry(param_frame, width=10)
        self.conf_entry.insert(0, "0.40")
        self.conf_entry.grid(row=0, column=1, sticky="w", padx=6, pady=8)

        ttk.Label(param_frame, text="NMS iou：").grid(row=0, column=2, sticky="e", padx=6, pady=8)
        self.iou_entry = ttk.Entry(param_frame, width=10)
        self.iou_entry.insert(0, "0.4")
        self.iou_entry.grid(row=0, column=3, sticky="w", padx=6, pady=8)

        ttk.Label(param_frame, text="imgsz：").grid(row=0, column=4, sticky="e", padx=6, pady=8)
        self.pred_imgsz_entry = ttk.Entry(param_frame, width=10)
        self.pred_imgsz_entry.insert(0, "960")
        self.pred_imgsz_entry.grid(row=0, column=5, sticky="w", padx=6, pady=8)

        ttk.Label(param_frame, text="max_det：").grid(row=1, column=0, sticky="e", padx=6, pady=8)
        self.max_det_entry = ttk.Entry(param_frame, width=10)
        self.max_det_entry.insert(0, "30")
        self.max_det_entry.grid(row=1, column=1, sticky="w", padx=6, pady=8)

        ttk.Label(param_frame, text="二次去重 IoU：").grid(row=1, column=2, sticky="e", padx=6, pady=8)
        self.post_nms_iou_entry = ttk.Entry(param_frame, width=10)
        self.post_nms_iou_entry.insert(0, "0.4")
        self.post_nms_iou_entry.grid(row=1, column=3, sticky="w", padx=6, pady=8)

        self.step_preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            param_frame,
            text="逐张显示，点击下一张才继续",
            variable=self.step_preview_var,
        ).grid(row=1, column=4, columnspan=2, sticky="w", padx=6, pady=8)

        hint = ttk.Label(
            parent,
            text="说明：图片不会显示在主界面里，预测时会打开一个独立大窗口，避免遮挡和界面混乱。",
            foreground="#555555",
        )
        hint.grid(row=2, column=0, columnspan=4, sticky="w", padx=12, pady=(2, 10))

        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=10, pady=8)
        ttk.Button(button_frame, text="开始预测并打开图片窗口", command=self.start_predict_thread).pack(side="left", padx=4)
        ttk.Button(button_frame, text="只打开图片窗口", command=self.open_viewer_window).pack(side="left", padx=4)
        ttk.Button(button_frame, text="检测 GPU 状态", command=self.show_device_info).pack(side="left", padx=4)

    def build_train_tab(self, parent: ttk.Frame) -> None:
        parent.grid_columnconfigure(1, weight=1)

        train_frame = ttk.LabelFrame(parent, text="训练设置")
        train_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        train_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(train_frame, text="数据集 YAML：").grid(row=0, column=0, sticky="w", padx=8, pady=7)
        self.data_yaml_entry = ttk.Entry(train_frame)
        self.data_yaml_entry.grid(row=0, column=1, sticky="ew", padx=8, pady=7)
        ttk.Button(train_frame, text="选择 YAML", command=self.choose_data_yaml).grid(row=0, column=2, padx=8, pady=7)

        ttk.Label(train_frame, text="epochs：").grid(row=1, column=0, sticky="w", padx=8, pady=7)
        self.epochs_entry = ttk.Entry(train_frame, width=12)
        self.epochs_entry.insert(0, "200")
        self.epochs_entry.grid(row=1, column=1, sticky="w", padx=8, pady=7)

        ttk.Label(train_frame, text="模型权重：").grid(row=2, column=0, sticky="w", padx=8, pady=7)
        self.arch_entry = ttk.Entry(train_frame, width=18)
        self.arch_entry.insert(0, "yolo11n.pt")
        self.arch_entry.grid(row=2, column=1, sticky="w", padx=8, pady=7)

        ttk.Label(train_frame, text="训练 imgsz：").grid(row=3, column=0, sticky="w", padx=8, pady=7)
        self.imgsz_entry = ttk.Entry(train_frame, width=12)
        self.imgsz_entry.insert(0, "832")
        self.imgsz_entry.grid(row=3, column=1, sticky="w", padx=8, pady=7)

        ttk.Label(train_frame, text="batch：").grid(row=4, column=0, sticky="w", padx=8, pady=7)
        self.batch_entry = ttk.Entry(train_frame, width=12)
        self.batch_entry.insert(0, "4")
        self.batch_entry.grid(row=4, column=1, sticky="w", padx=8, pady=7)

        ttk.Label(train_frame, text="最多训练图片数：").grid(row=5, column=0, sticky="w", padx=8, pady=7)
        self.max_images_entry = ttk.Entry(train_frame, width=12)
        self.max_images_entry.insert(0, "0")
        self.max_images_entry.grid(row=5, column=1, sticky="w", padx=8, pady=7)
        ttk.Label(train_frame, text="0 或留空 = 全部图片").grid(row=5, column=1, sticky="w", padx=(110, 8), pady=7)

        ttk.Label(train_frame, text="训练设备：").grid(row=6, column=0, sticky="w", padx=8, pady=7)
        self.device_var = tk.StringVar(value="cuda")
        ttk.OptionMenu(train_frame, self.device_var, self.device_var.get(), "cuda", "auto", "cpu").grid(row=6, column=1, sticky="w", padx=8, pady=7)

        ttk.Button(train_frame, text="开始训练", command=self.start_train_thread).grid(row=7, column=1, sticky="w", padx=8, pady=12)

    # ============================================================
    # 文件选择
    # ============================================================
    def choose_data_yaml(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml *.yml")])
        if path:
            self.data_yaml_entry.delete(0, tk.END)
            self.data_yaml_entry.insert(0, path)

    def choose_model_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pt")])
        if path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, path)
            self.pretrained_var.set("自定义")

    def choose_images_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.images_entry.delete(0, tk.END)
            self.images_entry.insert(0, path)

    def choose_output_file(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)

    # ============================================================
    # 日志
    # ============================================================
    def append_log(self, message: str) -> None:
        self.root.after(0, self._append_log, message)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, str(message) + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def show_device_info(self) -> None:
        self.append_log(get_torch_device_info())

    # ============================================================
    # 训练
    # ============================================================
    def start_train_thread(self) -> None:
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showwarning("提示", "训练正在进行中，请不要重复点击。")
            return
        self.train_thread = threading.Thread(target=self.train_model, daemon=True)
        self.train_thread.start()

    def train_model(self) -> None:
        data_yaml = self.data_yaml_entry.get().strip()
        if not data_yaml:
            self.root.after(0, lambda: messagebox.showerror("错误", "请先选择数据集配置文件！"))
            return

        try:
            epochs_int = int(self.epochs_entry.get().strip())
            img_size_int = int(self.imgsz_entry.get().strip())
            batch_size_int = int(self.batch_entry.get().strip())
            max_images_text = self.max_images_entry.get().strip()
            max_train_images = int(max_images_text) if max_images_text else 0
        except ValueError:
            self.root.after(0, lambda: messagebox.showerror("错误", "训练轮数、图片尺寸、批大小、最多训练图片数必须是整数！"))
            return

        model_arch = self.arch_entry.get().strip() or "yolo26m.pt"
        device = self.device_var.get().strip()

        self.append_log("=" * 80)
        self.append_log("开始训练")
        self.append_log(f"模型权重: {model_arch}")
        self.append_log(f"数据集配置: {data_yaml}")
        self.append_log(f"训练轮数: {epochs_int}")
        self.append_log(f"图片尺寸: {img_size_int}")
        self.append_log(f"批大小: {batch_size_int}")
        self.append_log(f"最多训练图片数: {max_train_images if max_train_images > 0 else '全部图片'}")
        self.append_log(f"训练设备: {device}")

        try:
            train_yolo(
                data_yaml=data_yaml,
                model_arch=model_arch,
                epochs=epochs_int,
                img_size=img_size_int,
                batch_size=batch_size_int,
                device=device,
                max_train_images=max_train_images,
            )
            self.append_log("训练完成！请到 runs/detect/train*/weights/best.pt 找最优权重。")
        except Exception as e:
            self.append_log(f"训练过程中出现错误：{e}")
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                self.append_log("如果是显存不足：优先把 batch 改 2；还不行把 imgsz 改 768 或模型改 yolo26s.pt。")

    # ============================================================
    # 预测主流程
    # ============================================================
    def start_predict_thread(self) -> None:
        if self.predict_thread and self.predict_thread.is_alive():
            messagebox.showwarning("提示", "预测正在进行中，请不要重复点击。")
            return

        if Image is None or ImageDraw is None or ImageTk is None:
            messagebox.showerror("缺少依赖", "图片预览需要 Pillow，请先运行：pip install pillow")
            return

        self.stop_predict_flag.clear()
        self.next_event.clear()
        self.open_viewer_window()
        self.predict_thread = threading.Thread(target=self.predict, daemon=True)
        self.predict_thread.start()

    def predict(self) -> None:
        selected = self.pretrained_var.get().strip()
        model_path = selected if selected != "自定义" else self.model_entry.get().strip()
        images_dir = self.images_entry.get().strip()
        output_json = self.output_entry.get().strip()

        if not model_path:
            self.root.after(0, lambda: messagebox.showerror("错误", "请先选择模型权重文件！"))
            return
        if not images_dir or not output_json:
            self.root.after(0, lambda: messagebox.showerror("错误", "请先选择图片目录和输出路径！"))
            return

        try:
            conf = float(self.conf_entry.get().strip())
            iou = float(self.iou_entry.get().strip())
            pred_imgsz = int(self.pred_imgsz_entry.get().strip())
            max_det = int(self.max_det_entry.get().strip())
            post_nms_iou = float(self.post_nms_iou_entry.get().strip())
        except ValueError:
            self.root.after(0, lambda: messagebox.showerror("错误", "conf、iou、imgsz、max_det、二次去重 IoU 参数格式不正确！"))
            return

        self.append_log("=" * 80)
        self.append_log(f"加载模型: {model_path}")
        self.append_log(f"预测图片文件夹: {images_dir}")
        self.append_log(f"预测参数: conf={conf}, iou={iou}, imgsz={pred_imgsz}, max_det={max_det}, 二次去重IoU={post_nms_iou}")
        self.append_log("预测策略：优先使用 agnostic_nms=True + 二次手动去重。")

        try:
            model = load_model(model_path)
            self.process_directory_step_by_step(
                image_dir=images_dir,
                model=model,
                output_json_path=output_json,
                conf=conf,
                iou=iou,
                imgsz=pred_imgsz,
                max_det=max_det,
                post_nms_iou=post_nms_iou,
                step_preview=bool(self.step_preview_var.get()),
            )
            if not self.stop_predict_flag.is_set():
                self.append_log(f"预测完成，结果已保存到: {output_json}")
                self.root.after(0, self.mark_viewer_finished)
        except Exception as e:
            self.append_log(f"推断过程中出现错误：{e}")
            self.root.after(0, lambda: self.viewer_status_var.set(f"预测出错：{e}"))
            if "out of memory" in str(e).lower():
                self.append_log("预测爆显存：把预测 imgsz 从 960 改成 832 或 640。")

    def process_directory_step_by_step(
        self,
        image_dir: str,
        model: Any,
        output_json_path: str,
        conf: float = 0.60,
        iou: float = 0.25,
        imgsz: int = 960,
        max_det: int = 30,
        post_nms_iou: float = 0.25,
        step_preview: bool = True,
    ) -> None:
        image_root = Path(image_dir)
        if not image_root.exists():
            raise FileNotFoundError(f"图片文件夹不存在: {image_dir}")

        image_paths = [
            p for p in sorted(image_root.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if not image_paths:
            raise RuntimeError(f"没有在文件夹中找到图片: {image_dir}")

        names = getattr(model, "names", {})
        all_results: Dict[str, Dict[str, int]] = {}

        for idx, image_path in enumerate(image_paths, start=1):
            if self.stop_predict_flag.is_set():
                self.append_log("预测已被用户停止。")
                break

            self.append_log(f"[{idx}/{len(image_paths)}] 预测: {image_path.name}")

            results = self.predict_yolo26m_with_traditional_nms(
                model=model,
                image_path=image_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
            )

            result = results[0]
            boxes = result.boxes
            counts: Dict[str, int] = {}
            detections: List[Dict[str, Any]] = []

            if boxes is not None and len(boxes) > 0:
                keep_indices = self.second_stage_class_agnostic_nms(boxes=boxes, iou_thres=post_nms_iou)
                self.append_log(f"    模型输出框数: {len(boxes)}，二次去重后计数框数: {len(keep_indices)}")

                cls_list = boxes.cls.cpu().tolist()
                xyxy_list = boxes.xyxy.cpu().tolist()
                conf_list = boxes.conf.cpu().tolist()

                for keep_idx in keep_indices:
                    cls_id = int(cls_list[keep_idx])
                    class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    counts[class_name] = counts.get(class_name, 0) + 1
                    detections.append({
                        "xyxy": [float(v) for v in xyxy_list[keep_idx]],
                        "class_name": class_name,
                        "conf": float(conf_list[keep_idx]),
                    })
            else:
                self.append_log("    未检测到目标。")

            all_results[image_path.name] = counts

            # 先保存一次临时 JSON，避免中途停止后结果全丢
            self.save_json(output_json_path, all_results)

            self.show_image_in_viewer(
                image_path=image_path,
                detections=detections,
                counts=counts,
                index=idx,
                total=len(image_paths),
                step_preview=step_preview,
            )

            if step_preview and idx < len(image_paths):
                self.next_event.clear()
                self.append_log("    已暂停：请在图片窗口点击“下一张”继续。")
                self.next_event.wait()

        self.save_json(output_json_path, all_results)

    @staticmethod
    def save_json(output_json_path: str, data: Dict[str, Dict[str, int]]) -> None:
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ============================================================
    # 独立查看窗口
    # ============================================================
    def open_viewer_window(self) -> None:
        if self.viewer_window is not None and self.viewer_window.winfo_exists():
            self.viewer_window.lift()
            return

        self.viewer_window = tk.Toplevel(self.root)
        self.viewer_window.title("预测结果查看窗口 - 点击下一张继续")
        self.viewer_window.geometry("1200x900")
        self.viewer_window.minsize(900, 650)
        self.viewer_window.protocol("WM_DELETE_WINDOW", self.on_close_viewer)
        self.viewer_window.grid_columnconfigure(0, weight=1)
        self.viewer_window.grid_rowconfigure(1, weight=1)

        info_frame = ttk.Frame(self.viewer_window)
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        info_frame.grid_columnconfigure(0, weight=1)

        info_label = ttk.Label(info_frame, textvariable=self.viewer_info_var, font=("Microsoft YaHei", 12, "bold"), anchor="w")
        info_label.grid(row=0, column=0, sticky="ew", padx=4, pady=2)

        count_label = ttk.Label(info_frame, textvariable=self.viewer_count_var, anchor="w")
        count_label.grid(row=1, column=0, sticky="ew", padx=4, pady=2)

        self.viewer_canvas = tk.Canvas(self.viewer_window, bg="#222222", highlightthickness=0)
        self.viewer_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)
        self.viewer_canvas.bind("<Configure>", self.redraw_viewer_canvas)

        bottom = ttk.Frame(self.viewer_window)
        bottom.grid(row=2, column=0, sticky="ew", padx=10, pady=(6, 10))
        bottom.grid_columnconfigure(0, weight=1)

        status_label = ttk.Label(bottom, textvariable=self.viewer_status_var, anchor="w")
        status_label.grid(row=0, column=0, sticky="ew", padx=4)

        self.next_button = tk.Button(
            bottom,
            text="下一张",
            font=("Microsoft YaHei", 12, "bold"),
            width=12,
            command=self.go_next_image,
            state="disabled",
        )
        self.next_button.grid(row=0, column=1, padx=8)

        self.stop_button = tk.Button(
            bottom,
            text="停止预测",
            font=("Microsoft YaHei", 11),
            width=12,
            command=self.stop_prediction,
        )
        self.stop_button.grid(row=0, column=2, padx=4)

        self.viewer_info_var.set("预测开始后，这里会显示当前图片。")
        self.viewer_count_var.set("")
        self.viewer_status_var.set("提示：图片显示在独立窗口，不会遮挡主界面的参数和日志。")

    def on_close_viewer(self) -> None:
        # 关闭查看窗口时不强制关闭主程序，但如果预测正在等待下一张，需要解除等待
        self.viewer_window.destroy()
        self.viewer_window = None
        self.go_next_image()

    def stop_prediction(self) -> None:
        self.stop_predict_flag.set()
        self.go_next_image()
        self.viewer_status_var.set("已请求停止预测。当前图片处理结束后会停止。")
        self.append_log("已请求停止预测。")

    def go_next_image(self) -> None:
        if self.next_button is not None:
            self.next_button.configure(state="disabled")
        self.next_event.set()

    def show_image_in_viewer(
        self,
        image_path: Path,
        detections: List[Dict[str, Any]],
        counts: Dict[str, int],
        index: int,
        total: int,
        step_preview: bool,
    ) -> None:
        detections_copy = [dict(item) for item in detections]
        counts_copy = dict(counts)
        self.root.after(
            0,
            lambda: self._show_image_in_viewer_main_thread(
                image_path=image_path,
                detections=detections_copy,
                counts=counts_copy,
                index=index,
                total=total,
                step_preview=step_preview,
            ),
        )

    def _show_image_in_viewer_main_thread(
        self,
        image_path: Path,
        detections: List[Dict[str, Any]],
        counts: Dict[str, int],
        index: int,
        total: int,
        step_preview: bool,
    ) -> None:
        if self.viewer_window is None or not self.viewer_window.winfo_exists():
            self.open_viewer_window()

        try:
            annotated = self.make_annotated_image(image_path, detections)
            self.viewer_pil_image = annotated
            self.update_viewer_canvas_image()

            self.viewer_info_var.set(f"[{index}/{total}] 当前图片：{image_path.name}")
            self.viewer_count_var.set(f"计数结果：{self.format_count_summary(counts)}")

            if index >= total:
                self.viewer_status_var.set("已经是最后一张。预测完成后 JSON 会保存到你选择的位置。")
                if self.next_button is not None:
                    self.next_button.configure(state="disabled")
            elif step_preview:
                self.viewer_status_var.set("检查当前检测框，没有问题就点击“下一张”继续预测。")
                if self.next_button is not None:
                    self.next_button.configure(state="normal")
            else:
                self.viewer_status_var.set("未开启逐张暂停，正在自动预测下一张。")
        except Exception as e:
            self.viewer_status_var.set(f"图片显示失败：{e}")
            self.append_log(f"图片显示失败：{e}")
            if step_preview:
                if self.next_button is not None:
                    self.next_button.configure(state="normal")

    def make_annotated_image(self, image_path: Path, detections: List[Dict[str, Any]]) -> Any:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        line_width = max(2, round(min(img.size) / 250))
        font_size = max(14, round(min(img.size) / 35))
        font = self.get_preview_font(font_size)

        for det in detections:
            x1, y1, x2, y2 = det["xyxy"]
            class_name = str(det["class_name"])
            score = float(det.get("conf", 0.0))
            label = f"{class_name} {score:.2f}"

            draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)

            try:
                bbox = draw.textbbox((x1, y1), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w = max(80, len(label) * 9)
                text_h = 20

            label_y1 = max(0, y1 - text_h - 6)
            label_y2 = label_y1 + text_h + 6
            label_x2 = min(img.width, x1 + text_w + 10)
            draw.rectangle([x1, label_y1, label_x2, label_y2], fill="red")
            draw.text((x1 + 5, label_y1 + 3), label, fill="white", font=font)

        if not detections:
            font = self.get_preview_font(max(18, font_size))
            msg = "未检测到目标"
            draw.rectangle([12, 12, 190, 52], fill="red")
            draw.text((24, 22), msg, fill="white", font=font)

        return img

    def redraw_viewer_canvas(self, event: Any = None) -> None:
        self.update_viewer_canvas_image()

    def update_viewer_canvas_image(self) -> None:
        if self.viewer_canvas is None or self.viewer_pil_image is None or ImageTk is None:
            return

        canvas_w = max(1, self.viewer_canvas.winfo_width())
        canvas_h = max(1, self.viewer_canvas.winfo_height())
        img = self.resize_image_to_fit(self.viewer_pil_image, canvas_w, canvas_h)
        self.viewer_photo = ImageTk.PhotoImage(img)

        self.viewer_canvas.delete("all")
        x = canvas_w // 2
        y = canvas_h // 2
        self.viewer_canvas.create_image(x, y, image=self.viewer_photo, anchor="center")

    @staticmethod
    def resize_image_to_fit(img: Any, max_w: int, max_h: int) -> Any:
        w, h = img.size
        if w <= 0 or h <= 0:
            return img
        scale = min(max_w / w, max_h / h, 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        if (new_w, new_h) == (w, h):
            return img
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        return img.resize((new_w, new_h), resample)

    def mark_viewer_finished(self) -> None:
        if self.viewer_window is not None and self.viewer_window.winfo_exists():
            self.viewer_status_var.set("预测完成，JSON 已保存。可以关闭这个窗口。")
            if self.next_button is not None:
                self.next_button.configure(state="disabled")

    # ============================================================
    # YOLO26m 预测和去重
    # ============================================================
    def predict_yolo26m_with_traditional_nms(
        self,
        model: Any,
        image_path: Path,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
    ) -> Any:
        predict_kwargs = dict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            agnostic_nms=True,
            verbose=False,
        )

        try:
            return model.predict(**predict_kwargs, end2end=False)
        except TypeError as e:
            self.append_log(f"    当前 Ultralytics 不支持 end2end 参数，已自动降级；建议升级 ultralytics。错误信息: {e}")
            return model.predict(**predict_kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "end2end" in msg or "unexpected keyword" in msg:
                self.append_log(f"    end2end=False 不可用，已自动降级；建议升级 ultralytics。错误信息: {e}")
                return model.predict(**predict_kwargs)
            raise

    def second_stage_class_agnostic_nms(self, boxes: Any, iou_thres: float = 0.25) -> List[int]:
        n = len(boxes)
        if n == 0:
            return []
        if iou_thres <= 0:
            return list(range(n))

        xyxy = boxes.xyxy.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        order = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)
        keep: List[int] = []

        while order:
            current = order.pop(0)
            keep.append(current)
            remaining = []
            for idx in order:
                overlap = self.box_iou_xyxy(xyxy[current], xyxy[idx])
                if overlap <= iou_thres:
                    remaining.append(idx)
            order = remaining

        return keep

    @staticmethod
    def box_iou_xyxy(box1: List[float], box2: List[float]) -> float:
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
        area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))

        union = area1 + area2 - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    @staticmethod
    def get_preview_font(size: int) -> Any:
        if ImageFont is None:
            return None
        candidate_fonts = [
            "msyh.ttc",
            "simhei.ttf",
            "arial.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for font_path in candidate_fonts:
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    @staticmethod
    def format_count_summary(counts: Dict[str, int]) -> str:
        if not counts:
            return "未检测到目标"
        return "，".join(f"{name}:{num}" for name, num in counts.items())


def main() -> None:
    root = tk.Tk()
    AnimalDetectApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
