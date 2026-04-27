import sys
import math
import time
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PyQt6.QtWidgets import (QApplication, QWidget, QFileDialog, QHBoxLayout,
                              QVBoxLayout, QPushButton, QSpinBox, QLabel, QFrame)
from PyQt6.QtGui import QPainter, QImage, QColor
from PyQt6.QtCore import Qt, QTimer
from PIL import Image

# ── 模型超参（必须和train_fm.py一致）─────────────────────
WINDOW      = 30
D_MODEL     = 256
N_HEADS     = 8
N_LAYERS    = 6
DROPOUT     = 0.1
CANVAS_SIZE = (224, 224)
BRUSH_MAX   = 436
STROKE_DIM  = 8
CANVAS_FEAT = 128
EPS         = 1e-4
FM_STEPS    = 10

CKPT_PATH   = r"D:\Github\painter\train-6-consistant-flow\base_model_fm.pt"

CANVAS_SIZE_W = 178
CANVAS_SIZE_H = 218
BRUSH_TIP = True
SOFT_ROUND = False


# ── 模型定义（和train_fm.py完全一致）────────────────────

class CanvasEncoder(nn.Module):
    def __init__(self, out_dim=CANVAS_FEAT):
        super().__init__()
        self.proj = nn.Linear(384, out_dim)

    def forward(self, x):
        return self.proj(x)


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb


class FMHead(nn.Module):
    def __init__(self, h_dim=D_MODEL, a_dim=STROKE_DIM, t_emb_dim=32):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(t_emb_dim)
        in_dim = h_dim + a_dim + t_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, a_dim),
        )

    def forward(self, h, at, t):
        t_emb = self.time_emb(t)
        x = torch.cat([h, at, t_emb], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def sample(self, h, steps=FM_STEPS):
        B = h.shape[0]
        device = h.device
        a = torch.randn(B, STROKE_DIM, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i * dt + EPS
            t = torch.full((B, 1), t_val, device=device)
            v = self.forward(h, a, t)
            a = a + v * dt
        return a.clamp(0.0, 1.0)


class StrokeARFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.canvas_encoder = CanvasEncoder(CANVAS_FEAT)
        self.stroke_proj = nn.Linear(STROKE_DIM, D_MODEL)
        self.pos_emb = nn.Embedding(WINDOW + 1, D_MODEL)
        self.canvas_proj = nn.Linear(CANVAS_FEAT, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4,
            batch_first=True, dropout=DROPOUT
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.fm_head = FMHead(h_dim=D_MODEL, a_dim=STROKE_DIM)

    def encode(self, strokes, canvas_feat):
        B, T, _ = strokes.shape
        canvas_token = self.canvas_proj(self.canvas_encoder(canvas_feat)).unsqueeze(1)
        pos = torch.arange(T, device=strokes.device)
        stroke_h = self.stroke_proj(strokes) + self.pos_emb(pos)
        seq = torch.cat([canvas_token, stroke_h], dim=1)
        mask = nn.Transformer.generate_square_subsequent_mask(T + 1, device=strokes.device)
        mask[:, 0] = 0.0
        h = self.transformer(seq, mask=mask, is_causal=False)
        return h[:, -1, :]

    @torch.no_grad()
    def sample(self, strokes, canvas_feat, steps=FM_STEPS):
        h = self.encode(strokes, canvas_feat)
        return self.fm_head.sample(h, steps=steps)


# ── DINOv2特征提取 ────────────────────────────────────

canvas_transform = T.Compose([
    T.Resize(CANVAS_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


def extract_canvas_feat(dino_model, canvas_pixels, device):
    rgba = np.clip(canvas_pixels, 0, 1)
    rgb_linear = rgba[:, :, :3]
    rgb_srgb = np.where(
        rgb_linear <= 0.0031308,
        rgb_linear * 12.92,
        1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055
    )
    alpha = rgba[:, :, 3:4]
    rgb_flat = rgb_srgb * alpha + np.ones_like(rgb_srgb) * (1 - alpha)
    rgb_flat = np.clip(rgb_flat, 0, 1)
    img = Image.fromarray((rgb_flat * 255).astype(np.uint8), "RGB")
    tensor = canvas_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = dino_model(tensor)   # (1, 384)
    return feat


def encode_stroke(s, W, H):
    p = s["samples"][0]
    return [
        p["x"] / W,
        p["y"] / H,
        p["pressure"],
        s["brush_size"] / BRUSH_MAX,
        (s["angle"] + math.pi) / (2 * math.pi),
        s["color"][0],
        s["color"][1],
        s["color"][2],
    ]


def decode_stroke(pred, W, H, stroke_idx):
    return {
        "tool": "brush",
        "brush_size": int(np.clip(round(pred[3] * BRUSH_MAX), 1, BRUSH_MAX)),
        "color": [float(np.clip(pred[5], 0, 1)),
                  float(np.clip(pred[6], 0, 1)),
                  float(np.clip(pred[7], 0, 1))],
        "smoothing": 0.7,
        "minimum_diameter": 0.0,
        "angle": float(pred[4] * 2 * math.pi - math.pi),
        "samples": [{
            "x": float(np.clip(pred[0] * W, 0, W)),
            "y": float(np.clip(pred[1] * H, 0, H)),
            "pressure": float(np.clip(pred[2], 0, 1)),
            "t": float(stroke_idx * 50)
        }]
    }


# ── Canvas / Brush（原版，不改动）────────────────────

class CanvasBuffer:

    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.pixels = np.zeros((h, w, 4), dtype=np.float32)
        self._dirty = True
        self._cache = None

    def blend_brush(self, stamp, x, y):
        h, w = stamp.shape[:2]
        x0 = int(x - w // 2)
        y0 = int(y - h // 2)
        cx0 = max(0, x0)
        cy0 = max(0, y0)
        cx1 = min(self.w, x0 + w)
        cy1 = min(self.h, y0 + h)
        if cx0 >= cx1 or cy0 >= cy1:
            return
        sx0 = cx0 - x0
        sy0 = cy0 - y0
        sx1 = sx0 + (cx1 - cx0)
        sy1 = sy0 + (cy1 - cy0)
        src = stamp[sy0:sy1, sx0:sx1]
        dst = self.pixels[cy0:cy1, cx0:cx1]
        a = src[:, :, 3:4]
        dst[:, :, :3] = src[:, :, :3] * a + dst[:, :, :3] * (1 - a)
        dst[:, :, 3:4] = a + dst[:, :, 3:4] * (1 - a)
        self.pixels[cy0:cy1, cx0:cx1] = dst
        self._dirty = True

    def to_qimage(self):
        if not self._dirty and self._cache is not None:
            return self._cache
        rgba = np.clip(self.pixels, 0, 1)
        rgb_linear = rgba[:, :, :3]
        rgb_srgb = np.where(
            rgb_linear <= 0.0031308,
            rgb_linear * 12.92,
            1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055
        )
        rgba = np.concatenate([rgb_srgb, rgba[:, :, 3:4]], axis=2)
        rgba = np.clip(rgba, 0, 1)
        rgba = (rgba * 255).astype(np.uint8)
        self._cache = QImage(rgba.data, self.w, self.h, self.w * 4,
                             QImage.Format.Format_RGBA8888).copy()
        self._dirty = False
        return self._cache


class Brush:

    TIP_IMAGE_PATH = None

    def __init__(self, size):
        self.size = size
        self.minimum_diameter = 0.0
        self.tip = self.generate_tip(size)
        self.prev_pressure = 0.0
        self.smoothing = 0.7

    def generate_tip(self, size):
        if BRUSH_TIP and Brush.TIP_IMAGE_PATH and os.path.exists(Brush.TIP_IMAGE_PATH):
            img = Image.open(Brush.TIP_IMAGE_PATH).convert("L")
            img = img.resize((size, size), Image.LANCZOS)
            alpha = np.array(img, dtype=np.float32) / 255.0
            return alpha
        cx = size / 2
        cy = size / 2
        r = size / 2
        ys, xs = np.mgrid[0:size, 0:size]
        dx = xs - cx
        dy = ys - cy
        d2 = dx * dx + dy * dy
        if SOFT_ROUND:
            sigma = r / 2.0
            alpha = np.exp(-d2 / (2.0 * sigma ** 2)).astype(np.float32)
        else:
            alpha = (np.sqrt(d2) <= r).astype(np.float32)
        return alpha

    def stamp(self, color, pressure, angle=0.0):

        print(f"pressure: {pressure}")  # DO NOT DELETE

        pressure = max(0.0, min(1.0, pressure))
        pressure = (
            self.smoothing * self.prev_pressure
            + (1 - self.smoothing) * pressure
        )
        self.prev_pressure = pressure
        max_size = self.size
        min_size = max(1, int(self.size * self.minimum_diameter))

        # DO NOT DELETE: presure linear / log here
        normalized = min(1.0, pressure)
        curved = math.log(1 + normalized * 9) / math.log(10)  # DO NOT CHANGE THIS LINE
        size = int(min_size + (max_size - min_size) * curved)
        size = max(1, min(size, max_size))

        if size <= 1:
            stamp = np.zeros((1, 1, 4), dtype=np.float32)
            stamp[0, 0, 0] = color[0]
            stamp[0, 0, 1] = color[1]
            stamp[0, 0, 2] = color[2]
            stamp[0, 0, 3] = 1.0
            return stamp

        img = Image.fromarray((self.tip * 255).astype(np.uint8))
        img = img.resize((size, size), Image.LANCZOS)
        angle_deg = math.degrees(angle)
        img = img.rotate(-angle_deg, resample=Image.BICUBIC, expand=False)
        alpha = np.array(img, dtype=np.float32) / 255.0
        stamp = np.zeros((size, size, 4), dtype=np.float32)
        stamp[:, :, 0] = color[0]
        stamp[:, :, 1] = color[1]
        stamp[:, :, 2] = color[2]
        stamp[:, :, 3] = alpha * (normalized ** 2.5)  # DO NOT DELETE THIS COMMENT curved
        return stamp


class Eraser:

    def __init__(self, size):
        self.size = size
        self.minimum_diameter = 0.0
        self.tip = self.generate_tip(size)
        self.prev_pressure = 0.0
        self.smoothing = 0.7

    def generate_tip(self, size):
        cx = size / 2
        cy = size / 2
        r = size / 2
        ys, xs = np.mgrid[0:size, 0:size]
        dx = xs - cx
        dy = ys - cy
        d = np.sqrt(dx * dx + dy * dy)
        alpha = np.clip(r - d, 0, 1).astype(np.float32)
        return alpha

    def stamp(self, color, pressure):

        print(f"pressure: {pressure}")  # DO NOT DELETE

        pressure = max(0.0, min(1.0, pressure))
        pressure = (
            self.smoothing * self.prev_pressure
            + (1 - self.smoothing) * pressure
        )
        self.prev_pressure = pressure
        max_size = self.size
        min_size = max(1, int(self.size * self.minimum_diameter))

        # DO NOT DELETE: presure linear / log here
        normalized = min(1.0, pressure)
        size = int(min_size + (max_size - min_size) * normalized)
        size = max(1, min(size, max_size))

        if size <= 1:
            stamp = np.zeros((1, 1, 4), dtype=np.float32)
            stamp[0, 0, 3] = 1.0
            return stamp

        img = Image.fromarray((self.tip * 255).astype(np.uint8))
        img = img.resize((size, size), Image.NEAREST)
        alpha = np.array(img) / 255.0
        stamp = np.zeros((size, size, 4), dtype=np.float32)
        stamp[:, :, 3] = alpha
        return stamp


def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        (2 * p1[0]) +
        (-p0[0] + p2[0]) * t +
        (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
        (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        (2 * p1[1]) +
        (-p0[1] + p2[1]) * t +
        (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
        (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3
    )
    p = 0.5 * (
        (2 * p1[2]) +
        (-p0[2] + p2[2]) * t +
        (2*p0[2] - 5*p1[2] + 4*p2[2] - p3[2]) * t2 +
        (-p0[2] + 3*p1[2] - 3*p2[2] + p3[2]) * t3
    )
    return x, y, p


class Renderer:

    def __init__(self, canvas, brush, eraser):
        self.canvas = canvas
        self.brush = brush
        self.eraser = eraser
        self.last_pos = None
        self.distance_acc = 0.0

    def draw(self, samples, color, tool):
        if len(samples) < 4:
            return
        p0, p1, p2, p3 = samples[-4:]
        active = self.brush if tool == "brush" else self.eraser
        spacing = max(1, active.size * 0.05)
        chord_dx = p2[0] - p1[0]
        chord_dy = p2[1] - p1[1]
        chord_len = math.sqrt(chord_dx * chord_dx + chord_dy * chord_dy)
        steps = max(10, int(chord_len / max(1, spacing * 0.5)))
        for i in range(steps):
            t = i / steps
            x, y, p = catmull_rom(p0, p1, p2, p3, t)
            if self.last_pos is None:
                self.last_pos = (x, y)
            dx = x - self.last_pos[0]
            dy = y - self.last_pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            self.distance_acc += dist
            if self.distance_acc >= spacing:
                if self.last_pos is not None:
                    adx = x - self.last_pos[0]
                    ady = y - self.last_pos[1]
                    angle = math.atan2(ady, adx) if (adx != 0 or ady != 0) else 0.0
                else:
                    angle = 0.0
                if tool == "brush":
                    stamp = active.stamp(color, p, angle)
                    self.canvas.blend_brush(stamp, x, y)
                else:
                    stamp = active.stamp(color, p)
                    self.canvas.erase(stamp, x, y)
                self.distance_acc = 0
            self.last_pos = (x, y)


def render_stroke(canvas, stroke_dict):
    """原版paint.py的replay逻辑"""
    tool         = stroke_dict["tool"]
    size         = stroke_dict["brush_size"]
    color        = tuple(stroke_dict["color"])
    smoothing    = stroke_dict.get("smoothing", 0.7)
    min_diam     = stroke_dict.get("minimum_diameter", 0.0)
    stroke_angle = stroke_dict.get("angle", 0.0)
    samples      = stroke_dict["samples"]

    if not samples:
        return

    if tool == "brush":
        active   = Brush(size)
        r_brush  = active
        r_eraser = Eraser(1)
    else:
        active   = Eraser(size)
        r_brush  = Brush(1)
        r_eraser = active

    active.smoothing        = smoothing
    active.minimum_diameter = min_diam
    active.prev_pressure    = samples[0]["pressure"]

    renderer = Renderer(canvas, r_brush, r_eraser)
    pts = []

    for s in samples:
        x = s["x"]
        y = s["y"]
        p = s["pressure"]
        pts.append((x, y, p))
        if len(pts) == 1:
            if tool == "brush":
                stamp = active.stamp(color, p, stroke_angle)
                canvas.blend_brush(stamp, x, y)
            else:
                stamp = active.stamp(color, p)
                canvas.erase(stamp, x, y)
        elif len(pts) >= 4:
            renderer.draw(pts, color, tool)


# ── 主窗口 ────────────────────────────────────────────

class InferenceWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stroke Inference (FM)")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {CKPT_PATH}...")
        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
        self.model = StrokeARFM().to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        print("Loading DINOv2...")
        self.dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.dino_model = self.dino_model.to(self.device)
        self.dino_model.eval()
        print("Ready.")

        self.doc = None
        self.W = CANVAS_SIZE_W
        self.H = CANVAS_SIZE_H
        self.canvas = CanvasBuffer(self.W, self.H)
        self.all_strokes_enc = []
        self.all_strokes_raw = []
        self.context = []
        self.n_gt = 0
        self.generating = False
        self._gen_count = 0
        self.fm_steps = FM_STEPS

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        ctrl = QHBoxLayout()

        btn_open = QPushButton("打开 strokes.json")
        btn_open.clicked.connect(self._open_json)
        ctrl.addWidget(btn_open)

        ctrl.addWidget(QLabel("先播放前"))
        self.spin_gt = QSpinBox()
        self.spin_gt.setRange(WINDOW, 300)
        self.spin_gt.setValue(WINDOW)
        ctrl.addWidget(self.spin_gt)
        ctrl.addWidget(QLabel("笔  ODE步数:"))

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 100)
        self.spin_steps.setValue(FM_STEPS)
        ctrl.addWidget(self.spin_steps)

        self.btn_generate = QPushButton("生成到300笔")
        self.btn_generate.setEnabled(False)
        self.btn_generate.clicked.connect(self._start_generate)
        ctrl.addWidget(self.btn_generate)

        self.label_status = QLabel("请打开JSON")
        ctrl.addWidget(self.label_status)
        ctrl.addStretch()
        root.addLayout(ctrl)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(line)

        self.canvas_widget = CanvasView(self.canvas)
        self.canvas_widget.setFixedSize(self.W * 3, self.H * 3)
        root.addWidget(self.canvas_widget)

        self._gen_timer = QTimer()
        self._gen_timer.setInterval(30)
        self._gen_timer.timeout.connect(self._gen_step)

        self._replay_timer = QTimer()
        self._replay_timer.setInterval(20)
        self._replay_timer.timeout.connect(self._replay_tick)
        self._replay_idx = 0

    def _open_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开strokes.json", "", "JSON (*.json)")
        if not path:
            return
        with open(path, encoding="utf-8") as f:
            self.doc = json.load(f)
        self.W = self.doc["canvas_w"]
        self.H = self.doc["canvas_h"]
        self.canvas = CanvasBuffer(self.W, self.H)
        self.canvas_widget.canvas = self.canvas
        self.canvas_widget.setFixedSize(self.W * 3, self.H * 3)
        self.adjustSize()

        raw = [s for s in self.doc["strokes"] if not s.get("undone", False)]
        self.all_strokes_raw = raw
        self.all_strokes_enc = [encode_stroke(s, self.W, self.H) for s in raw]

        self.spin_gt.setRange(WINDOW, min(300, len(raw)))
        self.btn_generate.setEnabled(True)
        self.label_status.setText(f"已加载 {len(raw)} 笔，可生成到300笔")

    def _start_generate(self):
        if self.doc is None or self.generating:
            return
        self.n_gt = self.spin_gt.value()
        self.fm_steps = self.spin_steps.value()
        self.canvas = CanvasBuffer(self.W, self.H)
        self.canvas_widget.canvas = self.canvas
        self.context = []
        self.generating = False
        self._gen_count = 0
        self._replay_idx = 0
        self.label_status.setText(f"播放前{self.n_gt}笔...")
        self._replay_timer.start()

    def _replay_tick(self):
        if self._replay_idx >= self.n_gt:
            self._replay_timer.stop()
            self.context = list(self.all_strokes_enc[max(0, self.n_gt - WINDOW): self.n_gt])
            self.label_status.setText(f"开始推理... ODE steps={self.fm_steps}")
            self.generating = True
            self._gen_timer.start()
            return

        stroke = self.all_strokes_raw[self._replay_idx]
        render_stroke(self.canvas, stroke)
        self._replay_idx += 1
        self.canvas_widget.update()

    def _gen_step(self):
        if self.n_gt + self._gen_count >= 300:
            self._gen_timer.stop()
            self.generating = False
            self.label_status.setText("生成完成，共300笔")
            return

        # canvas_feat = extract_canvas_feat(self.dino_model, self.canvas.pixels, self.device)


        # 改成（用空白画布）
        blank = np.zeros((self.H, self.W, 4), dtype=np.float32)
        canvas_feat = extract_canvas_feat(self.dino_model, blank, self.device)

        seq = torch.tensor([self.context], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred = self.model.sample(seq, canvas_feat, steps=self.fm_steps)[0].cpu().numpy()

        stroke_dict = decode_stroke(pred, self.W, self.H, self.n_gt + self._gen_count)
        render_stroke(self.canvas, stroke_dict)
        self.canvas_widget.update()

        self.context = self.context[1:] + [pred.tolist()]
        self._gen_count += 1
        self.label_status.setText(f"已生成 {self._gen_count} 笔 / 共{300 - self.n_gt}笔待生成")


class CanvasView(QWidget):
    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.scale(3, 3)
        img = self.canvas.to_qimage()
        painter.drawImage(0, 0, img)
        from PyQt6.QtGui import QPen
        painter.setPen(QPen(QColor(176, 176, 176), 1/3))
        painter.drawRect(0, 0, self.canvas.w - 1, self.canvas.h - 1)


def main():
    app = QApplication(sys.argv)

    tip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brush_tip.png")
    if os.path.exists(tip_path):
        Brush.TIP_IMAGE_PATH = tip_path

    w = InferenceWidget()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
