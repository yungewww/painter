import sys
import math
import json
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QFileDialog, QHBoxLayout,
                              QVBoxLayout, QPushButton, QLabel, QFrame, QSpinBox)
from PyQt6.QtGui import QPainter, QImage, QColor
from PyQt6.QtCore import Qt, QTimer
from PIL import Image

CANVAS_SIZE_W = 178
CANVAS_SIZE_H = 218
BRUSH_TIP     = True
SOFT_ROUND    = False


# ── Canvas ────────────────────────────────────────────

class CanvasBuffer:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.pixels = np.zeros((h, w, 4), dtype=np.float32)
        self._dirty = True
        self._cache = None

    def blend_brush(self, stamp, x, y):
        h, w = stamp.shape[:2]
        x0 = int(x - w // 2); y0 = int(y - h // 2)
        cx0 = max(0, x0); cy0 = max(0, y0)
        cx1 = min(self.w, x0 + w); cy1 = min(self.h, y0 + h)
        if cx0 >= cx1 or cy0 >= cy1:
            return
        sx0 = cx0 - x0; sy0 = cy0 - y0
        sx1 = sx0 + (cx1 - cx0); sy1 = sy0 + (cy1 - cy0)
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
        rgba_out = np.concatenate([rgb_srgb, rgba[:, :, 3:4]], axis=2)
        rgba_out = (np.clip(rgba_out, 0, 1) * 255).astype(np.uint8)
        self._cache = QImage(rgba_out.data, self.w, self.h, self.w * 4,
                             QImage.Format.Format_RGBA8888).copy()
        self._dirty = False
        return self._cache


# ── Brush / Eraser ────────────────────────────────────

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
            return np.array(img, dtype=np.float32) / 255.0
        cx = cy = size / 2; r = size / 2
        ys, xs = np.mgrid[0:size, 0:size]
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        if SOFT_ROUND:
            return np.exp(-d2 / (2.0 * (r / 2.0) ** 2)).astype(np.float32)
        return (np.sqrt(d2) <= r).astype(np.float32)

    def stamp(self, color, pressure, angle=0.0):
        print(f"pressure: {pressure}")  # DO NOT DELETE
        pressure = max(0.0, min(1.0, pressure))
        pressure = self.smoothing * self.prev_pressure + (1 - self.smoothing) * pressure
        self.prev_pressure = pressure
        max_size = self.size
        min_size = max(1, int(self.size * self.minimum_diameter))
        normalized = min(1.0, pressure)
        curved = math.log(1 + normalized * 9) / math.log(10)  # DO NOT CHANGE THIS LINE
        size = max(1, min(int(min_size + (max_size - min_size) * curved), max_size))
        if size <= 1:
            stamp = np.zeros((1, 1, 4), dtype=np.float32)
            stamp[0, 0, :3] = color; stamp[0, 0, 3] = 1.0
            return stamp
        img = Image.fromarray((self.tip * 255).astype(np.uint8))
        img = img.resize((size, size), Image.LANCZOS)
        img = img.rotate(-math.degrees(angle), resample=Image.BICUBIC, expand=False)
        alpha = np.array(img, dtype=np.float32) / 255.0
        stamp = np.zeros((size, size, 4), dtype=np.float32)
        stamp[:, :, :3] = color
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
        cx = cy = size / 2; r = size / 2
        ys, xs = np.mgrid[0:size, 0:size]
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        return np.clip(r - d, 0, 1).astype(np.float32)

    def stamp(self, color, pressure):
        print(f"pressure: {pressure}")  # DO NOT DELETE
        pressure = max(0.0, min(1.0, pressure))
        pressure = self.smoothing * self.prev_pressure + (1 - self.smoothing) * pressure
        self.prev_pressure = pressure
        max_size = self.size
        min_size = max(1, int(self.size * self.minimum_diameter))
        normalized = min(1.0, pressure)
        size = max(1, min(int(min_size + (max_size - min_size) * normalized), max_size))
        if size <= 1:
            stamp = np.zeros((1, 1, 4), dtype=np.float32); stamp[0, 0, 3] = 1.0
            return stamp
        img = Image.fromarray((self.tip * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
        alpha = np.array(img) / 255.0
        stamp = np.zeros((size, size, 4), dtype=np.float32); stamp[:, :, 3] = alpha
        return stamp


# ── Renderer ──────────────────────────────────────────

def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t; t3 = t2 * t
    def interp(a, b, c, d):
        return 0.5 * ((2*b) + (-a+c)*t + (2*a-5*b+4*c-d)*t2 + (-a+3*b-3*c+d)*t3)
    return interp(*[p[0] for p in [p0,p1,p2,p3]]), interp(*[p[1] for p in [p0,p1,p2,p3]]), interp(*[p[2] for p in [p0,p1,p2,p3]])


class Renderer:
    def __init__(self, canvas, brush, eraser):
        self.canvas = canvas; self.brush = brush; self.eraser = eraser
        self.last_pos = None; self.distance_acc = 0.0

    def draw(self, samples, color, tool):
        if len(samples) < 4:
            return
        p0, p1, p2, p3 = samples[-4:]
        active = self.brush if tool == "brush" else self.eraser
        spacing = max(1, active.size * 0.05)
        chord_dx = p2[0] - p1[0]; chord_dy = p2[1] - p1[1]
        chord_len = math.sqrt(chord_dx**2 + chord_dy**2)
        steps = max(10, int(chord_len / max(1, spacing * 0.5)))
        for i in range(steps):
            t = i / steps
            x, y, p = catmull_rom(p0, p1, p2, p3, t)
            if self.last_pos is None:
                self.last_pos = (x, y)
            dx = x - self.last_pos[0]; dy = y - self.last_pos[1]
            self.distance_acc += math.sqrt(dx**2 + dy**2)
            if self.distance_acc >= spacing:
                adx = x - self.last_pos[0]; ady = y - self.last_pos[1]
                angle = math.atan2(ady, adx) if (adx != 0 or ady != 0) else 0.0
                if tool == "brush":
                    self.canvas.blend_brush(active.stamp(color, p, angle), x, y)
                else:
                    self.canvas.erase(active.stamp(color, p), x, y)
                self.distance_acc = 0
            self.last_pos = (x, y)


def render_stroke(canvas, stroke_dict):
    tool = stroke_dict["tool"]
    size = stroke_dict["brush_size"]
    color = tuple(stroke_dict["color"])
    smoothing = stroke_dict.get("smoothing", 0.7)
    min_diam = stroke_dict.get("minimum_diameter", 0.0)
    stroke_angle = stroke_dict.get("angle", 0.0)
    samples = stroke_dict["samples"]
    if not samples:
        return
    if tool == "brush":
        active = Brush(size); r_brush = active; r_eraser = Eraser(1)
    else:
        active = Eraser(size); r_brush = Brush(1); r_eraser = active
    active.smoothing = smoothing
    active.minimum_diameter = min_diam
    active.prev_pressure = samples[0]["pressure"]
    renderer = Renderer(canvas, r_brush, r_eraser)
    pts = []
    for s in samples:
        x = s["x"]; y = s["y"]; p = s["pressure"]
        pts.append((x, y, p))
        if len(pts) == 1:
            if tool == "brush":
                canvas.blend_brush(active.stamp(color, p, stroke_angle), x, y)
            else:
                canvas.erase(active.stamp(color, p), x, y)
        elif len(pts) >= 4:
            renderer.draw(pts, color, tool)


# ── Canvas view ───────────────────────────────────────

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


# ── Main window ───────────────────────────────────────

class DemoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stroke Demo")

        self.doc = None
        self.W = CANVAS_SIZE_W
        self.H = CANVAS_SIZE_H
        self.canvas = CanvasBuffer(self.W, self.H)
        self.all_strokes = []
        self._replay_idx = 0
        self._playing = False

        self._build_ui()

        self._timer = QTimer()
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._tick)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        ctrl = QHBoxLayout()

        btn_open = QPushButton("打开 strokes.json")
        btn_open.clicked.connect(self._open_json)
        ctrl.addWidget(btn_open)

        self.btn_play = QPushButton("播放")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_play)
        ctrl.addWidget(self.btn_play)

        self.btn_reset = QPushButton("重置")
        self.btn_reset.setEnabled(False)
        self.btn_reset.clicked.connect(self._reset)
        ctrl.addWidget(self.btn_reset)

        ctrl.addWidget(QLabel("间隔(ms):"))
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(1, 500)
        self.spin_interval.setValue(20)
        self.spin_interval.valueChanged.connect(lambda v: self._timer.setInterval(v))
        ctrl.addWidget(self.spin_interval)

        self.label_status = QLabel("请打开 strokes.json")
        ctrl.addWidget(self.label_status)
        ctrl.addStretch()
        root.addLayout(ctrl)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(line)

        self.canvas_widget = CanvasView(self.canvas)
        self.canvas_widget.setFixedSize(self.W * 3, self.H * 3)
        root.addWidget(self.canvas_widget)

    def _open_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开 strokes.json", "", "JSON (*.json)")
        if not path:
            return
        with open(path, encoding="utf-8") as f:
            self.doc = json.load(f)
        self.W = self.doc["canvas_w"]
        self.H = self.doc["canvas_h"]
        self.all_strokes = [s for s in self.doc["strokes"] if not s.get("undone", False)]
        self._reset()
        self.btn_play.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.label_status.setText(f"已加载 {len(self.all_strokes)} 笔")

    def _reset(self):
        self._timer.stop()
        self._playing = False
        self._replay_idx = 0
        self.canvas = CanvasBuffer(self.W, self.H)
        self.canvas_widget.canvas = self.canvas
        self.canvas_widget.setFixedSize(self.W * 3, self.H * 3)
        self.adjustSize()
        self.canvas_widget.update()
        self.btn_play.setText("播放")
        if self.all_strokes:
            self.label_status.setText(f"已加载 {len(self.all_strokes)} 笔")

    def _toggle_play(self):
        if not self.all_strokes:
            return
        if self._playing:
            self._timer.stop()
            self._playing = False
            self.btn_play.setText("播放")
        else:
            self._playing = True
            self.btn_play.setText("暂停")
            self._timer.start()

    def _tick(self):
        if self._replay_idx >= len(self.all_strokes):
            self._timer.stop()
            self._playing = False
            self.btn_play.setText("播放")
            self.label_status.setText(f"完成，共 {len(self.all_strokes)} 笔")
            return
        stroke = self.all_strokes[self._replay_idx]
        render_stroke(self.canvas, stroke)
        self._replay_idx += 1
        self.canvas_widget.update()
        self.label_status.setText(f"{self._replay_idx} / {len(self.all_strokes)}")


def main():
    app = QApplication(sys.argv)

    tip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brush_tip.png")
    if os.path.exists(tip_path):
        Brush.TIP_IMAGE_PATH = tip_path

    w = DemoWidget()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()