import sys
import json
import math
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
import multiprocessing as mp_mod
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel

# ── Config ────────────────────────────────────────────
IMAGE_DIR   = "images"
OUTPUT_DIR  = "output"
BRUSH_TIP   = "brush_tip.png"
NUM_WORKERS = 4   # 建议等于GPU数量

DINOV2_DIR  = "models/dinov2"
CLIP_MODEL  = "openai/clip-vit-base-patch32"
EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIZE    = 218

# 68个关键点的MediaPipe索引
LANDMARK_68 = [
    10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,
    107,66,105,63,70,
    336,296,334,293,300,
    168,6,197,195,5,4,1,19,94,
    33,160,158,133,153,144,
    362,385,387,263,373,380,
    61,185,40,39,37,0,267,269,270,409,291,375,
    78,95,88,178,87,14,317,402,
]
assert len(LANDMARK_68) == 68

DINOV2_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std= [0.26862954, 0.26130258, 0.27577711]),
])

# ── Color utils ───────────────────────────────────────
def srgb_to_linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(linear):
    return np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * np.power(np.clip(linear, 0, 1), 1.0 / 2.4) - 0.055)

def load_brush_tip(path):
    p = Path(path)
    if not p.exists():
        return None
    img = Image.open(p).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)

# ── Render ────────────────────────────────────────────
def render_single_stroke(canvas, i, xy, pressure_l, color_l, angle_l, brush_sizes, H, W, brush_tip):
    x        = xy[:, 0].clamp(0, W - 1)
    y        = xy[:, 1].clamp(0, H - 1)
    pressure = torch.sigmoid(pressure_l)
    color    = torch.sigmoid(color_l)
    angle    = angle_l
    brush_size = brush_sizes[i]
    p_val      = pressure[i].clamp(0, 1)
    curved     = torch.log(1.0 + p_val * 9.0) / math.log(10.0)
    tip_size   = max(1, min(int(1 + (brush_size - 1) * curved.item()), brush_size))
    xi = int(x[i].item()); yi = int(y[i].item())

    if tip_size <= 1:
        if 0 <= xi < W and 0 <= yi < H:
            ci    = color[i]
            src_a = (p_val ** 2.5).clamp(0, 1)
            dst_a = canvas[yi, xi, 3]
            out_a = src_a + dst_a * (1 - src_a)
            new_rgb   = (ci * src_a + canvas[yi, xi, :3] * dst_a * (1 - src_a)) / out_a if out_a > 1e-6 else ci
            new_pixel = torch.cat([new_rgb, out_a.unsqueeze(0)])
            pixel_row = torch.cat([canvas[yi, :xi], new_pixel.unsqueeze(0), canvas[yi, xi+1:]], dim=0)
            canvas    = torch.cat([canvas[:yi], pixel_row.unsqueeze(0), canvas[yi+1:]], dim=0)
        return canvas

    half = tip_size // 2
    if brush_tip is not None:
        tip   = F.interpolate(brush_tip, size=(tip_size, tip_size), mode="bilinear", align_corners=False)
        cos_a = torch.cos(angle[i:i+1]); sin_a = torch.sin(angle[i:i+1])
        zeros = torch.zeros_like(cos_a)
        theta = torch.stack([torch.stack([cos_a, sin_a, zeros], dim=1),
                              torch.stack([-sin_a, cos_a, zeros], dim=1)], dim=1)
        grid      = F.affine_grid(theta, tip.shape, align_corners=False)
        rotated   = F.grid_sample(tip, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        tip_alpha = rotated[0, 0]
    else:
        ys_t = torch.arange(tip_size, dtype=torch.float32) - tip_size / 2.0
        xs_t = torch.arange(tip_size, dtype=torch.float32) - tip_size / 2.0
        dy, dx = torch.meshgrid(ys_t, xs_t, indexing="ij")
        tip_alpha = (torch.sqrt(dx*dx + dy*dy) <= tip_size/2.0).float()

    ai_map = (tip_alpha * (p_val ** 2.5)).clamp(0, 1)
    x0 = xi-half; x1 = x0+tip_size; y0 = yi-half; y1 = y0+tip_size
    cx0=max(0,x0); cy0=max(0,y0); cx1=min(W,x1); cy1=min(H,y1)
    if cx0>=cx1 or cy0>=cy1: return canvas
    sx0=cx0-x0; sy0=cy0-y0; sx1=sx0+(cx1-cx0); sy1=sy0+(cy1-cy0)
    src_a  = ai_map[sy0:sy1, sx0:sx1].unsqueeze(-1)
    ci     = color[i][None, None, :]
    region = canvas[cy0:cy1, cx0:cx1]
    new_rgb = ci * src_a + region[:,:,:3] * (1-src_a)
    new_a   = src_a + region[:,:,3:4] * (1-src_a)
    canvas  = torch.cat([canvas[:cy0],
                          torch.cat([canvas[cy0:cy1,:cx0], torch.cat([new_rgb,new_a],dim=-1), canvas[cy0:cy1,cx1:]], dim=1),
                          canvas[cy1:]], dim=0)
    return canvas

def render_up_to(xy, pressure_l, color_l, angle_l, brush_sizes, H, W, n, brush_tip=None):
    canvas = torch.zeros(H, W, 4)
    for i in range(n):
        canvas = render_single_stroke(canvas, i, xy, pressure_l, color_l, angle_l, brush_sizes, H, W, brush_tip)
    rgb   = canvas[:,:,:3]; alpha = canvas[:,:,3:4]
    linear = (rgb*alpha + torch.ones(H,W,3)*(1-alpha)).clamp(0,1).numpy()
    return (np.clip(linear_to_srgb(linear),0,1)*255).astype(np.uint8)

# ── Worker process ────────────────────────────────────
def worker_main(task_chunk, brush_tip_path, worker_id):
    """每个worker进程独立加载模型到自己的GPU"""
    torch.set_num_threads(2)
    n_gpu  = torch.cuda.device_count()
    device = torch.device(f"cuda:{worker_id % n_gpu}") if n_gpu > 0 else torch.device("cpu")
    print(f"Worker {worker_id} using {device}")

    # DINOv2
    dinov2_model = torch.hub.load(DINOV2_DIR, "dinov2_vits14", source="local").to(device)
    dinov2_model.eval()

    # CLIP
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    clip_model.eval()

    # MediaPipe
    _mp_fm    = mp.solutions.face_mesh
    face_mesh = _mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Brush tip
    brush_tip = load_brush_tip(brush_tip_path)

    def extract_clip_patches(pil_img):
        inp = CLIP_TRANSFORM(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = clip_model.vision_model(pixel_values=inp)
            patch_feats = out.last_hidden_state[:, 1:, :]  # (1,49,512)
        return patch_feats.squeeze(0).cpu().numpy()

    def extract_landmarks(image_rgb):
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        coords = []
        for idx in LANDMARK_68:
            lm = landmarks[idx]
            coords.extend([lm.x, lm.y])
        return np.array(coords, dtype=np.float32)

    for args in task_chunk:
        img_path_str, output_dir_str, idx, total = args
        img_path = Path(img_path_str)
        out_dir  = Path(output_dir_str) / img_path.stem

        if not out_dir.exists() or not (out_dir / "strokes.json").exists():
            print(f"[{idx}/{total}] {img_path.name} SKIP (no strokes.json)")
            continue

        need_clip  = not (out_dir / "clip_patch_feats.npy").exists()
        need_dino  = not (out_dir / "target_dino_cls.npy").exists()
        need_lmk   = not (out_dir / "face_landmarks.npy").exists()

        if not need_clip and not need_dino and not need_lmk:
            print(f"[{idx}/{total}] {img_path.name} SKIP (all done)")
            continue

        t0 = time.time()
        try:
            image     = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            H_orig, W_orig = image_rgb.shape[:2]
            scale = MAX_SIZE / max(H_orig, W_orig)
            W_s   = int(W_orig * scale)
            H_s   = int(H_orig * scale)

            # target DINOv2 CLS
            if need_dino:
                pil_orig = Image.fromarray(image_rgb).resize((W_s, H_s), Image.LANCZOS)
                inp = DINOV2_TRANSFORM(pil_orig).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = dinov2_model(inp)
                np.save(str(out_dir / "target_dino_cls.npy"), feat.squeeze(0).cpu().numpy())

            # face landmarks
            if need_lmk:
                lmk = extract_landmarks(image_rgb)
                if lmk is None:
                    lmk = np.zeros(136, dtype=np.float32)
                np.save(str(out_dir / "face_landmarks.npy"), lmk)

            # CLIP patch feats
            if need_clip:
                with open(out_dir / "strokes.json") as f:
                    doc = json.load(f)
                W_c = doc["canvas_w"]; H_c = doc["canvas_h"]
                strokes = [s for s in doc["strokes"] if not s.get("undone", False)]
                N = len(strokes)

                xs        = np.array([s["samples"][0]["x"] * W_s / W_c for s in strokes], np.float32)
                ys        = np.array([s["samples"][0]["y"] * H_s / H_c for s in strokes], np.float32)
                pressures = np.array([s["samples"][0]["pressure"] for s in strokes], np.float32)
                colors    = np.array([s["color"] for s in strokes], np.float32)
                angles    = np.array([s["angle"] for s in strokes], np.float32)
                brush_sizes = [s["brush_size"] for s in strokes]

                pl = np.log(np.clip(pressures,0.02,0.98)/(1-np.clip(pressures,0.02,0.98)))
                cl = np.log(np.clip(colors,0.02,0.98)/(1-np.clip(colors,0.02,0.98)))

                xy_t       = torch.tensor(np.stack([xs,ys],axis=1))
                pressure_l = torch.tensor(pl)
                color_l    = torch.tensor(cl)
                angle_l    = torch.tensor(angles)

                clip_patch_feats = np.zeros((N,49,512), dtype=np.float32)
                for si in range(N):
                    canvas_rgb = render_up_to(xy_t, pressure_l, color_l, angle_l,
                                              brush_sizes, H_s, W_s, si, brush_tip)
                    clip_patch_feats[si] = extract_clip_patches(Image.fromarray(canvas_rgb))

                np.save(str(out_dir / "clip_patch_feats.npy"), clip_patch_feats)

            elapsed = time.time() - t0
            print(f"[{idx}/{total}] {img_path.name} done {elapsed:.0f}s")

        except Exception as e:
            print(f"[{idx}/{total}] {img_path.name} ERROR: {e}")
            import traceback; traceback.print_exc()


# ── Main ──────────────────────────────────────────────
def main():
    base_dir   = Path(__file__).parent
    images_dir = base_dir / IMAGE_DIR
    output_dir = base_dir / OUTPUT_DIR

    img_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in EXTS])
    total     = len(img_paths)
    print(f"Found {total} images.")

    tasks = [(str(p), str(output_dir), idx+1, total) for idx, p in enumerate(img_paths)]

    # 按worker数量切分任务
    chunk_size = math.ceil(len(tasks) / NUM_WORKERS)
    chunks     = [tasks[i:i+chunk_size] for i in range(0, len(tasks), chunk_size)]
    tip_path   = str(base_dir / BRUSH_TIP)

    procs = []
    for wid, chunk in enumerate(chunks):
        p = mp_mod.Process(target=worker_main, args=(chunk, tip_path, wid))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    print("\nAll done.")

if __name__ == "__main__":
    main()
