import sys
import json
import math
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
from pathlib import Path
from PIL import Image
from torchvision import transforms

DINOV2_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
from transformers import AutoModel, AutoImageProcessor

# ── sys.path for bisenet ──────────────────────────────
sys.path.insert(0, "models")
import resnet
resnet.Resnet18.init_weight = lambda self: None
from model import BiSeNet

# ── Config ────────────────────────────────────────────
IMAGE_DIR   = "images"
OUTPUT_DIR  = "output"
BRUSH_TIP   = "brush_tip.png"
NUM_WORKERS = 48

BISENET_MODEL = "models/79999_iter.pth"
DSINE_DIR     = "models/DSINE"
DSINE_CKPT    = "models/DSINE/checkpoints/dsine.pt"
DINO_DIR      = "models/dino-vits16"
DINOV2_DIR    = "models/dinov2"
EXTS          = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

N_ITER              = 30
LR                  = 0.05
MAX_SIZE            = 218
EARLY_STOP_PATIENCE = 3

# ── Brush size schedule ───────────────────────────────
BRUSH_SIZE_MODE     = "linear"
BRUSH_SIZE_K        = 3
BRUSH_SIZE_ARCTAN_C = 0.3
BRUSH_SIZE_ARCTAN_S = 10

# ── Label definition ──────────────────────────────────
LABEL_DEF = [
    ( 0, "background",       10),
    ( 1, "neck",             30),
    ( 2, "hair",             30),
    ( 3, "cloth",            30),
    ( 4, "hat",              30),
    ( 5, "face",             50),
    ( 6, "left_ear",         10),
    ( 7, "right_ear",        10),
    ( 8, "left_eye",         30),
    ( 9, "right_eye",        30),
    (10, "nose",             30),
    (11, "mouth",            30),
    (12, "left_eye_detail",  30),
    (13, "right_eye_detail", 30),
    (14, "mouth_detail",     30),
]
LABEL_MAX_ID = 14

# ── Segment config (from segment.py, unchanged) ───────
BISENET_MAP = {
    "neck": "neck", "cloth": "cloth", "hair": "hair",
    "hat": "hat", "left_ear": "left_ear", "right_ear": "right_ear",
}
BISENET_NAMES = [
    "background", "skin", "left_brow", "right_brow",
    "left_eye", "right_eye", "eye_glass",
    "left_ear", "right_ear", "ear_ring",
    "nose", "mouth", "upper_lip", "lower_lip",
    "neck", "neck_lace", "cloth", "hair", "hat"
]
_LABEL_DEF_FULL = [
    ( 0, "background",        1, "bisenet"),
    ( 1, "face",              2, "mesh"),
    ( 2, "neck",              2, "bisenet"),
    ( 3, "hair",              2, "bisenet"),
    ( 4, "cloth",             2, "bisenet"),
    ( 5, "hat",               2, "bisenet"),
    ( 6, "left_ear",          3, "bisenet"),
    ( 7, "right_ear",         3, "bisenet"),
    ( 8, "left_eye",          3, "mesh"),
    ( 9, "right_eye",         3, "mesh"),
    (10, "nose",              3, "mesh"),
    (11, "mouth",             3, "mesh"),
    (12, "left_eye_detail",   4, "mesh"),
    (13, "right_eye_detail",  4, "mesh"),
    (14, "mouth_detail",      4, "mesh"),
]
LABEL_ID = {d[1]: d[0] for d in _LABEL_DEF_FULL}

MESH_REGIONS = {
    "face":             [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],
    "left_eye":         [107,66,105,63,70,156,143,111,117,118,101,100,47,114,188,122,193,55],
    "right_eye":        [336,296,334,293,300,383,372,340,346,347,330,329,277,343,412,351,417,285],
    "nose":             [9,107,55,221,189,244,128,114,47,100,36,206,165,167,164,393,391,426,266,329,277,343,357,464,413,441,285,336],
    "mouth":            [2,97,98,203,206,216,212,202,204,194,201,200,421,418,424,422,432,436,426,423,327,326],
    "left_eye_detail":  [243,112,26,22,23,24,110,25,130,247,30,29,27,28,56,190],
    "right_eye_detail": [463,414,286,258,257,259,260,467,359,255,339,254,253,252,256,341],
    "mouth_detail":     [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146],
}

# ── Color utils ───────────────────────────────────────
def srgb_to_linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(linear):
    return np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * np.power(np.clip(linear, 0, 1), 1.0 / 2.4) - 0.055)

# ── Brush tip ─────────────────────────────────────────
def load_brush_tip(path):
    p = Path(path)
    if not p.exists():
        return None
    img = Image.open(p).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)

# ── Helpers (unchanged from run2.py) ──────────────────
def get_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def brush_size_schedule(i, n, brush_min, brush_max):
    t = i / max(n - 1, 1)
    if BRUSH_SIZE_MODE == "curve_down":
        t_mapped = np.exp(-BRUSH_SIZE_K * (1 - t))
    elif BRUSH_SIZE_MODE == "arctan":
        c = BRUSH_SIZE_ARCTAN_C
        s = BRUSH_SIZE_ARCTAN_S
        t_mapped = np.arctan((t - c) * s) / np.arctan(s) * 0.5 + 0.5
    else:
        t_mapped = t
    return max(brush_min, int(brush_max - t_mapped * (brush_max - brush_min)))

def local_normal_variance_batch(normal_map, xs, ys, H, W, radius=8):
    variances = np.zeros(len(xs), dtype=np.float32)
    for i in range(len(xs)):
        xi = int(np.clip(xs[i], 0, W - 1))
        yi = int(np.clip(ys[i], 0, H - 1))
        x0, x1 = max(0, xi - radius), min(W, xi + radius)
        y0, y1 = max(0, yi - radius), min(H, yi + radius)
        patch = normal_map[y0:y1, x0:x1]
        variances[i] = patch.var(axis=(0, 1)).mean()
    return variances

def control_xy_masked(attn_map, mask, N, H, W):
    flat      = attn_map.flatten().astype(np.float64)
    flat_mask = mask.flatten().astype(np.float64)
    weights   = (flat + 0.01) * flat_mask
    if weights.sum() < 1e-10:
        weights = flat_mask + 1e-10
    weights /= weights.sum()
    indices = np.random.choice(len(weights), size=N, replace=True, p=weights)
    ys = (indices // W).astype(np.float32)
    xs = (indices  % W).astype(np.float32)
    xs += np.random.uniform(-1, 1, N).astype(np.float32)
    ys += np.random.uniform(-1, 1, N).astype(np.float32)
    xs  = np.clip(xs, 0, W - 1)
    ys  = np.clip(ys, 0, H - 1)
    return xs, ys

# ── Render (unchanged from run2.py) ───────────────────
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
    xi = int(x[i].item())
    yi = int(y[i].item())

    if tip_size <= 1:
        if 0 <= xi < W and 0 <= yi < H:
            ci    = color[i]
            src_a = (p_val ** 2.5).clamp(0, 1)
            dst_a = canvas[yi, xi, 3]
            out_a = src_a + dst_a * (1 - src_a)
            new_rgb = (ci * src_a + canvas[yi, xi, :3] * dst_a * (1 - src_a)) / out_a if out_a > 1e-6 else ci
            new_pixel = torch.cat([new_rgb, out_a.unsqueeze(0)])
            pixel_row = torch.cat([canvas[yi, :xi], new_pixel.unsqueeze(0), canvas[yi, xi+1:]], dim=0)
            canvas    = torch.cat([canvas[:yi], pixel_row.unsqueeze(0), canvas[yi+1:]], dim=0)
        return canvas

    half = tip_size // 2
    if brush_tip is not None:
        tip    = F.interpolate(brush_tip, size=(tip_size, tip_size), mode="bilinear", align_corners=False)
        cos_a  = torch.cos(angle[i:i+1])
        sin_a  = torch.sin(angle[i:i+1])
        zeros  = torch.zeros_like(cos_a)
        theta  = torch.stack([torch.stack([cos_a, sin_a, zeros], dim=1),
                               torch.stack([-sin_a, cos_a, zeros], dim=1)], dim=1)
        grid      = F.affine_grid(theta, tip.shape, align_corners=False)
        rotated   = F.grid_sample(tip, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        tip_alpha = rotated[0, 0]
    else:
        ys_t      = torch.arange(tip_size, dtype=torch.float32) - tip_size / 2.0
        xs_t      = torch.arange(tip_size, dtype=torch.float32) - tip_size / 2.0
        dy, dx    = torch.meshgrid(ys_t, xs_t, indexing="ij")
        tip_alpha = (torch.sqrt(dx * dx + dy * dy) <= tip_size / 2.0).float()

    ai_map = (tip_alpha * (p_val ** 2.5)).clamp(0, 1)
    x0 = xi - half; x1 = x0 + tip_size
    y0 = yi - half; y1 = y0 + tip_size
    cx0 = max(0, x0); cy0 = max(0, y0)
    cx1 = min(W, x1); cy1 = min(H, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return canvas
    sx0 = cx0 - x0; sy0 = cy0 - y0
    sx1 = sx0 + (cx1 - cx0); sy1 = sy0 + (cy1 - cy0)
    src_a  = ai_map[sy0:sy1, sx0:sx1].unsqueeze(-1)
    ci     = color[i][None, None, :]
    region = canvas[cy0:cy1, cx0:cx1]
    new_rgb = ci * src_a + region[:, :, :3] * (1 - src_a)
    new_a   = src_a + region[:, :, 3:4] * (1 - src_a)
    canvas  = torch.cat([
        canvas[:cy0],
        torch.cat([canvas[cy0:cy1, :cx0], torch.cat([new_rgb, new_a], dim=-1), canvas[cy0:cy1, cx1:]], dim=1),
        canvas[cy1:]
    ], dim=0)
    return canvas

def render(xy, pressure_l, color_l, angle_l, brush_sizes, H, W, brush_tip=None):
    canvas = torch.zeros(H, W, 4)
    for i in range(xy.shape[0]):
        canvas = render_single_stroke(canvas, i, xy, pressure_l, color_l, angle_l, brush_sizes, H, W, brush_tip)
    rgb   = canvas[:, :, :3]
    alpha = canvas[:, :, 3:4]
    return (rgb * alpha + torch.ones(H, W, 3) * (1 - alpha)).clamp(0, 1)

def render_up_to(xy, pressure_l, color_l, angle_l, brush_sizes, H, W, n, brush_tip=None):
    """渲染前n笔，返回sRGB uint8 numpy (H,W,3)"""
    canvas = torch.zeros(H, W, 4)
    for i in range(n):
        canvas = render_single_stroke(canvas, i, xy, pressure_l, color_l, angle_l, brush_sizes, H, W, brush_tip)
    rgb   = canvas[:, :, :3]
    alpha = canvas[:, :, 3:4]
    linear = (rgb * alpha + torch.ones(H, W, 3) * (1 - alpha)).clamp(0, 1).numpy()
    srgb   = linear_to_srgb(linear)
    return (np.clip(srgb, 0, 1) * 255).astype(np.uint8)

# ── Worker init ───────────────────────────────────────
def init_worker(brush_tip_path):
    global g_bisenet, g_dsine, g_dsine_utils, g_intrins_from_fov
    global g_dsine_normalize, g_dino_processor, g_dino_model
    global g_dinov2_model
    global g_face_mesh, g_brush_tip

    torch.set_num_threads(2)

    # BiSeNet
    import resnet as _resnet
    _resnet.Resnet18.init_weight = lambda self: None
    from model import BiSeNet as _BiSeNet
    g_bisenet = _BiSeNet(n_classes=19)
    g_bisenet.load_state_dict(torch.load(BISENET_MODEL, map_location="cpu"))
    g_bisenet.eval()

    # DSINE
    sys.path.insert(0, DSINE_DIR)
    import utils.utils as _dsine_utils
    from utils.projection import intrins_from_fov as _intrins_from_fov
    from models.dsine.v02 import DSINE_v02 as _DSINE

    class _DSINEArgs:
        NNET_encoder_B=5; NNET_decoder_NF=2048; NNET_decoder_BN=False
        NNET_decoder_down=8; NNET_learned_upsampling=False
        NNET_output_dim=3; NNET_output_type="R"
        NNET_feature_dim=64; NNET_hidden_dim=64
        NRN_prop_ps=5; NRN_num_iter_train=5; NRN_num_iter_test=5
        NRN_ray_relu=False

    g_dsine = _DSINE(_DSINEArgs())
    g_dsine = _dsine_utils.load_checkpoint(DSINE_CKPT, g_dsine)
    g_dsine.eval()
    for m in g_dsine.modules():
        for name, buf in m.named_buffers(recurse=False):
            m.register_buffer(name, buf.to("cpu"))
    g_dsine_utils      = _dsine_utils
    g_intrins_from_fov = _intrins_from_fov
    g_dsine_normalize  = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    # DINO
    g_dino_processor = AutoImageProcessor.from_pretrained(DINO_DIR)
    g_dino_model     = AutoModel.from_pretrained(DINO_DIR, output_attentions=True)
    g_dino_model.eval()

    # DINOv2
    g_dinov2_model = torch.hub.load(DINOV2_DIR, "dinov2_vits14", source="local")
    g_dinov2_model.eval()

    # MediaPipe
    _mp_fm = mp.solutions.face_mesh
    g_face_mesh = _mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Brush tip
    g_brush_tip = load_brush_tip(brush_tip_path)


# ── Segment (from segment.py, unchanged logic) ────────
def run_segment(image_rgb):
    """Returns masks_data dict, attn_map, normal — all at original resolution"""
    H, W = image_rgb.shape[:2]

    # BiSeNet
    inp_bs = cv2.resize(image_rgb[:,:,::-1], (512,512)).astype(np.float32) / 255.0
    inp_bs = (inp_bs - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
    inp_bs = torch.from_numpy(inp_bs.transpose(2,0,1).copy()).unsqueeze(0).float()
    with torch.no_grad():
        parsing = g_bisenet(inp_bs)[0].squeeze(0).numpy().argmax(0)
    parsing = cv2.resize(parsing.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    # MediaPipe
    results    = g_face_mesh.process(image_rgb)
    mesh_masks = {}
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        for region_name, indices in MESH_REGIONS.items():
            points = np.array([
                [int(landmarks[i].x * W), int(landmarks[i].y * H)]
                for i in indices
            ], dtype=np.int32)
            m = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(m, [points], 255)
            mesh_masks[region_name] = m.astype(bool)

    # DSINE
    img_t  = torch.from_numpy(image_rgb.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    lrtb   = g_dsine_utils.get_padding(H, W)
    img_tp = F.pad(img_t, lrtb, mode="constant", value=0.0)
    img_tp = g_dsine_normalize(img_tp)
    intrins = g_intrins_from_fov(new_fov=60.0, H=H, W=W, device=torch.device("cpu")).unsqueeze(0)
    intrins[:, 0, 2] += lrtb[0]; intrins[:, 1, 2] += lrtb[2]
    with torch.no_grad():
        pred = g_dsine(img_tp, intrins=intrins)[-1]
        pred = pred[:, :, lrtb[2]:lrtb[2]+H, lrtb[0]:lrtb[0]+W]
    normal = pred.squeeze(0).permute(1,2,0).numpy()
    nlen   = np.linalg.norm(normal, axis=-1, keepdims=True).clip(1e-6)
    normal = normal / nlen  # (H,W,3)

    # DINO attention map
    pil_img = Image.fromarray(image_rgb)
    inputs  = g_dino_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        dino_out = g_dino_model(**inputs)
    h_feat, w_feat = 14, 14
    attn_last    = dino_out.attentions[-1][0].cpu()
    attn_avg     = attn_last.mean(0)
    center_patch = (h_feat//2)*w_feat + (w_feat//2) + 1
    center_attn  = attn_avg[center_patch, 1:].numpy()
    center_attn  = (center_attn - center_attn.min()) / (center_attn.max() - center_attn.min() + 1e-8)
    center_map   = center_attn.reshape(h_feat, w_feat)
    attn_map = np.array(Image.fromarray((center_map*255).astype(np.uint8)).resize((W,H), Image.BILINEAR),
                        dtype=np.float32) / 255.0
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Raw masks (same as segment.py)
    raw_masks = {"background": np.ones((H, W), dtype=bool)}
    for bisenet_name, our_name in BISENET_MAP.items():
        bidx = BISENET_NAMES.index(bisenet_name)
        raw_masks[our_name] = (parsing == bidx)
    for region_name, mask in mesh_masks.items():
        raw_masks[region_name] = mask

    # Label map (same as segment.py)
    label_map = np.zeros((H, W), dtype=np.uint8)
    for bisenet_name, our_name in BISENET_MAP.items():
        label_map[raw_masks[our_name]] = LABEL_ID[our_name]
    if "face" in mesh_masks:
        label_map[mesh_masks["face"]] = LABEL_ID["face"]
    for our_name in ["left_ear", "right_ear"]:
        label_map[raw_masks[our_name]] = LABEL_ID[our_name]
    for region_name in ["left_eye", "right_eye", "nose", "mouth"]:
        if region_name in mesh_masks:
            label_map[mesh_masks[region_name]] = LABEL_ID[region_name]
    for region_name in ["left_eye_detail", "right_eye_detail", "mouth_detail"]:
        if region_name in mesh_masks:
            label_map[mesh_masks[region_name]] = LABEL_ID[region_name]

    masks_dict = {name: raw_masks.get(name, (label_map == lid))
                  for lid, name, _, _ in _LABEL_DEF_FULL}

    return masks_dict, attn_map, normal


# ── optimize (unchanged from run2.py, reads masks_data/attn/normal instead of files) ──
def optimize(img_path, masks_data, attn_full, normal_full, out_dir, brush_tip):
    out_dir  = Path(out_dir)

    def log(msg):
        print(msg)

    img    = Image.open(img_path).convert("RGB")
    W0, H0 = img.size
    scale  = MAX_SIZE / max(W0, H0)
    img    = img.resize((int(W0 * scale), int(H0 * scale)), Image.LANCZOS)
    srgb_np   = np.array(img, dtype=np.float32) / 255.0
    target_np = srgb_to_linear(srgb_np)
    H, W      = target_np.shape[:2]
    target_t  = torch.tensor(target_np)
    log(f"  Resolution: {W}x{H}")

    attn_map = np.array(Image.fromarray((attn_full * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    normal_vis = ((normal_full + 1) / 2 * 255).astype(np.uint8)
    normal = np.array(Image.fromarray(normal_vis).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0 * 2 - 1
    norm_len = np.linalg.norm(normal, axis=-1, keepdims=True).clip(1e-6)
    normal = normal / norm_len

    all_xs          = []
    all_ys          = []
    all_brush_sizes = []
    all_colors      = []
    all_meta        = []

    np.random.seed(0)
    torch.manual_seed(0)

    for label_id, label_name, n_strokes in LABEL_DEF:
        if label_name not in masks_data:
            continue
        mask_full = masks_data[label_name].astype(bool)
        mask = np.array(Image.fromarray(mask_full.astype(np.uint8) * 255).resize((W, H), Image.NEAREST), dtype=bool)
        if not mask.any():
            continue

        # background 单独处理
        if label_name == "background":
            brush_fixed = max(H, W) * 2
            log(f"  Label {label_id:2d} background            brush={brush_fixed}px  n={n_strokes}")
            bg_xs = np.random.uniform(0, W - 1, n_strokes).astype(np.float32)
            bg_ys = np.random.uniform(0, H - 1, n_strokes).astype(np.float32)
            bg_colors = np.zeros((n_strokes, 3), np.float32)
            for i in range(n_strokes):
                xi_ = int(np.clip(bg_xs[i], 0, W - 1))
                yi_ = int(np.clip(bg_ys[i], 0, H - 1))
                r = brush_fixed // 2
                x0b = max(0, xi_ - r); x1b = min(W, xi_ + r)
                y0b = max(0, yi_ - r); y1b = min(H, yi_ + r)
                bg_colors[i] = target_np[y0b:y1b, x0b:x1b].mean(axis=(0, 1))
            all_xs.append(bg_xs)
            all_ys.append(bg_ys)
            all_brush_sizes.extend([brush_fixed] * n_strokes)
            all_colors.append(np.clip(bg_colors, 0.02, 0.98))
            for i in range(n_strokes):
                all_meta.append({"label_id": label_id, "label_name": label_name, "normal": [0.0, 0.0, 1.0]})
            continue

        bbox = get_bbox(mask)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        bbox_short = min(x1 - x0, y1 - y0)
        brush_max  = max(2, bbox_short)
        if label_name in ("face", "neck", "hair", "cloth", "hat"):
            brush_min = 10
        elif label_name in ("left_eye_detail", "right_eye_detail", "mouth_detail"):
            brush_min = max(5, bbox_short // 4)
        else:
            brush_min = max(5, bbox_short // 4)

        log(f"  Label {label_id:2d} {label_name:20s}  bbox={bbox_short}px  brush {brush_min}~{brush_max}  n={n_strokes}")

        xs, ys = control_xy_masked(attn_map, mask, n_strokes, H, W)

        roughness = local_normal_variance_batch(normal, xs, ys, H, W)
        r_min, r_max = roughness.min(), roughness.max()
        roughness_norm = (roughness - r_min) / (r_max - r_min + 1e-8)
        attn_scores = np.array([attn_map[int(np.clip(ys[i], 0, H-1)), int(np.clip(xs[i], 0, W-1))]
                                 for i in range(len(xs))])
        priority = roughness_norm * 100 + attn_scores
        order = np.argsort(priority)
        xs = xs[order]
        ys = ys[order]

        sizes = [brush_size_schedule(i, len(xs), brush_min, brush_max) for i in range(len(xs))]

        colors = np.zeros((len(xs), 3), np.float32)
        normals_at = np.zeros((len(xs), 3), np.float32)
        for i in range(len(xs)):
            yi_ = int(np.clip(ys[i], 0, H - 1))
            xi_ = int(np.clip(xs[i], 0, W - 1))
            colors[i]     = target_np[yi_, xi_]
            normals_at[i] = normal[yi_, xi_]

        all_xs.append(xs)
        all_ys.append(ys)
        all_brush_sizes.extend(sizes)
        all_colors.append(colors)
        for i in range(len(xs)):
            all_meta.append({
                "label_id":   label_id,
                "label_name": label_name,
                "normal":     normals_at[i].tolist(),
            })

    if len(all_xs) == 0:
        log("  No strokes generated.")
        return None, None, None, None, None

    all_xs     = np.concatenate(all_xs)
    all_ys     = np.concatenate(all_ys)
    all_colors = np.concatenate(all_colors)
    N          = len(all_xs)
    log(f"  Total strokes: {N}  (mode={BRUSH_SIZE_MODE})")

    all_colors = np.clip(all_colors, 0.02, 0.98)

    xy_t       = torch.tensor(np.stack([all_xs, all_ys], axis=1), requires_grad=True)
    pr_init    = np.full(N, 0.7, dtype=np.float32)
    pressure_l = torch.tensor(np.log(pr_init / (1 - pr_init)), requires_grad=True)
    color_l    = torch.tensor(np.log(all_colors / (1 - all_colors)), requires_grad=True)
    angle_l    = torch.zeros(N, requires_grad=True)

    optimizer = torch.optim.Adam([xy_t, pressure_l, color_l, angle_l], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_ITER, eta_min=LR * 0.05)

    last_losses = []
    t_start     = time.time()
    log(f"  Start: {time.strftime('%H:%M:%S')}")

    for it in range(1, N_ITER + 1):
        t0 = time.time()
        optimizer.zero_grad()
        rendered = render(xy_t, pressure_l, color_l, angle_l, all_brush_sizes, H, W, brush_tip)
        loss     = F.mse_loss(rendered, target_t)
        lv       = loss.item()

        last_losses.append(lv)
        if len(last_losses) > EARLY_STOP_PATIENCE:
            last_losses.pop(0)
        if len(last_losses) == EARLY_STOP_PATIENCE and (max(last_losses) - min(last_losses)) < 1e-4:
            log(f"  Early stop at iter {it} (loss flat: {lv:.5f})")
            break

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            xy_t[:, 0].clamp_(0, W - 1)
            xy_t[:, 1].clamp_(0, H - 1)

        t1 = time.time()
        log(f"  iter {it:4d}/{N_ITER}  loss={lv:.5f}  {t1-t0:.2f}s")

    elapsed = time.time() - t_start
    log(f"  Total: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    return xy_t, pressure_l, color_l, angle_l, all_brush_sizes, all_meta, N, H, W


# ── Extract DinoFeats per stroke ──────────────────────
def extract_feats(xy_t, pressure_l, color_l, angle_l, brush_sizes, H, W, N):
    feats = np.zeros((N, 384), dtype=np.float32)
    with torch.no_grad():
        for stroke_i in range(N):
            canvas_rgb = render_up_to(xy_t, pressure_l, color_l, angle_l, brush_sizes, H, W, stroke_i, g_brush_tip)
            pil_img    = Image.fromarray(canvas_rgb)
            inp        = DINOV2_TRANSFORM(pil_img).unsqueeze(0)
            feat       = g_dinov2_model(inp)
            feats[stroke_i] = feat.squeeze(0).cpu().numpy()
    return feats


# ── process_one ───────────────────────────────────────
def process_one(args):
    img_path_str, output_dir_str, idx, total = args
    img_path = Path(img_path_str)
    out_dir  = Path(output_dir_str) / img_path.stem

    if out_dir.exists():
        print(f"[{idx}/{total}] {img_path.name} SKIP")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"[{idx}/{total}] {img_path.name} start")

    try:
        # 1. load image
        image     = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. segment
        masks_data, attn_map, normal = run_segment(image_rgb)

        # 3. optimize
        result = optimize(str(img_path), masks_data, attn_map, normal, str(out_dir), g_brush_tip)
        if result[0] is None:
            print(f"[{idx}/{total}] {img_path.name} no strokes, skip")
            return
        xy_t, pressure_l, color_l, angle_l, all_brush_sizes, all_meta, N, H, W = result

        # 4. save final.png
        with torch.no_grad():
            final = render(xy_t, pressure_l, color_l, angle_l, all_brush_sizes, H, W, g_brush_tip)
        Image.fromarray((np.clip(linear_to_srgb(final.cpu().numpy()), 0, 1) * 255).astype(np.uint8)).save(
            out_dir / "final.png")

        # 5. save strokes.json
        sx = 178 / W; sy = 218 / H
        x_out   = xy_t.detach().cpu().numpy()
        pr_out  = torch.sigmoid(pressure_l).detach().cpu().numpy()
        col_out = torch.sigmoid(color_l).detach().cpu().numpy()
        ang_out = angle_l.detach().cpu().numpy()

        strokes = []
        for i in range(N):
            strokes.append({
                "tool":             "brush",
                "brush_size":       all_brush_sizes[i],
                "color":            [float(col_out[i,0]), float(col_out[i,1]), float(col_out[i,2])],
                "smoothing":        0.7,
                "minimum_diameter": 0.0,
                "angle":            float(ang_out[i]),
                "label_id":         all_meta[i]["label_id"],
                "label_name":       all_meta[i]["label_name"],
                "normal":           all_meta[i]["normal"],
                "samples": [{
                    "x":        float(x_out[i,0]) * sx,
                    "y":        float(x_out[i,1]) * sy,
                    "pressure": float(pr_out[i]),
                    "t":        float(i * 50),
                }]
            })
        doc = {"canvas_w": 178, "canvas_h": 218, "strokes": strokes}
        with open(out_dir / "strokes.json", "w") as f:
            json.dump(doc, f, indent=2)

        # 6. extract DinoFeats
        print(f"[{idx}/{total}] {img_path.name} extracting {N} DINO feats...")
        feats = extract_feats(xy_t, pressure_l, color_l, angle_l, all_brush_sizes, H, W, N)
        np.save(str(out_dir / "feats.npy"), feats)

        elapsed = time.time() - t_start
        print(f"[{idx}/{total}] {img_path.name} done {elapsed:.0f}s  strokes={N}")

    except Exception as e:
        print(f"[{idx}/{total}] {img_path.name} ERROR: {e}")
        import traceback; traceback.print_exc()


# ── Main ──────────────────────────────────────────────
def main():
    from multiprocessing import Pool

    base_dir   = Path(__file__).parent
    images_dir = base_dir / IMAGE_DIR
    output_dir = base_dir / OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    exts      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])
    total     = len(img_paths)
    print(f"Found {total} images.")

    tasks = [(str(p), str(output_dir), idx+1, total) for idx, p in enumerate(img_paths)]

    with Pool(processes=NUM_WORKERS,
              initializer=init_worker,
              initargs=(str(base_dir / BRUSH_TIP),)) as pool:
        pool.map(process_one, tasks)

    print("\nAll done.")

if __name__ == "__main__":
    main()
