import json
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

# ─── Constants ────────────────────────────────────────────────────
N_STROKES_BG        = 10
N_STROKES_FG        = 100
N_STROKES_DT        = 100
N_STROKES_FN        = 100
N_ITER              = 30
LR                  = 0.05
MAX_SIZE            = 218
EARLY_STOP_PATIENCE = 3

XY_MODE         = "curve_up"
XY_PARAMS       = {"brush_k": 3}
SIZE_MODE       = "curve_up"
SIZE_PARAMS     = {"brush_k": 3}

DT_XY_MODE      = "curve_up"
DT_XY_PARAMS    = {"brush_k": 1}
DT_SIZE_MODE    = "curve_up"
DT_SIZE_PARAMS  = {"brush_k": 1}

FN_XY_MODE      = "curve_down"
FN_XY_PARAMS    = {"brush_k": 8}
FN_SIZE_MODE    = "curve_down"
FN_SIZE_PARAMS  = {"brush_k": 8}

# ─── Device ───────────────────────────────────────────────────────
def get_device():
    print("CPU")
    return torch.device("cpu")

# ─── Color utils ──────────────────────────────────────────────────
def srgb_to_linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(linear):
    return np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * np.power(np.clip(linear, 0, 1), 1.0 / 2.4) - 0.055)

# ─── Brush tip ────────────────────────────────────────────────────
def load_brush_tip(path, device):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr, device=device).unsqueeze(0).unsqueeze(0)

def load_attention_map(attn_path, H, W):
    attn    = Image.open(attn_path).convert("L").resize((W, H), Image.BILINEAR)
    attn_np = np.array(attn, dtype=np.float32) / 255.0
    attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
    return attn_np

# ─── Step 1: Attention map ────────────────────────────────────────
def get_attention_map(img_path, output_dir, processor, model, device):
    img_path = Path(img_path)
    image    = Image.open(img_path).convert("RGB")
    inputs   = processor(images=image, return_tensors="pt")
    inputs   = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    h_feat, w_feat = 14, 14
    attn_last    = outputs.attentions[-1][0].cpu()
    attn_avg     = attn_last.mean(0)
    center_patch = (h_feat // 2) * w_feat + (w_feat // 2) + 1
    center_attn  = attn_avg[center_patch, 1:].numpy()
    center_attn  = (center_attn - center_attn.min()) / (center_attn.max() - center_attn.min() + 1e-8)
    center_map   = center_attn.reshape(h_feat, w_feat)

    mask = Image.fromarray((center_map * 255).astype(np.uint8)).resize(
        image.size, resample=Image.BILINEAR
    )

    out_path = Path(output_dir) / f"{img_path.stem}_mask.png"
    mask.save(out_path)
    print(f"  Saved attention map: {out_path}")
    return str(out_path)

# ─── control_xy ───────────────────────────────────────────────────
def control_xy(attn_map, N, H, W, mode, mask_percentile=0,
               brush_center=None, brush_steepness=None, brush_k=None):
    flat = attn_map.flatten().astype(np.float64)

    if mode == "linear":
        weights = flat + 0.01
    elif mode == "curve_up":
        weights = 1 - np.exp(-brush_k * flat) + 0.01
    elif mode == "curve_down":
        weights = np.exp(-brush_k * (1 - flat)) + 0.01
    elif mode == "arctan":
        weights = np.arctan((flat - brush_center) * brush_steepness) / np.arctan(brush_steepness) * 0.5 + 0.5 + 0.01
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mask_percentile > 0:
        threshold = np.percentile(flat, mask_percentile)
        weights = weights * (flat >= threshold)

    weights /= weights.sum()

    indices = np.random.choice(len(weights), size=N, replace=True, p=weights)
    ys = (indices // W).astype(np.float32)
    xs = (indices  % W).astype(np.float32)
    xs += np.random.uniform(-1, 1, N).astype(np.float32)
    ys += np.random.uniform(-1, 1, N).astype(np.float32)
    xs  = np.clip(xs, 0, W - 1)
    ys  = np.clip(ys, 0, H - 1)
    return xs, ys

# ─── control_size ─────────────────────────────────────────────────
def control_size(xs, ys, attn_map, H, W, brush_min, brush_max,
                 mode, brush_center=None, brush_steepness=None, brush_k=None):
    sizes  = []
    ah, aw = attn_map.shape
    for i in range(len(xs)):
        xi       = int(np.clip(xs[i] / W * aw, 0, aw - 1))
        yi       = int(np.clip(ys[i] / H * ah, 0, ah - 1))
        attn_val = attn_map[yi, xi]
        if mode == "linear":
            t = attn_val
        elif mode == "curve_up":
            t = 1 - np.exp(-brush_k * attn_val)
        elif mode == "curve_down":
            t = np.exp(-brush_k * (1 - attn_val))
        elif mode == "arctan":
            t = np.arctan((attn_val - brush_center) * brush_steepness) / np.arctan(brush_steepness) * 0.5 + 0.5
        else:
            raise ValueError(f"Unknown mode: {mode}")
        size = int(brush_max - t * (brush_max - brush_min))
        size = max(brush_min, size)
        sizes.append(size)
    return sizes

# ─── conv average color ───────────────────────────────────────────
def get_conv_avg_color(target_np, xi, yi, H, W, radius):
    x0 = max(0, xi - radius)
    x1 = min(W, xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(H, yi + radius + 1)
    region = target_np[y0:y1, x0:x1]
    return region.mean(axis=(0, 1))

# ─── Render ───────────────────────────────────────────────────────
def render_single_stroke(canvas, i, xy, pressure_l, color_l, angle_l, brush_sizes, H, W, device, brush_tip):
    x        = xy[:, 0].clamp(0, W - 1)
    y        = xy[:, 1].clamp(0, H - 1)
    pressure = torch.sigmoid(pressure_l)
    color    = torch.sigmoid(color_l)
    angle    = angle_l

    brush_size = brush_sizes[i]
    max_size   = brush_size
    min_size   = 1
    p_val      = pressure[i].clamp(0, 1)
    normalized = p_val
    curved     = torch.log(1.0 + normalized * 9.0) / math.log(10.0)
    tip_size_f = min_size + (max_size - min_size) * curved
    tip_size   = int(tip_size_f.item())
    tip_size   = max(1, min(tip_size, max_size))
    xi = int(x[i].item())
    yi = int(y[i].item())

    if tip_size <= 1:
        if 0 <= xi < W and 0 <= yi < H:
            ci    = color[i]
            src_a = (normalized ** 2.5).clamp(0, 1)
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
        theta  = torch.stack([
            torch.stack([ cos_a, sin_a, zeros], dim=1),
            torch.stack([-sin_a, cos_a, zeros], dim=1),
        ], dim=1)
        grid      = F.affine_grid(theta, tip.shape, align_corners=False)
        rotated   = F.grid_sample(tip, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        tip_alpha = rotated[0, 0]
    else:
        ys_t      = torch.arange(tip_size, dtype=torch.float32, device=device) - tip_size / 2.0
        xs_t      = torch.arange(tip_size, dtype=torch.float32, device=device) - tip_size / 2.0
        dy, dx    = torch.meshgrid(ys_t, xs_t, indexing="ij")
        tip_alpha = (torch.sqrt(dx * dx + dy * dy) <= tip_size / 2.0).float()

    ai_map = (tip_alpha * (normalized ** 2.5)).clamp(0, 1)
    x0 = xi - half; x1 = x0 + tip_size
    y0 = yi - half; y1 = y0 + tip_size
    cx0 = max(0, x0); cy0 = max(0, y0)
    cx1 = min(W, x1); cy1 = min(H, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return canvas

    sx0 = cx0 - x0; sy0 = cy0 - y0
    sx1 = sx0 + (cx1 - cx0); sy1 = sy0 + (cy1 - cy0)

    src_a    = ai_map[sy0:sy1, sx0:sx1].unsqueeze(-1)
    ci       = color[i][None, None, :]
    region   = canvas[cy0:cy1, cx0:cx1]
    new_rgb  = ci * src_a + region[:, :, :3] * (1 - src_a)
    new_a    = src_a + region[:, :, 3:4] * (1 - src_a)
    new_rgba = torch.cat([new_rgb, new_a], dim=-1)
    canvas   = torch.cat([
        canvas[:cy0],
        torch.cat([canvas[cy0:cy1, :cx0], new_rgba, canvas[cy0:cy1, cx1:]], dim=1),
        canvas[cy1:]
    ], dim=0)
    return canvas


def render(xy, pressure_l, color_l, angle_l, brush_sizes, H, W, device, brush_tip=None):
    canvas = torch.zeros(H, W, 4, device=device)
    for i in range(xy.shape[0]):
        canvas = render_single_stroke(canvas, i, xy, pressure_l, color_l, angle_l, brush_sizes, H, W, device, brush_tip)
    rgb   = canvas[:, :, :3]
    alpha = canvas[:, :, 3:4]
    white = torch.ones(H, W, 3, device=device)
    return (rgb * alpha + white * (1 - alpha)).clamp(0, 1)


def render_and_save_strokes(xy, pressure_l, color_l, angle_l, brush_sizes, H, W, device, brush_tip, save_dir, stroke_offset=0):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    canvas = torch.zeros(H, W, 4, device=device)
    for i in range(xy.shape[0]):
        canvas = render_single_stroke(canvas, i, xy, pressure_l, color_l, angle_l, brush_sizes, H, W, device, brush_tip)
        rgb   = canvas[:, :, :3]
        alpha = canvas[:, :, 3:4]
        white = torch.ones(H, W, 3, device=device)
        current_srgb = linear_to_srgb((rgb * alpha + white * (1 - alpha)).clamp(0, 1).detach().cpu().numpy())
        Image.fromarray((np.clip(current_srgb, 0, 1) * 255).astype(np.uint8)).save(
            save_dir / f"stroke_{stroke_offset + i:04d}.png"
        )
    rgb   = canvas[:, :, :3]
    alpha = canvas[:, :, 3:4]
    white = torch.ones(H, W, 3, device=device)
    return (rgb * alpha + white * (1 - alpha)).clamp(0, 1)

# ─── Step 2: Optimize ─────────────────────────────────────────────
def optimize(img_path, attn_path, out_dir, device, brush_tip):
    out_dir  = Path(out_dir)
    iter_dir = out_dir / "iterations"
    iter_dir.mkdir(exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    img    = Image.open(img_path).convert("RGB")
    W0, H0 = img.size
    scale  = MAX_SIZE / max(W0, H0)
    img    = img.resize((int(W0 * scale), int(H0 * scale)), Image.LANCZOS)
    srgb_np   = np.array(img, dtype=np.float32) / 255.0
    target_np = srgb_to_linear(srgb_np)
    H, W      = target_np.shape[:2]
    target_t  = torch.tensor(target_np, device=device)
    log(f"  Resolution: {W}x{H}")

    attn_map = load_attention_map(attn_path, H, W)

    long_side    = max(H, W)
    BRUSH_BG     = long_side * 2
    BRUSH_MIN    = long_side // 16
    BRUSH_MAX    = long_side
    BRUSH_DT_MIN = long_side // 32
    BRUSH_DT_MAX = long_side // 4
    BRUSH_FN_MIN = 2
    BRUSH_FN_MAX = long_side // 16

    np.random.seed(0); torch.manual_seed(0)

    log(f"  Background: {N_STROKES_BG} strokes, brush_size={BRUSH_BG}")
    bg_xs = np.random.uniform(0, W - 1, N_STROKES_BG).astype(np.float32)
    bg_ys = np.random.uniform(0, H - 1, N_STROKES_BG).astype(np.float32)
    bg_brush_sizes = [BRUSH_BG] * N_STROKES_BG

    conv_radius = BRUSH_BG // 2
    bg_colors   = np.zeros((N_STROKES_BG, 3), np.float32)
    for i in range(N_STROKES_BG):
        xi = int(np.clip(bg_xs[i], 0, W - 1))
        yi = int(np.clip(bg_ys[i], 0, H - 1))
        bg_colors[i] = get_conv_avg_color(target_np, xi, yi, H, W, conv_radius)
    bg_colors = np.clip(bg_colors, 0.02, 0.98)

    bg_xy         = torch.tensor(np.stack([bg_xs, bg_ys], axis=1), device=device, requires_grad=True)
    bg_pr_init    = np.full(N_STROKES_BG, 0.8, dtype=np.float32)
    bg_pressure_l = torch.tensor(np.log(bg_pr_init / (1 - bg_pr_init)), device=device, requires_grad=True)
    bg_color_l    = torch.tensor(np.log(bg_colors / (1 - bg_colors)), device=device, requires_grad=True)
    bg_angle_l    = torch.zeros(N_STROKES_BG, device=device, requires_grad=True)

    log(f"  Foreground: {N_STROKES_FG} strokes, brush_max={BRUSH_MAX}, sampling 25%~100% attention")
    fg_xs, fg_ys = control_xy(attn_map, N_STROKES_FG, H, W, XY_MODE, mask_percentile=25, **XY_PARAMS)
    fg_brush_sizes = control_size(fg_xs, fg_ys, attn_map, H, W, BRUSH_MIN, BRUSH_MAX, SIZE_MODE, **SIZE_PARAMS)
    log(f"  FG brush sizes: min={min(fg_brush_sizes)}, max={max(fg_brush_sizes)}, mean={np.mean(fg_brush_sizes):.1f}")

    fg_order       = np.argsort(fg_brush_sizes)[::-1]
    fg_xs          = fg_xs[fg_order]
    fg_ys          = fg_ys[fg_order]
    fg_brush_sizes = [fg_brush_sizes[i] for i in fg_order]

    fg_xy         = torch.tensor(np.stack([fg_xs, fg_ys], axis=1), device=device, requires_grad=True)
    fg_pr_init    = np.full(N_STROKES_FG, 0.6, dtype=np.float32)
    fg_pressure_l = torch.tensor(np.log(fg_pr_init / (1 - fg_pr_init)), device=device, requires_grad=True)
    fg_colors = np.zeros((N_STROKES_FG, 3), np.float32)
    for i in range(N_STROKES_FG):
        yi = int(np.clip(fg_ys[i], 0, H - 1))
        xi = int(np.clip(fg_xs[i], 0, W - 1))
        fg_colors[i] = target_np[yi, xi]
    fg_colors  = np.clip(fg_colors, 0.02, 0.98)
    fg_color_l = torch.tensor(np.log(fg_colors / (1 - fg_colors)), device=device, requires_grad=True)
    fg_angle_l = torch.zeros(N_STROKES_FG, device=device, requires_grad=True)

    log(f"  Detail: {N_STROKES_DT} strokes, brush_max={BRUSH_DT_MAX}, brush_min={BRUSH_DT_MIN}, sampling 75%~100% attention")
    dt_xs, dt_ys = control_xy(attn_map, N_STROKES_DT, H, W, DT_XY_MODE, mask_percentile=75, **DT_XY_PARAMS)
    dt_brush_sizes = control_size(dt_xs, dt_ys, attn_map, H, W, BRUSH_DT_MIN, BRUSH_DT_MAX, DT_SIZE_MODE, **DT_SIZE_PARAMS)
    log(f"  DT brush sizes: min={min(dt_brush_sizes)}, max={max(dt_brush_sizes)}, mean={np.mean(dt_brush_sizes):.1f}")

    dt_order       = np.argsort(dt_brush_sizes)[::-1]
    dt_xs          = dt_xs[dt_order]
    dt_ys          = dt_ys[dt_order]
    dt_brush_sizes = [dt_brush_sizes[i] for i in dt_order]

    dt_xy         = torch.tensor(np.stack([dt_xs, dt_ys], axis=1), device=device, requires_grad=True)
    dt_pr_init    = np.full(N_STROKES_DT, 0.6, dtype=np.float32)
    dt_pressure_l = torch.tensor(np.log(dt_pr_init / (1 - dt_pr_init)), device=device, requires_grad=True)
    dt_colors = np.zeros((N_STROKES_DT, 3), np.float32)
    for i in range(N_STROKES_DT):
        yi = int(np.clip(dt_ys[i], 0, H - 1))
        xi = int(np.clip(dt_xs[i], 0, W - 1))
        dt_colors[i] = target_np[yi, xi]
    dt_colors  = np.clip(dt_colors, 0.02, 0.98)
    dt_color_l = torch.tensor(np.log(dt_colors / (1 - dt_colors)), device=device, requires_grad=True)
    dt_angle_l = torch.zeros(N_STROKES_DT, device=device, requires_grad=True)

    log(f"  Fine: {N_STROKES_FN} strokes, brush_max={BRUSH_FN_MAX}, brush_min={BRUSH_FN_MIN}, sampling 95%~100% attention")
    fn_xs, fn_ys = control_xy(attn_map, N_STROKES_FN, H, W, FN_XY_MODE, mask_percentile=90, **FN_XY_PARAMS)
    fn_brush_sizes = control_size(fn_xs, fn_ys, attn_map, H, W, BRUSH_FN_MIN, BRUSH_FN_MAX, FN_SIZE_MODE, **FN_SIZE_PARAMS)
    log(f"  FN brush sizes: min={min(fn_brush_sizes)}, max={max(fn_brush_sizes)}, mean={np.mean(fn_brush_sizes):.1f}")

    fn_order       = np.argsort(fn_brush_sizes)[::-1]
    fn_xs          = fn_xs[fn_order]
    fn_ys          = fn_ys[fn_order]
    fn_brush_sizes = [fn_brush_sizes[i] for i in fn_order]

    fn_xy         = torch.tensor(np.stack([fn_xs, fn_ys], axis=1), device=device, requires_grad=True)
    fn_pr_init    = np.full(N_STROKES_FN, 0.6, dtype=np.float32)
    fn_pressure_l = torch.tensor(np.log(fn_pr_init / (1 - fn_pr_init)), device=device, requires_grad=True)
    fn_colors = np.zeros((N_STROKES_FN, 3), np.float32)
    for i in range(N_STROKES_FN):
        yi = int(np.clip(fn_ys[i], 0, H - 1))
        xi = int(np.clip(fn_xs[i], 0, W - 1))
        fn_colors[i] = target_np[yi, xi]
    fn_colors  = np.clip(fn_colors, 0.02, 0.98)
    fn_color_l = torch.tensor(np.log(fn_colors / (1 - fn_colors)), device=device, requires_grad=True)
    fn_angle_l = torch.zeros(N_STROKES_FN, device=device, requires_grad=True)

    all_brush_sizes = bg_brush_sizes + fg_brush_sizes + dt_brush_sizes + fn_brush_sizes

    all_params = [
        bg_xy, bg_pressure_l, bg_color_l, bg_angle_l,
        fg_xy, fg_pressure_l, fg_color_l, fg_angle_l,
        dt_xy, dt_pressure_l, dt_color_l, dt_angle_l,
        fn_xy, fn_pressure_l, fn_color_l, fn_angle_l,
    ]
    optimizer = torch.optim.Adam(all_params, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_ITER, eta_min=LR * 0.05)

    def get_combined():
        xy_all       = torch.cat([bg_xy, fg_xy, dt_xy, fn_xy], dim=0)
        pressure_all = torch.cat([bg_pressure_l, fg_pressure_l, dt_pressure_l, fn_pressure_l], dim=0)
        color_all    = torch.cat([bg_color_l, fg_color_l, dt_color_l, fn_color_l], dim=0)
        angle_all    = torch.cat([bg_angle_l, fg_angle_l, dt_angle_l, fn_angle_l], dim=0)
        return xy_all, pressure_all, color_all, angle_all

    last_losses = []
    t_start     = time.time()
    log(f"  Start: {time.strftime('%H:%M:%S')}")

    for it in range(1, N_ITER + 1):
        t0 = time.time()
        optimizer.zero_grad()
        xy_all, pressure_all, color_all, angle_all = get_combined()
        rendered = render(xy_all, pressure_all, color_all, angle_all, all_brush_sizes, H, W, device, brush_tip)
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
            bg_xy[:, 0].clamp_(0, W - 1); bg_xy[:, 1].clamp_(0, H - 1)
            fg_xy[:, 0].clamp_(0, W - 1); fg_xy[:, 1].clamp_(0, H - 1)
            dt_xy[:, 0].clamp_(0, W - 1); dt_xy[:, 1].clamp_(0, H - 1)
            fn_xy[:, 0].clamp_(0, W - 1); fn_xy[:, 1].clamp_(0, H - 1)

        t1 = time.time()
        log(f"  iter {it:4d}/{N_ITER}  loss={lv:.5f}  iter_time={t1-t0:.2f}s")

        if it % 10 == 0:
            with torch.no_grad():
                xy_all, pressure_all, color_all, angle_all = get_combined()
                snap = render(xy_all, pressure_all, color_all, angle_all, all_brush_sizes, H, W, device, brush_tip)
            snap_srgb = linear_to_srgb(snap.cpu().numpy())
            Image.fromarray((np.clip(snap_srgb, 0, 1) * 255).astype(np.uint8)).save(
                iter_dir / f"iter_{it:04d}.png")

    t_end   = time.time()
    elapsed = t_end - t_start
    log(f"  End  : {time.strftime('%H:%M:%S')}")
    log(f"  Total: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    stroke_dir = out_dir / "stroke_frames"
    with torch.no_grad():
        xy_all, pressure_all, color_all, angle_all = get_combined()
        render_and_save_strokes(
            xy_all, pressure_all, color_all, angle_all,
            all_brush_sizes, H, W, device, brush_tip, stroke_dir, stroke_offset=0
        )

    with torch.no_grad():
        xy_all, pressure_all, color_all, angle_all = get_combined()
        final = render(xy_all, pressure_all, color_all, angle_all, all_brush_sizes, H, W, device, brush_tip)
    final_srgb = linear_to_srgb(final.cpu().numpy())
    Image.fromarray((np.clip(final_srgb, 0, 1) * 255).astype(np.uint8)).save(out_dir / "final.png")

    sx = 178 / W; sy = 218 / H
    with torch.no_grad():
        xy_all, pressure_all, color_all, angle_all = get_combined()
    x_out   = xy_all.detach().cpu().numpy()
    pr_out  = torch.sigmoid(pressure_all).detach().cpu().numpy()
    col_out = torch.sigmoid(color_all).detach().cpu().numpy()
    ang_out = angle_all.detach().cpu().numpy()

    strokes = []
    for i in range(len(all_brush_sizes)):
        strokes.append({
            "tool": "brush",
            "brush_size": all_brush_sizes[i],
            "color": [float(col_out[i, 0]), float(col_out[i, 1]), float(col_out[i, 2])],
            "smoothing": 0.7,
            "minimum_diameter": 0.0,
            "angle": float(ang_out[i]),
            "samples": [{
                "x": float(x_out[i, 0]) * sx,
                "y": float(x_out[i, 1]) * sy,
                "pressure": float(pr_out[i]),
                "t": float(i * 50)
            }]
        })
    doc = {"canvas_w": 178, "canvas_h": 218, "strokes": strokes}
    with open(out_dir / "strokes.json", "w") as f:
        json.dump(doc, f, indent=2)
    log(f"  Saved: final.png + strokes.json")

    with open(out_dir / "log.txt", "w") as f:
        f.write("\n".join(log_lines))


# ─── Worker ───────────────────────────────────────────────────────
def init_worker(base_dir_str, tip_path_str):
    global g_processor, g_model, g_brush_tip, g_device
    torch.set_num_threads(8)
    g_device = get_device()
    g_processor = AutoImageProcessor.from_pretrained(str(Path(base_dir_str) / "dino-vits16"))
    g_model = AutoModel.from_pretrained(
        str(Path(base_dir_str) / "dino-vits16"),
        output_attentions=True,
    )
    g_model.to(g_device).eval()
    tip_path = Path(tip_path_str)
    g_brush_tip = load_brush_tip(str(tip_path), g_device) if tip_path.exists() else None


def process_one(args):
    img_path_str, output_dir_str, idx, total = args
    img_path = Path(img_path_str)
    out_dir  = Path(output_dir_str) / img_path.stem

    if out_dir.exists():
        print(f"\n[{idx}/{total}] {img_path.name} — [SKIP] already exists.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{idx}/{total}] {img_path.name}")

    t_start   = time.time()
    attn_path = get_attention_map(img_path, out_dir, g_processor, g_model, g_device)
    optimize(str(img_path), attn_path, out_dir, g_device, g_brush_tip)
    elapsed   = time.time() - t_start
    print(f"  [{idx}/{total}] done in {elapsed:.1f}s ({elapsed/60:.1f}min)")


# ─── Main ─────────────────────────────────────────────────────────
def main():
    from multiprocessing import Pool

    base_dir   = Path(__file__).parent
    images_dir = base_dir / "images"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])
    total = len(img_paths)
    print(f"Found {total} images.")

    tip_path = str(base_dir / "brush_tip.png")

    tasks = [(str(p), str(output_dir), idx + 1, total) for idx, p in enumerate(img_paths)]

    NUM_WORKERS = 12

    with Pool(processes=NUM_WORKERS, initializer=init_worker, initargs=(str(base_dir), tip_path)) as pool:
        pool.map(process_one, tasks)

    print("\nAll done.")


if __name__ == "__main__":
    main()