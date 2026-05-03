import os
import json
import math
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np

# ── 超参 ──────────────────────────────────────────────
WINDOW      = 50
STRIDE      = 1
D_MODEL     = 256
N_HEADS     = 8
N_LAYERS    = 6
DROPOUT     = 0.1
LR          = 1e-4
WARMUP      = 10
EPOCHS      = 500
BATCH_SIZE  = 256   # cross-attention更耗显存，减小batch
CANVAS_FEAT = 256
TEXT_FEAT   = 512   # CLIP text
CLIP_PATCH_DIM = 512  # CLIP patch token dim
CLIP_PATCH_N   = 49   # 7x7 patches

# STROKE_DIM = 8 (inference时只用8维)
STROKE_DIM  = 8
LABEL_MAX   = 14
N_LANDMARKS = 136   # 68个关键点×2

# loss weights
W_FM       = 1.0
W_LABEL    = 0.1
W_NORMAL   = 0.1
W_LANDMARK = 0.1
W_DINO     = 0.1

EPS      = 1e-4
FM_STEPS = 1

DATA_ROOT = "output"
BASE_CKPT = "model_fm_v4.pt"
CKPT      = "model_fm_v4.pt"

# ── encode_stroke (8维，inference对齐) ────────────────
def encode_stroke(s, W, H):
    p        = s["samples"][0]
    x_norm   = p["x"] / W
    y_norm   = p["y"] / H
    press    = p["pressure"]
    bs_norm  = s["brush_size"] / max(W, H)
    ang_norm = (s["angle"] + math.pi) / (2 * math.pi)
    r, g, b  = s["color"]
    return [x_norm, y_norm, press, bs_norm, ang_norm, r, g, b]

# ── Dataset ───────────────────────────────────────────
class StrokeDataset(Dataset):
    def __init__(self, data_root, window=WINDOW, stride=STRIDE):
        self.samples = []
        json_paths = sorted(glob.glob(os.path.join(data_root, "*", "strokes.json")))
        print(f"  Found {len(json_paths)} stroke files.")
        skipped = 0
        for json_path in json_paths:
            stem = os.path.basename(os.path.dirname(json_path))
            # 必需文件
            feat_path      = os.path.join(data_root, stem, "feats.npy")
            text_path      = os.path.join(data_root, stem, "text.npy")
            clip_path      = os.path.join(data_root, stem, "clip_patch_feats.npy")
            dino_tgt_path  = os.path.join(data_root, stem, "target_dino_cls.npy")
            lmk_path       = os.path.join(data_root, stem, "face_landmarks.npy")
            for p in [feat_path, text_path, clip_path, dino_tgt_path, lmk_path]:
                if not os.path.exists(p):
                    skipped += 1
                    break
            else:
                try:
                    feats          = np.load(feat_path)
                    text_feat      = np.load(text_path)
                    clip_patches   = np.load(clip_path)
                    target_dino    = np.load(dino_tgt_path)
                    face_landmarks = np.load(lmk_path)
                except Exception:
                    skipped += 1
                    continue
                if clip_patches.ndim != 3 or clip_patches.shape[1:] != (49, 512):
                    skipped += 1
                    continue

                with open(json_path) as f:
                    doc = json.load(f)
                W, H = doc["canvas_w"], doc["canvas_h"]
                strokes_raw = [s for s in doc["strokes"] if not s.get("undone", False)]
                strokes_enc = [encode_stroke(s, W, H) for s in strokes_raw]
                N = len(strokes_enc)
                if N <= window:
                    skipped += 1
                    continue

                for i in range(0, N - window, stride):
                    seq        = strokes_enc[i: i + window]
                    target_s   = strokes_enc[i + window]
                    raw_s      = strokes_raw[i + window]
                    frame_idx  = min(i + window - 1, len(feats) - 1)

                    # label (15分类)
                    label_id   = raw_s.get("label_id", 0)

                    # normal (3维，归一化到[0,1])
                    nx, ny, nz = raw_s.get("normal", [0.0, 0.0, 1.0])
                    normal     = [(nx+1)/2, (ny+1)/2, (nz+1)/2]

                    self.samples.append((
                        torch.tensor(seq,                       dtype=torch.float32),  # (50, 8)
                        torch.tensor(feats[frame_idx],          dtype=torch.float32),  # (384,) canvas dinov2 cls
                        torch.tensor(clip_patches[frame_idx],   dtype=torch.float32),  # (49, 512) canvas clip patch
                        torch.tensor(text_feat,                 dtype=torch.float32),  # (512,)
                        torch.tensor(target_s,                  dtype=torch.float32),  # (8,) next stroke
                        torch.tensor(label_id,                  dtype=torch.long),     # scalar
                        torch.tensor(normal,                    dtype=torch.float32),  # (3,)
                        torch.tensor(face_landmarks,            dtype=torch.float32),  # (136,)
                        torch.tensor(target_dino,               dtype=torch.float32),  # (384,)
                    ))
        print(f"  Skipped {skipped}, loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Encoders ──────────────────────────────────────────
class CanvasDINOEncoder(nn.Module):
    """DINOv2 CLS → AdaLN condition"""
    def __init__(self, out_dim=CANVAS_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(384, 512), nn.SiLU(),
            nn.Linear(512, out_dim),
        )
    def forward(self, x):
        return self.net(x)  # (B, CANVAS_FEAT)


class TextEncoder(nn.Module):
    """CLIP text → AdaLN condition"""
    def __init__(self, out_dim=CANVAS_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TEXT_FEAT, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)  # (B, CANVAS_FEAT)


class ClipPatchProjector(nn.Module):
    """CLIP patch tokens → cross-attention KV"""
    def __init__(self, out_dim=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(CLIP_PATCH_DIM, out_dim)
    def forward(self, x):
        return self.proj(x)  # (B, 49, D_MODEL)


# ── AdaLN ─────────────────────────────────────────────
class AdaLN(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, d_model * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, c):
        scale, shift = self.proj(c).chunk(2, dim=-1)
        return (1 + scale.unsqueeze(1)) * self.norm(x) + shift.unsqueeze(1)


# ── Transformer Layer with self-attn + cross-attn + 2×AdaLN ──────
class V4TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, cond_dim):
        super().__init__()
        # self-attention
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # cross-attention (canvas clip patch)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # FFN
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )
        # AdaLN for canvas dinov2
        self.adaln_canvas1 = AdaLN(d_model, cond_dim)
        self.adaln_canvas2 = AdaLN(d_model, cond_dim)
        # AdaLN for text
        self.adaln_text1   = AdaLN(d_model, cond_dim)
        self.adaln_text2   = AdaLN(d_model, cond_dim)
        # norm for cross-attn
        self.norm_cross    = nn.LayerNorm(d_model)

    def forward(self, x, c_canvas, c_text, kv_patch, attn_mask=None):
        # 1. self-attention with AdaLN
        x2 = self.adaln_canvas1(x, c_canvas) + self.adaln_text1(x, c_text)
        x2, _ = self.self_attn(x2, x2, x2, attn_mask=attn_mask, is_causal=False)
        x = x + x2

        # 2. cross-attention to canvas clip patch
        x2 = self.norm_cross(x)
        x2, _ = self.cross_attn(x2, kv_patch, kv_patch)
        x = x + x2

        # 3. FFN with AdaLN
        x2 = self.adaln_canvas2(x, c_canvas) + self.adaln_text2(x, c_text)
        x2 = self.ff(x2)
        return x + x2


class V4Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, cond_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            V4TransformerLayer(d_model, nhead, dim_feedforward, dropout, cond_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, c_canvas, c_text, kv_patch, attn_mask=None):
        for layer in self.layers:
            x = layer(x, c_canvas, c_text, kv_patch, attn_mask=attn_mask)
        return x


# ── Sinusoidal time embedding ─────────────────────────
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args  = t * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ── FM Head (next stroke 8维) ─────────────────────────
class FMHead(nn.Module):
    def __init__(self, h_dim=D_MODEL, a_dim=STROKE_DIM, t_emb_dim=32):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(t_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(h_dim + a_dim + t_emb_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, a_dim),
        )

    def forward(self, h, at, t):
        return self.net(torch.cat([h, at, self.time_emb(t)], dim=-1))

    @torch.no_grad()
    def sample(self, h, steps=FM_STEPS):
        B = h.shape[0]; device = h.device
        a = torch.randn(B, STROKE_DIM, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B, 1), i * dt + EPS, device=device)
            a = a + self.forward(h, a, t) * dt
        return a.clamp(0.0, 1.0)


# ── FM Loss ───────────────────────────────────────────
def fm_loss(fm_head, h, a_tar):
    B, device = h.shape[0], h.device
    a_src = torch.randn_like(a_tar)
    t     = torch.empty(B, 1, device=device).uniform_(EPS, 1 - EPS)
    at    = (1 - t) * a_src + t * a_tar
    u     = a_tar - a_src
    return F.mse_loss(fm_head(h, at, t), u)


# ── Main Model ────────────────────────────────────────
class StrokeARFMv4(nn.Module):
    def __init__(self):
        super().__init__()
        # encoders
        self.canvas_dino_enc  = CanvasDINOEncoder(CANVAS_FEAT)
        self.text_enc         = TextEncoder(CANVAS_FEAT)
        self.clip_patch_proj  = ClipPatchProjector(D_MODEL)

        # stroke embedding
        self.stroke_proj = nn.Linear(STROKE_DIM, D_MODEL)
        self.pos_emb     = nn.Embedding(WINDOW, D_MODEL)

        # transformer
        self.transformer = V4Transformer(
            d_model=D_MODEL, nhead=N_HEADS, num_layers=N_LAYERS,
            dim_feedforward=D_MODEL*4, dropout=DROPOUT, cond_dim=CANVAS_FEAT,
        )
        self.out_norm = nn.LayerNorm(D_MODEL)

        # prediction heads
        self.fm_head       = FMHead(h_dim=D_MODEL, a_dim=STROKE_DIM)
        self.label_head    = nn.Linear(D_MODEL, LABEL_MAX + 1)     # 15分类
        self.normal_head   = nn.Linear(D_MODEL, 3)                  # xyz
        self.landmark_head = nn.Linear(D_MODEL, N_LANDMARKS)        # 136
        self.dino_head     = nn.Linear(D_MODEL, 384)                # target dinov2 cls

    def encode(self, strokes, canvas_dino, canvas_clip_patch, text_feat):
        B, T, _ = strokes.shape
        c_canvas  = self.canvas_dino_enc(canvas_dino)          # (B, CANVAS_FEAT)
        c_text    = self.text_enc(text_feat)                   # (B, CANVAS_FEAT)
        kv_patch  = self.clip_patch_proj(canvas_clip_patch)    # (B, 49, D_MODEL)

        pos = torch.arange(T, device=strokes.device)
        x   = self.stroke_proj(strokes) + self.pos_emb(pos)   # (B, T, D_MODEL)

        mask = nn.Transformer.generate_square_subsequent_mask(T, device=strokes.device)
        x    = self.transformer(x, c_canvas, c_text, kv_patch, attn_mask=mask)
        x    = self.out_norm(x)
        return x[:, -1, :]  # (B, D_MODEL)

    def forward(self, strokes, canvas_dino, canvas_clip_patch, text_feat,
                a_tar, label_tar, normal_tar, landmark_tar, dino_tar):
        h = self.encode(strokes, canvas_dino, canvas_clip_patch, text_feat)

        loss_fm       = fm_loss(self.fm_head, h, a_tar) * W_FM
        loss_label    = F.cross_entropy(self.label_head(h), label_tar) * W_LABEL
        loss_normal   = F.mse_loss(torch.sigmoid(self.normal_head(h)), normal_tar) * W_NORMAL
        loss_landmark = F.mse_loss(torch.sigmoid(self.landmark_head(h)), landmark_tar) * W_LANDMARK
        loss_dino     = F.mse_loss(self.dino_head(h), dino_tar) * W_DINO

        total = loss_fm + loss_label + loss_normal + loss_landmark + loss_dino
        return total, {
            "fm": loss_fm.item(),
            "label": loss_label.item(),
            "normal": loss_normal.item(),
            "landmark": loss_landmark.item(),
            "dino": loss_dino.item(),
        }

    @torch.no_grad()
    def sample(self, strokes, canvas_dino, canvas_clip_patch, text_feat, steps=FM_STEPS):
        h = self.encode(strokes, canvas_dino, canvas_clip_patch, text_feat)
        return self.fm_head.sample(h, steps=steps)


# ── LR schedule ───────────────────────────────────────
def get_lr(epoch, warmup=WARMUP, total=EPOCHS, lr_max=LR, lr_min=0.0):
    if epoch <= warmup:
        return lr_max * epoch / warmup
    progress = (epoch - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ── Train ─────────────────────────────────────────────
def train():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device    = torch.device(f"cuda:{local_rank}")
    is_master = local_rank == 0

    if is_master:
        print(f"STROKE_DIM={STROKE_DIM}  WINDOW={WINDOW}  D_MODEL={D_MODEL}")

    dataset = StrokeDataset(DATA_ROOT)
    if is_master:
        print(f"Dataset: {len(dataset)} samples")

    sampler = DistributedSampler(dataset, shuffle=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         sampler=sampler, num_workers=8, pin_memory=True)

    model = StrokeARFMv4().to(device)

    start_epoch = 1
    best_loss   = float("inf")
    V3_CKPT = "model_fm_v3_.pt"

    if os.path.exists(BASE_CKPT):
        # 直接加载v4 checkpoint继续训
        if is_master:
            print(f"Loading v4 checkpoint {BASE_CKPT}...")
        ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss   = ckpt.get("loss", float("inf"))
        if is_master:
            print(f"Resumed epoch {start_epoch-1}, best_loss={best_loss:.6f}")

    elif os.path.exists(V3_CKPT):
        # 用v3权重初始化，名字不同的手动映射
        if is_master:
            print(f"No v4 checkpoint found. Initializing from v3 {V3_CKPT}...")
        v3_ckpt = torch.load(V3_CKPT, map_location=device, weights_only=False)
        v3_state = v3_ckpt["model_state"]

        # 把v3的canvas_encoder映射到v4的canvas_dino_enc
        remapped = {}
        for k, v in v3_state.items():
            if k.startswith("canvas_encoder."):
                new_k = k.replace("canvas_encoder.", "canvas_dino_enc.")
                remapped[new_k] = v
            elif k.startswith("transformer.layers."):
                # transformer层：self_attn/ff/adaln1/adaln2 key都一样
                remapped[k] = v
            else:
                remapped[k] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if is_master:
            print(f"  Loaded from v3: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"  Missing (new in v4): {missing[:5]}...")
        start_epoch = 1
        best_loss   = float("inf")

    else:
        if is_master:
            print("No checkpoint found. Training from scratch.")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    if is_master:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

    for epoch in range(start_epoch, EPOCHS + 1):
        lr_now = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        sampler.set_epoch(epoch)
        t0         = time.time()
        model.train()
        epoch_loss = 0.0
        n_samples  = 0
        loss_dict_acc = {"fm": 0., "label": 0., "normal": 0., "landmark": 0., "dino": 0.}

        for batch in loader:
            (strokes, canvas_dino, canvas_clip, text_feat,
             target, label, normal, landmark, dino_tar) = [b.to(device) for b in batch]

            loss, loss_dict = model(strokes, canvas_dino, canvas_clip, text_feat,
                                    target, label, normal, landmark, dino_tar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = strokes.size(0)
            epoch_loss += loss.item() * bs
            n_samples  += bs
            for k in loss_dict_acc:
                loss_dict_acc[k] += loss_dict[k] * bs

        epoch_loss /= n_samples
        for k in loss_dict_acc:
            loss_dict_acc[k] /= n_samples
        elapsed = time.time() - t0

        if is_master:
            if epoch % 10 == 0 or epoch == start_epoch:
                parts = " ".join([f"{k}={v:.4f}" for k, v in loss_dict_acc.items()])
                print(f"epoch {epoch:4d}  loss={epoch_loss:.6f}  [{parts}]  lr={lr_now:.2e}  {elapsed:.1f}s")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    "epoch":           epoch,
                    "model_state":     model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss":            best_loss,
                    "config": {
                        "WINDOW":         WINDOW,
                        "STRIDE":         STRIDE,
                        "D_MODEL":        D_MODEL,
                        "N_HEADS":        N_HEADS,
                        "N_LAYERS":       N_LAYERS,
                        "STROKE_DIM":     STROKE_DIM,
                        "CANVAS_FEAT":    CANVAS_FEAT,
                        "TEXT_FEAT":      TEXT_FEAT,
                        "CLIP_PATCH_DIM": CLIP_PATCH_DIM,
                        "CLIP_PATCH_N":   CLIP_PATCH_N,
                        "FM_STEPS":       FM_STEPS,
                        "LABEL_MAX":      LABEL_MAX,
                        "N_LANDMARKS":    N_LANDMARKS,
                    }
                }, CKPT)

    if is_master:
        print(f"Done. Best loss={best_loss:.6f}, saved to {CKPT}")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()