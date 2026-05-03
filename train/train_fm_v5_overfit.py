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
N_LAYERS_S1 = 3
N_LAYERS_S2 = 3
DROPOUT     = 0.1
LR          = 1e-4
WARMUP      = 10
EPOCHS      = 500
BATCH_SIZE  = 512
CANVAS_FEAT = 256
TEXT_FEAT   = 512
CLIP_PATCH_DIM = 512
CLIP_PATCH_N   = 49

STROKE_DIM  = 8
LABEL_MAX   = 14
N_LANDMARKS = 136

# loss weights — stage1
W_DINO     = 0.1
W_LABEL    = 0.1
W_NORMAL   = 0.1
W_LANDMARK = 0.1
# loss weights — stage2
W_FM       = 1.0

STAGE2_START_EPOCH = 150   # 前N epoch只训stage1，之后stage1+stage2端到端联合训
EPS      = 1e-4
FM_STEPS = 1

DATA_ROOT  = "output"
BASE_CKPT  = "model_fm_v5_s2best.pt"  # 从s2best初始化
CKPT       = "model_fm_v5_overfit.pt"


# ── encode_stroke ─────────────────────────────────────
def encode_stroke(s, W, H):
    p        = s["samples"][0]
    x_norm   = p["x"] / W
    y_norm   = p["y"] / H
    press    = p["pressure"]
    bs_norm  = s["brush_size"] / max(W, H)
    ang_norm = (s["angle"] + math.pi) / (2 * math.pi)
    r, g, b  = s["color"]
    return [x_norm, y_norm, press, bs_norm, ang_norm, r, g, b]


# ── Dataset (lazy loading) ────────────────────────────
class StrokeDataset(Dataset):
    """
    Lazy loading版本：只在__init__时记录索引，
    __getitem__时才读文件，大幅降低RAM占用。
    """
    def __init__(self, data_root, window=WINDOW, stride=STRIDE):
        # self.index: list of (stem, i, frame_idx, label_id, normal)
        self.index     = []
        self.data_root = data_root
        self.window    = window

        all_json = sorted(glob.glob(os.path.join(data_root, "*", "strokes.json")))
        json_paths = [p for p in all_json
                      if os.path.basename(os.path.dirname(p)) in
                      [f"{i:04d}" for i in range(1, 17)]]
        print(f"  Found {len(json_paths)} stroke files.")
        skipped = 0

        for json_path in json_paths:
            stem          = os.path.basename(os.path.dirname(json_path))
            feat_path     = os.path.join(data_root, stem, "feats.npy")
            text_path     = os.path.join(data_root, stem, "text.npy")
            clip_path     = os.path.join(data_root, stem, "clip_patch_feats.npy")
            dino_tgt_path = os.path.join(data_root, stem, "target_dino_cls.npy")
            lmk_path      = os.path.join(data_root, stem, "face_landmarks.npy")

            for p in [feat_path, text_path, clip_path, dino_tgt_path, lmk_path]:
                if not os.path.exists(p):
                    skipped += 1; break
            else:
                try:
                    # 只读shape/长度，不加载内容
                    clip_patches = np.load(clip_path, mmap_mode="r")
                    if clip_patches.ndim != 3 or clip_patches.shape[1:] != (49, 512):
                        skipped += 1; continue
                    feats_len = np.load(feat_path, mmap_mode="r").shape[0]
                except Exception:
                    skipped += 1; continue

                with open(json_path) as f:
                    doc = json.load(f)
                W, H        = doc["canvas_w"], doc["canvas_h"]
                strokes_raw = [s for s in doc["strokes"] if not s.get("undone", False)]
                strokes_enc = [encode_stroke(s, W, H) for s in strokes_raw]
                N = len(strokes_enc)
                if N <= window:
                    skipped += 1; continue

                for i in range(0, N - window, stride):
                    raw_s     = strokes_raw[i + window]
                    frame_idx = min(i + window - 1, feats_len - 1)
                    label_id  = raw_s.get("label_id", 0)
                    nx, ny, nz = raw_s.get("normal", [0.0, 0.0, 1.0])
                    normal    = [(nx+1)/2, (ny+1)/2, (nz+1)/2]
                    # 把stroke序列和target也存下来（轻量，只是float list）
                    self.index.append((
                        stem, i, frame_idx, label_id, normal,
                        strokes_enc[i: i+window],   # list of list, 轻量
                        strokes_enc[i+window],       # list
                        W, H,
                    ))

        print(f"  Skipped {skipped}, loaded {len(self.index)} samples.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        stem, i, frame_idx, label_id, normal, seq, target_s, W, H = self.index[idx]

        # 按需读文件
        feats          = np.load(os.path.join(self.data_root, stem, "feats.npy"),
                                 mmap_mode="r")
        clip_patches   = np.load(os.path.join(self.data_root, stem, "clip_patch_feats.npy"),
                                 mmap_mode="r")
        text_feat      = np.load(os.path.join(self.data_root, stem, "text.npy"))
        target_dino    = np.load(os.path.join(self.data_root, stem, "target_dino_cls.npy"))
        face_landmarks = np.load(os.path.join(self.data_root, stem, "face_landmarks.npy"))

        return (
            torch.tensor(seq,                          dtype=torch.float32),  # (50,8)
            torch.tensor(feats[frame_idx].copy(),      dtype=torch.float32),  # (384,)
            torch.tensor(clip_patches[frame_idx].copy(),dtype=torch.float32), # (49,512)
            torch.tensor(text_feat,                    dtype=torch.float32),  # (512,)
            torch.tensor(target_s,                     dtype=torch.float32),  # (8,)
            torch.tensor(label_id,                     dtype=torch.long),
            torch.tensor(normal,                       dtype=torch.float32),  # (3,)
            torch.tensor(face_landmarks,               dtype=torch.float32),  # (136,)
            torch.tensor(target_dino,                  dtype=torch.float32),  # (384,)
        )


# ── Shared Encoders ───────────────────────────────────
class CanvasDINOEncoder(nn.Module):
    """canvas DINOv2 CLS → AdaLN condition (B, CANVAS_FEAT)"""
    def __init__(self, out_dim=CANVAS_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(384, 512), nn.SiLU(),
            nn.Linear(512, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class TextEncoder(nn.Module):
    """CLIP text → AdaLN condition (B, CANVAS_FEAT)"""
    def __init__(self, out_dim=CANVAS_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TEXT_FEAT, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class ClipPatchProjector(nn.Module):
    """CLIP patch tokens → cross-attn KV (B, 49, D_MODEL)"""
    def __init__(self, out_dim=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(CLIP_PATCH_DIM, out_dim)
    def forward(self, x):
        return self.proj(x)


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


# ── Stage 1 Transformer Layer (no cross-attn) ─────────
class Stage1TransformerLayer(nn.Module):
    """Self-attn + FFN，条件：AdaLN(canvas_dino) + AdaLN(text)"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, cond_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )
        self.adaln_canvas1 = AdaLN(d_model, cond_dim)
        self.adaln_text1   = AdaLN(d_model, cond_dim)
        self.adaln_canvas2 = AdaLN(d_model, cond_dim)
        self.adaln_text2   = AdaLN(d_model, cond_dim)

    def forward(self, x, c_canvas, c_text, attn_mask=None):
        x2 = self.adaln_canvas1(x, c_canvas) + self.adaln_text1(x, c_text)
        x2, _ = self.self_attn(x2, x2, x2, attn_mask=attn_mask, is_causal=False)
        x = x + x2
        x2 = self.adaln_canvas2(x, c_canvas) + self.adaln_text2(x, c_text)
        x2 = self.ff(x2)
        return x + x2


class Stage1Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, cond_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            Stage1TransformerLayer(d_model, nhead, dim_feedforward, dropout, cond_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, c_canvas, c_text, attn_mask=None):
        for layer in self.layers:
            x = layer(x, c_canvas, c_text, attn_mask=attn_mask)
        return x


# ── Stage 2 Transformer Layer (with cross-attn) ───────
class Stage2TransformerLayer(nn.Module):
    """Self-attn + Cross-attn(CLIP patch) + FFN
    条件：AdaLN(stage1_h) — stage1的hidden state作为单一condition
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, cond_dim):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )
        self.adaln1     = AdaLN(d_model, cond_dim)
        self.adaln2     = AdaLN(d_model, cond_dim)
        self.norm_cross = nn.LayerNorm(d_model)

    def forward(self, x, c_stage1, kv_patch, attn_mask=None):
        # self-attn conditioned on stage1 output
        x2 = self.adaln1(x, c_stage1)
        x2, _ = self.self_attn(x2, x2, x2, attn_mask=attn_mask, is_causal=False)
        x = x + x2
        # cross-attn on CLIP patch tokens
        x2 = self.norm_cross(x)
        x2, _ = self.cross_attn(x2, kv_patch, kv_patch)
        x = x + x2
        # FFN conditioned on stage1 output
        x2 = self.adaln2(x, c_stage1)
        x2 = self.ff(x2)
        return x + x2


class Stage2Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, cond_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            Stage2TransformerLayer(d_model, nhead, dim_feedforward, dropout, cond_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, c_stage1, kv_patch, attn_mask=None):
        for layer in self.layers:
            x = layer(x, c_stage1, kv_patch, attn_mask=attn_mask)
        return x


# ── Sinusoidal time embedding ──────────────────────────
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args  = t * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ── FM Head ───────────────────────────────────────────
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
        a  = torch.randn(B, STROKE_DIM, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B, 1), i * dt + EPS, device=device)
            a = a + self.forward(h, a, t) * dt
        return a.clamp(0.0, 1.0)


def fm_loss(fm_head, h, a_tar):
    B, device = h.shape[0], h.device
    a_src = torch.randn_like(a_tar)
    t     = torch.empty(B, 1, device=device).uniform_(EPS, 1 - EPS)
    at    = (1 - t) * a_src + t * a_tar
    u     = a_tar - a_src
    return F.mse_loss(fm_head(h, at, t), u)


# ── Stage 1 Encoder: stage1_h → condition for stage2 ─
class Stage1CondEncoder(nn.Module):
    """把stage1的hidden state (D_MODEL)映射到stage2的AdaLN condition (CANVAS_FEAT)"""
    def __init__(self, in_dim=D_MODEL, out_dim=CANVAS_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)


# ── Full Two-Stage Model ───────────────────────────────
class StrokeARFMv5(nn.Module):
    def __init__(self):
        super().__init__()

        # ── shared encoders ──
        self.canvas_dino_enc = CanvasDINOEncoder(CANVAS_FEAT)
        self.text_enc        = TextEncoder(CANVAS_FEAT)
        self.clip_patch_proj = ClipPatchProjector(D_MODEL)   # only used in stage2

        # ── shared stroke embedding ──
        self.stroke_proj = nn.Linear(STROKE_DIM, D_MODEL)
        self.pos_emb     = nn.Embedding(WINDOW, D_MODEL)

        # ── stage 1 ──
        self.s1_transformer = Stage1Transformer(
            d_model=D_MODEL, nhead=N_HEADS, num_layers=N_LAYERS_S1,
            dim_feedforward=D_MODEL*4, dropout=DROPOUT, cond_dim=CANVAS_FEAT,
        )
        self.s1_out_norm    = nn.LayerNorm(D_MODEL)
        # stage1 prediction heads
        self.s1_dino_head     = nn.Linear(D_MODEL, 384)          # predicted final DINOv2
        self.s1_label_head    = nn.Linear(D_MODEL, LABEL_MAX + 1) # this stroke label
        self.s1_normal_head   = nn.Linear(D_MODEL, 3)             # this stroke normal
        self.s1_landmark_head = nn.Linear(D_MODEL, N_LANDMARKS)   # final landmark

        # ── stage1 → stage2 condition ──
        self.s1_cond_enc = Stage1CondEncoder(D_MODEL, CANVAS_FEAT)

        # ── stage 2 ──
        self.s2_transformer = Stage2Transformer(
            d_model=D_MODEL, nhead=N_HEADS, num_layers=N_LAYERS_S2,
            dim_feedforward=D_MODEL*4, dropout=DROPOUT, cond_dim=CANVAS_FEAT,
        )
        self.s2_out_norm = nn.LayerNorm(D_MODEL)
        self.fm_head     = FMHead(h_dim=D_MODEL, a_dim=STROKE_DIM)

    def _embed_strokes(self, strokes):
        """共享stroke embedding，stage1和stage2复用"""
        B, T, _ = strokes.shape
        pos = torch.arange(T, device=strokes.device)
        return self.stroke_proj(strokes) + self.pos_emb(pos)  # (B, T, D_MODEL)

    def encode_stage1(self, strokes, canvas_dino, text_feat, x_emb=None):
        """Stage 1 encode：stroke history + canvas DINOv2 + text → h1"""
        c_canvas = self.canvas_dino_enc(canvas_dino)   # (B, CANVAS_FEAT)
        c_text   = self.text_enc(text_feat)             # (B, CANVAS_FEAT)

        x    = x_emb if x_emb is not None else self._embed_strokes(strokes)
        T    = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x    = self.s1_transformer(x, c_canvas, c_text, attn_mask=mask)
        x    = self.s1_out_norm(x)
        return x[:, -1, :]  # h1: (B, D_MODEL)

    def encode_stage2(self, canvas_clip_patch, c_stage1, x_emb):
        """Stage 2 encode：stroke embedding + CLIP patch + stage1 condition → h2"""
        kv_patch = self.clip_patch_proj(canvas_clip_patch)  # (B, 49, D_MODEL)
        T    = x_emb.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x_emb.device)
        x    = self.s2_transformer(x_emb, c_stage1, kv_patch, attn_mask=mask)
        x    = self.s2_out_norm(x)
        return x[:, -1, :]  # h2: (B, D_MODEL)

    def forward(self, strokes, canvas_dino, canvas_clip_patch, text_feat,
                a_tar, label_tar, normal_tar, landmark_tar, dino_tar,
                train_stage2=True):
        # ── 共享embedding，只算一次 ──
        x_emb = self._embed_strokes(strokes)

        # ── Stage 1 forward ──
        h1 = self.encode_stage1(strokes, canvas_dino, text_feat, x_emb=x_emb)

        loss_dino     = F.mse_loss(self.s1_dino_head(h1), dino_tar) * W_DINO
        loss_label    = F.cross_entropy(self.s1_label_head(h1), label_tar) * W_LABEL
        loss_normal   = F.mse_loss(torch.sigmoid(self.s1_normal_head(h1)), normal_tar) * W_NORMAL
        loss_landmark = F.mse_loss(torch.sigmoid(self.s1_landmark_head(h1)), landmark_tar) * W_LANDMARK

        loss_s1 = loss_dino + loss_label + loss_normal + loss_landmark

        if not train_stage2:
            # 只训stage1，跳过stage2
            return loss_s1, {
                "fm":       0.0,
                "dino":     loss_dino.item(),
                "label":    loss_label.item(),
                "normal":   loss_normal.item(),
                "landmark": loss_landmark.item(),
            }

        # ── epoch>=STAGE2_START_EPOCH：端到端，不detach，Stage 2梯度反传到Stage 1 ──
        c_stage1 = self.s1_cond_enc(h1)  # (B, CANVAS_FEAT)，不detach

        # ── Stage 2 forward ──
        h2 = self.encode_stage2(canvas_clip_patch, c_stage1, x_emb=x_emb)

        loss_fm = fm_loss(self.fm_head, h2, a_tar) * W_FM

        total = loss_s1 + loss_fm
        return total, {
            "fm":       loss_fm.item(),
            "dino":     loss_dino.item(),
            "label":    loss_label.item(),
            "normal":   loss_normal.item(),
            "landmark": loss_landmark.item(),
        }

    @torch.no_grad()
    def sample(self, strokes, canvas_dino, canvas_clip_patch, text_feat, steps=FM_STEPS):
        x_emb    = self._embed_strokes(strokes)
        h1       = self.encode_stage1(strokes, canvas_dino, text_feat, x_emb=x_emb)
        c_stage1 = self.s1_cond_enc(h1)

        # stage1 predictions
        dino_pred     = self.s1_dino_head(h1)
        label_pred    = self.s1_label_head(h1).argmax(dim=-1)
        normal_pred   = torch.sigmoid(self.s1_normal_head(h1))
        landmark_pred = torch.sigmoid(self.s1_landmark_head(h1))

        h2     = self.encode_stage2(canvas_clip_patch, c_stage1, x_emb=x_emb)
        stroke = self.fm_head.sample(h2, steps=steps)

        return stroke, label_pred, normal_pred, landmark_pred, dino_pred


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
        print(f"V5  WINDOW={WINDOW}  D_MODEL={D_MODEL}  two-stage + detach")

    dataset = StrokeDataset(DATA_ROOT)
    if is_master:
        print(f"Dataset: {len(dataset)} samples")

    sampler = DistributedSampler(dataset, shuffle=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                         num_workers=16, pin_memory=True, persistent_workers=True)

    model = StrokeARFMv5().to(device)

    start_epoch  = 1
    best_loss    = float("inf")  # stage1 only best
    best_loss_s2 = float("inf")  # stage2 best (fm included)

    if os.path.exists(CKPT):
        if is_master:
            print(f"Loading v5 checkpoint {CKPT}...")
        ckpt = torch.load(CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_loss    = ckpt.get("loss", float("inf"))
        best_loss_s2 = ckpt.get("loss_s2", float("inf"))
        if is_master:
            print(f"Resumed epoch {start_epoch-1}, best_loss={best_loss:.6f}")

    elif os.path.exists(BASE_CKPT):
        # 从v4初始化stage1的兼容权重
        if is_master:
            print(f"Initializing from v4 {BASE_CKPT}...")
        v4_ckpt  = torch.load(BASE_CKPT, map_location=device, weights_only=False)
        v4_state = v4_ckpt["model_state"]

        remapped = {}
        for k, v in v4_state.items():
            # canvas_dino_enc / text_enc / stroke_proj / pos_emb / fm_head 直接复用
            if any(k.startswith(p) for p in [
                "canvas_dino_enc.", "text_enc.", "stroke_proj.", "pos_emb.", "fm_head."
            ]):
                remapped[k] = v
            # transformer → s1_transformer（结构兼容：v4无cross-attn层在stage1里也没有）
            elif k.startswith("transformer."):
                new_k = k.replace("transformer.", "s1_transformer.")
                remapped[new_k] = v
            # out_norm → s1_out_norm
            elif k.startswith("out_norm."):
                new_k = k.replace("out_norm.", "s1_out_norm.")
                remapped[new_k] = v
            # prediction heads → s1 heads
            elif k.startswith("label_head."):
                remapped["s1_label_head." + k[len("label_head."):]] = v
            elif k.startswith("normal_head."):
                remapped["s1_normal_head." + k[len("normal_head."):]] = v
            elif k.startswith("landmark_head."):
                remapped["s1_landmark_head." + k[len("landmark_head."):]] = v
            elif k.startswith("dino_head."):
                remapped["s1_dino_head." + k[len("dino_head."):]] = v
            # clip_patch_proj 复用
            elif k.startswith("clip_patch_proj."):
                remapped[k] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if is_master:
            print(f"  Loaded from v4: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"  Missing (new in v5): {missing[:8]}")
    else:
        if is_master:
            print("No checkpoint found. Training from scratch.")

    model     = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    if is_master:
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(start_epoch, EPOCHS + 1):
        lr_now = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        sampler.set_epoch(epoch)
        t0            = time.time()
        model.train()
        epoch_loss    = 0.0
        n_samples     = 0
        loss_dict_acc = {"fm": 0., "dino": 0., "label": 0., "normal": 0., "landmark": 0.}

        train_stage2 = (epoch >= STAGE2_START_EPOCH)
        if is_master and epoch == STAGE2_START_EPOCH:
            print(f"epoch {epoch}: Stage 2 training starts.")

        for batch in loader:
            (strokes, canvas_dino, canvas_clip, text_feat,
             target, label, normal, landmark, dino_tar) = [b.to(device) for b in batch]

            loss, loss_dict = model(
                strokes, canvas_dino, canvas_clip, text_feat,
                target, label, normal, landmark, dino_tar,
                train_stage2=train_stage2,
            )

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

            cfg = {
                    "WINDOW": WINDOW, "STRIDE": STRIDE, "D_MODEL": D_MODEL,
                    "N_HEADS": N_HEADS, "N_LAYERS_S1": N_LAYERS_S1, "N_LAYERS_S2": N_LAYERS_S2,
                    "STROKE_DIM": STROKE_DIM, "CANVAS_FEAT": CANVAS_FEAT,
                    "TEXT_FEAT": TEXT_FEAT, "CLIP_PATCH_DIM": CLIP_PATCH_DIM,
                    "CLIP_PATCH_N": CLIP_PATCH_N, "FM_STEPS": FM_STEPS,
                    "LABEL_MAX": LABEL_MAX, "N_LANDMARKS": N_LANDMARKS,
                }
            payload = {
                "epoch":            epoch,
                "model_state":      model.module.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "loss":             best_loss,
                "loss_s2":          best_loss_s2,
                "config":           cfg,
            }

            if train_stage2:
                # stage2阶段：按fm loss保存最好的s2 pt
                fm_loss_now = loss_dict_acc["fm"]
                if fm_loss_now < best_loss_s2:
                    best_loss_s2 = fm_loss_now
                    payload["loss_s2"] = best_loss_s2
                    torch.save(payload, CKPT.replace(".pt", "_s2best.pt"))  # model_fm_v5_overfit_s2best.pt
                # stage1 loss也同步更新（保存完整状态）
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    payload["loss"] = best_loss
                    torch.save(payload, CKPT)
            else:
                # stage1阶段：只保存stage1 best
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(payload, CKPT)

    if is_master:
        print(f"Done. Best loss={best_loss:.6f}, saved to {CKPT}")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()