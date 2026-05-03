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

# ── 解压 ──────────────────────────────────────────────
DINOFEATS_TAR   = "DinoFeats.tar"
STROKESFLAT_TAR = "StrokesFlat_.tar"
STROKESFLAT_DIR = "StrokesFlat"

def extract_if_needed():
    import tarfile
    if not os.path.exists("DinoFeats"):
        print(f"Extracting {DINOFEATS_TAR}...")
        with tarfile.open(DINOFEATS_TAR, 'r') as t:
            t.extractall(".")
        print("Done.")
    else:
        print("DinoFeats already extracted.")

    if not os.path.exists(STROKESFLAT_DIR):
        print(f"Extracting {STROKESFLAT_TAR}...")
        with tarfile.open(STROKESFLAT_TAR, 'r') as t:
            t.extractall(".")
        print("Done.")
    else:
        print("StrokesFlat already extracted.")

# ── 超参 ──────────────────────────────────────────────
WINDOW      = 30
STRIDE      = 1
D_MODEL     = 256
N_HEADS     = 8
N_LAYERS    = 6
DROPOUT     = 0.1
LR          = 1e-4
WARMUP      = 10        # warmup epochs
EPOCHS      = 500
BATCH_SIZE  = 1024
CANVAS_SIZE = (224, 224)

BRUSH_MAX   = 436

FEAT_ROOT   = "DinoFeats"
DATA_ROOT   = STROKESFLAT_DIR
BASE_CKPT   = "base_model_fm_v2.pt"
CKPT        = "base_model_fm_v2.pt"

STROKE_DIM  = 8
CANVAS_FEAT = 256       # 扩大，给AdaLN更多容量

# ── Flow Matching 超参 ─────────────────────────────────
EPS      = 1e-4
FM_STEPS = 1

# ─────────────────────────────────────────────────────

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


class StrokeDataset(Dataset):
    def __init__(self, data_root, feat_root, window=WINDOW, stride=STRIDE):
        self.samples = []
        json_paths = sorted(glob.glob(os.path.join(data_root, "*.json")))
        for json_path in json_paths:
            painting_id = os.path.splitext(os.path.basename(json_path))[0]
            feat_path = os.path.join(feat_root, f"{painting_id}.npy")
            if not os.path.exists(feat_path):
                continue
            feats = np.load(feat_path)
            with open(json_path) as f:
                doc = json.load(f)
            W, H = doc["canvas_w"], doc["canvas_h"]
            strokes = [encode_stroke(s, W, H) for s in doc["strokes"]
                       if not s.get("undone", False)]
            N = len(strokes)
            for i in range(0, N - window, stride):
                seq = strokes[i: i + window]
                target = strokes[i + window]
                frame_idx = min(i + window - 1, len(feats) - 1)
                canvas_feat = feats[frame_idx]
                self.samples.append((
                    torch.tensor(seq, dtype=torch.float32),
                    torch.tensor(canvas_feat, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Canvas Encoder ────────────────────────────────────

class CanvasEncoder(nn.Module):
    def __init__(self, out_dim=CANVAS_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(384, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.net(x)   # (B, CANVAS_FEAT)


# ── AdaLN：用canvas_feat调制每层LayerNorm ────────────
#
# 给定条件向量 c，输出 scale 和 shift：
#   y = scale * LayerNorm(x) + shift
#
# 这让 canvas_feat 直接影响每一层的激活分布，
# 而不是只作为序列中的一个token被忽略。

class AdaLN(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, d_model * 2)  # → scale, shift
        # 初始化为恒等变换（scale=1, shift=0）
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, c):
        """
        x : (B, T, d_model)
        c : (B, cond_dim)
        """
        scale, shift = self.proj(c).chunk(2, dim=-1)   # 各 (B, d_model)
        scale = scale.unsqueeze(1)                      # (B, 1, d_model)
        shift = shift.unsqueeze(1)
        return (1 + scale) * self.norm(x) + shift


# ── 带AdaLN的Transformer层 ───────────────────────────

class AdaLNTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, cond_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.adaln1 = AdaLN(d_model, cond_dim)
        self.adaln2 = AdaLN(d_model, cond_dim)

    def forward(self, x, c, attn_mask=None):
        # Self-attention with AdaLN pre-norm
        x2 = self.adaln1(x, c)
        x2, _ = self.self_attn(x2, x2, x2, attn_mask=attn_mask, is_causal=False)
        x = x + x2

        # FFN with AdaLN pre-norm
        x2 = self.adaln2(x, c)
        x2 = self.ff(x2)
        x = x + x2

        return x


class AdaLNTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, cond_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            AdaLNTransformerLayer(d_model, nhead, dim_feedforward, dropout, cond_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, c, attn_mask=None):
        for layer in self.layers:
            x = layer(x, c, attn_mask=attn_mask)
        return x


# ── 正弦时间嵌入 ──────────────────────────────────────

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


# ── Flow Matching Head ────────────────────────────────

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


# ── Flow Matching Loss ────────────────────────────────

def fm_loss(fm_head, h, a_tar):
    B, device = h.shape[0], h.device
    a_src = torch.randn_like(a_tar)
    t = torch.empty(B, 1, device=device).uniform_(EPS, 1 - EPS)
    at = (1 - t) * a_src + t * a_tar
    u  = a_tar - a_src
    v_pred = fm_head(h, at, t)
    return F.mse_loss(v_pred, u)


# ── Stroke AR Model（主模型）─────────────────────────

class StrokeARFMv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.canvas_encoder = CanvasEncoder(CANVAS_FEAT)
        self.stroke_proj = nn.Linear(STROKE_DIM, D_MODEL)
        self.pos_emb = nn.Embedding(WINDOW, D_MODEL)

        # AdaLN Transformer：canvas_feat 作为条件注入每一层
        self.transformer = AdaLNTransformer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            num_layers=N_LAYERS,
            dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT,
            cond_dim=CANVAS_FEAT,
        )
        self.out_norm = nn.LayerNorm(D_MODEL)
        self.fm_head = FMHead(h_dim=D_MODEL, a_dim=STROKE_DIM)

    def encode(self, strokes, canvas_feat):
        """
        strokes     : (B, T, STROKE_DIM)
        canvas_feat : (B, 384)
        返回 h      : (B, D_MODEL)
        """
        B, T, _ = strokes.shape

        # 编码canvas条件
        c = self.canvas_encoder(canvas_feat)   # (B, CANVAS_FEAT)

        # 编码笔触序列
        pos = torch.arange(T, device=strokes.device)
        x = self.stroke_proj(strokes) + self.pos_emb(pos)   # (B, T, D_MODEL)

        # causal mask（让每个位置只看前面的笔触）
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=strokes.device)

        # AdaLN Transformer：c 注入每一层
        x = self.transformer(x, c, attn_mask=mask)   # (B, T, D_MODEL)
        x = self.out_norm(x)

        return x[:, -1, :]   # (B, D_MODEL) 取最后一步

    def forward(self, strokes, canvas_feat, a_tar):
        h = self.encode(strokes, canvas_feat)
        return fm_loss(self.fm_head, h, a_tar)

    @torch.no_grad()
    def sample(self, strokes, canvas_feat, steps=FM_STEPS):
        h = self.encode(strokes, canvas_feat)
        return self.fm_head.sample(h, steps=steps)


# ── LR Warmup + Cosine ───────────────────────────────

def get_lr(epoch, warmup=WARMUP, total=EPOCHS, lr_max=LR, lr_min=0.0):
    if epoch <= warmup:
        return lr_max * epoch / warmup
    progress = (epoch - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ── 训练 ──────────────────────────────────────────────

def train():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_master = local_rank == 0

    if is_master:
        extract_if_needed()
    dist.barrier()

    dataset = StrokeDataset(DATA_ROOT, FEAT_ROOT)
    if is_master:
        print(f"Dataset: {len(dataset)} samples")

    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        sampler=sampler, num_workers=16, pin_memory=True)

    model = StrokeARFMv2().to(device)

    start_epoch = 1
    best_loss = float("inf")
    if os.path.exists(BASE_CKPT):
        if is_master:
            print(f"Loading checkpoint from {BASE_CKPT}...")
        ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("loss", float("inf"))
        if is_master:
            print(f"Resumed from epoch {start_epoch - 1}, best_loss={best_loss:.6f}")
    else:
        if is_master:
            print("No checkpoint found, training from scratch.")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    if is_master:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

    for epoch in range(start_epoch, EPOCHS + 1):
        # 手动设置lr（warmup + cosine）
        lr_now = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        sampler.set_epoch(epoch)
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        for strokes, canvas_feat, target in loader:
            strokes     = strokes.to(device)
            canvas_feat = canvas_feat.to(device)
            target      = target.to(device)

            loss = model(strokes, canvas_feat, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * strokes.size(0)
            n_samples  += strokes.size(0)

        epoch_loss /= n_samples
        elapsed = time.time() - t0

        if is_master:
            if epoch % 10 == 0 or epoch == start_epoch:
                print(f"epoch {epoch:4d}  loss={epoch_loss:.6f}  lr={lr_now:.2e}  {elapsed:.1f}s")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    "epoch": epoch,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": {
                        "WINDOW": WINDOW, "D_MODEL": D_MODEL,
                        "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS,
                        "CANVAS_SIZE": CANVAS_SIZE,
                        "FM_STEPS": FM_STEPS,
                        "CANVAS_FEAT": CANVAS_FEAT,
                    }
                }, CKPT)

    if is_master:
        print(f"Done. Best loss={best_loss:.6f}, saved to {CKPT}")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
