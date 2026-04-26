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
EPOCHS      = 500
BATCH_SIZE  = 1024
CANVAS_SIZE = (224, 224)

BRUSH_MAX   = 436

FEAT_ROOT   = "DinoFeats"
DATA_ROOT   = STROKESFLAT_DIR
BASE_CKPT   = "base_model_fm.pt"
CKPT        = "base_model_fm.pt"

STROKE_DIM  = 8
CANVAS_FEAT = 128

# ── Flow Matching 超参 ─────────────────────────────────
EPS      = 1e-4   # 时间范围 [eps, 1-eps]
FM_STEPS = 10     # 推理 ODE 步数

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
        self.proj = nn.Linear(384, out_dim)

    def forward(self, x):
        return self.proj(x)


# ── 正弦时间嵌入 ──────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B, 1) → (B, dim)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb


# ── Flow Matching Head ────────────────────────────────

class FMHead(nn.Module):
    """
    预测速度场 v_θ(t, at | h)。
    训练目标：v_θ ≈ a1 - a0（线性插值路径的真实速度，与 t 无关）。
    """
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
        """
        h  : (B, h_dim)
        at : (B, a_dim)
        t  : (B, 1)
        """
        t_emb = self.time_emb(t)
        x = torch.cat([h, at, t_emb], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def sample(self, h, steps=FM_STEPS):
        """Euler ODE 积分从 a0~N(0,I) 到 a1。"""
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
    """
    标准 Conditional Flow Matching 损失。
    线性插值路径：at = (1-t)*a0 + t*a1
    真实速度：    u = a1 - a0（常数，与 t 无关）
    损失：        E[ ||v_θ(t, at | h) - u||^2 ]
    """
    B, device = h.shape[0], h.device
    a_src = torch.randn_like(a_tar)                          # a0 ~ N(0,I)
    t = torch.empty(B, 1, device=device).uniform_(EPS, 1 - EPS)

    at = (1 - t) * a_src + t * a_tar                        # 插值点
    u  = a_tar - a_src                                       # 真实速度

    v_pred = fm_head(h, at, t)
    return F.mse_loss(v_pred, u)


# ── Stroke AR Model（主模型）─────────────────────────

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
        """返回 Transformer 最后一步的隐状态 h: (B, D_MODEL)"""
        B, T, _ = strokes.shape
        canvas_token = self.canvas_proj(self.canvas_encoder(canvas_feat)).unsqueeze(1)
        pos = torch.arange(T, device=strokes.device)
        stroke_h = self.stroke_proj(strokes) + self.pos_emb(pos)
        seq = torch.cat([canvas_token, stroke_h], dim=1)
        mask = nn.Transformer.generate_square_subsequent_mask(T + 1, device=strokes.device)
        mask[:, 0] = 0.0
        h = self.transformer(seq, mask=mask, is_causal=False)
        return h[:, -1, :]   # (B, D_MODEL)

    def forward(self, strokes, canvas_feat, a_tar):
        """训练时调用，返回 FM loss"""
        h = self.encode(strokes, canvas_feat)
        return fm_loss(self.fm_head, h, a_tar)

    @torch.no_grad()
    def sample(self, strokes, canvas_feat, steps=FM_STEPS):
        """推理时调用，返回生成的下一笔动作"""
        h = self.encode(strokes, canvas_feat)
        return self.fm_head.sample(h, steps=steps)


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

    model = StrokeARFM().to(device)

    start_epoch = 1
    best_loss = float("inf")
    if os.path.exists(BASE_CKPT):
        if is_master:
            print(f"Loading checkpoint from {BASE_CKPT}...")
        ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("loss", float("inf"))
        if is_master:
            print(f"Resumed from epoch {start_epoch - 1}, best_loss={best_loss:.6f}")
    else:
        if is_master:
            print("No checkpoint found, training from scratch.")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    for _ in range(start_epoch - 1):
        scheduler.step()

    if is_master:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

    for epoch in range(start_epoch, EPOCHS + 1):
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

        scheduler.step()
        epoch_loss /= n_samples
        elapsed = time.time() - t0

        if is_master:
            if epoch % 10 == 0 or epoch == start_epoch:
                lr_now = scheduler.get_last_lr()[0]
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
                    }
                }, CKPT)

    if is_master:
        print(f"Done. Best loss={best_loss:.6f}, saved to {CKPT}")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
