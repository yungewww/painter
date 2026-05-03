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
LR          = 1e-5
EPOCHS      = 500
BATCH_SIZE  = 1024
CANVAS_SIZE = (224, 224)

BRUSH_MAX   = 436

FEAT_ROOT   = "DinoFeats"
DATA_ROOT   = STROKESFLAT_DIR
BASE_CKPT   = "base_model_cfm.pt"
CKPT        = "base_model_cfm.pt"

STROKE_DIM  = 8
CANVAS_FEAT = 128

# ── Consistency FM 超参 ────────────────────────────────
CFM_K       = 2       # 分段数
CFM_ALPHA   = 1.0     # 速度一致性损失权重
CFM_DELTA_T = 0.01    # 时间间隔 ∆t
EPS         = 1e-4    # 时间范围 [eps, 1-eps]
CFM_STEPS   = 1       # 推理步数（1=单步，>1=多步ODE）

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


# ── Consistency Flow Matching Head ────────────────────
#
# 替换原来的 MDNHead。
# 输入：Transformer隐状态 h (B, D_MODEL)
#        插值后的动作   at (B, STROKE_DIM)
#        时间           t  (B, 1)
# 输出：速度场预测 v_θ(t, at | h)  (B, STROKE_DIM)
#
# 用正弦时间嵌入，将 h、at、t 拼接后过MLP。

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
        args = t * freqs.unsqueeze(0)          # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return emb


class CFMHead(nn.Module):
    """
    给定条件隐状态 h，预测速度场 v_θ(t, at | h)。
    结构：MLP([h; at; t_emb]) → STROKE_DIM
    """
    def __init__(self, h_dim=D_MODEL, a_dim=STROKE_DIM, t_emb_dim=32, hidden=256):
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
        t_emb = self.time_emb(t)           # (B, t_emb_dim)
        x = torch.cat([h, at, t_emb], dim=-1)
        return self.net(x)                 # (B, a_dim)

    @torch.no_grad()
    def sample(self, h, steps=CFM_STEPS):
        """
        从高斯噪声单步（或多步ODE）生成动作。
        steps=1 对应 Consistency-FM 单步推理。
        """
        B = h.shape[0]
        device = h.device
        a = torch.randn(B, STROKE_DIM, device=device)   # a0 ~ N(0,I)

        dt = 1.0 / steps
        for i in range(steps):
            t_val = i * dt + EPS
            t = torch.full((B, 1), t_val, device=device)
            v = self.forward(h, a, t)
            a = a + v * dt

        return a.clamp(0.0, 1.0)


# ── Consistency FM Loss ───────────────────────────────

def cfm_loss(cfm_head, h, a_tar, K=CFM_K, alpha=CFM_ALPHA, delta_t=CFM_DELTA_T):
    """
    两段 Consistency Flow Matching 损失（对应论文公式5）。

    对每段 i ∈ {0, ..., K-1}：
      - 时间均匀采样自 [i/K, (i+1)/K - ∆t]
      - 计算插值点 at, at+∆t
      - 预测速度并施加一致性约束

    h     : (B, D_MODEL)   — Transformer输出（已detach，梯度只走cfm_head）
    a_tar : (B, STROKE_DIM) — 目标动作 a1
    """
    B, device = h.shape[0], h.device
    h = h.detach()  # 梯度只走cfm_head，不回传到Transformer
    a_src = torch.randn_like(a_tar)   # a0 ~ N(0,I)

    losses = []

    # 段权重：中间段更难，给更大权重
    seg_weights = []
    for i in range(K):
        w = 1.0 + float(i == K // 2)
        seg_weights.append(w)

    for i in range(K):
        t_lo = i / K
        t_hi = (i + 1) / K - delta_t

        # 均匀采样 t 和 t+∆t
        t_val = torch.empty(B, 1, device=device).uniform_(t_lo, t_hi)
        t_next = (t_val + delta_t).clamp(max=(i + 1) / K)

        # 线性插值
        at      = (1 - t_val)  * a_src + t_val  * a_tar
        at_next = (1 - t_next) * a_src + t_next * a_tar

        # 预测速度（当前参数 θ）
        v_t      = cfm_head(h, at,      t_val)

        # EMA参数预测（stop gradient）
        with torch.no_grad():
            v_t_next = cfm_head(h, at_next, t_next)

        # 预测终点 f^i_θ(t, at) = at + ((i+1)/K - t) * v
        seg_end = (i + 1) / K
        f_t      = at      + (seg_end - t_val)  * v_t
        f_t_next = at_next + (seg_end - t_next) * v_t_next

        # 一致性损失
        loss_consistency = F.mse_loss(f_t, f_t_next)

        # 速度一致性损失
        loss_velocity = F.mse_loss(v_t, v_t_next)

        lam = seg_weights[i]
        losses.append(lam * (loss_consistency + alpha * loss_velocity))

    return sum(losses) / K


# ── Stroke AR Model（主模型）─────────────────────────

class StrokeARCFM(nn.Module):
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
        self.cfm_head = CFMHead(h_dim=D_MODEL, a_dim=STROKE_DIM)
        self.null_region = nn.Parameter(torch.zeros(D_MODEL))
        self.null_text   = nn.Parameter(torch.zeros(D_MODEL))

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
        """训练时调用，返回 CFM loss"""
        h = self.encode(strokes, canvas_feat)
        return cfm_loss(self.cfm_head, h, a_tar)

    @torch.no_grad()
    def sample(self, strokes, canvas_feat, steps=CFM_STEPS):
        """推理时调用，返回生成的下一笔动作"""
        h = self.encode(strokes, canvas_feat)
        return self.cfm_head.sample(h, steps=steps)


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

    model = StrokeARCFM().to(device)

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

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

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
                        "CFM_K": CFM_K, "CFM_ALPHA": CFM_ALPHA,
                        "CFM_STEPS": CFM_STEPS,
                    }
                }, CKPT)

    if is_master:
        print(f"Done. Best loss={best_loss:.6f}, saved to {CKPT}")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()