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
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ── 超参 ──────────────────────────────────────────────
WINDOW      = 50
STRIDE      = 1           # 每笔都看前50笔
D_MODEL     = 256
N_HEADS     = 8
N_LAYERS    = 6
DROPOUT     = 0.1
LR          = 1e-4
WARMUP      = 10
EPOCHS      = 500
BATCH_SIZE  = 16          # 单位是"画"，每个画独立维护KV
CANVAS_FEAT = 256
TEXT_FEAT   = 512
CLIP_PATCH_DIM = 512
CLIP_PATCH_N   = 49

STROKE_DIM  = 8
LABEL_MAX   = 14
N_LANDMARKS = 136

MAX_KV_WINDOWS = 3        # 保留最近N个window的KV（超参，训练=2或3，推理可设4）

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
CKPT      = "model_fm_v4_1.pt"


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


# ── Dataset ───────────────────────────────────────────
# v4.1: 按画组织，每个样本是一幅画的所有顺序window
# 每个画返回: list of (strokes_window, canvas_dino, canvas_clip, text_feat,
#                      target, label, normal, landmark, dino_tar)
# DataLoader collate按画为单位，训练loop内按window顺序迭代

class PaintingDataset(Dataset):
    """
    Lazy loading版本：__init__只记录索引和轻量数据，
    __getitem__时用mmap按需读取大npy文件，大幅降低RAM占用。
    每个item是一幅画的所有window，顺序排列。
    """
    def __init__(self, data_root, window=WINDOW, stride=STRIDE):
        # paintings: list of painting_meta
        # painting_meta: dict with stem, text_feat, target_dino, face_landmarks,
        #                feats_len, windows_meta
        # windows_meta: list of (seq, target_s, frame_idx, label_id, normal)
        self.paintings  = []
        self.data_root  = data_root
        self.window     = window

        all_json_paths = sorted(glob.glob(os.path.join(data_root, "*", "strokes.json")))
        # 只用前10张图（00001~00010）
        json_paths = [p for p in all_json_paths
                      if os.path.basename(os.path.dirname(p)) in
                      [f"{i:04d}" for i in range(1, 17)]]
        print(f"  Found {len(json_paths)} stroke files.")
        skipped = 0

        for json_path in tqdm(json_paths, desc="Loading paintings"):
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
                    clip_mmap  = np.load(clip_path, mmap_mode="r")
                    if clip_mmap.ndim != 3 or clip_mmap.shape[1:] != (49, 512):
                        skipped += 1; continue
                    feats_len  = np.load(feat_path, mmap_mode="r").shape[0]
                    text_feat  = np.load(text_path)           # 小，直接加载
                    target_dino    = np.load(dino_tgt_path)   # 小
                    face_landmarks = np.load(lmk_path)        # 小
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

                windows_meta = []
                for i in range(0, N - window, stride):
                    raw_s     = strokes_raw[i + window]
                    frame_idx = min(i + window - 1, feats_len - 1)
                    label_id  = raw_s.get("label_id", 0)
                    nx, ny, nz = raw_s.get("normal", [0.0, 0.0, 1.0])
                    normal    = [(nx+1)/2, (ny+1)/2, (nz+1)/2]
                    windows_meta.append((
                        strokes_enc[i: i+window],  # list of list, 轻量
                        strokes_enc[i+window],      # list
                        frame_idx,
                        label_id,
                        normal,
                    ))

                if windows_meta:
                    self.paintings.append({
                        "stem":           stem,
                        "text_feat":      text_feat,
                        "target_dino":    target_dino,
                        "face_landmarks": face_landmarks,
                        "windows_meta":   windows_meta,
                    })

        total_windows = sum(len(p["windows_meta"]) for p in self.paintings)
        print(f"  Skipped {skipped}, loaded {len(self.paintings)} paintings, "
              f"{total_windows} total windows.")

    def __len__(self):
        return len(self.paintings)

    def __getitem__(self, idx):
        meta = self.paintings[idx]
        stem = meta["stem"]

        # 用mmap按需读取大文件
        feats        = np.load(os.path.join(self.data_root, stem, "feats.npy"), mmap_mode="r")
        clip_patches = np.load(os.path.join(self.data_root, stem, "clip_patch_feats.npy"), mmap_mode="r")

        text_feat      = meta["text_feat"]
        target_dino    = meta["target_dino"]
        face_landmarks = meta["face_landmarks"]

        windows = []
        for seq, target_s, frame_idx, label_id, normal in meta["windows_meta"]:
            windows.append((
                torch.tensor(seq,                              dtype=torch.float32),
                torch.tensor(feats[frame_idx].copy(),         dtype=torch.float32),
                torch.tensor(clip_patches[frame_idx].copy(),  dtype=torch.float32),
                torch.tensor(text_feat,                       dtype=torch.float32),
                torch.tensor(target_s,                        dtype=torch.float32),
                torch.tensor(label_id,                        dtype=torch.long),
                torch.tensor(normal,                          dtype=torch.float32),
                torch.tensor(face_landmarks,                  dtype=torch.float32),
                torch.tensor(target_dino,                     dtype=torch.float32),
            ))
        return windows


def painting_collate_fn(batch):
    """
    batch: list of paintings, 每个painting是list of window-tuples
    返回: list of paintings, 每个painting已stack成tensor
    即 [(strokes(W,50,8), canvas_dino(W,384), ...), ...]
    """
    result = []
    for painting_windows in batch:
        # painting_windows: list of tuples, 每个tuple有9个tensor
        stacked = tuple(torch.stack([w[i] for w in painting_windows], dim=0)
                        for i in range(9))
        result.append(stacked)
    return result


# ── Bucket Distributed Sampler ────────────────────────
class BucketDistributedSampler(Sampler):
    """
    按画的window数量分桶，同一batch内画长度相近，减少KV padding浪费。
    支持DDP：每个rank只取自己的子集。
    bucket_size: 每桶跨度（window数），默认10
    """
    def __init__(self, dataset, batch_size, bucket_size=10,
                 num_replicas=1, rank=0, shuffle=True, seed=0):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.bucket_size  = bucket_size
        self.num_replicas = num_replicas
        self.rank         = rank
        self.shuffle      = shuffle
        self.seed         = seed
        self.epoch        = 0

        # 按window数量分桶
        buckets = defaultdict(list)
        for idx, painting in enumerate(dataset.paintings):
            n_win = len(painting)
            bid   = n_win // bucket_size
            buckets[bid].append(idx)
        self.buckets = list(buckets.values())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)

        batches = []
        for bucket in self.buckets:
            indices = list(bucket)
            if self.shuffle:
                perm    = torch.randperm(len(indices), generator=rng).tolist()
                indices = [indices[i] for i in perm]
            # 切成batch
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i: i + self.batch_size])

        # shuffle batch顺序
        if self.shuffle:
            perm    = torch.randperm(len(batches), generator=rng).tolist()
            batches = [batches[i] for i in perm]

        # DDP: 每个rank取自己的batch子集
        rank_batches = batches[self.rank::self.num_replicas]

        for batch in rank_batches:
            yield from batch

    def __len__(self):
        total = sum(
            math.ceil(len(b) / self.batch_size)
            for b in self.buckets
        )
        return math.ceil(total / self.num_replicas) * self.batch_size


# ── Encoders ──────────────────────────────────────────
class CanvasDINOEncoder(nn.Module):
    def __init__(self, out_dim=CANVAS_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(384, 512), nn.SiLU(),
            nn.Linear(512, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class TextEncoder(nn.Module):
    def __init__(self, out_dim=CANVAS_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TEXT_FEAT, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class ClipPatchProjector(nn.Module):
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


# ── Transformer Layer with KV cache ───────────────────
class V41TransformerLayer(nn.Module):
    """
    v4.1改动：self_attn支持past_kv
    past_kv: tuple(k_past, v_past) shape (B, past_len, D_MODEL) 或 None
    返回 (x, (k_cur_full, v_cur_full)) 供下一window使用
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
        self.adaln_canvas1 = AdaLN(d_model, cond_dim)
        self.adaln_canvas2 = AdaLN(d_model, cond_dim)
        self.adaln_text1   = AdaLN(d_model, cond_dim)
        self.adaln_text2   = AdaLN(d_model, cond_dim)
        self.norm_cross    = nn.LayerNorm(d_model)

    def forward(self, x, c_canvas, c_text, kv_patch, attn_mask=None, past_kv=None):
        B, T, D = x.shape

        # ── 1. self-attention with KV cache ──
        x2 = self.adaln_canvas1(x, c_canvas) + self.adaln_text1(x, c_text)

        # query: 当前window (B, T, D)
        # key/value: past KV + 当前window
        if past_kv is not None:
            k_past, v_past = past_kv          # (B, past_len, D)
            k_full = torch.cat([k_past, x2], dim=1)   # (B, past_len+T, D)
            v_full = torch.cat([v_past, x2], dim=1)
        else:
            k_full = x2
            v_full = x2

        # causal mask: query T × key (past_len+T)
        # query中第i个只能看past全部 + 当前前i个
        past_len = k_full.shape[1] - T
        full_len = k_full.shape[1]
        # 构造 (T, full_len) mask: query i 可见 [0..past_len+i]
        causal_mask = torch.ones(T, full_len, device=x.device, dtype=torch.bool)
        for i in range(T):
            causal_mask[i, past_len + i + 1:] = False   # 不可见未来当前window
        # True=可见, MHA用 attn_mask: True means IGNORE → 取反
        mha_mask = ~causal_mask  # (T, full_len), True=blocked

        out, _ = self.self_attn(x2, k_full, v_full,
                                attn_mask=mha_mask.unsqueeze(0).expand(B * N_HEADS, -1, -1)
                                if False else mha_mask,
                                is_causal=False)
        x = x + out

        # 保存当前window的KV供下一window（detach，不跨window反传）
        new_kv = (k_full.detach(), v_full.detach())

        # ── 2. cross-attention to canvas clip patch ──
        x2 = self.norm_cross(x)
        x2, _ = self.cross_attn(x2, kv_patch, kv_patch)
        x = x + x2

        # ── 3. FFN with AdaLN ──
        x2 = self.adaln_canvas2(x, c_canvas) + self.adaln_text2(x, c_text)
        x2 = self.ff(x2)
        x  = x + x2

        return x, new_kv


class V41Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, cond_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            V41TransformerLayer(d_model, nhead, dim_feedforward, dropout, cond_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, c_canvas, c_text, kv_patch, attn_mask=None, past_kvs=None):
        """
        past_kvs: list of (k,v) per layer, 或 None
        返回 (x, new_kvs)
        """
        new_kvs = []
        for i, layer in enumerate(self.layers):
            past = past_kvs[i] if past_kvs is not None else None
            x, kv = layer(x, c_canvas, c_text, kv_patch, attn_mask=attn_mask, past_kv=past)
            new_kvs.append(kv)
        return x, new_kvs


# ── KV cache 管理：截断到 MAX_KV_WINDOWS ──────────────
def trim_kvs(kvs, max_windows=MAX_KV_WINDOWS):
    """
    kvs: list of (k, v) per layer
    k, v shape: (B, history_len, D)
    只保留最近 max_windows * WINDOW 个token
    """
    max_len = max_windows * WINDOW
    return [(k[:, -max_len:, :], v[:, -max_len:, :]) for k, v in kvs]


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


# ── Main Model ────────────────────────────────────────
class StrokeARFMv41(nn.Module):
    def __init__(self):
        super().__init__()
        self.canvas_dino_enc = CanvasDINOEncoder(CANVAS_FEAT)
        self.text_enc        = TextEncoder(CANVAS_FEAT)
        self.clip_patch_proj = ClipPatchProjector(D_MODEL)

        self.stroke_proj = nn.Linear(STROKE_DIM, D_MODEL)
        self.pos_emb     = nn.Embedding(WINDOW, D_MODEL)

        self.transformer = V41Transformer(
            d_model=D_MODEL, nhead=N_HEADS, num_layers=N_LAYERS,
            dim_feedforward=D_MODEL*4, dropout=DROPOUT, cond_dim=CANVAS_FEAT,
        )
        self.out_norm = nn.LayerNorm(D_MODEL)

        self.fm_head       = FMHead(h_dim=D_MODEL, a_dim=STROKE_DIM)
        self.label_head    = nn.Linear(D_MODEL, LABEL_MAX + 1)
        self.normal_head   = nn.Linear(D_MODEL, 3)
        self.landmark_head = nn.Linear(D_MODEL, N_LANDMARKS)
        self.dino_head     = nn.Linear(D_MODEL, 384)

    def encode(self, strokes, canvas_dino, canvas_clip_patch, text_feat, past_kvs=None):
        B, T, _ = strokes.shape
        c_canvas = self.canvas_dino_enc(canvas_dino)        # (B, CANVAS_FEAT)
        c_text   = self.text_enc(text_feat)                 # (B, CANVAS_FEAT)
        kv_patch = self.clip_patch_proj(canvas_clip_patch)  # (B, 49, D_MODEL)

        pos = torch.arange(T, device=strokes.device)
        x   = self.stroke_proj(strokes) + self.pos_emb(pos)  # (B, T, D_MODEL)

        # past_kvs决定是否有历史KV，causal mask在layer内部处理
        x, new_kvs = self.transformer(x, c_canvas, c_text, kv_patch, past_kvs=past_kvs)
        x = self.out_norm(x)
        h = x[:, -1, :]   # (B, D_MODEL)
        return h, new_kvs

    def forward(self, strokes, canvas_dino, canvas_clip_patch, text_feat,
                a_tar, label_tar, normal_tar, landmark_tar, dino_tar,
                past_kvs=None):
        h, new_kvs = self.encode(strokes, canvas_dino, canvas_clip_patch, text_feat, past_kvs)

        loss_fm       = fm_loss(self.fm_head, h, a_tar) * W_FM
        loss_label    = F.cross_entropy(self.label_head(h), label_tar) * W_LABEL
        loss_normal   = F.mse_loss(torch.sigmoid(self.normal_head(h)), normal_tar) * W_NORMAL
        loss_landmark = F.mse_loss(torch.sigmoid(self.landmark_head(h)), landmark_tar) * W_LANDMARK
        loss_dino     = F.mse_loss(self.dino_head(h), dino_tar) * W_DINO

        total = loss_fm + loss_label + loss_normal + loss_landmark + loss_dino
        return total, {
            "fm": loss_fm.item(), "label": loss_label.item(),
            "normal": loss_normal.item(), "landmark": loss_landmark.item(),
            "dino": loss_dino.item(),
        }, new_kvs

    @torch.no_grad()
    def sample(self, strokes, canvas_dino, canvas_clip_patch, text_feat,
               past_kvs=None, steps=FM_STEPS):
        h, new_kvs = self.encode(strokes, canvas_dino, canvas_clip_patch, text_feat, past_kvs)
        stroke = self.fm_head.sample(h, steps=steps)
        return stroke, new_kvs


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
        print(f"V4.1  WINDOW={WINDOW}  D_MODEL={D_MODEL}  MAX_KV_WINDOWS={MAX_KV_WINDOWS}")

    dataset = PaintingDataset(DATA_ROOT)
    if is_master:
        print(f"Dataset: {len(dataset)} paintings")

    sampler = BucketDistributedSampler(
        dataset, batch_size=BATCH_SIZE, bucket_size=10,
        num_replicas=dist.get_world_size(), rank=local_rank,
        shuffle=True, seed=42,
    )
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=16, pin_memory=True,
        collate_fn=painting_collate_fn,
        persistent_workers=True,
    )

    model = StrokeARFMv41().to(device)

    start_epoch = 1
    best_loss   = float("inf")

    if os.path.exists(CKPT):
        if is_master:
            print(f"Loading v4.1 checkpoint {CKPT}...")
        ckpt = torch.load(CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss   = ckpt.get("loss", float("inf"))
        if is_master:
            print(f"Resumed epoch {start_epoch-1}, best_loss={best_loss:.6f}")

    elif os.path.exists(BASE_CKPT):
        if is_master:
            print(f"Initializing from v4 {BASE_CKPT}...")
        v4_ckpt  = torch.load(BASE_CKPT, map_location=device, weights_only=False)
        v4_state = v4_ckpt["model_state"]
        # transformer层key名变了(V4TransformerLayer→V41TransformerLayer)，其余兼容
        remapped = {}
        for k, v in v4_state.items():
            remapped[k] = v
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if is_master:
            print(f"  Loaded from v4: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        if is_master:
            print("No checkpoint found. Training from scratch.")

    model     = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
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
        loss_dict_acc = {"fm": 0., "label": 0., "normal": 0., "landmark": 0., "dino": 0.}

        for batch_paintings in loader:
            # batch_paintings: list of painting-tuples，每个 tuple 9个 tensor (n_windows, ...)
            # 逐window顺序推进，维护各画的past_kvs

            # 初始化每幅画的KV为None
            batch_kvs = [None] * len(batch_paintings)

            # 取最长画的window数，短画提前结束
            max_n_windows = max(p[0].shape[0] for p in batch_paintings)

            for win_idx in range(max_n_windows):
                # 收集本window有效的画（有些画可能没有这么多window）
                valid_indices = [
                    pi for pi, p in enumerate(batch_paintings)
                    if win_idx < p[0].shape[0]
                ]
                if not valid_indices:
                    break

                # stack有效画的当前window
                strokes     = torch.stack([batch_paintings[pi][0][win_idx] for pi in valid_indices]).to(device)
                canvas_dino = torch.stack([batch_paintings[pi][1][win_idx] for pi in valid_indices]).to(device)
                canvas_clip = torch.stack([batch_paintings[pi][2][win_idx] for pi in valid_indices]).to(device)
                text_feat   = torch.stack([batch_paintings[pi][3][win_idx] for pi in valid_indices]).to(device)
                target      = torch.stack([batch_paintings[pi][4][win_idx] for pi in valid_indices]).to(device)
                label       = torch.stack([batch_paintings[pi][5][win_idx] for pi in valid_indices]).to(device)
                normal      = torch.stack([batch_paintings[pi][6][win_idx] for pi in valid_indices]).to(device)
                landmark    = torch.stack([batch_paintings[pi][7][win_idx] for pi in valid_indices]).to(device)
                dino_tar    = torch.stack([batch_paintings[pi][8][win_idx] for pi in valid_indices]).to(device)

                # 收集有效画的past_kvs（各画独立KV，需要拼成batch）
                # past_kvs_batch: list of (k,v) per layer, k shape (n_valid, past_len, D)
                past_kvs_batch = None
                if any(batch_kvs[pi] is not None for pi in valid_indices):
                    n_layers = N_LAYERS
                    past_kvs_batch = []
                    for li in range(n_layers):
                        ks, vs = [], []
                        for pi in valid_indices:
                            if batch_kvs[pi] is not None:
                                ks.append(batch_kvs[pi][li][0])  # (1, past_len, D) 或 (past_len, D)
                                vs.append(batch_kvs[pi][li][1])
                            else:
                                # 此画本window是第一个window，无history
                                ks.append(torch.zeros(1, 0, D_MODEL, device=device))
                                vs.append(torch.zeros(1, 0, D_MODEL, device=device))
                        # 对齐past_len（不同画的历史长度可能不同，pad到最长）
                        max_past = max(k.shape[-2] for k in ks)
                        if max_past > 0:
                            ks_pad = [F.pad(k, (0, 0, max_past - k.shape[-2], 0)) for k in ks]
                            vs_pad = [F.pad(v, (0, 0, max_past - v.shape[-2], 0)) for v in vs]
                            past_kvs_batch.append((
                                torch.cat(ks_pad, dim=0),
                                torch.cat(vs_pad, dim=0),
                            ))
                        else:
                            past_kvs_batch = None
                            break

                loss, loss_dict, new_kvs = model(
                    strokes, canvas_dino, canvas_clip, text_feat,
                    target, label, normal, landmark, dino_tar,
                    past_kvs=past_kvs_batch,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # 更新各画的KV（截断，detach已在layer内做）
                # new_kvs: list of (k,v) per layer, k shape (n_valid, new_len, D)
                for idx_in_valid, pi in enumerate(valid_indices):
                    painting_kvs = [
                        (new_kvs[li][0][idx_in_valid:idx_in_valid+1],
                         new_kvs[li][1][idx_in_valid:idx_in_valid+1])
                        for li in range(N_LAYERS)
                    ]
                    batch_kvs[pi] = trim_kvs(painting_kvs, MAX_KV_WINDOWS)

                B_valid = strokes.size(0)
                epoch_loss += loss.item() * B_valid
                n_samples  += B_valid
                for k in loss_dict_acc:
                    loss_dict_acc[k] += loss_dict[k] * B_valid

        if n_samples == 0:
            continue

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
                        "WINDOW":          WINDOW,
                        "STRIDE":          STRIDE,
                        "D_MODEL":         D_MODEL,
                        "N_HEADS":         N_HEADS,
                        "N_LAYERS":        N_LAYERS,
                        "STROKE_DIM":      STROKE_DIM,
                        "CANVAS_FEAT":     CANVAS_FEAT,
                        "TEXT_FEAT":       TEXT_FEAT,
                        "CLIP_PATCH_DIM":  CLIP_PATCH_DIM,
                        "CLIP_PATCH_N":    CLIP_PATCH_N,
                        "FM_STEPS":        FM_STEPS,
                        "LABEL_MAX":       LABEL_MAX,
                        "N_LANDMARKS":     N_LANDMARKS,
                        "MAX_KV_WINDOWS":  MAX_KV_WINDOWS,
                    }
                }, CKPT)

    if is_master:
        print(f"Done. Best loss={best_loss:.6f}, saved to {CKPT}")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()