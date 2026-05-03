import os
import json
import math
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ── 超参 ──────────────────────────────────────────────
WINDOW        = 30
STRIDE        = 1
D_MODEL       = 128
N_HEADS       = 4
N_LAYERS      = 4
DROPOUT       = 0.1
LR            = 5e-5
EPOCHS        = 500
BATCH_SIZE    = 64
CANVAS_SIZE   = (224, 224)

BRUSH_MAX     = 436
MDN_K         = 10

DATA_ROOT     = r"C:\Train\output"
FEAT_ROOT     = r"C:\Train\ClipFeats"
BASE_CKPT     = r"train-1-mse/base_model-mse.pt"
CKPT          = "base_model_mdn.pt"

STROKE_DIM    = 8
CANVAS_FEAT   = 128

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
    def __init__(self, data_root, feat_root=FEAT_ROOT, window=WINDOW, stride=STRIDE):
        self.samples = []
        painting_dirs = sorted(glob.glob(os.path.join(data_root, "*")))
        for pdir in painting_dirs:
            json_path = os.path.join(pdir, "strokes.json")
            painting_id = os.path.basename(pdir)
            feat_path = os.path.join(feat_root, f"{painting_id}.npy")
            if not os.path.exists(json_path) or not os.path.exists(feat_path):
                continue
            feats = np.load(feat_path)
            with open(json_path) as f:
                doc = json.load(f)
            W, H = doc["canvas_w"], doc["canvas_h"]
            strokes = [encode_stroke(s, W, H) for s in doc["strokes"]]
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
        self.proj = nn.Linear(512, out_dim)
        self.null_canvas = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return self.proj(x)


# ── MDN Head ─────────────────────────────────────────

class MDNHead(nn.Module):
    def __init__(self, in_dim=D_MODEL, out_dim=STROKE_DIM, K=MDN_K):
        super().__init__()
        self.K = K
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, K * (1 + out_dim + out_dim))

    def forward(self, h):
        out = self.fc(h)
        K, D = self.K, self.out_dim
        pi_raw    = out[:, :K]
        mu        = out[:, K: K + K*D].view(-1, K, D)
        sigma_raw = out[:, K + K*D:].view(-1, K, D)
        pi    = torch.softmax(pi_raw, dim=-1)
        sigma = torch.exp(sigma_raw).clamp(min=1e-4, max=1.0)
        return pi, mu, sigma

    def sample(self, h):
        pi, mu, sigma = self.forward(h)
        B, K, D = mu.shape
        idx = torch.multinomial(pi, num_samples=1).squeeze(1)
        batch_idx = torch.arange(B, device=h.device)
        mu_sel    = mu[batch_idx, idx]
        sigma_sel = sigma[batch_idx, idx]
        return mu_sel + sigma_sel * torch.randn_like(mu_sel)


def mdn_nll_loss(pi, mu, sigma, target):
    B, K, D = mu.shape
    target = target.unsqueeze(1).expand_as(mu)
    log_prob = -0.5 * (((target - mu) / sigma) ** 2 + 2 * torch.log(sigma) + math.log(2 * math.pi))
    log_prob = log_prob.sum(dim=-1)
    log_pi = torch.log(pi + 1e-8)
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)
    return -log_mix.mean()


# ── Stroke AR Model (MDN版) ───────────────────────────

class StrokeARMDN(nn.Module):
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
        self.mdn_head = MDNHead(in_dim=D_MODEL, out_dim=STROKE_DIM, K=MDN_K)
        self.null_region = nn.Parameter(torch.zeros(D_MODEL))
        self.null_text   = nn.Parameter(torch.zeros(D_MODEL))

    def forward(self, strokes, canvas_feat, region_feat=None, text_feat=None):
        B, T, _ = strokes.shape
        canvas_token = self.canvas_proj(self.canvas_encoder(canvas_feat)).unsqueeze(1)
        if region_feat is None:
            region_feat = self.null_region.unsqueeze(0).expand(B, -1)
        if text_feat is None:
            text_feat = self.null_text.unsqueeze(0).expand(B, -1)
        pos = torch.arange(T, device=strokes.device)
        stroke_h = self.stroke_proj(strokes) + self.pos_emb(pos)
        seq = torch.cat([canvas_token, stroke_h], dim=1)
        mask = nn.Transformer.generate_square_subsequent_mask(T + 1, device=strokes.device)
        mask[:, 0] = 0.0
        h = self.transformer(seq, mask=mask, is_causal=False)
        last = h[:, -1, :]
        return self.mdn_head(last)

    def sample(self, strokes, canvas_feat):
        B, T, _ = strokes.shape
        canvas_token = self.canvas_proj(self.canvas_encoder(canvas_feat)).unsqueeze(1)
        pos = torch.arange(T, device=strokes.device)
        stroke_h = self.stroke_proj(strokes) + self.pos_emb(pos)
        seq = torch.cat([canvas_token, stroke_h], dim=1)
        mask = nn.Transformer.generate_square_subsequent_mask(T + 1, device=strokes.device)
        mask[:, 0] = 0.0
        h = self.transformer(seq, mask=mask, is_causal=False)
        last = h[:, -1, :]
        return self.mdn_head.sample(last)


# ── 训练 ──────────────────────────────────────────────

def train():
    dataset = StrokeDataset(DATA_ROOT)
    print(f"Dataset: {len(dataset)} samples")
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found. Check DATA_ROOT='{DATA_ROOT}'")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = StrokeARMDN().to(device)

    # 加载原MSE checkpoint的body权重，跳过output_head
    print(f"Loading backbone from {BASE_CKPT}...")
    base_ckpt = torch.load(BASE_CKPT, map_location=device)
    base_state = base_ckpt["model_state"]
    model_state = model.state_dict()
    loaded = {}
    skipped = []
    for k, v in base_state.items():
        if k.startswith("output_head"):
            skipped.append(k)
            continue
        if k in model_state and model_state[k].shape == v.shape:
            loaded[k] = v
        else:
            skipped.append(k)
    model_state.update(loaded)
    model.load_state_dict(model_state)
    print(f"Loaded {len(loaded)} params, skipped: {skipped}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for strokes, canvas_feat, target in loader:
            strokes     = strokes.to(device)
            canvas_feat = canvas_feat.to(device)
            target      = target.to(device)

            pi, mu, sigma = model(strokes, canvas_feat)
            loss = mdn_nll_loss(pi, mu, sigma, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * strokes.size(0)

        scheduler.step()
        epoch_loss /= len(dataset)

        if epoch % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"epoch {epoch:4d}  loss={epoch_loss:.6f}  lr={lr_now:.2e}")

        if epoch == 1:
            print(f"Epoch 1 done, loss={epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": best_loss,
                "config": {
                    "WINDOW": WINDOW, "D_MODEL": D_MODEL,
                    "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS,
                    "CANVAS_SIZE": CANVAS_SIZE,
                    "MDN_K": MDN_K,
                }
            }, CKPT)

    print(f"Done. Best loss={best_loss:.6f}, saved to {CKPT}")


if __name__ == "__main__":
    train()