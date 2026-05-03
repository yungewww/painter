import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from datetime import datetime
from multiprocessing import Pool

DATA_ROOT   = r"/root/autodl-tmp/output"
FEAT_ROOT   = r"/root/autodl-tmp/DinoFeats"
CANVAS_SIZE = (224, 224)
BATCH_SIZE  = 64
NUM_WORKERS = 12

canvas_transform = T.Compose([
    T.Resize(CANVAS_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def init_worker():
    global g_model, g_device
    torch.set_num_threads(8)
    g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_model = torch.hub.load(
        "/root/.cache/torch/hub/facebookresearch_dinov2_main",
        "dinov2_vits14",
        source="local"
    )
    g_model.eval().to(g_device)

def process_one(args):
    i, total, pdir = args
    frames_dir = os.path.join(pdir, "stroke_frames")
    if not os.path.exists(frames_dir):
        return

    painting_id = os.path.basename(pdir)
    out_path = os.path.join(FEAT_ROOT, f"{painting_id}.npy")
    if os.path.exists(out_path):
        return

    png_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not png_paths:
        return

    all_feats = []
    for start in range(0, len(png_paths), BATCH_SIZE):
        batch_paths = png_paths[start:start + BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(canvas_transform(img))
            except Exception:
                continue

        if not imgs:
            continue

        imgs = torch.stack(imgs).to(g_device)
        with torch.no_grad():
            feats = g_model(imgs)
        all_feats.append(feats.cpu().float().numpy())

    if not all_feats:
        return

    all_feats = np.concatenate(all_feats, axis=0)
    np.save(out_path, all_feats)

    if (i + 1) % 10 == 0:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{i+1}/{total}] {painting_id}: {all_feats.shape}")

def main():
    os.makedirs(FEAT_ROOT, exist_ok=True)
    painting_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "*")))
    total = len(painting_dirs)
    print(f"Total paintings: {total}")

    tasks = [(i, total, pdir) for i, pdir in enumerate(painting_dirs)]

    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        pool.map(process_one, tasks)

    print("Done.")

if __name__ == "__main__":
    main()
