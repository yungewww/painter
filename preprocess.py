"""
预提取CLIP特征脚本
运行一次，把所有canvas图片的CLIP特征存成.npy
输出:  (每个painting一个.npy文件)
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import clip

DATA_ROOT  = r"C:\Train\output"
FEAT_ROOT  = r"C:\Train\ClipFeats"   # 存在SSD上
CANVAS_SIZE = (224, 224)
BATCH_SIZE  = 64

canvas_transform = T.Compose([
    T.Resize(CANVAS_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
])

def main():
    os.makedirs(FEAT_ROOT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    painting_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "*")))
    print(f"Total paintings: {len(painting_dirs)}")

    for i, pdir in enumerate(painting_dirs):
        frames_dir = os.path.join(pdir, "stroke_frames")
        if not os.path.exists(frames_dir):
            continue

        painting_id = os.path.basename(pdir)
        out_path = os.path.join(FEAT_ROOT, f"{painting_id}.npy")
        if os.path.exists(out_path):
            continue  # 已提取，跳过

        png_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not png_paths:
            continue

        # 分batch提取
        all_feats = []
        for start in range(0, len(png_paths), BATCH_SIZE):
            batch_paths = png_paths[start:start + BATCH_SIZE]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(canvas_transform(img))
            imgs = torch.stack(imgs).to(device)

            with torch.no_grad():
                feats = clip_model.visual(imgs.half()).float()  # (B, 512)
                # feats = clip_model.visual(imgs).float()  # (B, 512)
            all_feats.append(feats.cpu().numpy())

        all_feats = np.concatenate(all_feats, axis=0)  # (N_frames, 512)
        np.save(out_path, all_feats)

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(painting_dirs)}] {painting_id}: {all_feats.shape}")

    print("Done.")

if __name__ == "__main__":
    main()