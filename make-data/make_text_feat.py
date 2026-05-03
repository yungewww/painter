import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import CLIPTokenizer, CLIPTextModel
from multiprocessing import Pool

# ── Config ────────────────────────────────────────────
IMAGE_DIR  = "images"
OUTPUT_DIR = "run2_output"
EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_WORKERS = 8  # BLIP2较重，不要太多

BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
CLIP_MODEL  = "openai/clip-vit-base-patch32"

# ── Load models (global, single process) ─────────────
def init_worker():
    global g_blip2_processor, g_blip2_model, g_clip_tokenizer, g_clip_text_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Worker loading BLIP2...")
    g_blip2_processor = Blip2Processor.from_pretrained(BLIP2_MODEL)
    g_blip2_model     = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    g_blip2_model.eval()

    print(f"  Worker loading CLIP text encoder...")
    g_clip_tokenizer   = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    g_clip_text_model  = CLIPTextModel.from_pretrained(CLIP_MODEL).to(device)
    g_clip_text_model.eval()
    print(f"  Worker ready.")


def process_one(args):
    img_path_str, output_dir_str, idx, total = args
    img_path = Path(img_path_str)
    out_dir  = Path(output_dir_str) / img_path.stem
    out_path = out_dir / "text_feat.npy"

    if out_path.exists():
        print(f"[{idx}/{total}] {img_path.name} SKIP")
        return

    if not out_dir.exists():
        print(f"[{idx}/{total}] {img_path.name} SKIP (no output dir)")
        return

    try:
        device = next(g_blip2_model.parameters()).device

        # 1. BLIP2 生成描述
        image  = Image.open(str(img_path)).convert("RGB")
        inputs = g_blip2_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = g_blip2_model.generate(**inputs, max_new_tokens=50)
        caption = g_blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # 2. CLIP text encoder 编码
        tokens = g_clip_tokenizer(
            caption, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(device)
        with torch.no_grad():
            text_feat = g_clip_text_model(**tokens).pooler_output.squeeze(0).cpu().float().numpy()
        # text_feat: (512,)

        np.save(str(out_path), text_feat)
        print(f"[{idx}/{total}] {img_path.name}  \"{caption}\"  -> {out_path}")

    except Exception as e:
        print(f"[{idx}/{total}] {img_path.name} ERROR: {e}")


def main():
    base_dir   = Path(__file__).parent
    images_dir = base_dir / IMAGE_DIR
    output_dir = base_dir / OUTPUT_DIR

    exts      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])
    total     = len(img_paths)
    print(f"Found {total} images.")

    tasks = [(str(p), str(output_dir), idx+1, total) for idx, p in enumerate(img_paths)]

    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        pool.map(process_one, tasks)

    print("\nAll done.")

if __name__ == "__main__":
    main()
