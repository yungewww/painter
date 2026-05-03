import os
import shutil
import glob

print("starting...")


DATA_ROOT = r"C:\Train\output"
OUT_DIR   = r"C:\Train\StrokesFlat"

os.makedirs(OUT_DIR, exist_ok=True)

painting_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "*")))
count = 0
for pdir in painting_dirs:
    json_path = os.path.join(pdir, "strokes.json")
    if not os.path.exists(json_path):
        continue
    painting_id = os.path.basename(pdir)  # e.g. 000001
    dst = os.path.join(OUT_DIR, f"{painting_id}.json")
    shutil.copy(json_path, dst)
    count += 1

print(f"Done. {count} files copied to {OUT_DIR}")