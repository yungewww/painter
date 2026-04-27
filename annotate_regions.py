"""
自动标注脚本
用MediaPipe检测人脸关键点，生成各部位的bbox标注
输出: 每张图一个json，包含各部位的归一化bbox

用法: python annotate_regions.py
"""

import os
import json
import glob
import cv2
import numpy as np
import mediapipe as mp

# ── 路径 ──────────────────────────────────────────────
IMAGE_DIR  = r"C:\Train\images"    # 原始人脸图片目录
OUTPUT_DIR = r"C:\Train\RegionAnnotations"  # 输出json目录

# ── MediaPipe landmark索引 → 各部位 ──────────────────
# 每个部位取关键点，算convex hull的bbox
REGION_LANDMARKS = {
    "left_eye":   [33, 160, 158, 133, 153, 144, 163, 7],
    "right_eye":  [362, 385, 387, 263, 373, 380, 384, 249],
    "nose":       [1, 2, 98, 327, 168, 197, 195, 5],
    "mouth":      [61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],
    "left_brow":  [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_brow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "face":       [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                   397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                   172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
}

TEXT_LABELS = {
    "left_eye":  "左眼",
    "right_eye": "右眼",
    "nose":      "鼻子",
    "mouth":     "嘴巴",
    "left_brow": "左眉毛",
    "right_brow":"右眉毛",
    "face":      "脸部轮廓",
}

# ── 主逻辑 ────────────────────────────────────────────

def landmarks_to_bbox(landmarks, indices, w, h, padding=0.05):
    """
    给定landmark索引列表，返回归一化bbox [x1, y1, x2, y2]
    padding: 在bbox四周加一点余量
    """
    pts = np.array([
        [landmarks[i].x, landmarks[i].y]
        for i in indices
    ])
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()

    # 加padding
    pw = (x2 - x1) * padding
    ph = (y2 - y1) * padding
    x1 = max(0.0, x1 - pw)
    y1 = max(0.0, y1 - ph)
    x2 = min(1.0, x2 + pw)
    y2 = min(1.0, y2 + ph)

    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
    )

    img_paths = sorted(
        glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
        glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) +
        glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    )
    print(f"Total images: {len(img_paths)}")

    success, failed = 0, 0

    for i, img_path in enumerate(img_paths):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{stem}.json")
        if os.path.exists(out_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            failed += 1
            continue

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            failed += 1
            print(f"  No face: {stem}")
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        regions = {}
        for name, indices in REGION_LANDMARKS.items():
            bbox = landmarks_to_bbox(landmarks, indices, w, h)
            regions[name] = {
                "bbox": bbox,          # [x1, y1, x2, y2] 归一化
                "text": TEXT_LABELS[name],
            }

        annotation = {
            "image": os.path.basename(img_path),
            "width": w,
            "height": h,
            "regions": regions,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

        success += 1
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(img_paths)}] success={success} failed={failed}")

    print(f"Done. success={success}, failed={failed}")
    face_mesh.close()


if __name__ == "__main__":
    main()
