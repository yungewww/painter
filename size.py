import cv2
import mediapipe as mp
import numpy as np
import os
from glob import glob

# 初始化人脸检测 (mediapipe 默认CPU，不用额外设置)
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.8)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(BASE_DIR, "raw")
output_dir = os.path.join(BASE_DIR, "processed")
# # 输入/输出目录
# input_dir = "./raw"
# output_dir = "./processed"
os.makedirs(output_dir, exist_ok=True)

# 遍历所有图片
img_paths = sorted(glob(os.path.join(input_dir, "*.*")))
print(f"Found {len(img_paths)} images in {input_dir}")

target_w, target_h = 600, 800

for idx, path in enumerate(img_paths, 1):
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ 跳过无法读取的文件: {path}")
        continue

    h, w, _ = img.shape
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    final_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255  # 默认白底

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x1, y1 = int(bboxC.xmin * w), int(bboxC.ymin * h)
            x2, y2 = int((bboxC.xmin + bboxC.width) * w), int((bboxC.ymin + bboxC.height) * h)

            bw, bh = x2 - x1, y2 - y1

            # ---- 扩展框（头和肩膀）----
            expand_x = int(bw * 1.0)   # 左右扩展 100%
            expand_y = int(bh * 1.2)   # 上下扩展 120%

            nx1 = max(0, x1 - expand_x)
            ny1 = max(0, y1 - int(expand_y * 0.8))
            nx2 = min(w, x2 + expand_x)
            ny2 = min(h, y2 + expand_y)

            crop = img[ny1:ny2, nx1:nx2]
            ch, cw = crop.shape[:2]

            # ---- 保持比例缩放到 <= 600×800 ----
            scale = min(target_w / cw, target_h / ch)
            new_w, new_h = int(cw * scale), int(ch * scale)
            resized = cv2.resize(crop, (new_w, new_h))

            # ---- 居中贴到底图 ----
            canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

            # ---- 在此基础上放大 1.5 倍 ----
            crop_w, crop_h = int(target_w / 1.5), int(target_h / 1.5)  # 450x600
            center_x, center_y = target_w // 2, target_h // 2

            cx1 = center_x - crop_w // 2
            cy1 = center_y - crop_h // 2
            cx2 = cx1 + crop_w
            cy2 = cy1 + crop_h

            cropped = canvas[cy1:cy2, cx1:cx2]
            final_img = cv2.resize(cropped, (target_w, target_h))

            break  # 只取第一张人脸

    # 保存结果
    out_path = os.path.join(output_dir, f"{idx:04d}.jpg")
    cv2.imwrite(out_path, final_img)
    print(f"✅ 保存 {out_path}")

print("全部处理完成！")
