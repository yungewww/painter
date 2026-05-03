# import mediapipe as mp
# import cv2
# import numpy as np
# import os

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# img = cv2.imread("painting.png")
# h, w = img.shape[:2]
# rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# results = face_mesh.process(rgb)

# REGIONS = {
#     "nose":  [9, 107, 55, 221, 189, 244, 128, 114, 47, 100, 36, 206, 165, 167, 164, 393, 391, 426, 266, 329, 277, 343, 357, 464, 413, 441, 285, 336],
#     "mouth": [2, 97, 98, 203, 206, 216, 212, 202, 204, 194, 201, 200, 421, 418, 424, 422, 432, 436, 426, 423, 327, 326],
#     "eye_l": [107, 66, 105, 63, 70, 156, 143, 111, 117, 118, 101, 100, 47, 114, 188, 122, 193, 55],
# }

# os.makedirs("masks", exist_ok=True)

# if results.multi_face_landmarks:
#     landmarks = results.multi_face_landmarks[0].landmark

#     for name, indices in REGIONS.items():
#         points = np.array([
#             [int(landmarks[i].x * w), int(landmarks[i].y * h)]
#             for i in indices
#         ], dtype=np.int32)

#         mask = np.zeros((h, w), dtype=np.uint8)
#         cv2.fillPoly(mask, [points], 255)

#         out = np.ones((h, w, 3), dtype=np.uint8) * 255
#         out[mask == 255] = img[mask == 255]

#         cv2.imwrite(f"masks/{name}.png", out)
#         print(f"Saved masks/{name}.png")

# print("Done")

import mediapipe as mp
import cv2
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

REGIONS = {
    "nose":  [9, 107, 55, 221, 189, 244, 128, 114, 47, 100, 36, 206, 165, 167, 164, 393, 391, 426, 266, 329, 277, 343, 357, 464, 413, 441, 285, 336],
    "mouth": [2, 97, 98, 203, 206, 216, 212, 202, 204, 194, 201, 200, 421, 418, 424, 422, 432, 436, 426, 423, 327, 326],
    "eye_l": [107, 66, 105, 63, 70, 156, 143, 111, 117, 118, 101, 100, 47, 114, 188, 122, 193, 55],
}

INPUT_DIR = "images"
OUTPUT_DIR = "masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img = cv2.imread(os.path.join(INPUT_DIR, filename))
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    stem = os.path.splitext(filename)[0]

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        for name, indices in REGIONS.items():
            points = np.array([
                [int(landmarks[i].x * w), int(landmarks[i].y * h)]
                for i in indices
            ], dtype=np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)

            out = np.ones((h, w, 3), dtype=np.uint8) * 255
            out[mask == 255] = img[mask == 255]

            cv2.imwrite(f"{OUTPUT_DIR}/{stem}_{name}.png", out)
            print(f"Saved {stem}_{name}.png")
    else:
        print(f"No face detected: {filename}")

print("Done")