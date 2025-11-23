import cv2
import numpy as np
import os
from tqdm import tqdm

# === Create folders for dataset ===
os.makedirs("dataset/train/images", exist_ok=True)
os.makedirs("dataset/train/labels", exist_ok=True)
os.makedirs("dataset/val/images", exist_ok=True)
os.makedirs("dataset/val/labels", exist_ok=True)

# === Shapes and number of images ===
shapes = ["circle", "square", "triangle"]
num_images = 100  

def draw_shape(img, shape):
    h, w, _ = img.shape
    color = (255, 255, 255)
    thickness = -1
    center = (w // 2, h // 2)
    size = np.random.randint(40, 100)

    if shape == "circle":
        cv2.circle(img, center, size, color, thickness)
        return [0, center[0]/w, center[1]/h, size*2/w, size*2/h]

    elif shape == "square":
        pt1 = (center[0]-size, center[1]-size)
        pt2 = (center[0]+size, center[1]+size)
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return [1, center[0]/w, center[1]/h, size*2/w, size*2/h]

    elif shape == "triangle":
        pts = np.array([
            [center[0], center[1]-size],
            [center[0]-size, center[1]+size],
            [center[0]+size, center[1]+size]
        ], np.int32)
        cv2.drawContours(img, [pts], 0, color, thickness)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        return [2, (x_min+x_max)/(2*w), (y_min+y_max)/(2*h),
                (x_max-x_min)/w, (y_max-y_min)/h]

def generate_images(folder, start_idx, end_idx):
    for i in tqdm(range(start_idx, end_idx)):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        shape = np.random.choice(shapes)
        label = draw_shape(img, shape)

        img_path = f"{folder}/images/{i}.jpg"
        label_path = f"{folder}/labels/{i}.txt"

        cv2.imwrite(img_path, img)
        with open(label_path, "w") as f:
            f.write(" ".join(map(str, label)))

# === Generate dataset ===
generate_images("dataset/train", 0, int(num_images * 0.8))
generate_images("dataset/val", int(num_images * 0.8), num_images)

print("✅ Dataset generated successfully!")
