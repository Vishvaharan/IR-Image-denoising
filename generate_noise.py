import cv2
import numpy as np
import os

clean_folder = "dataset/clean_images"
noisy_folder = "dataset/noisy_images"

os.makedirs(noisy_folder, exist_ok=True)

for image_name in os.listdir(clean_folder):

    img_path = os.path.join(clean_folder, image_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    noise = np.random.normal(0, 25, img.shape)

    noisy_img = img + noise

    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    save_path = os.path.join(noisy_folder, image_name)

    cv2.imwrite(save_path, noisy_img)

print("Noisy images generated successfully")