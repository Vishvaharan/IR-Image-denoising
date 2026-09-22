import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load model
model = load_model("denoising_autoencoder.h5", compile=False)

# Load noisy image
img = cv2.imread("dataset/noisy_images/FLIR_00001.jpeg", cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (256,256))

# Normalize
img_norm = img / 255.0

input_img = img_norm.reshape(1,256,256,1)

# Predict denoised image
output = model.predict(input_img)

denoised = (output.reshape(256,256)*255).astype(np.uint8)

# -------- SHARPENING FILTER --------
kernel = np.array([[0,-1,0],
                   [-1,5,-1],
                   [0,-1,0]])

sharp = cv2.filter2D(denoised,-1,kernel)

# Show images
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Noisy Image")
plt.imshow(img,cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Denoised Image")
plt.imshow(denoised,cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Sharpened Image")
plt.imshow(sharp,cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()