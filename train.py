import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

clean_folder = "dataset/clean_images"
noisy_folder = "dataset/noisy_images"

clean_images = []
noisy_images = []

files = os.listdir(clean_folder)[:3000]   # use more images

for file in files:
    clean_path = os.path.join(clean_folder, file)
    noisy_path = os.path.join(noisy_folder, file)

    clean = cv2.imread(clean_path, 0)
    noisy = cv2.imread(noisy_path, 0)

    clean = cv2.resize(clean, (256,256))
    noisy = cv2.resize(noisy, (256,256))

    clean = clean / 255.0
    noisy = noisy / 255.0

    clean_images.append(clean)
    noisy_images.append(noisy)

clean_images = np.array(clean_images).reshape(-1,256,256,1)
noisy_images = np.array(noisy_images).reshape(-1,256,256,1)

input_img = Input(shape=(256,256,1))

# Encoder
x = Conv2D(64,(3,3),activation='relu',padding='same')(input_img)
x = MaxPooling2D((2,2),padding='same')(x)

x = Conv2D(32,(3,3),activation='relu',padding='same')(x)
x = MaxPooling2D((2,2),padding='same')(x)

# Decoder
x = Conv2D(32,(3,3),activation='relu',padding='same')(x)
x = UpSampling2D((2,2))(x)

x = Conv2D(64,(3,3),activation='relu',padding='same')(x)
x = UpSampling2D((2,2))(x)

decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(
    noisy_images,
    clean_images,
    epochs=30,
    batch_size=8,
    shuffle=True
)

autoencoder.save("denoising_autoencoder.h5")

print("Training complete")