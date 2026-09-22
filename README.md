# 🔥 IR Image Denoising using Convolutional Autoencoder

A deep learning project that denoises **Infrared (IR) / Thermal images** using a **Convolutional Autoencoder** built with TensorFlow/Keras. The model learns to reconstruct clean images from Gaussian-noisy counterparts, followed by a sharpening post-processing step.

---

## 📌 Project Overview

Infrared thermal images often suffer from noise due to sensor limitations, environmental conditions, or low-light captures. This project trains a convolutional autoencoder to:

- Learn the mapping from **noisy IR images → clean IR images**
- Remove Gaussian noise effectively
- Apply a **sharpening filter** post-denoising to enhance edge clarity

---

## 🗂️ Project Structure

```
IR_Image_Denoising_Project/
│
├── dataset/
│   ├── clean_images/          # Original clean IR images
│   └── noisy_images/          # Gaussian-noisy versions of clean images
│
├── generate_noise.py          # Script to add Gaussian noise to clean images
├── train.py                   # Model architecture definition and training
├── test.py                    # Inference + visualization (noisy → denoised → sharpened)
├── denoising_autoencoder.h5   # Saved trained model weights
└── README.md
```

---

## 🧠 Model Architecture

The autoencoder follows a symmetric **Encoder → Decoder** design:

```
Input (256×256×1)
    │
    ▼
[Encoder]
  Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
  Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
    │
    ▼
[Decoder]
  Conv2D(32, 3×3, ReLU) → UpSampling2D(2×2)
  Conv2D(64, 3×3, ReLU) → UpSampling2D(2×2)
    │
    ▼
Conv2D(1, 3×3, Sigmoid)   ← Reconstructed clean image
Output (256×256×1)
```

| Component     | Details                        |
|---------------|-------------------------------|
| Optimizer     | Adam                          |
| Loss Function | Mean Squared Error (MSE)      |
| Epochs        | 30                            |
| Batch Size    | 8                             |
| Input Size    | 256 × 256 (Grayscale)         |
| Max Samples   | 3000 images                   |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vishvaharan/IR-Image-denoising.git
cd IR-Image-denoising
```

### 2. Install dependencies

```bash
pip install tensorflow opencv-python numpy matplotlib
```

### 3. Prepare the dataset

Place your clean IR images (e.g., from [FLIR ADAS Dataset](https://www.flir.in/oem/adas/adas-dataset-form/)) inside:

```
dataset/clean_images/
```

---

## 🚀 Usage

### Step 1 — Generate Noisy Images

Adds Gaussian noise (mean=0, std=25) to all clean images:

```bash
python generate_noise.py
```

This populates `dataset/noisy_images/` automatically.

---

### Step 2 — Train the Model

```bash
python train.py
```

- Loads up to **3000 image pairs** from `dataset/`
- Trains for **30 epochs** with batch size **8**
- Saves the trained model to `denoising_autoencoder.h5`

---

### Step 3 — Test / Inference

```bash
python test.py
```

Displays a side-by-side comparison of:

| Panel          | Description                          |
|----------------|--------------------------------------|
| Noisy Image    | Input image with Gaussian noise       |
| Denoised Image | Autoencoder output                   |
| Sharpened Image| Post-processed with sharpening kernel|

> **Note:** Update the image path in `test.py` if testing a different image.

---

## 🖼️ Post-Processing: Sharpening Filter

After denoising, a **3×3 sharpening kernel** is applied via `cv2.filter2D` to enhance edges:

```
 [ 0, -1,  0]
 [-1,  5, -1]
 [ 0, -1,  0]
```

---

## 📦 Dependencies

| Library        | Purpose                          |
|----------------|----------------------------------|
| TensorFlow     | Model training & inference       |
| Keras          | High-level neural network API    |
| OpenCV         | Image I/O and processing         |
| NumPy          | Array operations                 |
| Matplotlib     | Visualization                    |

---

## 📈 Results

The model effectively suppresses Gaussian noise while preserving thermal gradient features in IR images. Post-sharpening further enhances object boundaries and edge clarity.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [FLIR ADAS Thermal Dataset](https://www.flir.in/oem/adas/adas-dataset-form/) — IR image source
- TensorFlow & Keras documentation
