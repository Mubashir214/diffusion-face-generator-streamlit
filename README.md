# 🎨 Diffusion Models for High-Resolution Image Generation & Reconstruction

## README.md

# 🌌 DDPM: High-Resolution Image Generation using Diffusion Models

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch using **PyTorch** for high-resolution image generation and reconstruction.

The system progressively adds noise to images and then learns to reverse the process to generate realistic images from pure random noise.

The implementation uses:

* Forward Diffusion (Noising)
* Reverse Diffusion (Denoising)
* Simplified U-Net Backbone
* Time-step Embeddings
* Residual Blocks

without using pretrained diffusion pipelines or HuggingFace Diffusers.

---

# 🚀 Live Demo

### Streamlit Application

[Diffusion Face Generator App](https://diffusion-face-generator-app-vrltc4rswyuqybk7fhfofy.streamlit.app/?utm_source=chatgpt.com)

The app allows users to:

* Generate images from random noise
* Visualize denoising steps
* Observe diffusion progression
* Reconstruct images interactively

---

# 📌 Project Objectives

This project aims to:

* Understand diffusion probabilistic models
* Implement DDPM using base PyTorch
* Learn forward and reverse diffusion processes
* Generate high-quality images from noise
* Reconstruct target images
* Visualize denoising behavior
* Deploy a real-time diffusion application

---

# 🧠 Concepts Covered

* Denoising Diffusion Probabilistic Models (DDPM)
* Forward Diffusion Process
* Reverse Diffusion Process
* U-Net Architecture
* Residual Learning
* Time-step Embeddings
* Noise Scheduling
* Image Reconstruction
* High-Resolution Image Synthesis

---

# 📂 Dataset Used

This project supports the following datasets:

## 1️⃣ CelebA-HQ Dataset

[CelebA-HQ Dataset](https://www.kaggle.com/datasets/denislukovnikov/celebahq256-images-only?utm_source=chatgpt.com)

High-quality celebrity face images.

---

## 2️⃣ FFHQ Dataset

[FFHQ Face Dataset](https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set?utm_source=chatgpt.com)

Flickr-Faces-HQ dataset for realistic face generation.

---

## 3️⃣ WikiArt Dataset

[WikiArt Dataset](https://www.kaggle.com/datasets/sairam3/wikiart?utm_source=chatgpt.com)

Artistic painting dataset for style-based generation.

---

# ⚙️ Environment Setup

## Platform

* Kaggle Notebook

## Hardware

* GPU: Tesla T4 ×2

---

# 📦 Libraries Used

```bash id="9z6lqb"
torch
torchvision
numpy
matplotlib
streamlit
Pillow
tqdm
scikit-image
```

Install dependencies:

```bash id="nv1xkj"
pip install torch torchvision matplotlib pillow tqdm streamlit scikit-image
```

---

# 🏗️ Model Architecture

# 🌫️ Denoising Diffusion Probabilistic Model (DDPM)

The diffusion model contains two major processes:

---

# 1️⃣ Forward Diffusion Process (Noising)

The forward process gradually adds Gaussian noise to an image over multiple timesteps.

## Features

* Fixed process
* Controlled by noise schedule
* Converts image → random noise

### Workflow

```text id="e3xwzi"
Original Image → Slight Noise → Medium Noise → Heavy Noise → Pure Noise
```

---

# 2️⃣ Reverse Diffusion Process (Denoising)

The reverse process learns to remove noise step-by-step.

Implemented using a neural network.

## Goal

Predict noise added at each timestep and reconstruct the original image.

---

# 🧩 U-Net Backbone

A simplified U-Net architecture is implemented.

## Features

* Encoder-Decoder structure
* Residual blocks
* Skip connections
* Time-step embeddings
* Downsampling & Upsampling layers

---

# 📐 Channel Progression

```text id="9e4ktn"
64 → 128 → 256
```

---

# 🧠 Network Inputs & Outputs

## Input

* Noisy image
* Timestep embedding

## Output

* Predicted noise
---

# 📊 Data Preprocessing

The preprocessing pipeline includes:

1. Load images
2. Resize images to:

   * 128×128 OR
   * 256×256
3. Convert images to tensors
4. Normalize pixel values
5. Create DataLoader

Example normalization:

```python id="aq6i8f"
transforms.Normalize((0.5,), (0.5,))
```

---

# 🌫️ Forward Diffusion Implementation

## Steps

1. Define noise schedule
2. Sample timestep `t`
3. Add Gaussian noise
4. Generate noisy image

---

# 🔍 Visualization of Noising

The project visualizes:

* Original image
* Light noise
* Medium noise
* Strong noise
* Fully noisy image

At least:

* 5 forward diffusion steps

---

# 🔄 Reverse Diffusion Process

## Steps

1. Input noisy image into U-Net
2. Predict noise
3. Remove predicted noise
4. Repeat iteratively
5. Generate clean image

---

# 📉 Loss Function

## Mean Squared Error (MSE)

The model compares:

```text id="txrq7j"
Predicted Noise vs Actual Noise
```

Formula:

```text id="jlwmkz"
Loss = MSE(predicted_noise, actual_noise)
```

---

# ⚡ Optimizer & Scheduler

| Component     | Configuration              |
| ------------- | -------------------------- |
| Optimizer     | Adam / AdamW               |
| Learning Rate | 0.0002                     |
| Scheduler     | Optional (Cosine / StepLR) |

---

# 🚀 Training Techniques

To fit Kaggle T4×2 GPUs:

* Mixed Precision Training (`torch.cuda.amp`)
* Small Batch Size (16–32)
* Gradient Clipping (optional)
* Reduced timesteps (200–500)
* Checkpoint saving every few epochs

---

# 🖼️ Image Reconstruction Task

The reconstruction task performs:

1. Start from random noise
2. Apply reverse diffusion
3. Generate image similar to target

---

# 🎨 Image Generation Task

The system generates:

* 5+ new images from pure noise

Generated outputs are:

* Sharp
* High quality
* Visually meaningful

---

# 📈 Visualization Module

The visualization utility displays:

| Visualization          | Description                      |
| ---------------------- | -------------------------------- |
| Original Image         | Input clean image                |
| Noisy Versions         | Multiple forward diffusion steps |
| Intermediate Denoising | Reverse diffusion process        |
| Final Output           | Generated image                  |

Minimum Requirements:

* 5 forward diffusion steps
* 5 reverse diffusion steps
* 5 generated images

---

# 📊 Quantitative Evaluation

# 1️⃣ PSNR (Peak Signal-to-Noise Ratio)

Measures reconstruction quality.

Higher PSNR = Better reconstruction.

---

# 2️⃣ SSIM (Structural Similarity Index)

Measures structural similarity between images.

Higher SSIM = Better image fidelity.

---

# 📱 Streamlit Deployment

The project includes a Streamlit application that:

✅ Starts from random noise
✅ Generates realistic images
✅ Displays denoising progression
✅ Visualizes intermediate steps
✅ Allows interactive testing

Run locally:

```bash id="3i0n0i"
streamlit run streamlit_app.py
```

Live deployed app:

[Open Diffusion Model App](https://diffusion-face-generator-app-vrltc4rswyuqybk7fhfofy.streamlit.app/?utm_source=chatgpt.com)

---

# 🔍 Results & Observations

## Model Performance

### Advantages

* Generates realistic images
* Learns complex data distributions
* Stable training compared to GANs
* Produces high-quality outputs

### Challenges

* Slow sampling process
* Computationally expensive
* Requires many diffusion steps

---

# 🎯 Applications

This system can be applied to:

* AI Art Generation
* Face Synthesis
* Image Restoration
* Super Resolution
* Medical Imaging
* Creative Design Tools

---

# 🔮 Future Improvements

* Faster diffusion sampling
* DDIM implementation
* Improved noise schedules
* Conditional diffusion models
* Higher-resolution training
* Multi-GPU distributed training

---

# 🎓 Conclusion

This project successfully demonstrates the implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch using PyTorch.

The system effectively:

* Learns forward and reverse diffusion
* Generates high-quality images
* Reconstructs target images
* Produces visually meaningful outputs from pure noise

The project highlights the power of diffusion models for modern generative AI tasks.

---

# 👨‍💻 Author

**Mubashir Siddique**

AI / Deep Learning / Computer Vision Enthusiast

---

# 📜 License

This project is developed for educational and research purposes.
