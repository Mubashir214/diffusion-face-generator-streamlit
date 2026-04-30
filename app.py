import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
from torchvision import transforms

from model import UNet
from utils import sample_with_steps, denormalize, q_sample
from config import Config

cfg = Config()

st.set_page_config(page_title="DDPM Face Generator", layout="wide")

st.title("🧠 DDPM Face Generator")
st.write("Generate faces OR upload image for reconstruction")

# =====================================
# Load Model
# =====================================
@st.cache_resource
def load_model():
    model = UNet(cfg.CHANNELS)
    model.load_state_dict(torch.load("ddpm_weights_only.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =====================================
# Sidebar
# =====================================
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Select Mode", ["Generate", "Upload & Reconstruct"])

# =====================================
# TRANSFORM
# =====================================
transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =====================================
# MODE 1: GENERATE FROM NOISE
# =====================================
if mode == "Generate":
    if st.sidebar.button("🎨 Generate Face"):
        steps = sample_with_steps(
            model,
            (1, cfg.CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
        )

        cols = st.columns(len(steps))

        for i, img in enumerate(steps):
            img = denormalize(img[0]).clamp(0,1)
            img = img.permute(1,2,0).cpu().numpy()
            cols[i].image(img, caption=f"Step {i}")

        final_img = denormalize(steps[-1][0]).clamp(0,1)
        final_img = final_img.permute(1,2,0).cpu().numpy()

        st.image(final_img, caption="Final Output")

# =====================================
# MODE 2: UPLOAD + RECONSTRUCT
# =====================================
elif mode == "Upload & Reconstruct":
    uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image")

        img_tensor = transform(image).unsqueeze(0)

        if st.button("Reconstruct Image"):
            t = torch.tensor([cfg.TIMESTEPS//2])

            # Add noise
            noisy = q_sample(img_tensor, t)

            # Denoise
            steps = sample_with_steps(
                model,
                img_tensor.shape
            )

            recon = steps[-1]

            recon_img = denormalize(recon[0]).clamp(0,1)
            recon_img = recon_img.permute(1,2,0).cpu().numpy()

            st.image(recon_img, caption="Reconstructed Image")
