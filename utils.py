import streamlit as st
import torch
import numpy as np
from model import UNet
from utils import sample_with_steps, denormalize
from config import Config

# =========================
# Config
# =========================
cfg = Config()

st.set_page_config(page_title="DDPM Image Generator", layout="wide")

st.title("🧠 DDPM Image Generation from Noise")
st.write("Generate images using reverse diffusion process")

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    model = UNet(cfg.CHANNELS)
    model.load_state_dict(torch.load("ddpm_weights_only.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =========================
# Generate Button
# =========================
if st.button("🎨 Generate Image from Noise"):

    st.subheader("🔄 Intermediate Denoising Steps")

    # START FROM RANDOM NOISE → GENERATION
    steps = sample_with_steps(
        model,
        (1, cfg.CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
    )

    # Show intermediate steps (REQUIRED)
    cols = st.columns(len(steps))

    for i, img in enumerate(steps):
        img = denormalize(img[0]).clamp(0, 1)
        img = img.permute(1, 2, 0).cpu().numpy()
        cols[i].image(img, caption=f"Step {i+1}")

    # FINAL OUTPUT
    st.subheader("🖼️ Final Generated Image")

    final_img = denormalize(steps[-1][0]).clamp(0, 1)
    final_img = final_img.permute(1, 2, 0).cpu().numpy()

    st.image(final_img, use_container_width=True)

    st.success("Image generated successfully from random noise!")
