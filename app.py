import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import UNet
from utils import sample_with_steps, denormalize
from config import Config

cfg = Config()

st.title("🧠 DDPM Face Generator")
st.write("Generate faces from random noise with diffusion")

# Load model
@st.cache_resource
def load_model():
    model = UNet(cfg.CHANNELS)
    model.load_state_dict(torch.load("ddpm_weights_only.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

if st.button("Generate Image"):
    steps = sample_with_steps(model, (1, cfg.CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE))

    st.subheader("Denoising Process")

    cols = st.columns(len(steps))

    for i, img in enumerate(steps):
        img = denormalize(img[0]).permute(1, 2, 0).numpy()
        cols[i].image(img, caption=f"Step {i}", use_column_width=True)

    st.success("Final Image Generated!")
