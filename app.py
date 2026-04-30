```python
import streamlit as st
import torch
import numpy as np
from model import UNet
from utils import sample_with_steps, denormalize
from config import Config

# =====================================
# Config
# =====================================
cfg = Config()

st.set_page_config(page_title="DDPM Face Generator", layout="wide")

st.title("🧠 DDPM Face Generator")
st.write("Generate faces from random noise and visualize denoising process")

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
# Sidebar Controls
# =====================================
st.sidebar.title("⚙️ Controls")

generate_btn = st.sidebar.button("🎨 Generate New Face")

# =====================================
# Generate Images
# =====================================
if generate_btn:
    st.subheader("🔄 Denoising Process")

    with st.spinner("Generating image... please wait ⏳"):
        steps = sample_with_steps(
            model,
            (1, cfg.CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
        )

    cols = st.columns(len(steps))

    for i, img in enumerate(steps):
        # FIX: Proper normalization + clamp
        img = denormalize(img[0]).clamp(0, 1)
        img = img.permute(1, 2, 0).cpu().numpy()

        cols[i].image(img, caption=f"Step {i}", use_container_width=True)

    st.success("✅ Final Image Generated!")

    # Show final image bigger
    st.subheader("🖼️ Final Output")
    final_img = denormalize(steps[-1][0]).clamp(0, 1)
    final_img = final_img.permute(1, 2, 0).cpu().numpy()

    st.image(final_img, use_container_width=True)

    # =====================================
    # Download Button
    # =====================================
    import io
    from PIL import Image

    img_pil = Image.fromarray((final_img * 255).astype(np.uint8))
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")

    st.download_button(
        label="📥 Download Image",
        data=buf.getvalue(),
        file_name="generated_face.png",
        mime="image/png"
    )

# =====================================
# Footer
# =====================================
st.markdown("---")
st.markdown("👨‍💻 Built with PyTorch + Streamlit")
```
