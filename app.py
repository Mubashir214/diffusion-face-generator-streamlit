# app.py
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from io import BytesIO

# Import custom modules
from config import Config, cfg
from model import UNet
from utils import setup_diffusion, extract, denormalize, tensor_to_pil

# Page configuration
st.set_page_config(
    page_title="DDPM Face Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎨 DDPM Face Generator")
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='font-size: 18px;'>
            Generate realistic human faces from random noise using 
            <b>Denoising Diffusion Probabilistic Models (DDPM)</b>
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Generation Settings")
    
    # Generation parameters
    num_images = st.slider("Number of images to generate", 1, 4, 1)
    
    st.markdown("---")
    st.header("🎮 Animation Settings")
    
    show_animation = st.checkbox("Show denoising animation", value=True)
    animation_speed = st.slider("Animation speed", 0.5, 2.0, 1.0, 0.1)
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
        **DDPM**: Denoising Diffusion Probabilistic Models
        
        - Trained on FFHQ face dataset
        - Image size: 128x128
        - Timesteps: 300
        - Architecture: U-Net with attention
        
        The model progressively denoises random noise to generate realistic faces.
    """)
    
    # Model info
    st.markdown("---")
    st.caption(f"Device: {cfg.DEVICE.upper()}")
    st.caption("Model trained on Kaggle")

# Load model
@st.cache_resource
def load_model():
    """Load the trained model with caching."""
    device = cfg.DEVICE
    
    # Initialize model
    model = UNet(cfg).to(device)
    
    try:
        # Try to load trained weights
        checkpoint = torch.load(cfg.MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        st.success("✅ Model loaded successfully!")
    except:
        st.warning("⚠️ No pre-trained weights found. Using randomly initialized model (will generate noise).")
    
    model.eval()
    
    # Setup diffusion parameters
    diffusion_params = setup_diffusion(cfg, device)
    
    return model, diffusion_params

# Sampling functions
@torch.no_grad()
def p_sample(model, x, t, t_index, diffusion_params, device):
    """Single step of the reverse process."""
    betas = diffusion_params['betas']
    alphas = diffusion_params['alphas']
    sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod']
    
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(1.0 / torch.sqrt(alphas), t, x.shape)
    
    # Model prediction
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(betas, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def generate_images(model, diffusion_params, num_images, device, return_intermediate=False, timesteps=None):
    """Generate images from noise."""
    if timesteps is None:
        timesteps = cfg.TIMESTEPS
    
    # Start from random noise
    img = torch.randn((num_images, cfg.CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE), device=device)
    
    intermediate_images = []
    
    # Progressive denoising
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in reversed(range(timesteps)):
        t = torch.full((num_images,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, diffusion_params, device)
        
        # Store intermediate steps for animation
        if return_intermediate and i % (timesteps // 10) == 0:
            intermediate_images.append(img.clone().cpu())
        
        # Update progress
        if i % (timesteps // 20) == 0:
            progress = (timesteps - i) / timesteps
            progress_bar.progress(progress)
            status_text.text(f"Denoising step: {timesteps - i}/{timesteps}")
    
    progress_bar.progress(1.0)
    status_text.text("Generation complete!")
    
    if return_intermediate:
        return img.cpu(), intermediate_images
    return img.cpu()

# Main generation function
def generate_and_display():
    """Main function to generate and display images."""
    with st.spinner("Loading model..."):
        model, diffusion_params = load_model()
    
    device = cfg.DEVICE
    
    # Generate images with or without intermediate steps
    if show_animation and num_images == 1:
        # Show animation for single image
        final_img, intermediates = generate_images(
            model, diffusion_params, num_images, device, 
            return_intermediate=True
        )
        
        # Display animation
        st.subheader("🎬 Denoising Animation")
        
        # Create placeholder for animation
        animation_placeholder = st.empty()
        
        # Play animation
        for img in intermediates:
            pil_img = tensor_to_pil(img[0])
            animation_placeholder.image(pil_img, caption="Denoising...", use_column_width=True)
            time.sleep(0.05 / animation_speed)
        
        # Show final image
        st.subheader("✨ Final Generated Image")
        final_pil = tensor_to_pil(final_img[0])
        st.image(final_pil, caption="Generated Face", use_column_width=True)
        
    else:
        # Generate without animation (faster for multiple images)
        final_imgs = generate_images(
            model, diffusion_params, num_images, device, 
            return_intermediate=False
        )
        
        # Display generated images
        st.subheader("✨ Generated Images")
        
        cols = st.columns(num_images)
        for idx, col in enumerate(cols):
            pil_img = tensor_to_pil(final_imgs[idx])
            col.image(pil_img, caption=f"Image {idx+1}", use_column_width=True)
    
    return final_imgs

# Function to show intermediate denoising steps grid
def show_intermediate_steps_grid(model, diffusion_params, device):
    """Generate a grid showing intermediate denoising steps."""
    st.subheader("🖼️ Intermediate Denoising Steps")
    
    # Generate single image with all intermediates
    final_img, intermediates = generate_images(
        model, diffusion_params, 1, device, 
        return_intermediate=True
    )
    
    # Add final image to intermediates
    intermediates.append(final_img)
    
    # Create a grid of images
    num_steps = len(intermediates)
    cols = st.columns(min(num_steps, 5))
    
    for idx, (col, img) in enumerate(zip(cols * (num_steps // 5 + 1), intermediates)):
        pil_img = tensor_to_pil(img[0])
        step_progress = 100 - (idx * 10)
        col.image(pil_img, caption=f"Step: {step_progress}%", use_column_width=True)
        if idx >= 4:
            break

# Main app interface
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🎨 Generate New Faces", type="primary", use_container_width=True):
        with st.spinner("Generating faces... This may take a few seconds."):
            generated_images = generate_and_display()
            
            # Add download buttons
            st.markdown("---")
            st.subheader("💾 Download Results")
            
            if num_images == 1:
                # Single image download
                buf = BytesIO()
                pil_img = tensor_to_pil(generated_images[0])
                pil_img.save(buf, format="PNG")
                st.download_button(
                    label="Download Image",
                    data=buf.getvalue(),
                    file_name="generated_face.png",
                    mime="image/png"
                )
            else:
                # Multiple images download
                for idx, img in enumerate(generated_images):
                    buf = BytesIO()
                    pil_img = tensor_to_pil(img)
                    pil_img.save(buf, format="PNG")
                    st.download_button(
                        label=f"Download Image {idx+1}",
                        data=buf.getvalue(),
                        file_name=f"generated_face_{idx+1}.png",
                        mime="image/png",
                        key=f"download_{idx}"
                    )
    
    with col2:
        st.markdown("### 🌟 Features")
        st.markdown("""
        - ✅ Generate realistic faces from noise
        - ✅ Watch denoising animation
        - ✅ Adjust generation parameters
        - ✅ Download generated images
        - ✅ Multiple image generation
        """)
        
        st.markdown("### 📊 Model Architecture")
        st.markdown("""
        - **U-Net** with skip connections
        - **Sinusoidal** time embeddings
        - **300** diffusion steps
        - **128x128** image resolution
        """)

# Additional section - Show example of intermediate steps without generation button
st.markdown("---")
st.header("📖 Understanding Diffusion")

st.markdown("""
The DDPM works by:
1. **Starting from random noise** (complete randomness)
2. **Progressively denoising** the image through 300 steps
3. **Learning to remove noise** using a neural network
4. **Gradually forming** face structure and details
""")

# Create a simple visual explanation
if st.checkbox("Show denoising process example"):
    if st.button("Generate Example"):
        model, diffusion_params = load_model()
        device = cfg.DEVICE
        show_intermediate_steps_grid(model, diffusion_params, device)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit | DDPM Face Generator | Powered by PyTorch"
    "</div>",
    unsafe_allow_html=True
)
