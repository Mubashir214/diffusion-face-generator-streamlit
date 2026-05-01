# app.py - Updated for ddpm_weights_only.pth
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image
import time
from io import BytesIO
import matplotlib.pyplot as plt
import os

# ==========================================
# Configuration
# ==========================================
class Config:
    IMG_SIZE = 128
    TIMESTEPS = 300
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHANNELS = 3
    # Updated to look for ddpm_weights_only.pth in root directory
    MODEL_PATHS = [
        "ddpm_weights_only.pth",  # Your weights file in root
        "ddpm_ffhq.pth",  # Alternative name
        "model.pth",  # Alternative name
        "checkpoint.pth"  # Alternative name
    ]

cfg = Config()

# ==========================================
# Model Architecture (Exactly matching your training)
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bn1(self.relu(self.conv1(x)))
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.bn2(self.relu(self.conv2(h)))
        return h

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        self.down1 = Block(cfg.CHANNELS, 64, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(64, 128, time_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = Block(128, 256, time_dim)
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_block1 = Block(256, 128, time_dim)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_block2 = Block(128, 64, time_dim)
        
        self.out = nn.Conv2d(64, cfg.CHANNELS, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        
        x1 = self.down1(x, t)
        p1 = self.pool1(x1)
        x2 = self.down2(p1, t)
        p2 = self.pool2(x2)
        x3 = self.down3(p2, t)
        
        u1 = self.up1(x3)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.up_block1(u1, t)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.up_block2(u2, t)
        
        return self.out(u2)

# ==========================================
# Diffusion Utilities
# ==========================================
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def denormalize(tensor):
    return (tensor + 1.0) / 2.0

def tensor_to_numpy(tensor):
    img = denormalize(tensor).permute(1, 2, 0).cpu().detach().numpy()
    img = np.clip(img, 0, 1)
    return img

# ==========================================
# Find and Load Model
# ==========================================
def find_model_file():
    """Find ddpm_weights_only.pth in root directory"""
    for path in cfg.MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None

@st.cache_resource
def load_model():
    """Load your trained ddpm_weights_only.pth"""
    device = cfg.DEVICE
    
    # Initialize model
    model = UNet().to(device)
    
    # Find model file
    model_path = find_model_file()
    
    if model_path:
        try:
            st.info(f"📂 Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Try different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    st.success("✅ Loaded model (format: model_state_dict)")
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    st.success("✅ Loaded model (format: state_dict)")
                else:
                    # Try to load directly
                    model.load_state_dict(checkpoint)
                    st.success("✅ Loaded model (direct format)")
            else:
                st.warning("⚠️ Checkpoint format not recognized")
                
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.warning("⚠️ Using randomly initialized model")
            
            # Show checkpoint keys for debugging
            if 'checkpoint' in locals():
                st.write("Checkpoint type:", type(checkpoint))
                if isinstance(checkpoint, dict):
                    st.write("Keys found:", checkpoint.keys())
    else:
        st.warning(f"⚠️ ddpm_weights_only.pth not found!")
        st.info("Please make sure ddpm_weights_only.pth is in the app directory")
        
        # Option to upload
        uploaded_file = st.file_uploader("Or upload your model file here", type=['pth'])
        if uploaded_file:
            with open("ddpm_weights_only.pth", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("✅ Model uploaded! Please click Generate again.")
            st.rerun()
    
    model.eval()
    return model

def setup_diffusion(device):
    betas = linear_beta_schedule(cfg.TIMESTEPS)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    return {
        'betas': betas.to(device),
        'alphas': alphas.to(device),
        'alphas_cumprod': alphas_cumprod.to(device),
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod.to(device),
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod.to(device)
    }

# ==========================================
# Generation Function with Intermediates
# ==========================================
@torch.no_grad()
def p_sample(model, x, t, t_index, diffusion_params, device):
    betas = diffusion_params['betas']
    alphas = diffusion_params['alphas']
    sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod']
    
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(1.0 / torch.sqrt(alphas), t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x.to(device), t) / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(betas, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_with_steps(model, diffusion_params, device, num_intermediates=10):
    """Generate image and return intermediate steps"""
    timesteps = cfg.TIMESTEPS
    
    # Start from random noise
    img = torch.randn((1, cfg.CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE), device=device)
    
    # Store intermediates
    intermediates = [img.clone().cpu()]
    
    # Calculate steps to capture
    step_size = max(1, timesteps // num_intermediates)
    capture_steps = list(range(timesteps - 1, -1, -step_size)) + [0]
    capture_steps = list(set(capture_steps))
    capture_steps.sort(reverse=True)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Denoising loop
    for i in reversed(range(timesteps)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, diffusion_params, device)
        
        if i in capture_steps:
            intermediates.append(img.clone().cpu())
        
        if i % (timesteps // 20) == 0:
            progress = (timesteps - i) / timesteps
            progress_bar.progress(progress)
            status_text.text(f"Denoising: Step {timesteps - i}/{timesteps} ({int(progress*100)}%)")
    
    progress_bar.empty()
    status_text.empty()
    
    return img.cpu(), intermediates

# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(
    page_title="DDPM Face Generator",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
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
    .step-caption {
        text-align: center;
        font-size: 12px;
        margin-top: 5px;
    }
    .model-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎨 DDPM Face Generator")
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='font-size: 18px;'>
            <b>Starts from random noise → Progressive denoising → Final face</b>
        </p>
        <p style='font-size: 14px; color: #666;'>
            Using trained model: ddpm_weights_only.pth
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    num_intermediates = st.slider("Intermediate steps to show", 5, 15, 10, 
                                   help="Number of denoising steps to display")
    
    st.markdown("---")
    st.header("ℹ️ Model Info")
    st.info(f"""
    - **Model File**: ddpm_weights_only.pth
    - **Resolution**: {cfg.IMG_SIZE}x{cfg.IMG_SIZE}
    - **Steps**: {cfg.TIMESTEPS}
    - **Device**: {cfg.DEVICE.upper()}
    """)
    
    st.markdown("---")
    st.header("💾 Model Status")
    
    # Check for model file
    model_exists = os.path.exists("ddpm_weights_only.pth")
    
    if model_exists:
        model_size = os.path.getsize("ddpm_weights_only.pth") / (1024 * 1024)
        st.success(f"✅ Model loaded!")
        st.info(f"File: ddpm_weights_only.pth\nSize: {model_size:.2f} MB")
    else:
        st.error("❌ ddpm_weights_only.pth not found")
        st.warning("Please add ddpm_weights_only.pth to the app directory")
        
        # File uploader as fallback
        uploaded_file = st.file_uploader("Upload ddpm_weights_only.pth", type=['pth'])
        if uploaded_file:
            with open("ddpm_weights_only.pth", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("✅ File uploaded! Please refresh.")
            st.rerun()

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🎨 Generate Face", type="primary", use_container_width=True):
        try:
            # Check if model exists
            if not os.path.exists("ddpm_weights_only.pth"):
                st.error("❌ ddpm_weights_only.pth not found! Please upload the model file first.")
                st.stop()
            
            # Load model
            with st.spinner("Loading model..."):
                model = load_model()
                diffusion_params = setup_diffusion(cfg.DEVICE)
            
            # Generate with intermediates
            with st.spinner(f"Generating over {cfg.TIMESTEPS} denoising steps..."):
                start_time = time.time()
                final_image, intermediates = sample_with_steps(
                    model, diffusion_params, cfg.DEVICE, num_intermediates
                )
                generation_time = time.time() - start_time
            
            # Success
            st.success(f"✅ Generation complete in {generation_time:.1f} seconds!")
            
            # Display intermediate steps in grid
            st.markdown("---")
            st.header("🔄 Denoising Process (Noise → Face)")
            
            num_steps = len(intermediates)
            cols_per_row = min(5, num_steps)
            
            # Create grid rows
            for row in range(0, num_steps, cols_per_row):
                cols = st.columns(cols_per_row)
                for idx, col in enumerate(cols):
                    step_idx = row + idx
                    if step_idx < num_steps:
                        with col:
                            img_numpy = tensor_to_numpy(intermediates[step_idx][0])
                            st.image(img_numpy, use_column_width=True)
                            
                            # Add captions
                            if step_idx == 0:
                                st.markdown("<p class='step-caption'>🎲 START<br>t=300 (Noise)</p>", unsafe_allow_html=True)
                            elif step_idx == num_steps - 1:
                                st.markdown("<p class='step-caption'>✨ FINAL<br>t=0 (Face)</p>", unsafe_allow_html=True)
                            else:
                                step = cfg.TIMESTEPS - (step_idx * (cfg.TIMESTEPS // num_intermediates))
                                progress = int((cfg.TIMESTEPS - step) / cfg.TIMESTEPS * 100)
                                st.markdown(f"<p class='step-caption'>Step t={step}<br>{progress}% denoised</p>", unsafe_allow_html=True)
            
            # Animation
            st.markdown("---")
            st.header("🎬 Animation")
            st.markdown("*Watch the transformation from noise to face*")
            
            anim_placeholder = st.empty()
            for idx, img_tensor in enumerate(intermediates):
                img_numpy = tensor_to_numpy(img_tensor[0])
                if idx == 0:
                    caption = "Starting from random noise..."
                elif idx == len(intermediates) - 1:
                    caption = "✨ Final generated face! ✨"
                else:
                    step = cfg.TIMESTEPS - (idx * (cfg.TIMESTEPS // num_intermediates))
                    caption = f"Denoising... Step {step}/{cfg.TIMESTEPS}"
                
                anim_placeholder.image(img_numpy, caption=caption, use_column_width=True)
                time.sleep(0.1)
            
            # Download button
            st.markdown("---")
            st.header("💾 Download")
            
            final_pil = Image.fromarray((tensor_to_numpy(final_image[0]) * 255).astype(np.uint8))
            buf = BytesIO()
            final_pil.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Generated Face",
                data=buf.getvalue(),
                file_name="generated_face.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure ddpm_weights_only.pth is in the correct location")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "DDPM Face Generator | Model: ddpm_weights_only.pth | 300 Diffusion Steps"
    "</div>",
    unsafe_allow_html=True
)
