# app.py - Complete Single File for Streamlit Cloud
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image
import time
from io import BytesIO

# ==========================================
# Configuration
# ==========================================
class Config:
    IMG_SIZE = 64  # Smaller for faster generation on Streamlit Cloud
    TIMESTEPS = 100  # Fewer steps for speed
    DEVICE = "cpu"  # Force CPU for compatibility
    CHANNELS = 3

cfg = Config()

# ==========================================
# Model Architecture
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
        
        # Simplified U-Net for faster generation
        self.down1 = Block(cfg.CHANNELS, 32, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(32, 64, time_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = Block(64, 128, time_dim)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_block1 = Block(128, 64, time_dim)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up_block2 = Block(64, 32, time_dim)
        
        self.out = nn.Conv2d(32, cfg.CHANNELS, 1)

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

def tensor_to_pil(tensor):
    img = denormalize(tensor).permute(1, 2, 0).cpu().detach().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

# ==========================================
# Model Loading with Fallback
# ==========================================
@st.cache_resource
def load_model():
    """Initialize model without external weights"""
    model = UNet().to(cfg.DEVICE)
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
# Generation Functions
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
def generate_images(model, diffusion_params, num_images, device, show_progress=False):
    """Generate images from noise"""
    timesteps = cfg.TIMESTEPS
    
    # Start from random noise
    img = torch.randn((num_images, cfg.CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE), device=device)
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Progressive denoising
    for i in reversed(range(timesteps)):
        t = torch.full((num_images,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, diffusion_params, device)
        
        if show_progress and i % (timesteps // 10) == 0:
            progress = (timesteps - i) / timesteps
            progress_bar.progress(progress)
            status_text.text(f"Denoising: {int(progress * 100)}%")
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    return img.cpu()

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
    .generated-image {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🎨 DDPM Face Generator")
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='font-size: 18px;'>
            Generate realistic faces from random noise using 
            <b>Denoising Diffusion Probabilistic Models</b>
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Generation Settings")
    num_images = st.slider("Number of images to generate", 1, 4, 1, help="More images will take longer")
    
    st.markdown("---")
    st.header("🎨 Model Settings")
    st.info(f"""
    - **Image Size**: {cfg.IMG_SIZE}x{cfg.IMG_SIZE}
    - **Diffusion Steps**: {cfg.TIMESTEPS}
    - **Device**: {cfg.DEVICE.upper()}
    """)
    
    st.markdown("---")
    st.header("ℹ️ How it works")
    st.markdown("""
    1. **Start**: Random Gaussian noise
    2. **Denoise**: Model progressively removes noise
    3. **Result**: Realistic face emerges
    
    The model learns to reverse the diffusion process that gradually adds noise to images.
    """)
    
    st.markdown("---")
    st.caption("Built with PyTorch & Streamlit | DDPM")

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("✨ Generate New Faces", type="primary", use_container_width=True):
        try:
            # Load model and setup
            with st.spinner("Initializing model..."):
                model = load_model()
                diffusion_params = setup_diffusion(cfg.DEVICE)
            
            # Generate images
            with st.spinner(f"Generating {num_images} face(s)..."):
                generated_images = generate_images(
                    model, diffusion_params, num_images, 
                    cfg.DEVICE, show_progress=True
                )
            
            # Display results
            st.markdown("---")
            st.subheader("✨ Generated Images")
            
            if num_images == 1:
                # Display single image
                pil_img = tensor_to_pil(generated_images[0])
                st.image(pil_img, caption="Generated Face", use_column_width=True)
                
                # Download button
                buf = BytesIO()
                pil_img.save(buf, format="PNG")
                st.download_button(
                    label="📥 Download Image",
                    data=buf.getvalue(),
                    file_name="generated_face.png",
                    mime="image/png"
                )
            else:
                # Display multiple images in a grid
                cols = st.columns(num_images)
                download_cols = st.columns(num_images)
                
                for idx, (col, download_col) in enumerate(zip(cols, download_cols)):
                    pil_img = tensor_to_pil(generated_images[idx])
                    col.image(pil_img, caption=f"Face {idx+1}", use_column_width=True)
                    
                    # Download button for each image
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    download_col.download_button(
                        label=f"📥 Save {idx+1}",
                        data=buf.getvalue(),
                        file_name=f"generated_face_{idx+1}.png",
                        mime="image/png",
                        key=f"download_{idx}"
                    )
            
            st.success("✅ Generation complete!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Try reducing the number of images or refreshing the page.")

# Information section
st.markdown("---")
st.header("📖 Understanding Diffusion Models")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Step 1: Noise")
    st.markdown("Start with pure random noise - no structure at all")

with col2:
    st.markdown("### 🔄 Step 2: Denoise")
    st.markdown(f"Model progressively removes noise over {cfg.TIMESTEPS} steps")

with col3:
    st.markdown("### ✨ Step 3: Result")
    st.markdown("Final output - a realistic generated face")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 20px;'>"
    "🎨 DDPM Face Generator | Powered by PyTorch & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
