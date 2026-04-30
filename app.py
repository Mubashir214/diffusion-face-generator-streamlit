# app.py - Complete with Intermediate Denoising Steps Display
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image
import time
from io import BytesIO
import matplotlib.pyplot as plt

# ==========================================
# Configuration
# ==========================================
class Config:
    IMG_SIZE = 64  # Smaller for faster generation
    TIMESTEPS = 100  # Total diffusion steps
    DEVICE = "cpu"
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

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for display"""
    img = denormalize(tensor).permute(1, 2, 0).cpu().detach().numpy()
    img = np.clip(img, 0, 1)
    return img

# ==========================================
# Model Loading
# ==========================================
@st.cache_resource
def load_model():
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
# Generation with Intermediate Steps
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
def generate_with_intermediates(model, diffusion_params, device, num_intermediates=10):
    """Generate image and return intermediate denoising steps"""
    timesteps = cfg.TIMESTEPS
    
    # Start from random noise
    img = torch.randn((1, cfg.CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE), device=device)
    
    # Store initial noise
    intermediates = [img.clone().cpu()]
    
    # Calculate which steps to capture
    step_size = timesteps // num_intermediates
    capture_steps = list(range(timesteps - 1, -1, -step_size)) + [0]
    capture_steps = list(set(capture_steps))  # Remove duplicates
    capture_steps.sort(reverse=True)
    
    # Progressive denoising
    for i in reversed(range(timesteps)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, diffusion_params, device)
        
        # Capture intermediate step
        if i in capture_steps:
            intermediates.append(img.clone().cpu())
    
    return img.cpu(), intermediates

# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(
    page_title="DDPM Face Generator - Denoising Steps",
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
        margin: 10px 0;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .step-caption {
        text-align: center;
        font-size: 12px;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🎨 DDPM Face Generator")
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h3>Starting from Random Noise → Progressive Denoising → Final Face</h3>
        <p style='font-size: 16px;'>
            Watch as the model gradually transforms pure noise into a realistic face!
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Generation Settings")
    
    num_intermediates = st.slider(
        "Number of intermediate steps to show", 
        min_value=5, 
        max_value=20, 
        value=10,
        help="More steps show finer details of the denoising process"
    )
    
    st.markdown("---")
    st.header("🎨 Model Info")
    st.info(f"""
    - **Image Size**: {cfg.IMG_SIZE}x{cfg.IMG_SIZE}
    - **Total Diffusion Steps**: {cfg.TIMESTEPS}
    - **Device**: {cfg.DEVICE.upper()}
    - **Architecture**: U-Net with Attention
    """)
    
    st.markdown("---")
    st.header("📖 How It Works")
    st.markdown("""
    **Denoising Diffusion Process:**
    
    1. **Start** → Pure random noise
    2. **Step 1** → Begins to see patterns
    3. **Middle** → Face structure emerges
    4. **Final** → Clear face appears
    
    The model learns to reverse the process of gradually adding noise to images.
    """)
    
    st.markdown("---")
    st.caption("🎯 DDPM: Denoising Diffusion Probabilistic Models")

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🚀 Generate Face & Show Denoising Steps", type="primary", use_container_width=True):
        try:
            # Load model
            with st.spinner("Initializing diffusion model..."):
                model = load_model()
                diffusion_params = setup_diffusion(cfg.DEVICE)
            
            # Generate with intermediate steps
            with st.spinner(f"Generating face over {cfg.TIMESTEPS} denoising steps..."):
                start_time = time.time()
                final_image, intermediates = generate_with_intermediates(
                    model, diffusion_params, cfg.DEVICE, num_intermediates
                )
                generation_time = time.time() - start_time
            
            # Success message
            st.success(f"✅ Generation complete in {generation_time:.1f} seconds!")
            
            # ==========================================
            # Display Intermediate Denoising Steps
            # ==========================================
            st.markdown("---")
            st.header("🔄 Denoising Process: From Noise to Face")
            st.markdown("*Watch how the image evolves from pure random noise to a clear face*")
            
            # Display intermediates in a grid
            num_steps = len(intermediates)
            cols_per_row = min(5, num_steps)
            num_rows = (num_steps + cols_per_row - 1) // cols_per_row
            
            for row in range(num_rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    step_idx = row * cols_per_row + col_idx
                    if step_idx < num_steps:
                        with cols[col_idx]:
                            # Calculate step progress percentage
                            step_number = len(intermediates) - 1 - step_idx
                            progress_percent = int((step_number / cfg.TIMESTEPS) * 100)
                            
                            # Display image
                            img_numpy = tensor_to_numpy(intermediates[step_idx][0])
                            st.image(img_numpy, use_column_width=True)
                            
                            # Display caption
                            if step_idx == 0:
                                st.markdown("<p class='step-caption'>🎲 <b>START</b><br>Random Noise</p>", unsafe_allow_html=True)
                            elif step_idx == len(intermediates) - 1:
                                st.markdown("<p class='step-caption'>✨ <b>FINAL</b><br>Generated Face</p>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<p class='step-caption'><b>Step {step_number}</b><br>{progress_percent}% denoised</p>", unsafe_allow_html=True)
            
            # ==========================================
            # Side-by-side comparison: Start vs Final
            # ==========================================
            st.markdown("---")
            st.header("📊 Start vs Final Comparison")
            
            col_start, col_arrow, col_final = st.columns([2, 1, 2])
            
            with col_start:
                st.markdown("### 🎲 Initial State")
                start_img = tensor_to_numpy(intermediates[0][0])
                st.image(start_img, caption="Pure Random Noise", use_column_width=True)
                st.markdown("*No structure, completely random pixels*")
            
            with col_arrow:
                st.markdown("<br><br><br><h1 style='text-align: center;'>→</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Denoising<br>Process</p>", unsafe_allow_html=True)
            
            with col_final:
                st.markdown("### ✨ Final Result")
                final_img = tensor_to_numpy(final_image[0])
                st.image(final_img, caption="Generated Face", use_column_width=True)
                st.markdown("*Clear, realistic face emerges*")
            
            # ==========================================
            # Animation of denoising process
            # ==========================================
            st.markdown("---")
            st.header("🎬 Denoising Animation")
            st.markdown("*Watch the transformation in real-time*")
            
            animation_placeholder = st.empty()
            
            # Create animation
            for idx, img_tensor in enumerate(intermediates):
                img_numpy = tensor_to_numpy(img_tensor[0])
                step_number = len(intermediates) - 1 - idx
                progress = int((step_number / cfg.TIMESTEPS) * 100)
                
                caption = f"Denoising Step: {step_number} / {cfg.TIMESTEPS} ({progress}% complete)"
                animation_placeholder.image(img_numpy, caption=caption, use_column_width=True)
                time.sleep(0.1)  # Small delay for animation effect
            
            # Show final image again
            animation_placeholder.image(tensor_to_numpy(final_image[0]), caption="Final Generated Face!", use_column_width=True)
            
            # ==========================================
            # Download Options
            # ==========================================
            st.markdown("---")
            st.header("💾 Save Your Results")
            
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                # Download final image
                final_pil = Image.fromarray((tensor_to_numpy(final_image[0]) * 255).astype(np.uint8))
                buf = BytesIO()
                final_pil.save(buf, format="PNG")
                st.download_button(
                    label="📥 Download Final Face",
                    data=buf.getvalue(),
                    file_name="generated_face.png",
                    mime="image/png"
                )
            
            with col_download2:
                # Create and download a grid of all intermediate steps
                fig, axes = plt.subplots(2, (num_steps + 1) // 2, figsize=(15, 6))
                axes = axes.flatten() if num_steps > 2 else [axes]
                
                for idx, ax in enumerate(axes):
                    if idx < num_steps:
                        img_numpy = tensor_to_numpy(intermediates[idx][0])
                        ax.imshow(img_numpy)
                        step_number = len(intermediates) - 1 - idx
                        ax.set_title(f"Step {step_number}", fontsize=8)
                        ax.axis('off')
                    else:
                        ax.axis('off')
                
                plt.tight_layout()
                
                # Save grid to buffer
                buf_grid = BytesIO()
                plt.savefig(buf_grid, format='PNG', dpi=150, bbox_inches='tight')
                buf_grid.seek(0)
                plt.close()
                
                st.download_button(
                    label="🖼️ Download All Steps (Grid)",
                    data=buf_grid.getvalue(),
                    file_name="denoising_steps_grid.png",
                    mime="image/png"
                )
            
        except Exception as e:
            st.error(f"❌ Error during generation: {str(e)}")
            st.info("Please try again or refresh the page.")

# Information section at bottom
st.markdown("---")
st.header("🎓 Understanding the Denoising Process")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1️⃣ Random Noise")
    st.markdown("The process starts with pure Gaussian noise - no recognizable patterns or structures.")

with col2:
    st.markdown("### 2️⃣ Progressive Denoising")
    st.markdown(f"Over {cfg.TIMESTEPS} steps, the neural network gradually removes noise while preserving meaningful patterns.")

with col3:
    st.markdown("### 3️⃣ Face Emergence")
    st.markdown("Facial features slowly appear - first shapes, then details like eyes, nose, and mouth.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 20px;'>"
    "🎨 DDPM Face Generator | Showing Complete Denoising Process | Powered by PyTorch & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
