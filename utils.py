# utils.py
import torch
import numpy as np
from PIL import Image

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def denormalize(tensor):
    """Converts [-1, 1] tensor back to [0, 1] for visualization."""
    return (tensor + 1.0) / 2.0

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image."""
    img = denormalize(tensor).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def setup_diffusion(cfg, device):
    """Setup diffusion parameters."""
    betas = linear_beta_schedule(cfg.TIMESTEPS).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod
    }
