import torch
from config import Config

cfg = Config()
device = "cpu"

# =====================================
# Beta Schedule
# =====================================
def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)

betas = linear_beta_schedule(cfg.TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# =====================================
# Helper: Extract values at timestep
# =====================================
def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# =====================================
# Forward Diffusion (FOR UPLOAD FEATURE)
# =====================================
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alpha = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alpha * x_start + sqrt_one_minus * noise

# =====================================
# Reverse Step
# =====================================
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_recip_alphas = extract(1.0 / torch.sqrt(alphas), t, x.shape)
    sqrt_one_minus = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)

    model_mean = sqrt_recip_alphas * (
        x - betas_t * model(x, t) / sqrt_one_minus
    )

    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(betas_t) * noise

# =====================================
# Full Sampling (Noise → Image)
# =====================================
@torch.no_grad()
def sample_with_steps(model, shape):
    img = torch.randn(shape, device=device)

    steps = []
    step_interval = max(1, cfg.TIMESTEPS // 6)

    for i in reversed(range(cfg.TIMESTEPS)):
        t = torch.full((shape[0],), i, dtype=torch.long, device=device)
        img = p_sample(model, img, t, i)

        if i % step_interval == 0:
            steps.append(img.clone())

    return steps

# =====================================
# Denormalization
# =====================================
def denormalize(x):
    return torch.clamp((x + 1) / 2, 0, 1)
