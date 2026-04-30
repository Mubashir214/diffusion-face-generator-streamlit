import torch
from config import Config

cfg = Config()

def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)

betas = linear_beta_schedule(cfg.TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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

@torch.no_grad()
def sample_with_steps(model, shape):
    device = "cpu"
    img = torch.randn(shape, device=device)

    steps = []
    for i in reversed(range(cfg.TIMESTEPS)):
        t = torch.full((shape[0],), i, dtype=torch.long)
        img = p_sample(model, img, t, i)

        if i % (cfg.TIMESTEPS // 6) == 0:
            steps.append(img.clone())

    return steps

def denormalize(x):
    return (x + 1) / 2
