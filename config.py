# config.py
class Config:
    IMG_SIZE = 128
    TIMESTEPS = 300
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHANNELS = 3
    MODEL_PATH = "models/ddpm_ffhq.pth"  # Path to your trained model weights

cfg = Config()
