# model.py
import math
import torch
import torch.nn as nn

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
    def __init__(self, cfg):
        super().__init__()
        time_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Channel Progression: 64 -> 128 -> 256
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
