import torch
import torch.nn as nn
from torchsummary import summary
from einops.layers.torch import Rearrange
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.z_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        # z = batch_size*100(z_dim) dim tensor
        out = self.model(z)
        return out

class ConvGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 3, bias=False),
            nn.Tanh()
        )

    
    
    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(dim=-1)
        x = x.unsqueeze(dim=-1)
        x = self.model(x)
        return x
        
class DeepConvGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        x = x.unsqueeze(dim=-1)
        x = self.model(x)
        x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        return x
    

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # m = ConvGenerator(latent_dim=100).to(device)
    m = DeepConvGenerator(latent_dim=100).to(device)
    output = m(torch.Tensor(64,100).to(device))
    print(f"output shape: {output.shape}")
    
    m = Generator().to(device)
    summary(m, tuple([100]), batch_size=64,device="cuda")