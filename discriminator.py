import torch
import torch.nn as nn
from torchsummary import summary

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
                nn.Linear(28*28, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
        )
        
    def forward(self, x):
        # x is batch_size*28*28 dim tensor if sampled from the Fashion-MNIST
        # if its from the generator, a 784 tensor
        x = x.view(x.size(0), 28*28)
        out = self.model(x)
        return out

class ConvDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = self.get_stem_module()
        
    def get_stem_module(self):
        m = nn.Sequential(
            nn.Conv2d(1, 256, 2, 2, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.GELU()
            )
        return m
        
    def forward(self, x):
        x = self.stem(x) # b, 256, 14, 14
        return x
class ConvDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        validity = validity.squeeze()
        validity = validity.unsqueeze(dim=-1)
        return validity

class LightConvDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        validity = validity.squeeze()
        validity = validity.unsqueeze(dim=-1)
        return validity
    

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # m = Discriminator().to(device)
    m = LightConvDiscriminator().to(device)
    print(m(torch.Tensor(64,1,28,28).to(device)).shape)
    summary(m, (1,28,28), batch_size=64, device="cuda")