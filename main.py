import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid

import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from generator import Generator, ConvGenerator, DeepConvGenerator
from discriminator import Discriminator, ConvDiscriminator, LightConvDiscriminator


## hyper params
SEED = 42
BATCH_SIZE = 64 * 40
EPOCHS = 200
LEARNING_RATE = 1e-4
INFERENCE_INTERVAL = 10
OPTIM = "AdamW"
EXP_NAME = "LatentDim2048_DeepConvGenerator_LightConvDiscriminator"
SAVE_TAG = True
LATENT_DIM = 2048

device = "cuda:0" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=0.2860, std=0.3530)
     ]
)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# for update generator
def generator_step(batch_size, generator, discriminator, optim_g:torch.optim.Optimizer, criterion:nn.BCELoss):
    optim_g.zero_grad()
    
    z = Variable(torch.randn(batch_size, LATENT_DIM)).to(device)
    fake_imgs = generator(z)
    preds = discriminator(fake_imgs)
    
    loss_g = criterion(preds, Variable(torch.ones(batch_size)).unsqueeze(dim=1).to(device))
    loss_g.backward()
    optim_g.step()
    return loss_g.data

# for loss calc(per epoch)
def calc_mean(input:list):
    return sum(input)/len(input)

# for setting optim
def get_optimizer(optimizer_name, model:nn.Module, learning_rate):
    optimizer_cls = getattr(optim, optimizer_name, None)
    if optimizer_cls is None or not issubclass(optimizer_cls, optim.Optimizer):
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")
    # optimizer_cls(model.parameters(), lr=learning_rate)
    return optimizer_cls(model.parameters(), lr=learning_rate)

# for update discriminator params
def discriminator_step(generator, discriminator, optim_d, criterion:nn.BCELoss, real_imgs):
    optim_d.zero_grad()
    
    # real loss
    preds = discriminator(real_imgs)
    sample_batch_size = preds.size(0)
    loss_real = criterion(preds, Variable(torch.ones(sample_batch_size)).unsqueeze(dim=1).to(device))

    # fake loss
    z = Variable(torch.randn(sample_batch_size, LATENT_DIM)).to(device)
    
    fake_imgs = generator(z)
    preds = discriminator(fake_imgs)
    
    loss_fake = criterion(preds, 
                          Variable(torch.zeros(sample_batch_size)).unsqueeze(dim=1).to(device))
    
    loss_d = (loss_real + loss_fake) * 0.5 # mean
    loss_d.backward()
    optim_d.step()
    return loss_d.data
    
    

if __name__ == "__main__":
    seed_everything(SEED)
    
    # generator = Generator(latent_dim=LATENT_DIM).to(device) # mlp generator
    # generator = ConvGenerator(latent_dim=LATENT_DIM).to(device) # conv generator
    generator = DeepConvGenerator(latent_dim=LATENT_DIM).to(device) # conv generator
    
    # discriminator = Discriminator().to(device) # discriminator
    # discriminator = ConvDiscriminator().to(device)
    discriminator = LightConvDiscriminator().to(device)
    
    # dataset & dataloader
    os.makedirs("./data/mnist", exist_ok=True) # data download
    os.makedirs(f"./result/{EXP_NAME}", exist_ok=True) # for save imgs
    dataloader = DataLoader(
        datasets.FashionMNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=transform
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    ) # make dataloader
    
    
    # loss func
    criterion = nn.BCELoss()
    
    
    # optimizer
    optim_g = get_optimizer(optimizer_name=OPTIM, # Adam?
                            model=generator, 
                            learning_rate=LEARNING_RATE)
    
    optim_d = get_optimizer(optimizer_name=OPTIM, 
                            model=discriminator, 
                            learning_rate=LEARNING_RATE)

    # total loss
    total_loss_d = []
    total_loss_g = []
    
    # train + save imgs
    for epoch in range(EPOCHS):
        logger_d = [] # for each epoch log
        logger_g = []
        
        for i, (imgs, labels) in enumerate(dataloader):
            real_imgs = Variable(imgs).to(device)
            generator.train()
            loss_d = discriminator_step(generator, discriminator, optim_d, criterion, real_imgs)
            loss_g = generator_step(labels.size(0), generator, discriminator, optim_g, criterion)
            logger_d.append(loss_d.item())
            logger_g.append(loss_g.item())
        
        epoch_d_loss = calc_mean(logger_d)
        epoch_g_loss = calc_mean(logger_g)
        
        total_loss_d.append(epoch_d_loss)
        total_loss_g.append(epoch_g_loss)
        
        print(f"Epoch: {epoch+1}  loss_g: {epoch_g_loss:.4f}, loss_d: {epoch_d_loss:.4f}")
        
        if (epoch+1) % INFERENCE_INTERVAL == 0:
            print(f"    epoch {epoch+1} image save: epoch{epoch+1}_result.png")
            
            generator.eval()
            z = Variable(torch.randn(9, LATENT_DIM)).to(device)
            sample_imgs = generator(z)
            sample_imgs = sample_imgs.view(sample_imgs.size(0), 28, 28).unsqueeze(dim=1).data.cpu()
            
            grid = make_grid(sample_imgs, nrow=3, normalize=True).permute(1,2,0).numpy()
            plt.clf() # clean plt
            plt.imshow(grid)
            plt.axis('off')
            plt.savefig(f"./result/{EXP_NAME}/epoch{epoch+1}_result.png")
    
            plt.clf() # clean plt
            plt.plot(range(epoch+1), total_loss_g, 'b', label="generator loss")
            plt.plot(range(epoch+1), total_loss_d, 'r', label="discriminator loss")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Generator & Discriminator Loss')
            plt.legend()
            plt.savefig(f'./loss_plot_{EXP_NAME}.png')
    
    if SAVE_TAG is True:
        torch.save(generator.state_dict(), f'./result/{EXP_NAME}/_generator.pth')
        torch.save(discriminator.state_dict(), f'./result/{EXP_NAME}/_discriminator.pth')

