import numpy as np 
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Lab_Dataset
import torch.optim as optim
from model import Colorizer, Discriminator, initilize_weights
import random
import time

batch_size = 8
epochs = 100
learning_rate = 5e-5
extra_epochs = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Lab_Dataset(
    color_space="CIELAB", 
    train=False,
    device=device,
)

fixed_l, fixed_ab = dataset[-1]
fixed_l.unsqueeze_(0)
fixed_image = dataset.rgb_image(-1) 

loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
)

colorizer = Colorizer(features=32).to(device)
discriminator = Discriminator(features=32).to(device)
initilize_weights(discriminator)
initilize_weights(colorizer)

optim_color = optim.Adam(colorizer.parameters(), lr=learning_rate, betas=(0.5, 0.99))
optim_disc = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.99))

# TODO use the reduction='none'
# TODO make it focus on the worse ones in the batch
# TODO learn what BCEWithLogitsLoss() does
critic = nn.BCELoss()

for epochs in range(epochs):

    for i, (L, real_ab) in enumerate(loader):

        for _ in range(extra_epochs):

            
            fake_ab = colorizer(L)

            fake_score = discriminator(L, fake_ab.detach())
            fake_loss = critic(fake_score, torch.zeros_like(fake_score))

            real_score = discriminator(L, real_ab)
            real_loss = critic(real_score, torch.ones_like(real_score))
            print(real_loss, fake_loss)

            mixed_loss = (real_loss + fake_loss)

            discriminator.zero_grad()
            mixed_loss.backward()
            optim_disc.step()


        fake_ab = colorizer(L)

        score = discriminator(L, fake_ab)
        loss = critic(score, torch.ones_like(score))
        print(loss)

        colorizer.zero_grad()
        loss.backward()
        optim_color.step()
    

        if i % 200 == 0:

            fake_ab = colorizer(fixed_l).detach()

            L_ab = torch.cat([fixed_l, fake_ab], dim=1).squeeze(0)
            L_ab = L_ab.squeeze(0).permute(1, 2, 0).cpu()

            fig, axes = plt.subplots(1, 2, figsize=(30, 10))

            axes[0].imshow(lab2rgb(L_ab))
            axes[0].axis("off")
            axes[0].set_title("AI Colored")

            axes[1].imshow(fixed_image)
            axes[1].axis("off")
            axes[1].set_title("Orginal")

            plt.tight_layout()
            plt.show()


