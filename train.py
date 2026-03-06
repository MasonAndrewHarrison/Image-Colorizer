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

batch_size = 16
epochs = 100
learning_rate = 3e-3
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

optim_color = optim.Adam(colorizer.parameters(), lr=learning_rate, betas=(0.9, 0.999))
optim_disc = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999))

# TODO use the reduction='none'
# TODO make it focus on the worse ones in the batch
critic = nn.BCELoss()

for epochs in range(epochs):

    for i, (L, real_ab) in enumerate(loader):

        for i in range(extra_epochs):

            optim_disc.zero_grad()
            predicted_ab = colorizer(L)

            predicted_score = discriminator(L, predicted_ab)
            predicted_loss = critic(predicted_score, torch.ones_like(predicted_score))

            real_score = discriminator(L, real_ab)
            real_loss = critic(real_score, torch.ones_like(real_score))

            mixed_loss = real_loss + predicted_loss
            mixed_loss.backward()
            optim_disc.step()



        optim_color.zero_grad()

        predicted_ab = colorizer(L.detach())
        #print(predicted_ab.min(), predicted_ab.max())
        score = discriminator(L, predicted_ab)
        loss = critic(score, torch.ones_like(score))

        loss.backward()
        optim_color.step()

        


        
        #print(f"loss: {loss:.2f}, norm: {total_norm:.2f}")

        if i % 250 == 0:

            predicted_ab = colorizer(fixed_l).detach()

            L_ab = torch.cat([fixed_l, predicted_ab], dim=1).squeeze(0)
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


