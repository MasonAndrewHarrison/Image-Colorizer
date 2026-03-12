import numpy as np 
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Lab_Dataset
import torch.optim as optim
from utils import initilize_weights
from generator import Generator
from discriminator import Discriminator
import random
import time

batch_size = 12
epochs = 1000
learning_rate = 1e-4
extra_epochs = 5
lambda_color = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Lab_Dataset(
    color_space="CIELAB", 
    train=False,
    device=device,
)

fixed_l, fixed_ab = dataset[-11]
fixed_l.unsqueeze_(0)
fixed_image = dataset.rgb_image(-11) 

loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
)

gen = Generator(features=32).to(device)
disc = Discriminator(features=32).to(device)
initilize_weights(disc)
initilize_weights(gen)

optim_color = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optim_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# TODO implement VGG or DeltaE
# TODO create oklab dataset
# TODO create copic dataset

criterion = nn.BCEWithLogitsLoss(reduction='mean')

for epochs in range(epochs):

    for i, (L, real_ab) in enumerate(loader):

        for _ in range(extra_epochs):

            
            fake_ab = gen(L)

            fake_score = disc(L, fake_ab.detach())
            fake_loss = criterion(fake_score, torch.zeros_like(fake_score))

            real_score = disc(L, real_ab)
            real_loss = criterion(real_score, torch.ones_like(real_score))

            mixed_loss = (real_loss + fake_loss)

            disc.zero_grad()
            mixed_loss.backward()
            optim_disc.step()

        #TODO make this create just on output and add logits_to_ab in utils
        fake_ab = gen(L)
        logits = gen(L, return_logits=True)

        scores = disc(L, fake_ab)
        gan_loss = criterion(scores, torch.ones_like(scores))

        #TODO create ab_to_bins(ab, mode..., gen.pts_in_hull)
        #TODO create bin_weights_(mode)
        # loss = -weight[y] * log(softmax(logits)[y])
        """target_bins = ab_to_bins(real_ab)
        color_loss = F.cross_entropy(
            logits,
            target_bins,
            weight=bin_weights
        )"""

        loss = gan_loss + lambda_color * color_loss
        
        gen.zero_grad()
        loss.backward()
        optim_color.step()
    

        if i == 0 and epochs % 5 == 0:
 
            fake_ab = gen(fixed_l).detach()

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


