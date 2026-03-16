import numpy as np 
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Lab_Dataset
import torch.optim as optim
from utils import *
from generator import Generator
from discriminator import Discriminator
import random
from torch.amp import autocast, GradScaler
import time
import os

batch_size = 28
epochs = 1000
learning_rate = 5e-5
extra_epochs = 3
lambda_color = 10
render_batch = (5, 5)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Lab_Dataset(
    color_space="CIELAB", 
    train=True,
    metadata_mode='r',
)

fixed_l, fixed_ab = dataset[-11]
fixed_l = fixed_l.to(device)
fixed_ab = fixed_ab.to(device)
fixed_l.unsqueeze_(0)

fixed_l_batch,_ = dataset[-(render_batch[0] * render_batch[1]):]
fixed_image = dataset.rgb_image(-11)
fixed_l_batch = fixed_l_batch.to(device)

loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    num_workers=4,
    prefetch_factor=2,
)

gen = Generator(features=32).to(device)
disc = Discriminator(features=32).to(device)

scaler_gen = GradScaler(device.__str__())
scaler_disc = GradScaler(device.__str__())

initilize_weights(disc)
initilize_weights(gen)

bin_weights = torch.load(f"Bin-Weights/{gen.mode}_weights.pth").to(device)

optim_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optim_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# TODO implement DeltaE   ΔE*ab = sqrt( (ΔL*)^2 + (Δa*)^2 + (Δb*)^2 )
# TODO consider gradient penalies
# TODO create oklab dataset
# TODO create copic dataset

criterion = nn.BCEWithLogitsLoss(reduction='mean')

def disc_step(L, real_ab, extra_epochs: int = extra_epochs):

    with torch.no_grad():
        fake_ab = gen(L) 

    for _ in range(extra_epochs):

        optim_disc.zero_grad()

        with autocast(device_type=device.__str__(), dtype=torch.float16):

            fake_score = disc(L, fake_ab)
            fake_loss = criterion(fake_score, torch.zeros_like(fake_score))

            real_score = disc(L, real_ab)
            real_loss = criterion(real_score, torch.ones_like(real_score))

        mixed_loss = (real_loss + fake_loss)

        scaler_disc.scale(mixed_loss).backward()
        scaler_disc.step(optim_disc)
        scaler_disc.update()

def gen_step(L, real_ab):

    optim_gen.zero_grad()

    with autocast(device_type=device.__str__(), dtype=torch.float16):
        logits = gen(L, return_logits=True)
        fake_ab = logits_to_ab(logits, gen.pts_in_hull)

        scores = disc(L, fake_ab)
        gan_loss = criterion(scores, torch.ones_like(scores))

        with torch.no_grad():
            target_bins = ab_to_bins(
                real_ab.detach(), 
                gen.mode, 
                gen.pts_in_hull.detach(), 
                return_bin_index=True
            )

        color_loss = F.cross_entropy(
            logits,
            target_bins,
            weight=bin_weights
        )

    loss = gan_loss + lambda_color * color_loss

    print(loss.item(), color_loss.item(), gan_loss.item())
    print(
        f"loss generator: {loss.item():.4f} | "
        f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB / "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
    )  

    scaler_gen.scale(loss).backward()
    scaler_gen.step(optim_gen)
    scaler_gen.update()

print(f"Generator parameters:     {count_parameters(gen):,}")
print(f"Discriminator parameters: {count_parameters(disc):,}")

for epoch in range(epochs):

    for i, (L, real_ab) in enumerate(loader):

        L = L.to(device)
        real_ab = real_ab.to(device)

        disc_step(L, real_ab)
        torch.cuda.empty_cache()
        gen_step(L, real_ab)

        if i == 0:
            gen.eval()
            fig, axes = plt.subplots(render_batch[0], render_batch[1], figsize=(15, 15))

            with torch.no_grad():

                print(fixed_l_batch.shape)
                fake_ab = gen(fixed_l_batch).detach()
                L_ab = torch.cat([fixed_l_batch, fake_ab], dim=1).squeeze(0)   
                L_ab = L_ab.squeeze(0).permute(0, 2, 3, 1).cpu()
                rgb_image = lab2rgb(L_ab)
                print(rgb_image.shape)

                for idx, ax in enumerate(axes.flat):

                    ax.imshow(rgb_image[idx, :, :, :]) 
                    ax.axis("off")

            
            plt.tight_layout()
            plt.savefig("output.png")
            plt.close()
            gen.train()

        if i == 0 and epoch % 50 == -1:

            gen.eval()
            with torch.no_grad():

                fake_ab = gen(fixed_l).detach()

                L_ab = torch.cat([fixed_l, fake_ab], dim=1).squeeze(0)
                L_ab = L_ab.squeeze(0).permute(1, 2, 0).cpu()
                rgb_image = lab2rgb(L_ab)
                fig, axes = plt.subplots(1, 2, figsize=(30, 10))

                axes[0].imshow(rgb_image)
                axes[0].axis("off")
                axes[0].set_title("AI Colored")

                axes[1].imshow(fixed_image)
                axes[1].axis("off")
                axes[1].set_title("Orginal")

                plt.tight_layout()
                plt.show()                             
            gen.train()       


