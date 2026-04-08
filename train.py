import numpy as np 
from skimage.color import lab2rgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Lab_Dataset
import torch.optim as optim
from utils import *
from models import Generator, Discriminator
import random
from torch.amp import autocast, GradScaler
import time
import yaml
import os
import warnings

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)["training"]

batch_size = config["batch_size"]
epochs = config["epochs"]
learning_rate = float(config["learning_rate"])
lambda_color = config["lambda_color"]
render_batch = (6, 6)
gen_features = config["generator_features"]
disc_features = config["discriminator_features"]
gen_update_freq = config["gen_update_freq"]
disc_update_freq = config["disc_update_freq"]

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

gen = Generator(features=gen_features).to(device)
disc = Discriminator(features=disc_features).to(device)

scaler_gen = GradScaler(device.__str__())
scaler_disc = GradScaler(device.__str__())

initilize_weights(disc)
initilize_weights(gen)

bin_weights = torch.load(f"Bin-Weights/{gen.mode}_weights.pth").to(device)

optim_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optim_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# TODO implement DeltaE   ΔE*ab = sqrt( (ΔL*)^2 + (Δa*)^2 + (Δb*)^2 )
# TODO create oklab dataset
# TODO use jacobian for the patch gan

criterion = nn.BCEWithLogitsLoss(reduction='mean')



def disc_fake_step(L):
    with torch.no_grad():
        fake_ab = gen(L)
    optim_disc.zero_grad()
    with autocast(device_type=device.__str__(), dtype=torch.float16):
        fake_score = disc(L, fake_ab)
        fake_loss = criterion(fake_score, torch.zeros_like(fake_score))
    scaler_disc.scale(fake_loss).backward()
    scaler_disc.step(optim_disc)
    scaler_disc.update()

def disc_real_step(L, real_ab):
    optim_disc.zero_grad()
    with autocast(device_type=device.__str__(), dtype=torch.float16):
        real_score = disc(L, real_ab)
        real_loss = criterion(real_score, torch.ones_like(real_score))
    scaler_disc.scale(real_loss).backward()
    scaler_disc.step(optim_disc)
    scaler_disc.update()

def disc_r1_step(L, real_ab):
    optim_disc.zero_grad()
    real_ab_grad = real_ab.detach().requires_grad_(True)
    real_score = disc(L, real_ab_grad)
    penalty = r1_penalty(real_ab_grad, real_score)
    scaler_disc.scale(penalty).backward()
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

    print(
        f"Mixed Loss: {loss.item():.2f} | "
        f"Color Loss: {color_loss.item():.2f} | "
        f"Gan Loss {gan_loss.item():.2f} | "
        f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB/"
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
    )  

    scaler_gen.scale(loss).backward()
    scaler_gen.step(optim_gen)
    scaler_gen.update()

print(f"Generator parameters:     {count_parameters(gen):,}")
print(f"Discriminator parameters: {count_parameters(disc):,}")

print(bin_weights.min(), bin_weights.max(), bin_weights.mean())

for epoch in range(epochs):

    for i, (L, real_ab) in enumerate(loader):

        L = L.to(device)
        real_ab = real_ab.to(device)

        disc_fake_step(L)
        disc_real_step(L, real_ab)
        if i % 16 == 0:
            disc_r1_step(L, real_ab)
        gen_step(L, real_ab)

        if i % 10 == 0:
            print(f"epoch: {epoch}/{epochs} || idx of: {i}/{len(loader)}")

        if i % 2 == 0:
            save_images(fixed_l_batch, render_batch=render_batch, gen_mode=gen)



