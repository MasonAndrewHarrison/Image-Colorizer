import numpy as np 
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Lab_Dataset
import torch.optim as optim
from model import Colorizer, initilize_weights
import random
import time

batch_size = 16
epochs = 100
learning_rate = 3e-3

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

model = Colorizer(features=64).to(device)
initilize_weights(model)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# TODO use the reduction='none'
# TODO make it focus on the worse ones in the batch
critic = nn.HuberLoss(delta=10)


for epochs in range(epochs):

    for i, (L, real_ab) in enumerate(loader):

        optimizer.zero_grad()

        predicted_ab = model(L)
        print(predicted_ab.min(), predicted_ab.max())
        loss = critic(predicted_ab, real_ab)

        loss.backward()
        optimizer.step()

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        #print(f"loss: {loss:.2f}, norm: {total_norm:.2f}")

        if i % 25 == 0:

            predicted_ab = model(fixed_l).detach()

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


