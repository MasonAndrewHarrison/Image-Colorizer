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

batch_size = 64
epochs = 100
learning_rate = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Lab_Dataset(color_space="CIELAB", train=False)
loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    pin_memory=True,
)

model = Colorizer(features=32).to(device)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

critic = nn.MSELoss()


for epochs in range(epochs):

    for i, (L, real_ab) in enumerate(loader):

        L = L.to(device)
        real_ab = real_ab.to(device)

        predicted_ab = model(L)

        loss = critic(predicted_ab, real_ab)

        
