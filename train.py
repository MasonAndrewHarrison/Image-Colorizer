import numpy as np 
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import torch
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

start = time.time()

dataset = Lab_Dataset(train=False)
loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    pin_memory=True,
)

model = Colorizer(features=32)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

end = time.time()

print(end-start)