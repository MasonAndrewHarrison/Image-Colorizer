import numpy as np 
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import torch
from dataset import Lab_Dataset
import random


dataset = Lab_Dataset()
L, ab = dataset.__getitem__(0)

L_ab = torch.cat([L, ab], dim=3).squeeze(0)
image = lab2rgb(L_ab)
plt.imshow(image)
plt.show()