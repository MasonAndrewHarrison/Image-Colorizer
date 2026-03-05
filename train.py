import numpy as np 
import matplotlib.pyplot as plt
import torch


gray_image = np.load("dataset/l/gray_scale.npy")
data = np.load("dataset/ab/ab/ab1.npy")

img = torch.tensor([data[0, :, :, :]])
gray_img = torch.tensor([gray_image[0, :, :]]).unsqueeze(3)
image = torch.cat([gray_img, img], dim=3).squeeze(0)
plt.imshow(image)
plt.show()