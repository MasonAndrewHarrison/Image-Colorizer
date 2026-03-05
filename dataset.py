import torch
import numpy as np 
from skimage.color import lab2rgb

class Lab_Dataset():

    def __init__(self, train: bool = True):

        filename = "dataset/train" if train else "dataset/test"

        ab = np.load(f"{filename}/ab.npy")
        L = np.load(f"{filename}/L.npy")

        self.ab = torch.tensor(ab, dtype=torch.float32)
        self.L = torch.tensor(L, dtype=torch.float32)

    def __len__(self):

        return len(self.L)

    def __getitem__(self, idx):

        L = self.L[idx, :, :].unsqueeze(2)
        ab = self.ab[idx, :, : , :]
        return (L, ab)

    def rgb_image(self, idx):

        L, ab = self[idx]

        L_ab = torch.cat([L, ab], dim=2).squeeze(0)
        rgb_image = lab2rgb(L_ab)
        return rgb_image