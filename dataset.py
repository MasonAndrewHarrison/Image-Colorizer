import torch
import numpy as np 
from skimage.color import lab2rgb

class Lab_Dataset():

    def __init__(self, color_space, device, train: bool = True):
        """
        color_space is expected it be either "CIELAB" or "OKLAB".
        """

        self.device = device

        main_folder = f"{color_space}-Dataset"
        filename = f"{main_folder}/train" if train else f"{main_folder}/test"

        ab = np.load(f"{filename}/ab.npy")
        L = np.load(f"{filename}/L.npy")

        self.ab = torch.tensor(ab, dtype=torch.float32, device=device) - 128
        self.L = torch.tensor(L, dtype=torch.float32, device=device)

    def __len__(self):

        return len(self.L)

    def __getitem__(self, idx):

        L = self.L[idx, :, :].unsqueeze(0)
        ab = self.ab[idx, :, : , :].permute(2, 0, 1)

        return (L, ab)

    def rgb_image(self, idx):

        L, ab = self[idx]
        L = L.permute(1, 2, 0)
        ab = ab.permute(1, 2, 0)

        L_ab = torch.cat([L, ab], dim=2).cpu()
        rgb_image = lab2rgb(L_ab)
        return rgb_image