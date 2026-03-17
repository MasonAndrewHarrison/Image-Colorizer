import torch
import numpy as np 
from skimage.color import lab2rgb
import warnings

class Lab_Dataset():

    def __init__(self, color_space, train: bool = True, metadata_mode: str = 'r'):
        """
        color_space is expected it be either "CIELAB" or "OKLAB".
        """

        main_folder = f"{color_space}-Dataset"
        filename = f"{main_folder}/train" if train else f"{main_folder}/test"

        self.ab = np.load(f"{filename}/ab.npy", mmap_mode=metadata_mode)
        self.L = np.load(f"{filename}/L.npy", mmap_mode=metadata_mode)

    def __len__(self):

        return len(self.L)

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self.L))
            idx = list(range(start, stop, step))
            L = torch.tensor(self.L[idx], dtype=torch.float32).unsqueeze(1)
            ab = torch.tensor(self.ab[idx], dtype=torch.float32).permute(0, 3, 1, 2)
            ab.subtract_(128)
            return (L, ab)

        L = torch.tensor(self.L[idx], dtype=torch.float32).unsqueeze(0)
        ab = torch.tensor(self.ab[idx], dtype=torch.float32).permute(2, 0, 1)

        return (L, ab)

    def rgb_image(self, idx):

        L, ab = self[idx]
        L = L.permute(1, 2, 0)
        ab = ab.permute(1, 2, 0)

        L_ab = torch.cat([L, ab], dim=2).cpu()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb_image = lab2rgb(L_ab)

        return rgb_image