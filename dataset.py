import torch
import numpy as np 


def Lab_Dataset():

    def __init__(self, train: bool = True):

        self.filename = filename
        self.transform = transform

        filename = "dataset/train" if train else "data/test"

        self.ab = np.load(filename + "ab.npy")
        self.L = np.load(filename + "L.npy")

    def __len__(self):

        return len(self.L)

    def __getitem__(self, idx):

        L = self.L[idx, :, :]
        ab = self.ab[idx, :, : , :]
        print(ab.shape, L.shape)
        return (L, ab)