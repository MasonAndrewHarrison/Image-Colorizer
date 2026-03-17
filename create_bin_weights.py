import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from dataset import Lab_Dataset
from torch.utils.data import DataLoader
from utils import logits_to_ab, ab_to_bins

device = "cuda" if torch.cuda.is_available() else "cpu"
epsilon = 1e-5

dataset = Lab_Dataset(
    color_space="CIELAB", 
    train=True,
    metadata_mode='r',
)

loader = DataLoader(
    dataset=dataset,
    batch_size=64,
)

pts_in_hull = np.load('third_party/richzhang_colorization/pts_in_hull_cielab.npy')
pts_in_hull = torch.tensor(pts_in_hull, device=device)

bin_count = len(pts_in_hull)
total_bin_frequency = torch.zeros(bin_count, device=device)

for i, (_, ab) in enumerate(loader):

    ab = ab.to(device)
    bins = ab_to_bins(ab, "mode", pts_in_hull, return_bin_index=True)
    print(f"{i} / {len(loader)}")

    B, H, W = bins.shape
    bins = bins.view(B*H*W)
    bin_frequency = torch.bincount(bins)

    extra_padding = bin_count - len(bin_frequency)
    bin_frequency = F.pad(bin_frequency, (0, extra_padding))

    total_bin_frequency.add_(bin_frequency)


total_bin_frequency = 1.0 / (total_bin_frequency + epsilon)
total_bin_frequency.div_(total_bin_frequency.sum())
os.makedirs("Bin-Weights", exist_ok=True)
torch.save(total_bin_frequency, "Bin-Weights/cielab_weights.pth")

