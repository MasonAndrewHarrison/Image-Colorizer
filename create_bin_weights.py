import torch
import torch.nn.functional as F
import numpy as np
import os
from dataset import Lab_Dataset
from torch.utils.data import DataLoader
from utils import ab_to_bins
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)["create_bins"]

lambda_weight = float(config["lambda_weight"])
batch_size = int(config["batch_size"])

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = Lab_Dataset(color_space="CIELAB", train=True, metadata_mode='r')
loader = DataLoader(dataset=dataset, batch_size=batch_size)

pts_in_hull = np.load('third_party/richzhang_colorization/pts_in_hull_cielab.npy')
pts_in_hull = torch.tensor(pts_in_hull, device=device)
bin_count = len(pts_in_hull)
total_bin_frequency = torch.zeros(bin_count, device=device)

for i, (_, ab) in enumerate(loader):

    ab = ab.to(device)
    bins = ab_to_bins(ab, "cielab", pts_in_hull, return_bin_index=True)
    print(f"{i * batch_size} / {len(loader) * batch_size}")

    B, H, W = bins.shape
    bins = bins.view(B * H * W)
    bin_frequency = torch.bincount(bins)

    extra_padding = bin_count - len(bin_frequency)
    bin_frequency = F.pad(bin_frequency, (0, extra_padding))
    total_bin_frequency.add_(bin_frequency)


order = torch.argsort(total_bin_frequency, descending=True)
ranks = torch.arange(bin_count, device=device).float()
power_weights = (1.0 / (ranks + 1)) ** 0.15
weights = torch.zeros(bin_count, device=device)

weights[order] = power_weights
weights[total_bin_frequency == 0] = 0.0
nonzero_mask = weights > 0
weights[nonzero_mask] = weights[nonzero_mask] / weights[nonzero_mask].mean()

print(f"min: {weights.min():.4f}")
print(f"max: {weights.max():.4f}")
print(f"mean: {weights.mean():.4f}")
print(f"zero bins: {(weights < 0.01).sum()}")

os.makedirs("Bin-Weights", exist_ok=True)
torch.save(weights, "Bin-Weights/cielab_weights.pth")
