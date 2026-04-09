import torch
import numpy as np
import matplotlib.pyplot as plt

total_bin_frequency = torch.load("Bin-Weights/cielab_weights.pth").cpu().numpy()
sorted_freq = np.sort(total_bin_frequency)[::-1]

plt.bar(range(len(sorted_freq)), sorted_freq, color='steelblue', width=1.0)
plt.xlabel("Bin rank")
plt.ylabel("Weight")
plt.title("Bin Weights (sorted descending)")
plt.show()