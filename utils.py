import torch
import torch.nn as nn
import numpy as np
from dataset import Lab_Dataset
import matplotlib.pyplot as plt
import os
import warnings
from skimage.color import lab2rgb

def initilize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


def logits_to_ab(logits, pts_in_hull):

    logits = torch.softmax(logits, dim=1) 
    logits = logits.permute(0, 2, 3, 1)         
    ab = torch.matmul(logits, pts_in_hull)    
    ab = ab.permute(0, 3, 1, 2)   

    return ab

def ab_to_bins(ab, mode, pts_in_hull, return_bin_index: bool = False):

    #TODO make this work with copic

    B, C, H, W = ab.shape
    og_pts_in_hull = pts_in_hull.detach()

    bin_size,_ = pts_in_hull.shape
    ab = ab.view(B, C, H*W, 1).repeat(1, 1, 1, bin_size)
    pts_in_hull = pts_in_hull.permute(1, 0)
    pts_in_hull = pts_in_hull.unsqueeze(0).unsqueeze(2)

    pts_in_hull = pts_in_hull.expand(B, -1, -1, -1)

    color_space="CIELAB", 
    x_dist = ab[:, 0, :, :].subtract_(pts_in_hull[:, 0, :, :])
    y_dist = ab[:, 1, :, :].subtract_(pts_in_hull[:, 1, :, :])


    dist_sq = x_dist**2
    dist_sq.addcmul_(y_dist, y_dist)

    closest_idx = torch.argmin(dist_sq, dim=2)

    if return_bin_index:

        bins_index = closest_idx.view(B, H, W)
        return bins_index

    bins_ab = og_pts_in_hull[closest_idx]
    bins_ab = bins_ab.permute(0, 2, 1).view(B, 2, H, W)

    return bins_ab

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_images(fixed_l_batch, render_batch, gen_mode):

    gen_mode.eval()
    fig, axes = plt.subplots(render_batch[0], render_batch[1], figsize=(15, 15))

    with torch.no_grad():

        fake_ab = gen_mode(fixed_l_batch).detach()
        L_ab = torch.cat([fixed_l_batch, fake_ab], dim=1)
        L_ab = L_ab.squeeze(0).squeeze(0)   
        L_ab = L_ab.permute(0, 2, 3, 1).cpu().numpy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb_image = lab2rgb(L_ab)
        rgb_image = np.clip(rgb_image, 0, 1)

        for idx, ax in enumerate(axes.flat):

            ax.imshow(rgb_image[idx, :, :, :]) 
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()
    gen_mode.train()   