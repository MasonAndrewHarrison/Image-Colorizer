import torch
import torch.nn as nn
import numpy as np

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

    B, C, H, W = ab.shape
    og_pts_in_hull = pts_in_hull.detach()

    bin_size,_ = pts_in_hull.shape
    ab = ab.view(B, C, H*W, 1).repeat(1, 1, 1, bin_size)
    pts_in_hull = pts_in_hull.permute(1, 0)
    pts_in_hull = pts_in_hull.unsqueeze(0).unsqueeze(2)

    pts_in_hull = pts_in_hull.expand(B, -1, -1, -1)

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




if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ab = torch.rand(32, 2, 224, 224).to(device) * 128
    pts_in_hull = np.load('third_party/richzhang_colorization/pts_in_hull_cielab.npy')
    pts_in_hull = torch.tensor(pts_in_hull, device=device)

    bins = ab_to_bins(ab, 'cielab', pts_in_hull, return_bin_index=True)

    print(bins.shape)



