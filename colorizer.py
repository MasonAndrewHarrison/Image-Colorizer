import torch
import torch.nn as nn
import numpy as np

class Colorizer(nn.Module):
    def __init__(self, mode):
        super(Colorizer, self).__init__()

        self.mode = mode
        self.n_bins = {
            'cielab': 313,
            'oklab': 256, #change depending on my kmeans bin
            'copic': 358,
        }[mode]

        self.out_dim = {
            'cielab': 2,
            'oklab': 2,
            'copic': 3,
        }[mode]

        pts = {
            'cielab': np.load('third_party/richzhang_colorization/pts_in_hull_cielab.npy'),
            #'oklab': np.load('pts_in_hull_oklab.npy'),
            #'copic': np.load('pts_in_hull_copic.npy'),
        }[mode]
        self.register_buffer('pts_in_hull', torch.from_numpy(pts).float())
        
        self.center_l = 50
        self.norm_l = 100
        self.norm_ab = 105

    def normalize_l(self, l):
        return (l - self.center_l) / self.norm_l

    def normalize_ab(self, ab):
        return ab / self.norm_ab

    def unnormalize_1(self, l):
        return l * self.norm_l + self.center_l

    def unnormalize_ab(self, ab):
        return ab * self.norm_ab
    
    @staticmethod
    def conv_block(in_features, out_features, kernel, stride, padding, use_batch_norm: bool = False, dilation: int = 1):

        layers = [
            nn.Conv2d(
                in_channels=in_features, 
                out_channels=out_features, 
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not use_batch_norm,
            ),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_features))

        additional_layers = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.05),
            nn.Conv2d(
                in_channels=out_features,
                out_channels=out_features,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
            ),
            nn.LeakyReLU(negative_slope=0.05),
        )

        layers.append(additional_layers)

        return nn.Sequential(*layers)

    @staticmethod
    def conv_tran_block(in_features, out_features, kernel, stride, use_batch_norm: bool = False):

        layers = [
            nn.ConvTranspose2d(
                in_channels=in_features, 
                out_channels=in_features, 
                kernel_size=kernel,
                stride=stride,
                padding=1,
                output_padding=1,
                bias=not use_batch_norm,
            ),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(in_features))

        additional_layers = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.05),
            nn.ConvTranspose2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.05),
        )

        layers.append(additional_layers)

        return nn.Sequential(*layers)

    @staticmethod
    def cat_skip(skip_connect, conv_output):

        return torch.cat([skip_connect, conv_output], dim=1)



