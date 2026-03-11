import torch
import torch.nn as nn


class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()

        self.center_l = 50
        self.norm_l = 100
        self.norm_ab = 128

    def normalize_l(self, l):
        return (l - self.center_l) / self.norm_l

    def normalize_ab(self, ab):
        return ab / self.norm_ab

    def unnormalize_1(self, l):
        return l * self.norm_l + self.center_l

    def unnormalize_ab(self, ab):
        return ab * self.norm_ab
    
    @staticmethod
    def conv_block(in_features, out_features, kernel, stride, padding, use_batch_norm: bool = False):

        layers = [
            nn.Conv2d(
                in_channels=in_features, 
                out_channels=out_features, 
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            ),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_features))

        layers.append(nn.LeakyReLU(negative_slope=0.05))

        return nn.Sequential(*layers)

    @staticmethod
    def conv_tran_block(in_features, out_features, kernel, stride, use_batch_norm: bool = False):

        layers = [
            nn.ConvTranspose2d(
                in_channels=in_features, 
                out_channels=out_features, 
                kernel_size=kernel,
                stride=stride,
                padding=1,
                output_padding=1,
                bias=not use_batch_norm,
            ),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_features))

        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    @staticmethod
    def cat_skip(skip_connect, conv_output):

        return torch.cat([skip_connect, conv_output], dim=1)



