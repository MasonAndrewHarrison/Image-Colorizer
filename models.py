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
        }[mode]

        self.out_dim = {
            'cielab': 2,
            'oklab': 2,
        }[mode]

        pts = {
            'cielab': np.load('third_party/richzhang_colorization/pts_in_hull_cielab.npy'),
            #'oklab': np.load('pts_in_hull_oklab.npy'),
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



class Generator(Colorizer):
    def __init__(self, in_dim: int = 1, features: int = 64, mode: str = 'cielab'):
        super(Generator, self).__init__(mode=mode) 

        self.conv0 = self.conv_block(in_dim, features, 5, 1, 2)

        self.conv1 = self.conv_block(features, features*2, 3, 2, 1,)
        self.conv2 = self.conv_block(features*2, features*4, 3, 2, 1, use_batch_norm=True)
        self.conv3 = self.conv_block(features*4, features*8, 3, 2, 1)
        self.conv4 = self.conv_block(features*8, features*16, 3, 2, 2, use_batch_norm=True, dilation=2)
        self.conv5 = self.conv_block(features*16, features*32, 3, 2, 2, dilation=2)

        self.convT1 = self.conv_tran_block(features*32, features*16, 3, 2)
        self.convT2 = self.conv_tran_block(2*features*16, features*8, 3, 2, use_batch_norm=True)
        self.convT3 = self.conv_tran_block(2*features*8, features*4, 3, 2)
        self.convT4 = self.conv_tran_block(2*features*4, features*2, 3, 2, use_batch_norm=True)
        self.convT5 = self.conv_tran_block(2*features*2, self.n_bins, 3, 2)

        self.softmax = nn.Softmax(dim=1)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(313, self.out_dim, 5, 1, 2)
        )

    def forward(self, input_l, return_logits: bool = False):

        out = self.conv0(self.normalize_l(input_l))
        skip_connect1 = self.conv1(out)
        skip_connect2 = self.conv2(skip_connect1)
        skip_connect3 = self.conv3(skip_connect2)
        skip_connect4 = self.conv4(skip_connect3)
        latent_space = self.conv5(skip_connect4)

        out = self.convT1(latent_space)
        out = self.convT2(self.cat_skip(skip_connect4, out))
        out = self.convT3(self.cat_skip(skip_connect3, out))
        out = self.convT4(self.cat_skip(skip_connect2, out))
        out = self.convT5(self.cat_skip(skip_connect1, out))

        

        if return_logits:
            return out

        out = self.softmax(out)    
        
        out = out.permute(0, 2, 3, 1)         
        ab = torch.matmul(out, self.pts_in_hull)    
        ab = ab.permute(0, 3, 1, 2)   

        return ab



class Discriminator(nn.Module):

    def __init__(self, in_dim: int = 3, features: int = 16, oklab: bool = False):
        super(Discriminator, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(3, features, 5, 1, 2),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features*2, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(features*2, features*4, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(features*4, features*8, 3, 2, 1),
            nn.BatchNorm2d(features*8),
            nn.ReLU(),
            nn.Conv2d(features*8, features*16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(features*16, features*32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(features*32, 1, 3, 1, 1)
        )

    def forward(self, L, ab):

        L_ab = torch.cat([L, ab], dim=1)
        out = self.convolution(L_ab)

        return out


