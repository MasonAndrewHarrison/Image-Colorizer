import torch
import torch.nn as nn


class Colorizer(nn.Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 2, features: int = 64, oklab: bool = False):
        super(Colorizer, self).__init__()

        self.out_dim = out_dim 
        self.oklab = oklab

        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, features*4, 5, 1),
            nn.BatchNorm2d(features*4),
            nn.ReLU(),
            nn.Conv2d(features*4, features*8, 3, 2),
            nn.ReLU(),
            nn.Conv2d(features*8, features*16, 3, 2),
            nn.BatchNorm2d(features*16),
            nn.ReLU(),
            nn.Conv2d(features*16, features*64, 3, 2),
            nn.ELU(alpha=2.0),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features*64, features*16, 3, 2, 0, 1),
            nn.BatchNorm2d(features*16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(features*16, features*8, 3, 2, 0, 1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(features*8, features*4, 3, 2, 0, 1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(negative_slope=0.03),
            nn.ConvTranspose2d(features*4, out_dim, 5, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.05),
        )

        self.conv0 = self._conv_block(in_dim, features, 5, 1, 2)
        self.conv1 = self._conv_block(features, features*2, 3, 2, 1,)
        self.conv2 = self._conv_block(features*2, features*4, 3, 2, 1, use_batch_norm=True)
        self.conv3 = self._conv_block(features*4, features*8, 3, 2, 1)
        self.conv4 = self._conv_block(features*8, features*16, 3, 2, 1, use_batch_norm=True)
        self.conv5 = self._conv_block(features*16, features*32, 3, 2, 1)

        self.convT1 = self._conv_tran_block(features*32, features*16, 3, 2)
        self.convT2 = self._conv_tran_block(2*features*16, features*8, 3, 2, use_batch_norm=True)
        self.convT3 = self._conv_tran_block(2*features*8, features*4, 3, 2)
        self.convT4 = self._conv_tran_block(2*features*4, features*2, 3, 2, use_batch_norm=True)
        self.convT5 = self._conv_tran_block(2*features*2, features, 3, 2)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(features, out_dim, 5, 1, 2),
            #nn.Tanh(),
        )
    
    @staticmethod
    def _conv_block(in_features, out_features, kernel, stride, padding, use_batch_norm: bool = False):

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
            layers.append(nn.BatchNorm2d(out_features)),
            layers.append(nn.LeakyReLU(negative_slope=0.05)),

        return nn.Sequential(*layers)

    @staticmethod
    def _conv_tran_block(in_features, out_features, kernel, stride, use_batch_norm: bool = False):

        layers = [
            nn.ConvTranspose2d(
                in_channels=in_features, 
                out_channels=out_features, 
                kernel_size=kernel,
                stride=stride,
                padding=1,
                output_padding=1,
            ),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_features)),
            layers.append(nn.LeakyReLU(negative_slope=0.20)),

        return nn.Sequential(*layers)

    def cat_skip(self, skip_connect, conv_output):

        return torch.cat([skip_connect, conv_output], dim=1)

    def forward(self, x):

        out = self.conv0(x)
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

        final = self.final_layer(out)

        if not self.oklab:
            ...#final = final * 128

        return final


class Discriminator(nn.Module):

    def __init__(self, in_dim: int = 3, features: int = 16, oklab: bool = False):
        super(Discriminator, self).__init__()

        self.oklab = oklab

        self.convolution = nn.Sequential(
            nn.Conv2d(3, features, 5, 1, 2),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features*2, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(features*2, features*4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(features*4, features*8, 3, 2, 1),
            nn.BatchNorm2d(features*8),
            nn.ReLU(),
            nn.Conv2d(features*8, features*16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(features*16, features*32, 3, 2, 1),
            nn.ReLU(),
        )

        self.shared_mlp = nn.Sequential(
            nn.Linear(features*32, features*8),
            nn.ReLU(),
            nn.Linear(features*8, features),
            nn.ReLU(),
            nn.Linear(features, 1),
            nn.Sigmoid(),
        )

    def forward(self, L, ab):

        '''if not self.oklab:
            L = L / 128
            ab = ab / 128'''

        L_ab = torch.cat([L, ab], dim=1)
        out = self.convolution(L_ab)
        out = out.mean(3).mean(2)
        out = self.shared_mlp(out)

        return out


def initilize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant(m.bias.data, 0.0)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    L = torch.rand(20, 1, 224, 224).to(device)

    colorizer = Colorizer(features=64).to(device)
    initilize_weights(colorizer)

    print(L.shape)

    ab = colorizer(L)

    print(ab.shape)

    discriminator = Discriminator(features=64).to(device)

    print(ab.shape)

    score = discriminator(L, ab)


    print(score.shape)