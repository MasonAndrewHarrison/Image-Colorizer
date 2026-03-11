import torch
import torch.nn as nn
from colorizer import Colorizer


class Generator(Colorizer):
    def __init__(self, in_dim: int = 1, out_dim: int = 2, features: int = 64):
        super(Generator, self).__init__()

        self.out_dim = out_dim 

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
            nn.ConvTranspose2d(features, out_dim, 5, 1, 2)
        )

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

        return final



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    L = torch.rand(20, 1, 224, 224).to(device)

    colorizer = Generator(features=64).to(device)

    print(L.shape)

    ab = colorizer(L)

    print(ab.shape)

