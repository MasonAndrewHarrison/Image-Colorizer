import torch
import torch.nn as nn
from colorizer import Colorizer


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



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = torch.rand(20, 1, 224, 224).to(device)

    colorizer = Generator(features=32).to(device)
    print(L.shape)

    ab = colorizer(L)
    print(ab.shape)

    logits = colorizer(L, return_logits=True)
    print(logits.shape)

