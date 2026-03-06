import torch
import torch.nn as nn


class Colorizer(nn.Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 2, features: int = 16):
        super(Colorizer, self).__init__()

        self.out_dim = out_dim

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
            nn.GELU(),
            nn.ConvTranspose2d(features*16, features*8, 3, 2, 0, 1),
            nn.GELU(),
            nn.ConvTranspose2d(features*8, features*4, 3, 2, 0, 1),
            nn.BatchNorm2d(features*4),
            nn.GELU(),
            nn.ConvTranspose2d(features*4, out_dim, 5, 1, 1, 0),
        )


    def forward(self, x):

        #TODO do a U-net patter later if this works

        encoded_latent = self.encoder(x)
        out = self.decoder(encoded_latent)

        return out


    #TODO add a patch discrimitor


def initilize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        #TODO normalize batchnorm


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = torch.rand(1, 1, 64, 64).to(device)

    colorizer = Colorizer(features=32).to(device)
    initilize_weights(colorizer)

    print(image.shape)

    colored_img = colorizer(image)

    print(colored_img.shape)