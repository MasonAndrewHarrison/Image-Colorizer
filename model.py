import torch
import torch.nn as nn


class Colorizer(nn.Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 2, features: int = 16):
        super(Colorizer, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, features*4, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(features*4, features*8, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(features*8, features*16, 3, 2, 1,),
            nn.ReLU(),
            nn.Conv2d(features*16, features*64, 3, 2, 1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features*64, features*16, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(features*16, features*8, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(features*8, features*4, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(features*4, out_dim, 5, 1, 2),
        )

    def forward(self, x):

        encoded_latent = self.encoder(x)
        out = self.decoder(encoded_latent)
        out = out * x.repeat(1, self.out_dim, 1, 1)

        return out


def initilize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = torch.rand(1, 1, 64, 64).to(device)

    colorizer = Colorizer(features=32).to(device)
    initilize_weights(colorizer)

    colored_img = colorizer(image)

    print(colored_img.shape)