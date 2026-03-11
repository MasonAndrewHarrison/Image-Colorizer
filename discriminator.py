import torch
import torch.nn as nn

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

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    L = torch.rand(20, 1, 224, 224).to(device)
    ab = torch.rand(20, 2, 224, 224).to(device)

    disc = Discriminator(features=64).to(device)

    print(ab.shape)

    scores = disc(L, ab)

    print(scores.shape)
