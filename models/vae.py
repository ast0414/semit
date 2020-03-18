from torch import nn

from .blocks import Conv2dBlock, GaussianVAE2D, ConvTranspose2dBlock


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = []

        self.model += [Conv2dBlock(input_dim=1, output_dim=32, kernel_size=5, stride=2, padding=4, norm='bn', activation='lrelu')]
        self.model += [Conv2dBlock(input_dim=32, output_dim=64, kernel_size=3, stride=2, padding=1, norm='bn', activation='lrelu')]
        self.model += [Conv2dBlock(input_dim=64, output_dim=128, kernel_size=3, stride=2, padding=1, norm='bn', activation='lrelu')]
        self.model += [Conv2dBlock(input_dim=128, output_dim=256, kernel_size=3, stride=2, padding=1, norm='bn', activation='lrelu')]
        self.model += [Conv2dBlock(input_dim=256, output_dim=512, kernel_size=3, stride=2, padding=1, norm='bn', activation='lrelu')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = []

        self.model += [ConvTranspose2dBlock(512, 256, 3, 2, 1, 1, norm='bn', activation='lrelu')]
        self.model += [ConvTranspose2dBlock(256, 128, 3, 2, 1, 1, norm='bn', activation='lrelu')]
        self.model += [ConvTranspose2dBlock(128, 64, 3, 2, 1, 1, norm='bn', activation='lrelu')]
        self.model += [ConvTranspose2dBlock(64, 32, 3, 2, 1, 1, norm='bn', activation='lrelu')]
        self.model += [ConvTranspose2dBlock(32, 1, 5, 2, 4, 1, activation='sigmoid')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = Encoder()
        self.gauss_vae = GaussianVAE2D(512, 512, kernel_size=1, stride=1)
        self.decoder = Decoder()

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, sd = self.gauss_vae.sample(self.encoder(x))
        return self.decode(z), mu, sd
