import torch
import torch.nn as nn

from .blocks import Conv2dBlock, GaussianVAE2D, ConvTranspose2dBlock


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = []

        # self.model += [Conv2dBlock(input_dim=1, output_dim=32, kernel_size=5, stride=2, padding=4, norm='bn', activation='lrelu')]
        # self.model += [Conv2dBlock(input_dim=32, output_dim=64, kernel_size=3, stride=2, padding=1, norm='bn', activation='lrelu')]
        # self.model += [Conv2dBlock(input_dim=64, output_dim=128, kernel_size=3, stride=2, padding=1, norm='bn', activation='lrelu')]
        # self.model += [Conv2dBlock(input_dim=128, output_dim=256, kernel_size=3, stride=2, padding=1, norm='bn', activation='lrelu')]
        # self.model += [Conv2dBlock(input_dim=256, output_dim=512, kernel_size=3, stride=2, padding=1, norm='bn', activation='lrelu')]
        # self.model = nn.Sequential(*self.model)

        self.model += [Conv2dBlock(input_dim=1, output_dim=32, kernel_size=4, stride=2, padding=1, norm='bn', activation='lrelu')]
        self.model += [Conv2dBlock(input_dim=32, output_dim=64, kernel_size=4, stride=2, padding=1, norm='bn', activation='lrelu')]
        self.model += [Conv2dBlock(input_dim=64, output_dim=128, kernel_size=4, stride=2, padding=1, norm='bn', activation='lrelu')]
        self.model += [Conv2dBlock(input_dim=128, output_dim=256, kernel_size=4, stride=2, padding=1, norm='bn', activation='lrelu')]
        self.model += [Conv2dBlock(input_dim=256, output_dim=512, kernel_size=4, stride=2, padding=1, norm='bn', activation='lrelu')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = []

        # self.model += [ConvTranspose2dBlock(512, 256, 3, 2, 1, 1, norm='bn', activation='lrelu')]
        # self.model += [ConvTranspose2dBlock(256, 128, 3, 2, 1, 1, norm='bn', activation='lrelu')]
        # self.model += [ConvTranspose2dBlock(128, 64, 3, 2, 1, 1, norm='bn', activation='lrelu')]
        # self.model += [ConvTranspose2dBlock(64, 32, 3, 2, 1, 1, norm='bn', activation='lrelu')]
        # self.model += [ConvTranspose2dBlock(32, 1, 5, 2, 4, 1, activation='tanh')]
        # self.model = nn.Sequential(*self.model)

        self.model += [ConvTranspose2dBlock(512, 256, 4, 2, 1, 0, norm='bn', activation='lrelu')]
        self.model += [ConvTranspose2dBlock(256, 128, 4, 2, 1, 0, norm='bn', activation='lrelu')]
        self.model += [ConvTranspose2dBlock(128, 64, 4, 2, 1, 0, norm='bn', activation='lrelu')]
        self.model += [ConvTranspose2dBlock(64, 32, 4, 2, 1, 0, norm='bn', activation='lrelu')]
        self.model += [ConvTranspose2dBlock(32, 1, 4, 2, 1, 0, activation='tanh')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = []

        self.model += [Conv2dBlock(input_dim=1, output_dim=32, kernel_size=4, stride=2, padding=1, norm='none', activation='lrelu')]
        # self.model += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.model += [nn.Dropout(p=0.5)]
        self.model += [Conv2dBlock(input_dim=32, output_dim=64, kernel_size=4, stride=2, padding=1, norm='none', activation='lrelu')]
        # self.model += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.model += [nn.Dropout(p=0.5)]
        self.model += [Conv2dBlock(input_dim=64, output_dim=128, kernel_size=4, stride=2, padding=1, norm='none', activation='lrelu')]
        # self.model += [nn.MaxPool2d(kernel_size=2, stride=2, padding=1)]
        self.model += [nn.Dropout(p=0.5)]
        self.model += [Conv2dBlock(input_dim=128, output_dim=256, kernel_size=4, stride=2, padding=1, norm='none', activation='lrelu')]
        # self.model += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.model += [nn.Dropout(p=0.5)]
        self.model += [Conv2dBlock(input_dim=256, output_dim=512, kernel_size=4, stride=2, padding=1, norm='none', activation='lrelu')]
        # self.model += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.model = nn.Sequential(*self.model)

        self.drop = nn.Dropout(p=0.5)
        self.discriminator = nn.Linear(in_features=512, out_features=1)
        self.classifier = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        out = self.model(x)
        out = torch.flatten(out, start_dim=1)
        out = self.drop(out)
        logit_discr = self.discriminator(out)
        logit_class = self.classifier(out)
        # return out.view(-1, 1).squeeze(1)
        return logit_discr, logit_class


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
