import argparse
import os
import pickle

import numpy as np

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.dataset import CustomTensorDataset
from utils import transforms
from models.modules import Decoder, Discriminator


parser = argparse.ArgumentParser(description='GAN for MNIST and KMNIST')
parser.add_argument('--dataset', type=str, default='mnist', metavar='S',
                    help='name of dataset to use (default: mnist, kannada)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--iters', type=int, default=100000, metavar='N',
                    help='number of iterations to train (default: 100K)')
# parser.add_argument('--epochs', type=int, default=30, metavar='N',
#                     help='number of epochs to train (default: 10)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='/localscratch/san37/semit/GAN', type=str, metavar='S', help='save path')
args = parser.parse_args()

torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True if args.cuda else False
device = torch.device("cuda" if args.cuda else "cpu")

if args.save == '':
    args.save = os.path.join('results', args.dataset)
else:
    args.save = os.path.join(args.save, args.dataset)

os.makedirs(args.save, exist_ok=True)


if args.dataset == 'mnist':
    with open('data/mnist_full.pkl', 'rb') as f:
        mnist_full = pickle.load(f)
    x_train = mnist_full["x_train"]
    y_train = mnist_full["y_train"]
    x_val = mnist_full["x_val"]
    y_val = mnist_full["y_val"]
    x_test = mnist_full['x_test']
    y_test = mnist_full['y_test']
elif args.dataset == 'kannada':
    with open('data/kannada_semi_1pct.pkl', 'rb') as f:
        kannada_semi = pickle.load(f)
    x_train_labeled = kannada_semi["x_train_labeled"]
    y_train_labeled = kannada_semi["y_train_labeled"]
    x_train_unlabeled = kannada_semi["x_train_unlabeled"]
    y_train_unlabeled = kannada_semi["y_train_unlabeled"]
    x_train = np.concatenate((x_train_labeled, x_train_unlabeled), axis=0)
    y_train = np.concatenate((y_train_labeled, y_train_unlabeled), axis=0)
    x_val = kannada_semi["x_val"]
    y_val = kannada_semi["y_val"]
    x_test = kannada_semi['x_test']
    y_test = kannada_semi['y_test']
else:
    raise AttributeError


train_transform = transforms.Compose([
    transforms.Normalize(mean=0.5, std=0.5),
    transforms.Resize(size=(32, 32))
])

train_dataset = CustomTensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long(), transform=train_transform)
# val_dataset = CustomTensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
# test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
# val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


num_batch = len(train_loader)
num_epoch = args.iters // num_batch + 1
args.__dict__.update({'epochs': num_epoch})

netG = Decoder().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()

fixed_noise = torch.randn(args.batch_size, 512, 1, 1, device=device)
real_label = 1 - 0.1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

for epoch in range(1, args.epochs + 1):
    for i, data in enumerate(train_loader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, 512, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 10 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.epochs + 1, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    save_image(real_cpu,
               os.path.join(args.save, 'real_samples.png'),
               normalize=True)
    fake = netG(fixed_noise)
    save_image(fake.detach(),
               os.path.join(args.save, 'fake_samples_epoch_{:03d}.png'.format(epoch)),
               normalize=True)
