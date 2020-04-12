import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import os
import pickle
from utils.dataset import CustomTensorDataset
from utils import transforms
from utils.misc import save_checkpoint
from models.modules import VAE
from torchsummary import summary

parser = argparse.ArgumentParser(description='VAE for MNIST and KMNIST')
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
parser.add_argument('--save', default='/localscratch/san37/semit/VAE', type=str, metavar='S', help='save path')
args = parser.parse_args()

torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True if args.cuda else False
device = torch.device("cuda" if args.cuda else "cpu")

if args.save == '':
    args.save = os.path.join('results', args.dataset)
else:
    args.save = os.path.join(args.save, args.dataset)

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

eval_transform = transforms.Compose([
    transforms.Normalize(mean=0.5, std=0.5),
    transforms.Resize(size=(32, 32))
])

train_dataset = CustomTensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long(), transform=train_transform)
val_dataset = CustomTensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long(), transform=eval_transform)
# test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

num_batch = len(train_loader)
num_epoch = args.iters // num_batch + 1
args.__dict__.update({'epochs': num_epoch})

model = VAE().to(device)
summary(model, (1, 32, 32))

optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))


def recon_loss(recon_x, x):
    # Reconstruction loss summed over all elements and batch
    #return F.binary_cross_entropy(recon_x, x, reduction='sum')
    return F.l1_loss(recon_x, x, reduction='sum')


def gauss_kl_loss(mu, sd):
    # KL divergence loss summed over all elements and batch
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # -KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mu_2 = torch.pow(mu, 2)
    sd_2 = torch.pow(sd, 2)
    encoding_loss = 0.5 * (mu_2 + sd_2 - torch.log(sd_2) - 1).sum()
    return encoding_loss


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, sd = model(data)
        loss = recon_loss(recon_batch, data) + gauss_kl_loss(mu, sd)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return train_loss / len(train_loader.dataset)


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(val_loader):
            data = data.to(device)
            recon_batch, mu, sd = model(data)
            test_loss += (recon_loss(recon_batch, data) + gauss_kl_loss(mu, sd)).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 32, 32)[:n]])
                save_image(comparison.cpu(),
                           os.path.join(args.save, 'reconstruction_' + str(epoch) + '.png'), nrow=n)

    test_loss /= len(val_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss


if __name__ == "__main__":
    os.makedirs(args.save, exist_ok=True)
    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs + 1):
        train_losses.append(train(epoch))
        val_losses.append(test(epoch))
        with torch.no_grad():
            sample = torch.randn(64, 512, 1, 1).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 32, 32),
                       os.path.join(args.save, 'sample_' + str(epoch) + '.png'))

    save_checkpoint({
        'epoch': epoch,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(args.save, 'checkpoint.pth'))

    fig = plt.figure(figsize=(12, 9))
    plt.plot(np.arange(len(train_losses)), np.array(train_losses), label='Training Loss')
    plt.plot(np.arange(len(val_losses)), np.array(val_losses), label='Validation Loss')
    plt.title("Loss Curve", fontsize=20)
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", fontsize=16)
    plt.tight_layout()
    fig.savefig(os.path.join(args.save, 'loss.png'), format='png', bbox_inches='tight')
    plt.close('all')
