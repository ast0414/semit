import argparse
import os
import pickle

import numpy as np

import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import DataLoader

from utils.dataset import CustomTensorDataset
from utils import transforms
from models.modules import Decoder, Discriminator, Encoder, GaussianVAE2D

import umap
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description='Test UNIT/SUIT/SEMIT for MNIST, KMNIST, and Dig-MNIST')

parser.add_argument('model', type=str, help='path to the model')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True if args.cuda else False
device = torch.device("cuda" if args.cuda else "cpu")

with open('data/mnist_full.pkl', 'rb') as f:
    mnist_full = pickle.load(f)
mnist_x_train = mnist_full["x_train"]
mnist_y_train = mnist_full["y_train"]

with open('data/kannada_semi_1pct.pkl', 'rb') as f:
    kannada_semi = pickle.load(f)
kannada_x_train_labeled = kannada_semi["x_train_labeled"]
kannada_y_train_labeled = kannada_semi["y_train_labeled"]
kannada_x_train_unlabeled = kannada_semi["x_train_unlabeled"]
kannada_y_train_unlabeled = kannada_semi["y_train_unlabeled"]
kannada_x_train = np.concatenate((kannada_x_train_labeled, kannada_x_train_unlabeled), axis=0)
kannada_y_train = np.concatenate((kannada_y_train_labeled, kannada_y_train_unlabeled), axis=0)

train_transform = transforms.Compose([
    transforms.Normalize(mean=0.5, std=0.5),
    transforms.Resize(size=(32, 32))
])

mnist_train_dataset = CustomTensorDataset(torch.from_numpy(mnist_x_train).float(), torch.from_numpy(mnist_y_train).long(), transform=train_transform)
kannada_train_dataset = CustomTensorDataset(torch.from_numpy(kannada_x_train).float(), torch.from_numpy(kannada_y_train).long(), transform=train_transform)

mnist_loader = DataLoader(mnist_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
kannada_loader = DataLoader(kannada_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Modules
dim_z = 512
shared_latent = GaussianVAE2D(dim_z, dim_z, kernel_size=1, stride=1).to(device)
encoder_1 = Encoder().to(device)
encoder_2 = Encoder().to(device)
decoder_1 = Decoder().to(device)
decoder_2 = Decoder().to(device)
discriminator_1 = Discriminator().to(device)
discriminator_2 = Discriminator().to(device)

if os.path.isfile(args.model):
    print("===> Loading Checkpoint to Evaluate '{}'".format(args.model))
    checkpoint = torch.load(args.model)
    shared_latent.load_state_dict(checkpoint['shared_latent'])
    encoder_1.load_state_dict(checkpoint['encoder_1'])
    encoder_2.load_state_dict(checkpoint['encoder_2'])
    decoder_1.load_state_dict(checkpoint['decoder_1'])
    decoder_2.load_state_dict(checkpoint['decoder_2'])
    discriminator_1.load_state_dict(checkpoint['discriminator_1'])
    discriminator_2.load_state_dict(checkpoint['discriminator_2'])
    print("\t===> Loaded Checkpoint '{}' (epoch {})".format(args.model, checkpoint['epoch']))
else:
    raise FileNotFoundError("\t====> no checkpoint found at '{}'".format(args.model))

# Evaluation
shared_latent.eval()
encoder_1.eval()
encoder_2.eval()

with torch.no_grad():
    # MNIST Test
    mnist_latent = []

    for data, _ in mnist_loader:
        data = data.to(device)

        z, mu, sd = shared_latent.sample(encoder_1(data))
        mnist_latent.append(z.detach())

    mnist_latent = torch.cat(mnist_latent, dim=0).squeeze()
    mnist_latent = mnist_latent.cpu().numpy()

    # Kannda-MNIST Test
    kannada_latent = []
    for data, _ in kannada_loader:
        data = data.to(device)

        z, mu, sd = shared_latent.sample(encoder_2(data))
        kannada_latent.append(z.detach())

    kannada_latent = torch.cat(kannada_latent, dim=0).squeeze()
    kannada_latent = kannada_latent.cpu().numpy()

args.save = os.path.join(os.path.dirname(args.model), 'UMAP')
os.makedirs(args.save, exist_ok=True)

np.save(os.path.join(args.save, 'mnist_latent.npy'), mnist_latent)
np.save(os.path.join(args.save, 'kannada_latent.npy'), kannada_latent)

num_mnist = mnist_latent.shape[0]
all_latent = np.vstack((mnist_latent, kannada_latent))

# umap_reducer = umap.UMAP(n_components=2, verbose=1, random_state=args.seed)
# umap_transformed = umap_reducer.fit_transform(all_latent)
#
# mnist_umap = umap_transformed[:num_mnist]
# kannada_umap = umap_transformed[num_mnist:]
#
# np.save(os.path.join(args.save, 'mnist_umap.npy'), mnist_umap)
# np.save(os.path.join(args.save, 'kannada_umap.npy'), kannada_umap)

tsne_reducer = TSNE(n_components=2, verbose=1, random_state=args.seed, method='barnes_hut', n_jobs=-1)
tsne_transformed = tsne_reducer.fit_transform(all_latent)

mnist_tsne = tsne_transformed[:num_mnist]
kannada_tsne = tsne_transformed[num_mnist:]

np.save(os.path.join(args.save, 'mnist_tsne.npy'), mnist_tsne)
np.save(os.path.join(args.save, 'kannada_tsne.npy'), kannada_tsne)
