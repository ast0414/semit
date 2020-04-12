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

from sklearn.metrics import fowlkes_mallows_score, accuracy_score

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
mnist_x_test = mnist_full['x_test']
mnist_y_test = mnist_full['y_test']

with open('data/kannada_semi_1pct.pkl', 'rb') as f:
    kannada_semi = pickle.load(f)
kannada_x_test = kannada_semi['x_test']
kannada_y_test = kannada_semi['y_test']

with open('data/dig_test.pkl', 'rb') as f:
    dig_test = pickle.load(f)
dig_x_test = dig_test['x_test']
dig_y_test = dig_test['y_test']

train_transform = transforms.Compose([
    transforms.Normalize(mean=0.5, std=0.5),
    transforms.Resize(size=(32, 32))
])

mnist_test_dataset = CustomTensorDataset(torch.from_numpy(mnist_x_test).float(), torch.from_numpy(mnist_y_test).long(), transform=train_transform)
kannada_test_dataset = CustomTensorDataset(torch.from_numpy(kannada_x_test).float(), torch.from_numpy(kannada_y_test).long(), transform=train_transform)
dig_test_dataset = CustomTensorDataset(torch.from_numpy(dig_x_test).float(), torch.from_numpy(dig_y_test).long(), transform=train_transform)

mnist_loader = DataLoader(mnist_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
kannada_loader = DataLoader(kannada_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
dig_loader = DataLoader(dig_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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
encoder_2.eval()
decoder_1.eval()
discriminator_1.eval()

with torch.no_grad():
    # MNIST Test
    mnist_labels_true = []
    mnist_labels_pred = []
    for data, target in mnist_loader:
        data = data.to(device)
        target = target.to(device)

        _, class_translated = discriminator_1(data)

        mnist_labels_pred.extend(class_translated.argmax(dim=1, keepdim=False).detach().cpu().numpy().tolist())
        mnist_labels_true.extend(target.detach().cpu().numpy().tolist())

    mnist_acc = 100. * accuracy_score(mnist_labels_true, mnist_labels_pred)
    mnist_fms = fowlkes_mallows_score(mnist_labels_true, mnist_labels_pred)
    print("MNIST - ACC: {} FMS: {}".format(mnist_acc, mnist_fms))

    # Kannada-MNIST Test
    kannada_labels_true = []
    kannada_labels_pred = []
    for data, target in kannada_loader:
        data = data.to(device)
        target = target.to(device)

        z_val, mu_val, sd_val = shared_latent.sample(encoder_2(data))
        translated = decoder_1(z_val)
        _, class_translated = discriminator_1(translated)

        kannada_labels_pred.extend(class_translated.argmax(dim=1, keepdim=False).detach().cpu().numpy().tolist())
        kannada_labels_true.extend(target.detach().cpu().numpy().tolist())

    kannada_acc = 100. * accuracy_score(kannada_labels_true, kannada_labels_pred)
    kannada_fms = fowlkes_mallows_score(kannada_labels_true, kannada_labels_pred)
    print("Kannada - ACC: {} FMS: {}".format(kannada_acc, kannada_fms))

    # Dig-MNIST Test
    dig_labels_true = []
    dig_labels_pred = []
    for data, target in dig_loader:
        data = data.to(device)
        target = target.to(device)

        z_val, mu_val, sd_val = shared_latent.sample(encoder_2(data))
        translated = decoder_1(z_val)
        _, class_translated = discriminator_1(translated)

        dig_labels_pred.extend(class_translated.argmax(dim=1, keepdim=False).detach().cpu().numpy().tolist())
        dig_labels_true.extend(target.detach().cpu().numpy().tolist())

    dig_acc = 100. * accuracy_score(dig_labels_true, dig_labels_pred)
    dig_fms = fowlkes_mallows_score(dig_labels_true, dig_labels_pred)
    print("Dig - ACC: {} FMS: {}".format(dig_acc, dig_fms))

