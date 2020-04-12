import argparse
import json
import os
import pickle
import shutil

import numpy as np

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from utils.dataset import CustomTensorDataset
from utils import transforms
from utils.misc import save_checkpoint
from utils.metrics import recon_loss, gauss_kl_loss, compute_batch_accuracy
from models.modules import Decoder, Discriminator, Encoder, GaussianVAE2D

from tqdm import tqdm

from sklearn.metrics import fowlkes_mallows_score


parser = argparse.ArgumentParser(description='UNIT for MNIST and KMNIST')

parser.add_argument('--lam0', type=float, default=1.0, metavar='F', help='lambda_0 GAN weight (default: 10.0)')
parser.add_argument('--lam1', type=float, default=0.001, metavar='F', help='lambda_1 VAE KL weight (default: 0.1)')
parser.add_argument('--lam2', type=float, default=0.01, metavar='F', help='lambda_2 VAE Recon weight (default: 100.0)')
parser.add_argument('--lam3', type=float, default=0.0001, metavar='F', help='lambda_3 Cycle KL weight (default: 0.1)')
parser.add_argument('--lam4', type=float, default=0.001, metavar='F', help='lambda_4 Cycle Recon weight (default: 100.0)')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='/localscratch/san37/semit/UNIT', type=str, metavar='S', help='save path')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to a checkpoint (default: none)')
args = parser.parse_args()

torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True if args.cuda else False
device = torch.device("cuda" if args.cuda else "cpu")

if args.save == '':
    args.save = os.path.join(args.save, 'results')

os.makedirs(args.save, exist_ok=True)
with open(os.path.join(args.save, "config.txt"), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

with open('data/mnist_full.pkl', 'rb') as f:
    mnist_full = pickle.load(f)
mnist_x_train = mnist_full["x_train"]
mnist_y_train = mnist_full["y_train"]
mnist_x_val = mnist_full["x_val"]
mnist_y_val = mnist_full["y_val"]
mnist_x_test = mnist_full['x_test']
mnist_y_test = mnist_full['y_test']

with open('data/kannada_semi_1pct.pkl', 'rb') as f:
    kannada_semi = pickle.load(f)
kannada_x_train_labeled = kannada_semi["x_train_labeled"]
kannada_y_train_labeled = kannada_semi["y_train_labeled"]
kannada_x_train_unlabeled = kannada_semi["x_train_unlabeled"]
kannada_y_train_unlabeled = kannada_semi["y_train_unlabeled"]
kannada_x_train = np.concatenate((kannada_x_train_labeled, kannada_x_train_unlabeled), axis=0)
kannada_y_train = np.concatenate((kannada_y_train_labeled, kannada_y_train_unlabeled), axis=0)
kannada_x_val = kannada_semi["x_val"]
kannada_y_val = kannada_semi["y_val"]
kannada_x_test = kannada_semi['x_test']
kannada_y_test = kannada_semi['y_test']

train_transform = transforms.Compose([
    transforms.Normalize(mean=0.5, std=0.5),
    transforms.Resize(size=(32, 32))
])

mnist_train_dataset = CustomTensorDataset(torch.from_numpy(mnist_x_train).float(), torch.from_numpy(mnist_y_train).long(), transform=train_transform)
kannada_train_dataset = CustomTensorDataset(torch.from_numpy(kannada_x_train).float(), torch.from_numpy(kannada_y_train).long(), transform=train_transform)
kannada_val_dataset = CustomTensorDataset(torch.from_numpy(kannada_x_val).float(), torch.from_numpy(kannada_y_val).long(), transform=train_transform)

train_loader_1 = DataLoader(mnist_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
train_loader_2 = DataLoader(kannada_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(kannada_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

num_batch = min(len(train_loader_1), len(train_loader_2))

dim_z = 512

# Modules
shared_latent = GaussianVAE2D(dim_z, dim_z, kernel_size=1, stride=1).to(device)
encoder_1 = Encoder().to(device)
encoder_2 = Encoder().to(device)
decoder_1 = Decoder().to(device)
decoder_2 = Decoder().to(device)
discriminator_1 = Discriminator().to(device)
discriminator_2 = Discriminator().to(device)

criterion_discr = nn.BCELoss(reduction='sum')
criterion_class = nn.CrossEntropyLoss(reduction='sum')

fixed_noise = torch.randn(args.batch_size, dim_z, 1, 1, device=device)
real_label = 1
fake_label = 0
label_noise = 0.1

# setup optimizer
dis_params = list(discriminator_1.parameters()) + list(discriminator_2.parameters())
optimizerD = optim.AdamW(dis_params, lr=2e-4, betas=(0.5, 0.999), weight_decay=5e-4)

gen_params = list(shared_latent.parameters()) + list(encoder_1.parameters()) + list(encoder_2.parameters()) + list(decoder_1.parameters()) + list(decoder_2.parameters())
optimizerG = optim.AdamW(gen_params, lr=2e-4, betas=(0.5, 0.999), weight_decay=5e-4)

best_val_acc = 0.0
iteration = 0

if args.resume:
    if os.path.isfile(args.resume):
        print("===> Loading Checkpoint to Resume '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch'] + 1
        iteration = checkpoint['iteration']
        best_val_acc = checkpoint['best_val_acc']

        shared_latent.load_state_dict(checkpoint['shared_latent'])
        encoder_1.load_state_dict(checkpoint['encoder_1'])
        encoder_2.load_state_dict(checkpoint['encoder_2'])
        decoder_1.load_state_dict(checkpoint['decoder_1'])
        decoder_2.load_state_dict(checkpoint['decoder_2'])
        discriminator_1.load_state_dict(checkpoint['discriminator_1'])
        discriminator_2.load_state_dict(checkpoint['discriminator_2'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])

        print("\t===> Loaded Checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        raise FileNotFoundError("\t====> no checkpoint found at '{}'".format(args.resume))
else:
    try:
        shutil.rmtree(os.path.join(args.save, "tb_log/"))
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

summary_writer = SummaryWriter(log_dir=os.path.join(args.save, "tb_log"))

for epoch in tqdm(range(args.start_epoch, args.epochs), desc='Epoch'):

    train_iter_1 = iter(train_loader_1)
    train_iter_2 = iter(train_loader_2)

    shared_latent.train()
    encoder_1.train()
    encoder_2.train()
    decoder_1.train()
    decoder_2.train()
    discriminator_1.train()
    discriminator_2.train()
    for batch_idx in range(num_batch):

        data_1, target_1 = next(train_iter_1)
        data_1 = data_1.to(device)
        target_1 = target_1.to(device)

        data_2, _ = next(train_iter_2)
        data_2 = data_2.to(device)
        target_2 = None

        batch_size_1 = data_1.size(0)
        batch_size_2 = data_2.size(0)

        ############################
        # (0) Generate Data
        ###########################

        # Reconstruction stream
        z_1, mu_1, sd_1 = shared_latent.sample(encoder_1(data_1))
        recon_1_1 = decoder_1(z_1)

        z_2, mu_2, sd_2 = shared_latent.sample(encoder_2(data_2))
        recon_2_2 = decoder_2(z_2)

        # Translation stream
        recon_1_2 = decoder_2(z_1)
        recon_2_1 = decoder_1(z_2)

        # # Cycle-reconstruction stream
        # z_cycle_1_2, mu_cycle_1_2, sd_cycle_1_2 = shared_latent.sample(encoder_2(recon_1_2))
        # cycle_recon_1_2_1 = decoder_1(z_cycle_1_2)
        #
        # z_cycle_2_1, mu_cycle_2_1, sd_cycle_2_1 = shared_latent.sample(encoder_1(recon_2_1))
        # cycle_recon_2_1_2 = decoder_2(z_cycle_2_1)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        discriminator_1.zero_grad()
        discriminator_2.zero_grad()

        # train with the 1st dataset real
        label_real_1 = torch.full((batch_size_1,), real_label, device=device) - label_noise * torch.rand((batch_size_1,), device=device)
        disc_real_1, class_real_1 = discriminator_1(data_1)
        disc_real_1 = disc_real_1.view(-1, 1).squeeze(1).sigmoid()
        loss_d_real_1 = criterion_discr(disc_real_1, label_real_1)
        loss_c_real_1 = criterion_class(class_real_1, target_1)
        errD_real_1 = loss_d_real_1 + loss_c_real_1

        D_x_1 = disc_real_1.mean().item()
        summary_writer.add_scalar('Discriminator_1/Loss_D_Real', loss_d_real_1.item() / batch_size_1, iteration)
        summary_writer.add_scalar('Discriminator_1/Loss_C_Real', loss_c_real_1.item() / batch_size_1, iteration)
        summary_writer.add_scalar('Discriminator_1/Acc_Real', compute_batch_accuracy(class_real_1, target_1), iteration)

        # train with fake
        label_fake_1 = torch.full((batch_size_2,), fake_label, device=device)
        disc_fake_1, class_fake_1 = discriminator_1(recon_2_1.detach())
        disc_fake_1 = disc_fake_1.view(-1, 1).squeeze(1).sigmoid()
        loss_d_fake_1 = criterion_discr(disc_fake_1, label_fake_1)
        errD_fake_1 = loss_d_fake_1

        D_G_z11 = disc_fake_1.mean().item()
        summary_writer.add_scalar('Discriminator_1/Loss_D_Fake', loss_d_fake_1.item() / batch_size_2, iteration)

        errD_1 = errD_real_1 / batch_size_1 + errD_fake_1 / batch_size_2

        # train with the 2nd dataset real
        label_real_2 = torch.full((batch_size_2,), real_label, device=device) - label_noise * torch.rand((batch_size_2,), device=device)
        disc_real_2, class_real_2 = discriminator_2(data_2)
        disc_real_2 = disc_real_2.view(-1, 1).squeeze(1).sigmoid()
        loss_d_real_2 = criterion_discr(disc_real_2, label_real_2)
        errD_real_2 = loss_d_real_2

        D_x_2 = disc_real_2.mean().item()
        summary_writer.add_scalar('Discriminator_2/Loss_D_Real', loss_d_real_2.item() / batch_size_2, iteration)

        # train with fake
        label_fake_2 = torch.full((batch_size_1,), fake_label, device=device)
        disc_fake_2, class_fake_2 = discriminator_2(recon_1_2.detach())
        disc_fake_2 = disc_fake_2.view(-1, 1).squeeze(1).sigmoid()
        loss_d_fake_2 = criterion_discr(disc_fake_2, label_fake_2)
        errD_fake_2 = loss_d_fake_2

        D_G_z21 = disc_fake_2.mean().item()
        summary_writer.add_scalar('Discriminator_2/Loss_D_Fake', loss_d_fake_2.item() / batch_size_1, iteration)

        errD_2 = errD_real_2 / batch_size_2 + errD_fake_2 / batch_size_1

        errD = errD_1 + errD_2
        errD.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        shared_latent.zero_grad()
        encoder_1.zero_grad()
        encoder_2.zero_grad()
        decoder_1.zero_grad()
        decoder_2.zero_grad()

        # VAE losses
        loss_recon_1 = recon_loss(recon_1_1, data_1)
        loss_kl_1 = gauss_kl_loss(mu_1, sd_1)
        loss_vae_1 = args.lam1 * loss_kl_1 + args.lam2 * loss_recon_1

        loss_recon_2 = recon_loss(recon_2_2, data_2)
        loss_kl_2 = gauss_kl_loss(mu_2, sd_2)
        loss_vae_2 = args.lam1 * loss_kl_2 + args.lam2 * loss_recon_2

        summary_writer.add_scalar('Encoder_1/Loss_Recon', loss_recon_1.item() / batch_size_1, iteration)
        summary_writer.add_scalar('Encoder_1/Loss_KL', loss_kl_1.item() / batch_size_1, iteration)
        summary_writer.add_scalar('Encoder_2/Loss_Recon', loss_recon_2.item() / batch_size_2, iteration)
        summary_writer.add_scalar('Encoder_2/Loss_KL', loss_kl_2.item() / batch_size_2, iteration)

        # Cycle-consistency losses
        # l_cycle_1_2_1 = args.lam3 * gauss_kl_loss(mu_1, sd_1) + args.lam3 * gauss_kl_loss(mu_cycle_1_2, sd_cycle_1_2) + args.lam4 * recon_loss(cycle_recon_1_2_1, data_1)
        # l_cycle_2_1_2 = args.lam3 * gauss_kl_loss(mu_2, sd_2) + args.lam3 * gauss_kl_loss(mu_cycle_2_1, sd_cycle_2_1) + args.lam4 * recon_loss(cycle_recon_2_1_2, data_2)

        # GAN (for generators) losses
        # recon_1_2
        label_gen_1 = torch.full((batch_size_1,), real_label, device=device)
        discr_1_2, class_1_2 = discriminator_2(recon_1_2)
        discr_1_2 = discr_1_2.view(-1, 1).squeeze(1).sigmoid()
        loss_g1_d = criterion_discr(discr_1_2, label_gen_1)
        loss_g1_c = criterion_class(class_1_2, target_1)
        errG_1_2 = args.lam0 * (loss_g1_d + loss_g1_c)

        D_G_z12 = discr_1_2.mean().item()
        summary_writer.add_scalar('Decoder_1/Loss_D', loss_g1_d.item() / batch_size_1, iteration)
        summary_writer.add_scalar('Decoder_1/Loss_C', loss_g1_c.item() / batch_size_1, iteration)
        summary_writer.add_scalar('Decoder_1/Acc_Fake', compute_batch_accuracy(class_1_2, target_1), iteration)

        # recon_2_1
        label_gen_2 = torch.full((batch_size_2,), real_label, device=device)
        discr_2_1, class_2_1 = discriminator_1(recon_2_1)
        discr_2_1 = discr_2_1.view(-1, 1).squeeze(1).sigmoid()
        loss_g2_d = criterion_discr(discr_2_1, label_gen_2)
        errG_2_1 = args.lam0 * loss_g2_d

        D_G_z22 = discr_2_1.mean().item()
        summary_writer.add_scalar('Decoder_2/Loss_D', loss_g2_d.item() / batch_size_2, iteration)

        # errG = (l_vae_1 + l_cycle_1_2_1 + errG_1_2) / batch_size_1 + (l_vae_2 + l_cycle_2_1_2 + errG_2_1) / batch_size_2
        errG = (loss_vae_1 + errG_1_2) / batch_size_1 + (loss_vae_2 + errG_2_1) / batch_size_2
        errG.backward()
        optimizerG.step()

        if batch_idx % 100 == 0:
            tqdm.write('[%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f D(x1): %.3f D(G(z1)): %.3f / %.3f D(x2): %.3f D(G(z2)): %.3f / %.3f'
                       % (epoch+1, args.epochs, batch_idx+1, num_batch, errD.item(), errG.item(), D_x_1, D_G_z11, D_G_z12, D_x_2, D_G_z21, D_G_z22))

        save_img = False
        if iteration < 1000:
            if iteration % 10 == 0 and batch_idx != num_batch - 1:
                save_img = True
        elif iteration < 10000:
            if iteration % 100 == 0:
                save_img = True
        elif iteration <= 25000:
            if iteration % 1000 == 0:
                save_img = True

        if save_img:
            save_image(data_1, os.path.join(args.save, 'input_1_i{}.png'.format(iteration)), normalize=True)
            save_image(data_2, os.path.join(args.save, 'input_2_i{}.png'.format(iteration)), normalize=True)
            save_image(recon_1_1.detach(), os.path.join(args.save, 'recon_1_1_i{}.png'.format(iteration)), normalize=True)
            save_image(recon_1_2.detach(), os.path.join(args.save, 'recon_1_2_i{}.png'.format(iteration)), normalize=True)
            save_image(recon_2_1.detach(), os.path.join(args.save, 'recon_2_1_i{}.png'.format(iteration)), normalize=True)
            save_image(recon_2_2.detach(), os.path.join(args.save, 'recon_2_2_i{}.png'.format(iteration)), normalize=True)

        summary_writer.flush()
        iteration += 1

    # Evaluation (indirectly)
    shared_latent.eval()
    encoder_2.eval()
    decoder_1.eval()
    discriminator_1.eval()

    labels_true = []
    labels_pred = []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            z_val, mu_val, sd_val = shared_latent.sample(encoder_2(data))
            translated = decoder_1(z_val)
            _, class_translated = discriminator_1(translated)

            labels_pred.extend(class_translated.argmax(dim=1, keepdim=False).detach().cpu().numpy().tolist())
            labels_true.extend(target.detach().cpu().numpy().tolist())

    val_acc = 100. * fowlkes_mallows_score(labels_true, labels_pred)
    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc

    summary_writer.add_scalar('Validation/Acc_Trans', val_acc, epoch)
    tqdm.write('[%d/%d] Validation Accuracy %.3f' % (epoch+1, args.epochs, val_acc))

    save_checkpoint({
        'epoch': epoch,
        'iteration': iteration,
        'best_val_acc': best_val_acc,
        'shared_latent': shared_latent.state_dict(),
        'encoder_1': encoder_1.state_dict(),
        'encoder_2': encoder_2.state_dict(),
        'decoder_1': decoder_1.state_dict(),
        'decoder_2': decoder_2.state_dict(),
        'discriminator_1': discriminator_1.state_dict(),
        'discriminator_2': discriminator_2.state_dict(),
        'optimizerD': optimizerD.state_dict(),
        'optimizerG': optimizerG.state_dict(),
    }, is_best=is_best, filename=os.path.join(args.save, 'checkpoint.pth'))
