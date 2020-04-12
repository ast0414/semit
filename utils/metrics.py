import torch
from torch.nn import functional as F


def recon_loss(recon_x, x):
    # Reconstruction loss summed over all elements and batch
    #return F.binary_cross_entropy(recon_x, x, reduction='sum')
    # return F.l1_loss(recon_x, x, reduction='sum')
    return F.smooth_l1_loss(recon_x, x, reduction='sum')
    # return F.mse_loss(recon_x, x, reduction='sum')


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


def compute_batch_accuracy(output, target):
    """Computes the accuracy for a batch"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum()

    return correct * 100.0 / batch_size