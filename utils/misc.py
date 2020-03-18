import os
import shutil
import torch


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        directory = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(directory, 'best_checkpoint.pth'))