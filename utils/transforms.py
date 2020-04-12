import numpy as np
import torch
import torch.nn.functional as F


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class Normalize(object):
    """
    1D Normalization
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (ndarray): data to be normalized.

        Returns:
            ndarray: Normalized data array.
        """

        # Broadcated operation
        data -= self.mean
        data /= self.std

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(object):

    def __init__(self, size, mode='bilinear'):
        self.size = size
        self.mode = mode

    def __call__(self, data):
        data = F.interpolate(data.unsqueeze(0), size=self.size, mode=self.mode, align_corners=False)
        data = data.squeeze(0)
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
