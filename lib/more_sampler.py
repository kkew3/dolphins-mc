import torch
from torch.utils.data.sampler import Sampler


class RandomSubsetSampler(Sampler):
    """
    Samples a random subset of indices of specified size from the data source.
    """

    def __init__(self, data_source, n, shuffle=False):
        """
        :param data_source: the dataset object
        :param n: subset size
        :param shuffle: False to sort the indices after sampling
        """
        self.data_source = data_source
        self.n = n
        self.shuffle = shuffle

    def __len__(self):
        return min(self.n, len(self.data_source))

    def __iter__(self):
        indices = torch.randperm(len(self.data_source)).tolist()[:self.n]
        if not self.shuffle:
            indices.sort()
        return iter(indices)


class DummySampler(Sampler):
    """
    For debug.
    """
    def __init__(self, *args, **kwargs):
        self.indices = range(100)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)
