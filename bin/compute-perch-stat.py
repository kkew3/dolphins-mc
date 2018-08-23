#!/usr/bin/env python
import argparse
import sys
import os

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as trans

import vdata


def vdset(root):
    try:
        return root, vdata.VideoDataset(root, max_block_cached=2,
                                        transform=trans.ToTensor())
    except:
        raise argparse.ArgumentTypeError('Error loading dataset "{}"'
                                         .format(root))

def make_parser():
    parser = argparse.ArgumentParser(description='Compute the per-channel mean'
                                     ' and stdandard deviation for the '
                                     'specified video dataset by random '
                                     'sampling.',
                                     epilog='Returns a space-separated list of'
                                     ' real numbers for mean and a second list'
                                     ' of the same format for std.')
    parser.add_argument('root', type=vdset,
                        help='the root directory of the video dataset')
    parser.add_argument('-n', type=int, default=50,
                        help='number of random samples, default 10.0')
    parser.add_argument('-o', dest='outfile',
                        help='result file to write, which will be an npz file '
                             'with fields "mean" and "std"')
    parser.add_argument('-k', nargs='?', dest='skewness_ul', type=float,
                        const=10.0,
                        help='the absolute skewness upperbound of data; if '
                             'exceeded, the mean will be set to 0 and std '
                             'to 1; default to %(default)s when still '
                             'specifying "-k", otherwise default to infinity')
    parser.add_argument('-m', dest='median_when_skewed', action='store_true',
                        help='set mean to median rather than 0 when heavily '
                             'skewed (i.e. absolute skewness larger than '
                             '`SKEWNESS_UL`')
    parser.add_argument('-d', dest='hist_dir',
                        help='directory under which histogram images to '
                             'create, which can be used to evaluate the '
                             'skewness of the normalized video frames')
    return parser

class RandomSubsetSampler(Sampler):
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

def plot_hists(root, mean, std, n_samples, outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize(mean=mean, std=std),
    ])
    raw_dset = vdata.VideoDataset(root, transform=trans.ToTensor())
    dset = vdata.VideoDataset(root, transform=transform)
    sam = RandomSubsetSampler(dset, n_samples)
    raw_dataloader = DataLoader(raw_dset, sampler=sam, batch_size=1, num_workers=1)
    dataloader = DataLoader(dset, sampler=sam, batch_size=1, num_workers=1)

    plt.close('all')
    for fid, frame, raw_frame in zip(sam, dataloader, raw_dataloader):
        frame = frame.view(*frame.shape[1:])
        frame = frame.view(frame.shape[0], -1).numpy()
        raw_frame = raw_frame.view(*raw_frame.shape[1:])
        raw_frame = raw_frame.view(raw_frame.shape[0], -1).numpy()

        data = np.stack((frame, raw_frame))
        data_names = ['nml', 'raw']

        plt.figure(figsize=(28, 21))
        for c in range(frame.shape[0]):
            for d in range(2):
                plt.subplot(frame.shape[0], 2, 2*c+d+1)
                plt.hist(data[d,c])
                plt.title('{} c{}'.format(data_names[d], c))
        plt.savefig(os.path.join(outdir, 'f{}.png'.format(fid)))
        plt.close()

def main():
    args = make_parser().parse_args()
    root, dset = args.root
    sam = RandomSubsetSampler(dset, args.n)
    dataloader = DataLoader(dset, batch_size=args.n,
                            sampler=sam, num_workers=2)
    frames = next(iter(dataloader)).numpy()
    assert frames.shape[0] == args.n, 'frames.shape[0] = {} != args.n = {}'\
            .format(frames.shape[0], args.n)
    assert len(frames.shape) == 4, 'len(frames.shape) = {}'.format(len(frames.shape))
    frames = frames.transpose((1, 0) + tuple(range(len(frames.shape)))[2:])
    frames = frames.reshape((frames.shape[0], -1))

    means = np.mean(frames, axis=1)
    stds = np.std(frames, axis=1)
    skews = np.abs(scipy.stats.skew(frames, axis=1))

    if args.skewness_ul is None:
        args.skewness_ul = np.inf
    skewed_ind, = np.where(skews > args.skewness_ul)
    if len(skewed_ind):
        if args.median_when_skewed:
            means[skewed_ind] = np.median(frames[skewed_ind], axis=1)
        else:
            means[skewed_ind] = np.zeros(means[skewed_ind].shape)
        stds[skewed_ind] = np.ones(stds[skewed_ind].shape)

    print ' '.join(map(str, means))
    print ' '.join(map(str, stds))
    if args.outfile is not None:
        np.savez(args.outfile, mean=means, std=stds)

    if args.hist_dir is not None:
        means = tuple(map(float, means))
        stds = tuple(map(float, stds))
        plot_hists(root, means, stds, 10, args.hist_dir)

if __name__ == '__main__':
    main()
