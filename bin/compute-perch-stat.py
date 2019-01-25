#!/usr/bin/env python3
import pdb
import argparse
import sys
import os

import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trans

import vmdata
import more_trans
import more_sampler


def make_parser():
    parser = argparse.ArgumentParser(description='Compute the per-channel mean'
                                     ' and stdandard deviation for the '
                                     'specified video dataset by random '
                                     'sampling.',
                                     epilog='Returns a space-separated list of'
                                     ' real numbers for mean and a second list'
                                     ' of the same format for std.')
    parser.add_argument('root',
                        help='the root directory of the video dataset')
    parser.add_argument('-n', type=int, default=50,
                        help='number of random samples, default %(default)s')
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


def plot_hists(root, mean, std, n_samples, outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize(mean=mean, std=std),
    ])

    channel_data_raw = []
    with vmdata.VideoDataset(root, transform=trans.ToTensor()) as raw_dset:
        sam = more_sampler.RandomSubsetSampler(raw_dset, n_samples)
        dl_params = {'sampler': sam, 'batch_size': 1, 'num_workers': 1}
        raw_dl = more_trans.numpy_loader(DataLoader(raw_dset, **dl_params))
        for raw_frame in raw_dl:
            assert raw_frame.shape[0] == 1
            raw_frame = raw_frame.reshape((raw_frame.shape[1], -1))
            channel_data_raw.append(raw_frame)
    channel_data_nml = []
    with vmdata.VideoDataset(root, transform=transform) as dset:
        # using the same sampler, `sam`; and the same dl paraeters, `dl_params`
        dl = more_trans.numpy_loader(DataLoader(dset, **dl_params))
        for frame in dl:
            assert frame.shape[0] == 1
            frame = frame.reshape((frame.shape[1], -1))
            channel_data_nml.append(frame)
    plt.close('all')
    for fid, frame, raw_frame in zip(sam, channel_data_nml, channel_data_raw):
        plt.figure(figsize=(28, 21))
        for c in range(frame.shape[0]):
            plt.subplot(frame.shape[0], 2, 2*c+1)
            plt.hist(frame[c])
            plt.title('nml c{}'.format(c))
            plt.subplot(frame.shape[0], 2, 2*c+2)
            plt.hist(raw_frame[c])
            plt.title('raw c{}'.format(c))
        plt.savefig(os.path.join(outdir, 'f{}.png'.format(fid)))
        plt.close()

def main():
    args = make_parser().parse_args()
    with vmdata.VideoDataset(args.root, transform=trans.ToTensor()) as dset:
        sam = more_sampler.RandomSubsetSampler(dset, args.n)
        dataloader = more_trans.numpy_loader(
            DataLoader(dset, batch_size=args.n,
                       sampler=sam, num_workers=2))
        frames = next(iter(dataloader))
        assert frames.shape[0] == args.n, 'frames.shape[0] = {} != args.n = {}'\
                .format(frames.shape[0], args.n)
        frames = frames.transpose((1, 0) + tuple(range(len(frames.shape)))[2:])
        frames = frames.reshape((frames.shape[0], -1))

        means = np.mean(frames, axis=1)
        stds = np.std(frames, axis=1)
        skews = np.abs(scipy.stats.skew(frames, axis=1))

    if args.skewness_ul is None:
        args.skewness_ul = np.inf
    skewed_ind = np.where(skews > args.skewness_ul)[0]
    if len(skewed_ind):
        if args.median_when_skewed:
            means[skewed_ind] = np.median(frames[skewed_ind], axis=1)
        else:
            means[skewed_ind] = np.zeros(means[skewed_ind].shape)
        stds[skewed_ind] = np.ones(stds[skewed_ind].shape)

    print(' '.join(map(str, means)))
    print(' '.join(map(str, stds)))
    if args.outfile is not None:
        np.savez(args.outfile, mean=means, std=stds)

    if args.hist_dir is not None:
        means = tuple(map(float, means))
        stds = tuple(map(float, stds))
        plot_hists(args.root, means, stds, 10, args.hist_dir)

if __name__ == '__main__':
    main()
