#!/usr/bin/env python
import os
import argparse
from contextlib import suppress

import cv2
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import vmdata
import more_trans
import more_sampler


_num_workers = 0

def make_parser():
    parser = argparse.ArgumentParser(description='Compute the per-channel mean'
                                     ' and standard deviation for the '
                                     'specified video dataset in black&white '
                                     'by random sampling.',
                                     epilog='Returns two real number, one per '
                                     'line, for respectively the mean and '
                                     'std.')
    parser.add_argument('root',
                        help='the video dataset root')
    parser.add_argument('-n', type=int, default=50,
                        help='number of random samples, default %(default)s')
    parser.add_argument('-o', dest='outfile',
                        help='result file to write, which will be an npz file '
                             'with fields "mean" and "std"')
    parser.add_argument('-d', dest='hist_dir',
                        help='directory under which to create histogram plots,'
                             ' which can be used to evaluate the skewness of '
                             'the normalized video frames')
    return parser

def rgb2bw(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., np.newaxis]

def plot_hists(root, mean, std, n_samples, outdir):
    with suppress(OSError):
        os.mkdir(outdir)

    raw_transform = trans.Compose([
            trans.Lambda(lambda x: rgb2bw(x)),
            trans.ToTensor(),
    ])
    with vmdata.VideoDataset(root, transform=raw_transform) as raw_dset:
        sam = more_sampler.RandomSubsetSampler(raw_dset, n_samples)
        params = {'sampler': sam, 'batch_size': n_samples, 'num_workers': 0}
        rdata = next(iter(more_trans.numpy_loader(DataLoader(raw_dset, **params)))).reshape((n_samples, -1))

    nml_transform = trans.Compose([
            raw_transform,
            trans.Normalize(mean=mean, std=std),
    ])
    with vmdata.VideoDataset(root, transform=nml_transform) as dset:
        ndata = next(iter(more_trans.numpy_loader(DataLoader(dset, **params)))).reshape((n_samples, -1))

    plt.close('all')
    for fid, frame, raw_frame in zip(sam, ndata, rdata):
        plt.figure(figsize=(18, 8))
        plt.subplot(121)
        plt.hist(frame)
        plt.title('nml')
        plt.subplot(122)
        plt.hist(raw_frame)
        plt.title('raw')
        plt.savefig(os.path.join(outdir, 'f{}.png'.format(fid)))
        plt.close()



def main():
    args = make_parser().parse_args()
    transform = trans.Compose([
            trans.Lambda(lambda x: rgb2bw(x)),
            trans.ToTensor(),
    ])
    with vmdata.VideoDataset(args.root, transform=transform) as dset:
        sam = more_sampler.RandomSubsetSampler(dset, args.n)
        dataloader = more_trans.numpy_loader(
                DataLoader(dset, batch_size=args.n,
                           sampler=sam, num_workers=_num_workers))
        frames = next(iter(dataloader))
        assert frames.shape[0] == args.n
        pixels = frames.reshape(-1)
        mean = np.mean(pixels)
        std = np.std(pixels)

    print(mean)
    print(std)
    if args.outfile is not None:
        np.savez(args.outfile, mean=mean[np.newaxis], std=std[np.newaxis])

    if args.hist_dir is not None:
        plot_hists(args.root, (float(mean),), (float(std),), 10, args.hist_dir)


if __name__ == '__main__':
    main()
