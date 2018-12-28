#!/usr/bin/env python
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def indicestype(string: str):
    typeids = {
        's': str,
        'i': int,
    }
    return [typeids[c] for c in string]

def make_parser():
    parser = argparse.ArgumentParser(
        description='Plot 2D array into image by indexing the npy/npz file')
    parser.add_argument('-T', type=indicestype, default='',
                        dest='idt', metavar='SPECSTRING',
                        help='string with the ith character specifying the '
                             'data type of the ith index object, \'s\' '
                             'for string and \'i\' for integer; default to '
                             'all-ints')
    parser.add_argument('-I', nargs='*', dest='indices', metavar='INDEX',
                        help='the indices')
    parser.add_argument('-M', dest='cmap')
    parser.add_argument('npyzfile')
    parser.add_argument('tofile')
    return parser


def index_img(filename, indices, types):
    if not types:
        types = [int] * len(indices)
    if len(indices) != len(types):
        raise ValueError('len(indices) != len(types): {} != {}'
                         .format(len(indices), len(types)))
    data = np.load(filename)
    for idx, dt in zip(indices, types):
        idx = dt(idx)
        data = data[idx]
    return data

if __name__ == '__main__':
    args = make_parser().parse_args()
    img = index_img(args.npyzfile, args.indices, args.idt)
    kwargs = {}
    if args.cmap:
        kwargs['cmap'] = args.cmap
    plt.imsave(args.tofile, img, **kwargs)
