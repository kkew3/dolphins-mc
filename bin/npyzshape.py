#!/usr/bin/env python
import argparse

import numpy as np


def make_parser():
    parser = argparse.ArgumentParser(
        description='Print array shape. If the input file is an npy file, '
                    'the output will be a comma-separated list of integers, '
                    'signifying the dimension along each axis; if the input '
                    'file is an npz file, each line of the output will be '
                    '"${filename}:${comma-separated-list-of-dimensions}".')
    parser.add_argument('npyzfile')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    data = np.load(args.npyzfile)
    try:
        for k, v in data.items():
            if len(v.shape):
                print(k, ':', ','.join(map(str, v.shape)), sep='')
            else:
                print(k, ': <scalar>')
    except AttributeError:
        print(','.join(map(str, data.shape)))
