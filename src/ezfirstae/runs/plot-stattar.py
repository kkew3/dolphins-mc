#!/usr/bin/env python
import argparse
import tarfile
from functools import reduce
import os
import operator

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_parser():
    parser = argparse.ArgumentParser(description='Analzye and plot loss graph '
                                     'from stat.*.tar. The outputs will be '
                                     'stat.*.tar.png.')
    parser.add_argument('tarfiles', metavar='FILE', nargs='*')
    return parser


def stat_sortkey(filename: str):
    name = os.path.splitext(os.path.basename(filename))[0]
    return tuple(map(int, name.split('_')[1:]))


def loaddata(filename: str):
    with tarfile.TarFile(filename, 'r') as tar_infile:
        for member in tar_infile:
            if member.isfile():
                infile = tar_infile.extractfile(member)
                datadict = np.load(infile)
                datadict_local = {k: v for k, v in datadict.items()}
                yield member.name, datadict_local


def organize_data(filename: str):
    dicts = [y for _, y in sorted(loaddata(filename),
                                  key=lambda x: stat_sortkey(x[0]))]
    if not dicts:
        return [], []
    _keys = [set(d.keys()) for d in dicts]
    joint_keys = list(reduce(operator.and_, _keys))
    data = [tuple(d[k] for k in joint_keys) for d in dicts]
    lines = list(map(np.stack, zip(*data)))
    return lines, joint_keys


def plot(filename, lines, legend):
    plt.figure(figsize=(18,8))
    for l in lines:
        plt.plot(l)
    plt.yscale('log')
    plt.ylabel(r'$\log(\cdot)$')
    plt.legend(legend)
    plt.grid()
    plt.savefig(filename + '.png')


def main():
    args = make_parser().parse_args()
    for filename in args.tarfiles:
        plot(filename, *organize_data(filename))


if __name__ == '__main__':
    main()
