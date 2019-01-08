#!/usr/bin/env python -O
import argparse
import contextlib
import itertools
import operator
import os
import sys
from functools import reduce, partial
from typing import Optional, List, Union, Sequence, Any

import numpy as np
import matplotlib

if __debug__:
    import pdb

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_parser():
    parser = argparse.ArgumentParser(
        description='Plot 2D array into image(s) by indexing the npy/npz '
                    'file.')
    parser.add_argument('npyzfile')
    parser.add_argument('-d', dest='basedir', default='.',
                        help='the base directory of output images, which will '
                             'be created if not yet exists, default to '
                             '"%(default)s"')
    parser.add_argument('-f', '--overwrite', dest='overwrite',
                        action='store_true',
                        help='to overwrite existing file when extracting '
                             'images')
    parser.add_argument('-r', '--routine', metavar='ROUTINE_FILE',
                        help='post-processing routine of the image matrix. '
                             'ROUTINE_FILE should contain a function called '
                             '`process\' that takes the image matrix and '
                             'returns the processed matrix.')
    parser.add_argument('-I', nargs='*', dest='indices', metavar='INDEX',
                        default=[], action='append',
                        help='the indices; one may specify N times this '
                             'option to extract N images. One may use `-\' '
                             'to traverse one axis. For example, given a '
                             '3x4x5 tensor, one option of `-I -\' will be '
                             'expanded to three options `-I0 -I1 -I2\'. Note '
                             'that `-\' only applies to axis of numpy array, '
                             'rather than the string keys in an npz file')
    parser.add_argument('-M', dest='cmaps', action='append', default=[],
                        metavar='CMAP',
                        help='one may specify zero for color-map default, one '
                             'CMAP for all images, or the same number of '
                             'occurrences as option `-I\' for each image. In '
                             'the third option, one may use `-\' to denote '
                             'the default color-map')
    return parser


def expand_index(data, index: Sequence[str]) -> List[Sequence[Any]]:
    """
    Expand an underlying expandable index to type validated list of indices.
    If ``index`` is not expandable, the returned list will contain only the
    ``index`` itself.

    :param data: the data
    :param index: the index to expand
    :return: expanded indices
    :raise ValueError: if ``index`` contains '-' on non-array axis
    :raise IndexError: if ``index`` contains invalid index entry
    """
    xidx = []
    for i in index:
        if i == '-':
            try:
                xidx.append(range(data.shape[0]))
            except AttributeError as e:
                raise ValueError(e)
            else:
                data = data[0]
        else:
            try:
                data = data[i]
            except (TypeError, IndexError, KeyError):
                i = int(i)
                data = data[i]
            xidx.append([i])
    xidx = [i for i in itertools.product(*xidx)]
    return xidx


class IllegalArgsLengthError(BaseException):
    def __init__(self, argname: str):
        super().__init__('Illegal number of {} provided'.format(argname))


class prepare_plot_meta(object):
    def __init__(self, cmaplist: List[str]):
        self.n = len(cmaplist)
        self.cmap0 = cmaplist[0] if self.n else None
        self.cmaps = iter(cmaplist)

    def __iter__(self):
        return self

    def __next__(self):
        kwargs = {}
        if self.n == 1:
            kwargs['cmap'] = self.cmap0
        elif self.n > 1:
            try:
                cm = next(self.cmaps)
            except StopIteration:
                raise IllegalArgsLengthError('cmap')
            else:
                if cm and cm != '-':
                    kwargs['cmap'] = cm
        return kwargs


def prepare_out_name(dataname: str, all_indices: List[Sequence[Any]]):
    all_indices = iter(all_indices)
    for index in all_indices:
        yield '{}-{}.png'.format(dataname, '_'.join(map(str, index)))


def prepare_default_outnames(dataname: str, all_indices: List[Sequence[Any]]):
    for i in all_indices:
        yield '{}-{}.png'.format(dataname, '_'.join(map(str, i)))


def identity_routine(_):
    return _


if __name__ == '__main__':
    args = make_parser().parse_args()
    name, ext = os.path.splitext(os.path.basename(args.npyzfile))
    npyzdata = np.load(args.npyzfile)
    aug_indices_groups = list(map(partial(expand_index, npyzdata),
                                  args.indices))
    if args.routine:
        with open(args.routine) as infile:
            src = infile.read()
        o = compile(src, args.routine, 'exec')
        ns = {}
        exec(o, ns)
        routine = ns['process']
    else:
        routine = identity_routine
    if len(args.cmaps) > 1 and len(args.cmaps) != len(args.indices):
        raise IllegalArgsLengthError('cmaps')

    if __debug__:
        print('==> args.indices <==')
        print('\n'.join([','.join(map(str, x)) for x in args.indices]))
        print('==> end of args.indices <==')

        print()
        print('==> args.cmaps <==')
        print('\n'.join(map(str, args.cmaps)))
        print('==> end of args.cmaps <==')

        print()
        print('==> aug_indices_groups <==')
        print('\n'.join([', '.join(map(str, x)) for x in aug_indices_groups]))
        print('==> end of aug_indices_groups <==')

    os.makedirs(args.basedir, exist_ok=True)

    for j, grp in enumerate(aug_indices_groups):
        default_names = prepare_default_outnames(name, grp)
        if len(args.cmaps) == 0:
            mt = {}
        elif len(args.cmaps) == 1:
            mt = {'cmap': args.cmaps[0]}
        elif args.cmaps[j] == '-':
            mt = {}
        else:
            mt = {'cmap': args.cmaps[j]}
        metas = itertools.repeat(mt)
        outfiles = prepare_out_name(name, grp)
        for idx, cm, out in zip(grp, metas, outfiles):
            out = os.path.join(args.basedir, out)
            im = npyzdata
            for i in idx:
                im = im[i]
            im = routine(im)
            if os.path.exists(out) and not args.overwrite:
                print('Skipping writing {} (index={}) due to existing file'
                      .format(out, ','.join(map(str, idx))), file=sys.stderr)
            else:
                if not __debug__:
                    plt.imsave(out, im, **cm)
