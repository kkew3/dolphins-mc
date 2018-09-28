#!/usr/bin/env python
import argparse
import sys
import os
import operator as op
from functools import partial

from typing import List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import vmdata


supported_exts = ['.png', '.jpg']
default_todir = '.'
default_ext = '.png'
default_nametmpl = 'f{fid}'


def make_parser():
    parser = argparse.ArgumentParser(description='Extract frames in PNG images'
                                     ' from a video dataset.')
    parser.add_argument('-d', metavar='DIR', dest='todir', default='.',
                        help='directory to write extracted frames (in PNG)')
    parser.add_argument('root', help='the video dataset root')
    parser.add_argument('frames', nargs='*',
                        help='the frames INDICES to extract, which should be '
                             'nonnegative integers; if \'-\' is provided, '
                             'the frame indices will be read from stdin')
    parser.add_argument('-c', '--format', default=default_nametmpl,
                        help='the python str.format string to name the '
                             'extracted frames; the string "{fid}" will be '
                             'replaced by the index of each frame extracted. '
                             'Default to "%(default)s"')
    parser.add_argument('-x', '--ext', choices=supported_exts,
                        default=default_ext,
                        help='supported image format, default to %(default)s')
    parser.add_argument('-a', action='store_true', dest='fid_aligned',
                        help='make frame index in image filename right-aligned'
                             ' with \'0\'')
    return parser

def parse_frame_indices(indices: List[str]):
    """
    Return iterator of frame indices to extract.
    """
    if indices:
        if indices[0] == '-':
            if not sys.stdin.isatty():
                indices = map(int, sys.stdin)
            else:
                indices = []
        else:
            indices = map(int, indices)
    else:
        indices = []
    return indices

def save_frame(fid: int, frame: np.ndarray, todir: str=default_todir,
               ext: str=default_ext, nametmpl: str=default_nametmpl,
               align_njust: int=None):
    if align_njust:
        fid = str(fid).rjust(align_njust, '0')
    tofile = os.path.join(todir, nametmpl.format(fid=fid) + ext)
    plt.imsave(tofile, frame)
    plt.close()

def main():
    args = make_parser().parse_args()
    indices = list(parse_frame_indices(args.frames))
    njust = None
    if args.fid_aligned:
        njust = int(np.ceil(np.log10(max(indices))))
    args.root = os.path.normpath(args.root)
    with vmdata.VideoDataset(root=args.root) as vdset:
        for fid in indices:
            save_frame(fid, vdset[fid], todir=args.todir, ext=args.ext,
                       nametmpl=args.format, align_njust=njust)


if __name__ == '__main__':
    main()
