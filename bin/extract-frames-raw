#!/usr/bin/env python3
import argparse
import math
import os
import pdb
from typing import Optional

import cv2

import utils


def make_parser():
    parser = argparse.ArgumentParser(
        description='Extract frames from raw videos into images.')
    parser.add_argument('video', metavar='VIDEO',
                        help='the video file to extract frames')
    parser.add_argument('-s', '--start', type=int, help='start index')
    parser.add_argument('-t', '--stop', type=int, help='stop index')
    parser.add_argument('-d', '--todir',
                        help='directory under which to write the frames; if '
                             'not specified, the basename without extension '
                             'of VIDEO will be used as TODIR, relative to '
                             'current working directory. TODIR must be either '
                             'non-exist (and will be created), or empty')
    parser.add_argument('-c', '--color', choices=('r', 'g'), default='g',
                        help='whether (r)gb or (g)rayscale; default to '
                             '%(default)s')
    parser.add_argument('-p', '--prefix',
                        help='the images will be named as '
                             '"${PREFIX}${index}.png" under TODIR')
    parser.add_argument('-R', '--rename', action='store_true',
                        help='rename the images such that the lexicographical'
                             ' order is consistent with the numerical order '
                             'by prepending \'0\' before "${index}" after '
                             'all images have been written')
    return parser


def prepare_workspace(todir: str):
    try:
        os.rmdir(todir)
    except FileNotFoundError:
        pass
    except OSError:
        raise
    os.makedirs(todir, exist_ok=False)


def ensure_incompleted(rng, index: int):
    s, t = rng
    if t is not None and index >= t:
        raise StopIteration


def in_range(rng, index: int):
    s, t = rng
    if s is None and t is None:
        return True
    elif s is None:
        return index < t
    elif t is None:
        return index >= s
    else:
        return s <= index < t


def proc_frame(frame, color: str):
    if color == 'g':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


if __name__ == '__main__':
    args = make_parser().parse_args()
    args.ind_range = args.start, args.stop
    prepare_workspace(args.todir)


    def imgpath(index):
        return os.path.join(args.todir, '{}.png'.format(index))


    filenames = []
    with utils.capcontext(args.video) as cap:
        try:
            for i, frame in enumerate(utils.frameiter(cap)):
                ensure_incompleted(args.ind_range, i)
                if in_range(args.ind_range, i):
                    frame = proc_frame(frame, args.color)
                    name = imgpath(i)
                    filenames.append(name)
                    cv2.imwrite(name, frame)
        except StopIteration:
            pass

    if args.rename:
        indices = list(int(os.path.splitext(os.path.basename(x))[0])
                       for x in filenames)
        max_index = max(indices)
        width = math.ceil(math.log10(max_index))
        for i, fromname in zip(indices, filenames):
            toname = imgpath(str(i).rjust(width, '0'))
            os.rename(fromname, toname)
