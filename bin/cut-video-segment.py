#!/usr/bin/env python3
import argparse

import cv2

import vmdata
import utils


def make_parser():
    parser = argparse.ArgumentParser(description='Extract a segment from a '
                                                 'VideoDataset.')
    parser.add_argument('root', help='the dataset root')
    parser.add_argument('out', help='the output video file to write')
    parser.add_argument('-f', type=int, metavar='M', default=0, dest='fromf',
                        help='the segment will start with the M-th frame, '
                             'default to the first frame')
    parser.add_argument('-t', type=int, metavar='N', dest='tof',
                        help='the segment will ends with the N-th frame '
                             '(inclusive), default to the last frame')
    parser.add_argument('--fps', type=float, default=30.0)
    return parser


def extract(fromframe: int, toframe: int, dset: vmdata.VideoDataset,
            fps: float, outfile: str):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    h, w = dset.metainfo['resolution']
    with utils.videowritercontext(outfile, fourcc, fps, (w, h)) as writer:
        for i in range(fromframe, toframe + 1):
            frame = dset[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.waitKey(33)
            writer.write(frame)

def main():
    args = make_parser().parse_args()
    with vmdata.VideoDataset(args.root) as dset:
        if not (0 <= args.fromf < args.tof < len(dset)):
            raise RuntimeError('Invalid frame range: {}~{}'
                               .format(args.fromf, args.tof))
        extract(args.fromf, args.tof, dset, args.fps, args.out)

if __name__ == '__main__':
    main()
