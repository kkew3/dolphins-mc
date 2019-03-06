#!/usr/bin/env python3
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from rtc import rtc
import vmdata


def make_parser():
    parser = argparse.ArgumentParser(description='RTC demo')
    parser.add_argument('root', help='video dataset root')
    parser.add_argument('startframe', type=int,
                        help='id of the first (inclusive) frame')
    parser.add_argument('endframe', type=int,
                        help='id of the last (exclusive) frame')
    bbox_group = parser.add_mutually_exclusive_group(required=True)
    bbox_group.add_argument('-b', nargs=4, type=int,
                            metavar=('X', 'Y', 'W', 'H'), dest='object_box',
                            help='the initial object box (X,Y) coordinate and '
                                 '(W)idth and (H)eight of the box')
    bbox_group.add_argument('-l', metavar='NPYFILE', dest='fromfile',
                            help='npy file to load the initial object box')
    parser.add_argument('-d', dest='todir', help='where to write annotated frames')
    return parser

def parse_object_box(args):
    if args.object_box:
        object_box = rtc.Rect(*args.object_box)
    else:
        object_box = rtc.Rect(*np.load(args.fromfile).reshape(-1).astype(np.int32))
    return object_box

def draw_rectangle(frame: np.ndarray, object_box: rtc.Rect):
    annotated_frame = np.copy(frame)
    pt1 = object_box.x, object_box.y
    pt2 = object_box.x + object_box.width, object_box.y + object_box.height
    cv2.rectangle(annotated_frame, pt1, pt2, (200, 0, 0), 2)
    return annotated_frame

def savefig(handle, todir: str, digitwidth: int, frameid: int):
    filename = os.path.join(todir, 'f' + str(frameid).rjust(digitwidth, '0') + '.png')
    handle.savefig(filename)
    handle.close()

def main():
    args = make_parser().parse_args()
    os.makedirs(args.todir, exist_ok=True)
    object_box = parse_object_box(args)
    digitwidth = np.ceil(np.log10(args.endframe)).astype(np.int32)
    with vmdata.VideoDataset(args.root) as dset:
        frame = dset[args.startframe]
        annotframe = draw_rectangle(frame, object_box)
        plt.imshow(annotframe)
        savefig(plt, args.todir, digitwidth, args.startframe)
        grayframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        tracker = rtc.CompressiveTracker(grayframe, object_box)
        for i in range(1 + args.startframe, args.endframe):
            frame = dset[i]
            grayframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            object_box = tracker.process_frame(grayframe, object_box)
            annotframe = draw_rectangle(frame, object_box)
            plt.imshow(annotframe)
            savefig(plt, args.todir, digitwidth, i)

if __name__ == '__main__':
    main()
