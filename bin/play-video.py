#!/usr/bin/env python
import os
import time
import argparse

import cv2

import utils


def make_parser():
    parser = argparse.ArgumentParser(description='Play avi video')
    parser.add_argument('-f', dest='video', help='the video filename')
    parser.add_argument('-c', dest='cameraid', default=9, choices=[7,8,9],
                        help='the camera id, default to %(default)s; '
                             'ignored when `-f\' has been specified')
    parser.add_argument('-G', dest='rgb', action='store_false',
                        help='to play in gray color; default to RGB color')
    parser.add_argument('-s', dest='scale', type=int, default=1,
                        help='the scale factor on width/height, default to '
                             '%(default)s')
    return parser

def play_scaled(filename, scale, in_rgb):
    framename = 'frame-downscaled_{}x'.format(scale)
    with utils.capcontext(filename) as cap:
        for frame in utils.frameiter(cap, rgb=False):
            if not in_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scaled = cv2.resize(frame, (0, 0), fx=1./scale, fy=1./scale)
            cv2.imshow(framename, scaled)
            if cv2.waitKey(300) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    args = make_parser().parse_args()
    if args.video:
        filename = args.video
    else:
        filename = os.path.join(os.environ['PROJ_HOME'], 'res', 'videos',
                                'CH0{}-08_00_00.avi'.format(args.cameraid))
    play_scaled(filename, args.scale, args.rgb)
