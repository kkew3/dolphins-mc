import argparse
import operator
from functools import partial
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import vmdata
# noinspection PyUnresolvedReferences
from pyflow import coarse2fine_flow


flow_params = (
        0.012,   # alpha
        0.75,    # ratio
        20,      # minWidth
        7,       # nOuterFPIterations
        1,       # nInnerFPIterations
        30,      # nSORIterations
        0,       # colType (0 for RGB, 1 for GRAY)
)

inf = 1000000000000000000000000


def load_img_pair(vdset, fid):
    im1 = vdset[fid].astype(np.float64) / 255.0
    im2 = vdset[fid+1].astype(np.float64) / 255.0
    return im1, im2

def visualize_flow(frame, fid, u, v):
    hsv = np.zeros(frame.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(u, v)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float64) / 255.0
    plt.figure(figsize=(18,8))
    plt.imsave('results/f{fid}.png'.format(fid=fid),
               np.concatenate((frame, rgb), axis=1))
    plt.close()

def analyze_flow(vdset, fid):
    im1, im2 = load_img_pair(vdset, fid)
    u, v, _ = coarse2fine_flow(im1, im2, *flow_params)
    visualize_flow(im1, fid, u, v)

def make_parser():
    parser = argparse.ArgumentParser(description='Compute Optical Flow')
    parser.add_argument('-c', '--camera', default=9, choices=[7,8,9], type=int)
    parser.add_argument('fids', nargs='*', type=int)
    parser.add_argument('--debug', action='store_true')
    return parser

def main():
    args = make_parser().parse_args()
    root = vmdata.dataset_root(args.camera, (8, 0, 0))
    with vmdata.VideoDataset(root) as vdset:
        _analyze_flow = partial(analyze_flow, vdset)
        if args.debug:
            print(args.fids)
            return
        for fid in args.fids:
            print('Running on frame {}'.format(fid), file=sys.stderr)
            _analyze_flow(fid)

if __name__ == '__main__':
    main()
