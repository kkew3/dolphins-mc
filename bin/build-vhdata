#!/usr/bin/env python3
import argparse
import os
from itertools import count

import cv2
import numpy as np
import ddlogger

import vhdata
import utils


def make_parser():
    parser = argparse.ArgumentParser(description='Build HDF5-based video '
                                     'frame dataset.')
    parser.add_argument('video_file',
                        help='the video file from which to extract frames')
    parser.add_argument('dataset_rootdir',
                        help='parent directory of the root directory of the '
                             'dataset')
    test_group = parser.add_argument_group(title='these arguments are used to '
                                           'build dataset for development '
                                           'environment')
    test_group.add_argument('--block-size', type=int, metavar='N',
                            dest='block_size', default=1000,
                            help='the number of frames to write in batch; '
                                 'default to %(default)s')
    test_group.add_argument('--head', type=int, metavar='B', dest='max_blocks',
                            help='build dataset only using the first B blocks '
                                 'of the given video file')
    return parser

def save_frames(video_file, dataset_root, block_size, max_blocks):
    """
    :param video_file: the path of the video file
    :param dataset_root: root directory of the dataset; must exists
    :param block_size: the number of frames to be saved as a HDF5 block
    :param max_blocks: None to save blocks as many as possible, otherwise save
           up to this number of blocks
    """
    with utils.capcontext(video_file) as cap:
        shape = (3,
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                )
        with vhdata.VideoDatasetWriter(dataset_root, shape,
                                       dimension='NCHW') as writer:
            with ddlogger.ddlogger() as dl:
                for _ in range(max_blocks) if max_blocks else count():
                    frames = list(utils.frameiter(cap, block_size))  # frames of dimension HWC
                    if not frames:
                        break
                    frames = np.stack(frames)  # of dimension NHWC
                    frames = np.transpose(frames, (0, 3, 1, 2))  # NCHW
                    writer.write_batch(frames)
                    dl.update()

def main():
    args = make_parser().parse_args()
    root = os.path.join(args.dataset_rootdir,
                        os.path.splitext(os.path.basename(args.video_file))[0])
    if not os.path.isdir(root):
        os.mkdir(root)
    save_frames(args.video_file, root,
                args.block_size, args.max_blocks)

if __name__ == '__main__':
    main()
