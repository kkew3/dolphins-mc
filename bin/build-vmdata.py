#!/usr/bin/env python
import argparse
import os

import vmdata

def make_parser():
    parser = argparse.ArgumentParser(description='Build memory-mapped file '
                                     'based video frame dataset')
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

def main():
    args = make_parser().parse_args()
    vmdata.create_vdset(args.video_file, args.dataset_rootdir,
                        batch_size=args.block_size,
                        max_batches=args.max_blocks)

if __name__ == '__main__':
    main()
