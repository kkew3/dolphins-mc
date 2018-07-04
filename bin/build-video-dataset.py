#!/usr/bin/env python

r"""
Build video dataset based on Hierarchical Data Format 5 (HDF5) and `hickle`
module.

Directory structure of dataset:

    dataset_root/
    |- video_name_0/
    |  |- B0.hkl
    |  |- B1.hkl
    |  | ...
    |  |- BM.hkl
    |  \- video_name_0.sha256
    |- ...
    |- video_name_x/
    |  |- B0.hkl
    |  |- B1.hkl
    |  |- ...
    |  |- BN.hkl
    \  \- video_name_x.sha256
"""

import argparse
import os
import sys
import glob
from collections import OrderedDict
import itertools
import operator
import hashlib

import cv2
from tqdm import tqdm
import numpy as np
import hickle

import utils


BATCH_SIZE = 1000

def edir(string):
    if not os.path.isdir(string):
        raise argparse.ArgumentTypeError('"{}" not found'.format(string))
    return string

def efile(string):
    if not os.path.isfile(string):
        raise argparse.ArgumentTypeError('"{}" not found'.format(string))
    return string

def make_parser():
    parser = argparse.ArgumentParser(description='Build HDF5-based video '
            'frame dataset')
    parser.add_argument('-f', metavar='FILE', dest='video_files',
            help='the video file from which to extract frames', nargs='+',
            type=efile)
    parser.add_argument('dataset_root', help='root directory of the dataset',
            type=edir)
    return parser


def save_frames(video_file, dataset_root):
    """
    :param video_file: the path of the video file
    :param dataset_root: root directory of the dataset; must exists
    """
    with utils.capcontext(video_file) as cap:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vit = utils.FrameIterator(cap, max_len=min(BATCH_SIZE, total_frames))
        video_name, _ = os.path.splitext(os.path.basename(video_file))
        batches_dir = os.path.join(dataset_root, video_name)
        if not os.path.isdir(batches_dir):
            os.mkdir(batches_dir)
        dest_tmpl = 'B{}.hkl'
        with tqdm(total=total_frames, ascii=True) as t:
            for batch_id in itertools.count():
                frames = list(vit)
                if not len(frames):
                    break

                # frames of dimension:
                # [N x W x H x 3], where N <= BATCH_SIZE
                frames = np.stack(frames)
                frames = np.transpose(frames, (0, 3, 1, 2))
                tofile = os.path.join(batches_dir, dest_tmpl.format(batch_id))
                with open(tofile, 'w') as outfile:
                    hickle.dump(frames, outfile, compression='gzip')
                t.update(frames.shape[0])
                vit.reset_counter()

def verify_frames(video_file, dataset_root):
    """
    See if dumped hickle files can be opened successfully.

    :param video_file: the path of the video file
    :param dataset_root: root directory of the dataset; must exists
    """
    video_name, _ = os.path.splitext(os.path.basename(video_file))
    batches_dir = os.path.join(dataset_root, video_name)
    failed_filenames = list()
    for filename in glob.iglob(os.path.join(batches_dir, 'B*.hkl')):
        try:
            with open(filename, 'r') as _:
                pass
        except BaseException:
            failed_filenames.append(filename)
    return failed_filenames

def compute_checksum(video_file, dataset_root):
    """
    :param video_file: the path of the video file
    :param dataset_root: root directory of the dataset; must exists
    :return: a dict of checksums, of form (key=filename, value=sha256(hex))
    """
    video_name, _ = os.path.splitext(os.path.basename(video_file))
    batches_dir = os.path.join(dataset_root, video_name)
    hash_results = OrderedDict()
    data_batches = glob.glob(os.path.join(batches_dir, 'B*.hkl'))
    for filename in tqdm(data_batches, ascii=True):
        sha256 = hashlib.sha256()
        with open(filename, 'rb') as infile:
            for block in iter(lambda: infile.read(4096), b''):
                sha256.update(block)
        hash_results[filename] = sha256.hexdigest()
    return hash_results

def write_hashes(hash_results, video_file, dataset_root):
    """
    :param hash_results: the dict of checksums
    :param video_file: the path of the video file
    :param dataset_root: root directory of the dataset; must exists
    """
    video_name, _ = os.path.splitext(os.path.basename(video_file))
    batches_dir = os.path.join(dataset_root, video_name)
    line_tmpl = '{} *{}'
    tofile = os.path.join(batches_dir, video_name + '.sha256')
    with open(tofile, 'w') as outfile:
        for row in hash_results.items():
            outfile.write(line_tmpl.format(*row))

def main():
    args = make_parser().parse_args()
    failed_filenames = OrderedDict()

    # write frames
    for video_file in args.video_files:
        tqdm.write('>>> Dumping {}'.format(video_file))
        save_frames(video_file, args.dataset_root)
        failed_filenames[video_file] = verify_frames(video_file,
                                                     args.dataset_root)
    # check write failures
    all_failed_filenames = reduce(operator.add, failed_filenames.values())
    if len(all_failed_filenames):
        print 'Failed dumps:'
        for video_name in failed_filenames:
            print '  {}:'.format(video_name)
            for batch_name in failed_filenames[video_name]:
                print '    {}'.format(batch_name)
        sys.exit(1)
    # checksums
    for video_file in args.video_files:
        tqdm.write('>>> Hashing {}'.format(video_file))
        hash_results = compute_checksum(video_file, args.dataset_root)
        write_hashes(hash_results, video_file, args.dataset_root)

if __name__ == '__main__':
    main()
