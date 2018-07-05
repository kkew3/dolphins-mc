#!/usr/bin/env python

r"""
Build video dataset based on Hierarchical Data Format 5 (HDF5) and `hickle`
module.

Directory structure of dataset:

    dataset_root/
    |- video_name_0/
    |  |- video_name_0.json
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
import re
from collections import OrderedDict
import itertools
import operator
import hashlib
import json

import numpy as np
import hickle

import utils


BLOCKFILE_PAT = re.compile(r'^B\d+\.hkl$')
HASH_ALGORITHM = 'sha1'

def edir(string):
    if not os.path.isdir(string):
        raise argparse.ArgumentTypeError('"{}" not found'.format(string))
    return string

def efile(string):
    if not os.path.isfile(string):
        raise argparse.ArgumentTypeError('"{}" not found'.format(string))
    return string

def pint(string):
    try:
        value = int(string)
        if value <= 0:
            raise ValueError()
        return value
    except BaseException:
        raise argparse.ArgumentTypeError('"{}" is not a positive integer'
                                         .format(string))

def make_parser():
    parser = argparse.ArgumentParser(description='Build HDF5-based video '
                                     'frame dataset.')
    parser.add_argument('-f', metavar='FILE', dest='video_files',
                        help='the video file from which to extract frames',
                        nargs='+', type=efile)
    parser.add_argument('dataset_root', help='root directory of the dataset',
                        type=edir)
    parser.add_argument('-p', '--show-progress', action='store_true',
                        dest='show_progress',
                        help='print dots as an indicator of progress')
    test_group = parser.add_argument_group(title='these arguments are used to '
                                           'build dataset for development '
                                            'environment')
    test_group.add_argument('--block-size', type=pint, metavar='N',
                            dest='block_size', default=1000,
                            help='the number of contiguous frames to be treated'
                                 ' as a memory block (saved in one HDF5 file), '
                                 'default to %(default)s')
    test_group.add_argument('--head', type=pint, metavar='N', dest='max_blocks',
                            help='build dataset only '
                                 'using the first N blocks of the given video '
                                 'file')
    return parser

def save_frames(video_file, dataset_root, block_size, max_blocks, dot_progress):
    """
    :param video_file: the path of the video file
    :param dataset_root: root directory of the dataset; must exists
    :param block_size: the number of frames to be saved as a HDF5 block
    :param max_blocks: None to save blocks as many as possible, otherwise save
           up to this number of blocks
    :param dot_progress: True to print dots as an indicator of progress
    """
    video_name, _ = os.path.splitext(os.path.basename(video_file))
    batches_dir = os.path.join(dataset_root, video_name)
    n_frames = 0
    with utils.capcontext(video_file) as cap:
        vit = utils.FrameIterator(cap, max_len=block_size)
        if not os.path.isdir(batches_dir):
            os.mkdir(batches_dir)
        dest_tmpl = 'B{}.hkl'
        cit = range(max_blocks) if max_blocks else itertools.count()
        for block_id in cit:
            frames = list(vit)
            if not len(frames):
                break
            n_frames += len(frames)

            # frames of dimension:
            # [N x H x W x 3], where N <= BATCH_SIZE
            frames = np.stack(frames)
            # frames of dimension:
            # [N x 3 x H x W], where N <= BATCH_SIZE
            frames = np.transpose(frames, (0, 3, 1, 2))
            tofile = os.path.join(batches_dir, dest_tmpl.format(block_id))
            with open(tofile, 'w') as outfile:
                hickle.dump(frames, outfile, compression='gzip')
            vit.reset_counter()
            if dot_progress:
                sys.stdout.write('.')
                sys.stdout.flush()
        if dot_progress:
            print
    metafilename = os.path.join(batches_dir, video_name + '.json')
    with open(metafilename, 'w') as outfile:
        json.dump(dict(total_frames=n_frames,
                       dim_description=['B', 'C', 'W', 'H'],
                       block_size=block_size), outfile)

def verify_frames(video_file, dataset_root):
    """
    See if dumped hickle files can be opened successfully.

    :param video_file: the path of the video file
    :param dataset_root: root directory of the dataset; must exists
    :return: a list of hickle filenames that failed to be opened successfully
    """
    video_name, _ = os.path.splitext(os.path.basename(video_file))
    batches_dir = os.path.join(dataset_root, video_name)
    failed_filenames = list()
    data_batches = map(lambda name: os.path.join(batches_dir, name),
                       filter(BLOCKFILE_PAT.match, os.listdir(batches_dir)))
    for filename in data_batches:
        try:
            with open(filename) as infile:
                with utils.suppress_stdout():
                    _ = hickle.load(infile)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except:
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
    data_batches = map(lambda name: os.path.join(batches_dir, name),
                       filter(BLOCKFILE_PAT.match, os.listdir(batches_dir)))
    for filename in data_batches:
        hashbuf = hashlib.new(HASH_ALGORITHM)
        with open(filename, 'rb') as infile:
            for block in iter(lambda: infile.read(4096), b''):
                hashbuf.update(block)
        hash_results[os.path.basename(filename)] = hashbuf.hexdigest()
    return hash_results

def write_hashes(hash_results, video_file, dataset_root):
    """
    :param hash_results: the dict of checksums
    :param video_file: the path of the video file
    :param dataset_root: root directory of the dataset; must exists
    """
    video_name, _ = os.path.splitext(os.path.basename(video_file))
    batches_dir = os.path.join(dataset_root, video_name)
    line_tmpl = '{1} *{0}\n'
    tofile = os.path.join(batches_dir, '.'.join([video_name, HASH_ALGORITHM]))
    with open(tofile, 'w') as outfile:
        for row in hash_results.items():
            outfile.write(line_tmpl.format(*row))

def main():
    args = make_parser().parse_args()
    failed_filenames = OrderedDict()

    # write frames
    for video_file in args.video_files:
        print '>>> Dumping {}'.format(video_file)
        save_frames(video_file, args.dataset_root, args.block_size,
                    args.max_blocks, args.show_progress)
        print '>>> Verifying write {}'.format(video_file)
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
        print '>>> Hashing {}'.format(video_file)
        hash_results = compute_checksum(video_file, args.dataset_root)
        write_hashes(hash_results, video_file, args.dataset_root)

if __name__ == '__main__':
    main()
