"""
Store video frames as mini-batches of numpy array, 10 frames a batch, in a
zip file, following ``npz`` format.

Dataset directory hierarchy::

    root/
    |- data.npz
    |- meta.json
    |- nml-stats.ini

where ``nml-stats.ini`` contains two keys ``mean`` and ``std`` in each section.
The section name is ``rgb`` for colorful images, and ``bw`` for gray images.
"""

__all__ = [
    'create_vdset',
    'VideoDataset',
]


import os
import io
import zipfile
import json
import logging
import itertools
import configparser
from ast import literal_eval
import typing
import warnings
import zlib

import numpy as np
import torch.utils.data

import loglib
import utils

DATASETS_ROOT = os.path.join(os.path.normpath(os.environ['PYTORCH_DATA_HOME']),
                             'zdatasets')


def _l(self_or_functionname, methodname=None) -> logging.Logger:
    return logging.getLogger(loglib.loggername(
        __name__,
        self_or_functionname,
        methodname,
    ))


def write_batch_and_clear(bid: int, batch_size: int, frame_list, zfile) -> int:
    """
    :param bid: batch id
    :param batch_size: batch size
    :param frame_list: a list of frames of current batch
    :param zfile: the zip file to write
    :return: next batch id
    """
    batch = np.stack(frame_list, axis=0)
    s = bid * batch_size
    t = s + len(frame_list)
    name = '{}-{}.npy'.format(s, t)
    with io.BytesIO() as cbuf:
        np.save(cbuf, batch)
        cbuf.seek(0)
        zfile.writestr(name, cbuf.read())
    frame_list.clear()
    return bid + 1


def frame_location(batch_size: int, n_frame: int, fid: int):
    """
    :param batch_size: batch size
    :param fid: the frame id to read
    :return: which ``npy`` file the frame is expected to reside, and the index
             within that ``npy`` file
    """
    s = fid // batch_size * batch_size
    t = min(s + batch_size, n_frame + 1)
    return '-'.join(map(str, (s, t))), fid % batch_size


class VideoDataset(torch.utils.data.Dataset):
    """
    The video dataset based on npz format. Unlike ``vhdata`` and ``vmdata``,
    this dataset is multi-processing free. Moreover, this dataset implements
    ``closing`` protocol, which means it's save to use the context manager
    ``contextlib.closing`` to properly close the dataset after use::

        .. code-block::

            from contextlib import closing
            from vzdata import VideoDataset
            root = ...
            with closing(VideoDataset(root, transform)) as dataset:
                ...

    Note, however, that ``VideoDataset`` itself does **not** implement the
    context manager protocol, as it's not a common use in the training and
    evaluation pipeline.
    """

    FILE_ENTRIES = {
        'data': 'data.npz',
        'meta': 'meta.json',
        'nml-stat': 'nml-stat.ini',
    }

    def __init__(self, root: str, transform: typing.Callable = None, **kwargs):
        self.root = root
        self.transform = transform
        self._fe = self.file_entries(root)
        # FIXME
        # Reference: https://discuss.pytorch.org/t/dataloader-stucks/14087/5
        # However, it seems that this method causes file handler leaking.
        # There's not much I can do about it for now, though.
        self.data: np.lib.npyio.NpzFile = None
        if kwargs:
            warnings.warn('vzdata no longer accepts keyword arguments '
                          'as vmdata did',
                          DeprecationWarning)
        with open(self._fe['meta']) as mf:
            self.meta: typing.Dict[str, typing.Any] = json.load(mf)

    @classmethod
    def file_entries(cls, root: str):
        return {k: os.path.join(root, v) for k, v in cls.FILE_ENTRIES.items()}

    def __len__(self):
        return self.meta['n_frame']

    def __getitem__(self, index):
        if self.data is None:
            self.data = np.load(self._fe['data'])
        batchname, offset = frame_location(self.meta['batch_size'],
                                           self.meta['n_frame'],
                                           index)
        try:
            batch = self.data[batchname]
        except KeyError:
            raise IndexError()
        except TypeError:
            # see ``self.close()``
            raise ValueError('I/O operation on closed file.')
        except (zlib.error, zipfile.BadZipFile):
            print('###DEBUG### batchname={}'.format(batchname))
            raise
        else:
            img = batch[offset]
            if self.transform:
                img = self.transform(img)
        return img

    def close(self) -> None:
        """
        Close the dataset file handler.
        """
        if self.data is not None:
            self.data.close()
            self.data = None


def create_vdset(filename: str, root: str, batch_size: int = 10,
                 n_frames: int = None) -> None:
    """
    Make dataset named ``tofile`` from video ``filename``.

    :param filename: the video filename
    :param root: the dataset root directory
    :param batch_size: batch size, should be small enough
    :param n_frames: number of frames to use
    """
    logger = _l('create_vdset')

    try:
        os.mkdir(root)
    except FileNotFoundError:
        logger.error('parent directory of root "{}" not found'
                     .format(root))
        raise
    except FileExistsError:
        if os.listdir(root):
            logger.error('root "{}" is not empty')
            raise

    bid: int = 0
    fid: int = 0
    batch: typing.List[np.ndarray] = []
    file_entries = VideoDataset.file_entries(root)
    with zipfile.ZipFile(file_entries['data'], mode='x',
                         compression=zipfile.ZIP_DEFLATED) as zoutfile:
        with utils.capcontext(filename) as cap:
            for frame in utils.frameiter(cap, n_frames):
                ty = frame.dtype
                if len(batch) == batch_size:
                    bid = write_batch_and_clear(bid, batch_size, batch,
                                                zoutfile)
                    if fid and fid % 500 == 0:
                        logger.info('Added %d frames', fid)
                batch.append(frame)
                fid += 1
            if batch:
                bid = write_batch_and_clear(bid, batch_size, batch, zoutfile)
        logger.info('Completed dumping video "{}" ({} frames in {} batches)'
                    ' to "{}"'
                    .format(filename, fid, bid, file_entries['data']))
    meta = {
        'batch_size': batch_size,
        'n_batch': bid,
        'n_frame': fid,
        'color': 'rgb',
    }
    try:
        meta['dtype'] = str(ty)
    except NameError:
        pass
    with open(file_entries['meta'], 'w') as outfile:
        json.dump(meta, outfile)
    logger.info('Completed dumping meta info to "{}"'
                .format(file_entries['meta']))

def dataset_root(cam_channel: int,
                 video_index: typing.Union[int, typing.Tuple[int, int, int]],
                 day: typing.Union[int, str] = None) -> str:
    """
    A convenient function to get the root directory under DATASETS_ROOT.

    :param cam_channel: the camera channel ID
    :param video_index: the video index, either a 3-tuple of integers, or
           a single integer which will be expanded to a 3-tuple as
           ``(video_index, 0, 0)``. The first element of the tuple represents
           the starting hour of the video
    :param day: the day, can be either a string represeting the full date,
           e.g. '2Feb13', or an integer that will be expanded to '{day}Feb13'
    :return: the root directory
    """
    if isinstance(video_index, int):
        video_index = video_index, 0, 0
    if isinstance(day, int):
        day = '{}Feb13'.format(day)
    # tmpl_nd.format(cam_channel, video_index)
    tmpl_nd = '[CH{0:>02}]{1[0]:>02}_{1[1]:>02}_{1[2]:>02}'
    # tmpl_d.format(cam_channel, video_index, video_index[0]+3, day)
    tmpl_d = os.path.join('{3}.{1[0]}to{2}', tmpl_nd)

    tokens = [cam_channel, video_index]
    if day:
        tokens.extend([video_index[0] + 3, day])
        tmpl = tmpl_d
    else:
        tmpl = tmpl_nd
    root = os.path.join(DATASETS_ROOT, tmpl.format(*tokens))

    if not os.path.isdir(root):
        raise ValueError('root "{}" not found'.format(root))
    return root


def get_normalization_stats(root: str, bw=False, as_rgb=False) \
        -> typing.Tuple[typing.Tuple[float, ...], typing.Tuple[float, ...]]:
    """
    Returns the normalization statistics (mean, std) in preprocessing step
    when loading the dataset. The normalization data presents in a ``npz``
    file ``$NORMALIZATION_INFO_FILE`` under the dataset root directory.

    Usage example::

        .. code-block::

            import torchvision.transforms as trans
            normalize = trans.Normalize(*get_normalization_stats(root))

    :param root: the root directory of the video dataset
    :param bw: load the B&W mean/std instead
    :param as_rgb: repeat mean and std into three times when ``bw`` is True
    :return: the mean and std
    """
    sec = 'bw' if bw else 'rgb'
    filename = VideoDataset.file_entries(root)['nml-stat']
    cfg = configparser.ConfigParser()
    if not cfg.read(filename):
        raise FileNotFoundError('"{}" not found'.format(filename))
    mean = literal_eval(cfg[sec]['mean'])
    try:
        d = len(mean)  # TypeError hazard
    except TypeError:
        mean = (mean,)
        d = 1
    std = literal_eval(cfg[sec]['std'])
    try:
        if len(std) != d:  # TypeError hazard
            raise RuntimeError('dimension mismatch: len(mean)={} len(std)={}'
                               .format(len(mean), len(std)))
    except TypeError:
        std = (std,)
    if (bw and d != 1) or (not bw and d != 3):
        raise RuntimeError('wrong dimension: len(mean)={} len(std)={}'
                           .format(len(mean), len(std)))
    if bw and as_rgb:
        mean *= 3
        std *= 3
    return mean, std


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import argparse

    def make_parser():
        parser = argparse.ArgumentParser(description='Build video dataset')
        parser.add_argument('basedir', type=os.path.normpath,
                            help='the base directory of the root of the '
                                 'datasets to build; must already exist')
        parser.add_argument('video', type=os.path.normpath,
                            help='video filename')
        parser.add_argument('-B', '--batch-size', type=int, dest='batch_size',
                            default=10,
                            help='the batch size; larger BATCH_SIZE enhances '
                                 'compression rate, while at the same time '
                                 'increase reading overhead,')
        parser.add_argument('-n', '--num-frames', metavar='N', type=int,
                            help='build dataset using only the first N frames '
                                 'from each video')
        return parser

    def main():
        args = make_parser().parse_args()
        dirname = os.path.splitext(os.path.basename(args.video))[0]
        root = os.path.join(args.basedir, dirname)
        create_vdset(args.video, root, args.batch_size, args.num_frames)

    main()
