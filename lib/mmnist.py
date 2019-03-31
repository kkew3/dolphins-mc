"""
Defines Moving MNIST dataset and function to generate moving mnist.
Adapted from: https://gist.github.com/tencia/afb129122a64bde3bd0c
"""

import os
import math
import collections
import itertools
import bisect
import typing
import logging

import torch.utils.data
import torchvision
import numpy as np
from PIL import Image

from utils import loggername

__all__ = [
    'MMnistParams',
    'MovingMNIST',
]

LabelImgsDict = typing.Dict[int, np.ndarray]
GroupedMMnistDataset = typing.Dict[typing.Tuple[int, ...], np.ndarray]

MNIST_IMG_SIZE: int = 28
USE_MNIST_TRAIN: bool = True


class MMnistParams(collections.namedtuple('MMnistParams', (
        'shape', 'seq_len', 'seqs_per_class', 'nums_per_image',
        'noise_std'))):
    """
    Parameters:

        shape ((int, int)): (height, width) of generated videos
        seq_len (int): video length
        seqs_per_class (int): number of videos for each combination of digits
        nums_per_image (int): number of digits in each video
        noise_std (float): zero-mean Gaussian noise std over the video

    >>> par = MMnistParams((64, 64), 20, 100, 3, 0.015)
    >>> str(par)
    'r64x64_T20_m100_n3_s1.50e-02'
    >>> str(MMnistParams.parse('r64x64_T20_m100_n3_s1.50e-02'))
    'r64x64_T20_m100_n3_s1.50e-02'
    """

    @classmethod
    def parse(cls, string: str):
        """
        Parse instance from string.
        """
        key2attr_ty = {
            'r': ('shape', lambda x: tuple(map(int, x.split('x')))),
            'T': ('seq_len', int),
            'm': ('seqs_per_class', int),
            'n': ('nums_per_image', int),
            's': ('noise_std', float),
        }
        kwargs = {}
        for tok in string.split('_'):
            k, rest = tok[0], tok[1:]
            a, ty = key2attr_ty[k]
            kwargs[a] = ty(rest)
        return cls(**kwargs)

    def __str__(self):
        attr_tmpl = [
            ('shape', 'r{{0.{0}[0]}}x{{0.{0}[1]}}'),
            ('seq_len', 'T{{0.{0}}}'),
            ('seqs_per_class', 'm{{0.{0}}}'),
            ('nums_per_image', 'n{{0.{0}}}'),
            ('noise_std', 's{{0.{0}:.2e}}'),
        ]
        return '_'.join(y.format(x) for x, y in attr_tmpl).format(self)


def iter_(dataset):
    # pylint: disable=consider-using-enumerate
    for i in range(len(dataset)):
        yield dataset[i]


def _l(*args) -> logging.Logger:
    return logging.getLogger(loggername(__name__, *args))


def load_dataset() -> LabelImgsDict:
    """
    Load MNIST dataset grouped by label. Requires existence of
    MNIST dataset under PYTORCH_DATA_HOME.
    """
    root = os.path.join(os.environ['PYTORCH_DATA_HOME'], 'MNIST')
    mnist = torchvision.datasets.MNIST(root=root, train=USE_MNIST_TRAIN,
                                       download=False)
    mnist_dict = collections.defaultdict(list)
    for img, label in iter_(mnist):
        mnist_dict[label.item()].append(np.array(img))
    for label in mnist_dict:
        mnist_dict[label] = np.stack(mnist_dict[label])
    return mnist_dict


def arr_from_img(im: Image.Image, shift: float = 0) -> np.ndarray:
    r"""
    Convert from image to float array within range
    :math:`[-\text{shift}, 1-\text{shift}]`.

    :param im: uint8 image
    :param shift:
    :return: image array of shape (C, W, H)
    """
    arr = np.asarray(im).astype(np.float32) / 255
    if len(arr.shape) == 2:
        arr = arr[..., np.newaxis]
    arr = arr.transpose(2, 1, 0) - shift
    return arr


def get_random_images(mnist_dict: LabelImgsDict,
                      labels: typing.Sequence[int]) \
        -> typing.List[np.ndarray]:
    """
    Returns a stack of K digit images, of shape (K, 28, 28)

    :param mnist_dict: dictionary of MNIST label to images of that label
    :param labels: labels of digits to appear in one video
    """
    images = []
    for y in labels:
        i = np.random.randint(mnist_dict[y].shape[0])
        images.append(mnist_dict[y][i])
    return images


def generate_moving_mnist(params: MMnistParams) -> GroupedMMnistDataset:
    """
    Returns the Moving MNIST dataset to dump. The numpy array under each key
    is of shape (N, T, H, W).
    """
    logger = _l('generate_moving_mnist')
    mnist_dict = load_dataset()
    combs = itertools.combinations(mnist_dict, params.nums_per_image)
    combs = list(map(tuple, map(sorted, combs)))

    h, w = params.shape
    lims = h - MNIST_IMG_SIZE, w - MNIST_IMG_SIZE

    logger.info('Generating MovingMNIST')
    mmnist: GroupedMMnistDataset = {}
    for cid, cb in enumerate(combs):
        mmnist[cb] = []
        for k in range(params.seqs_per_class):
            # randomly generate direc/speed/position,
            # calculate velocity vector
            direcs = np.pi * (np.random.rand(params.nums_per_image) * 2 - 1)
            speeds = np.random.randint(5, size=params.nums_per_image) + 2
            velocs = [(v * math.cos(d), v * math.sin(d))
                      for d, v in zip(direcs, speeds)]
            posits = [(np.random.rand() * lims[0], np.random.rand() * lims[1])
                      for _ in range(params.nums_per_image)]
            mnist_images = get_random_images(mnist_dict, cb)

            video = []
            for _fid in range(params.seq_len):
                canvas = np.zeros((1, w, h), dtype=np.float32)
                for i in range(params.nums_per_image):
                    _c = Image.new('L', (w, h))
                    _c.paste(Image.fromarray(mnist_images[i]),
                             tuple(map(int, map(round, posits[i]))))
                    canvas += arr_from_img(_c)
                # update positions based on velocity and see which digits
                # go beyond the walls
                _next_posits = [tuple(map(sum, zip(*pv)))
                                for pv in zip(posits, velocs)]
                # bounce off wall if we hit one
                for i, pos in enumerate(_next_posits):
                    for j, coord in enumerate(pos):
                        if coord < -2 or coord > lims[j] + 2:
                            velocs[i] = tuple(itertools.chain(
                                list(velocs[i][:j]),
                                [-1 * velocs[i][j]],
                                velocs[i][j + 1:],
                            ))

                # update positions with bouncing for sure
                posits = [tuple(map(sum, zip(*pv)))
                          for pv in zip(posits, velocs)]

                canvas += np.random.randn(*canvas.shape) * params.noise_std
                image = (canvas * 255) \
                    .astype(np.uint8) \
                    .clip(0, 255) \
                    .transpose(2, 1, 0)
                assert image.shape[-1] == 1, str(image.shape)
                image = image[..., 0]

                video.append(image)
            mmnist[cb].append(video)
            logger.debug('Generated 1 video for {}; {}/{} completed'
                         .format(cb, k + 1, params.seqs_per_class))
        mmnist[cb] = np.array(mmnist[cb])
        logger.info('Generated {} videos for {}; {}/{} completed'
                    .format(mmnist[cb].shape[0], cb, cid + 1, len(combs)))
    logger.info('Completed generating MovingMNIST')
    return mmnist


def comb2str(comb_list: typing.Sequence[int]):
    """
    Convert combination tuple to string.

    >>> comb2str((2,3))
    '2,3'
    """
    return ','.join(map(str, comb_list))


def str2comb(string: str) -> typing.Tuple[int, ...]:
    """
    Parse combination tuple from string.
    """
    return tuple(map(int, string.split(',')))


def dump_dataset(dataset: GroupedMMnistDataset, root: str,
                 params: MMnistParams) -> None:
    """
    Dump dataset in binary format.
    :param dataset: the generated dataset
    :param root: an existing directory to store ``dataset``
    :param params: dataset parameters
    :raise FileNotFoundError: if ``root`` not exists
    """
    logger = _l('dump_dataset')
    root = os.path.normpath(root)
    subroot = os.path.join(root, str(params) + '.npz')
    try:
        _ = np.load(subroot)
    # pylint: disable=bare-except
    except:
        _dataset = {comb2str(k): v for k, v in dataset.items()}
        np.savez_compressed(subroot, **_dataset)
    else:
        logger.warning('Sub-dataset {} already exists; aborted'
                       .format(subroot))


class MovingMNIST(torch.utils.data.Dataset):
    """
    This is a labeled dataset for Moving MNIST.
    """

    def __init__(self, root: str, *args: MMnistParams,
                 transform: typing.Callable = None,
                 target_transform: typing.Callable = None,
                 generate: bool = False, train: bool = True,
                 **kwargs):
        """
        If neither ``args`` nor ``kwargs`` is specified, all sub-datasets
        under ``root`` will be used as one dataset. The order will be the
        lexicographical order of the sub-dataset name (filename without
        extension). If only ``args`` is specified, the sub-datasets implied
        by them are used. If only ``kwargs`` is specified, the sub-dataset
        of which the parameter is populated by ``kwargs`` will be used. If
        both ``args`` and ``kwargs`` are specified, ``kwargs`` will be
        omitted. If ``kwargs`` won't populate an ``MMnistParam`` instance,
        ``TypeError`` will be raised.

        The keys within the one sub-dataset will be ordered by the digits.
        For example, key ``(1, 2)`` will precede ``(1, 3)`` when
        ``nums_per_image`` is 2.

        :param root: an existing dataset root directory
        :param transform: the image transform; if not specified, each image
               will be a numpy array of shape (T, H, W) and of type uint8
        :param target_transform: the label transform; if not specified, each
               label will be a numby array of shape (M,) and of type int64
        :param generate: ``True`` to generate the dataset if not exists
        :param train: if ``False``, load the test set, in which case
               ``args``, ``kwargs``, ``generate``, and ``target_transform``
               will be ignored, and in which case the dataset is no longer
               labelled (so return video only)
        """
        logger = self.__l('__init__')
        self.train = train
        if train:
            params = args
            if not params and kwargs:
                params = [MMnistParams(**kwargs)]

            if not params:
                params = [MMnistParams.parse(os.path.splitext(x)[0])
                          for x in os.listdir(root) if x.endswith('.npz')]

            params.sort(key=str)

            subs = []
            keys = []
            for x in params:
                try:
                    data = np.load(os.path.join(root, str(x) + '.npz'))
                    subs.append(data)
                    keys.append(tuple(sorted(data, key=str2comb)))
                except BaseException as err:
                    if generate:
                        to_dump = generate_moving_mnist(x)
                        dump_dataset(to_dump, root, x)
                        data = np.load(os.path.join(root, str(x) + '.npz'))
                        subs.append(data)
                        keys.append(tuple(sorted(data, key=str2comb)))
                    else:
                        logger.error('Failed to load {} -- {}'.format(x, err))
                        raise

            self.params = tuple(params)
            self.keys = tuple(keys)
            self.subs = tuple(subs)
            self.root = root
            self.transform = transform
            self.target_transform = target_transform

            # build index -- (len, sub_id, key)
            self.lens: typing.List[typing.Tuple[int, int, str]] = []
            for sid, (kys, data) in enumerate(zip(self.keys, self.subs)):
                # kys: sorted keys of current sub-dataset
                # data: sub-dataset of shape (N, T, H, W)
                self.lens.extend((data[k].shape[0], sid, k) for k in kys)
            self.cuml = list(np.cumsum([0] + [x[0] for x in self.lens]))
        else:
            self.data = np.load(os.path.join(root, 'mnist_test_seq.npy'))

    def __len__(self):
        try:
            return self.cuml[-1]
        except AttributeError:
            return self.data.shape[0]

    def __getitem__(self, index):
        try:
            if index < 0:
                index += len(self)
            loc = bisect.bisect(self.cuml, index) - 1
            _, sid, k = self.lens[loc]
            video = self.subs[sid][k][index - self.cuml[loc]]
            if self.transform:
                video = self.transform(video)
            label = np.array(str2comb(k))
            if self.target_transform:
                label = self.target_transform(label)
            return video, label
        except AttributeError:
            video = self.data[index]
            if self.transform:
                video = self.transform(video)
            return video

    def __repr__(self):
        if self.train:
            return ('{}(root={}, train=True, params={})'
                    .format(type(self).__name__,
                            self.root, list(self.params)))
        return ('{}(root={}, train=False)'
                .format(type(self).__name__, self.root))

    def __l(self, method_name: str):
        return _l(self, method_name)
