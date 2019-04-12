"""
Defines Moving MNIST dataset (with or without Gaussian noise) and function to
generate moving mnist.
Adapted from: https://gist.github.com/tencia/afb129122a64bde3bd0c.

Also defines Moving MNIST dataset with caustic pattern.
"""

import os
import math
import collections
import itertools
import bisect
import tempfile
import multiprocessing
import contextlib
import shutil
import typing
from configparser import ConfigParser
import logging

import numpy as np
import cv2
import torch.utils.data
import torchvision
from PIL import Image

import utils

__all__ = [
    'MMnistParams',
    'MovingMNIST',
]

LabelImgsDict = typing.Dict[int, np.ndarray]
GroupedMMnistDataset = typing.Dict[typing.Tuple[int, ...], np.ndarray]

MNIST_IMG_SIZE: int = 28
USE_MNIST_TRAIN: bool = True

MP_BATCH_SIZE = 100


class MMnistParams:
    """
    >>> par = MMnistParams((64, 64), 20, 100, 3, 0.015, [])
    >>> str(par)
    'r64x64_T20_m100_n3_s1.50e-02_b'
    >>> par = MMnistParams((64, 64), 20, 100, 3, 0.015, [1, 0])
    >>> str(par)
    'r64x64_T20_m100_n3_s1.50e-02_b0,1'
    >>> str(MMnistParams.parse('r64x64_T20_m100_n3_s1.50e-02_b'))
    'r64x64_T20_m100_n3_s1.50e-02_b'
    >>> str(MMnistParams.parse('r64x64_T20_m100_n3_s1.50e-02_b1'))
    'r64x64_T20_m100_n3_s1.50e-02_b1'
    """

    def __init__(self, shape: typing.Tuple[int, int], seq_len: int,
                 seqs_per_class: int, nums_per_image: int,
                 noise_std: float = 0.0,
                 backgrounds: typing.Optional[typing.Sequence[int]] = None):
        self.shape: typing.Tuple[int, int] = shape
        self.seq_len: int = seq_len
        self.seqs_per_class: int = seqs_per_class
        self.nums_per_image: int = nums_per_image
        self.noise_std: float = noise_std
        self.backgrounds: typing.List[int] = sorted(backgrounds or [])

    @classmethod
    def parse(cls, string: str):
        """
        Parse instance from string.
        """
        key2attr_ty = {
            'r': ('shape', lambda x: tuple(
                map(int, filter(None, x.split('x'))))),
            'T': ('seq_len', int),
            'm': ('seqs_per_class', int),
            'n': ('nums_per_image', int),
            's': ('noise_std', float),
            'b': ('backgrounds', lambda x: sorted(
                map(int, filter(None, x.split(','))))),
        }
        kwargs = {}
        for tok in string.split('_'):
            k, rest = tok[0], tok[1:]
            a, ty = key2attr_ty[k]
            kwargs[a] = ty(rest)
        return cls(**kwargs)

    def __str__(self):
        tmpl = '_'.join((
            'r{shape[0]}x{shape[1]}',
            'T{seq_len}',
            'm{seqs_per_class}',
            'n{nums_per_image}',
            's{noise_std:.2e}',
            'b{backgrounds}',
        ))
        kwargs = self.__dict__.copy()
        kwargs['backgrounds'] = ','.join(map(str, kwargs['backgrounds']))
        return tmpl.format(**kwargs)

    def __repr__(self):
        return ''.join((
            self.__class__.__name__, '(',
            'shape={}, '.format(self.shape),
            'seq_len={}, '.format(self.seq_len),
            'seqs_per_class={}, '.format(self.seqs_per_class),
            'nums_per_image={}, '.format(self.nums_per_image),
            'noise_std={:.2e}, '.format(self.noise_std),
            'backgrounds={}'.format(self.backgrounds),
            ')',
        ))

    def __iter__(self):
        return iter([
            self.shape,
            self.seq_len,
            self.seqs_per_class,
            self.nums_per_image,
            self.noise_std,
            self.backgrounds,
        ])

    def __len__(self):
        return 6

    def __bool__(self):
        return True

    def __eq__(self, other):
        try:
            return tuple(self) == tuple(other)
        except TypeError:
            return NotImplemented

    def __hash__(self):
        return hash(tuple(self))


def iter_(dataset):
    # pylint: disable=consider-using-enumerate
    for i in range(len(dataset)):
        yield dataset[i]


def _l(*args) -> logging.Logger:
    return logging.getLogger(utils.loggername(__name__, *args))


def load_dataset() -> LabelImgsDict:
    """
    Load MNIST dataset grouped by label. Requires existence of
    MNIST dataset under PYTORCH_DATA_HOME.
    """
    root = os.path.join(os.environ['PYTORCH_DATA_HOME'])
    mnist = torchvision.datasets.MNIST(root=root, train=USE_MNIST_TRAIN,
                                       download=False)
    mnist_dict = collections.defaultdict(list)
    for img, label in iter_(mnist):
        mnist_dict[label].append(np.asarray(img))
    for label in mnist_dict:
        mnist_dict[label] = np.stack(mnist_dict[label])
    return mnist_dict


def arr_from_img(im: Image.Image, shift: float = 0) -> np.ndarray:
    r"""
    Convert from image to float array within range
    :math:`[-\text{shift}, 1-\text{shift}]`.

    :param im: uint8 image
    :param shift:
    :return: image array of shape (C, H, W)
    """
    arr = np.asarray(im).astype(np.float32) / 255
    if len(arr.shape) == 2:
        arr = arr[np.newaxis]
    arr -= shift
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


def load_background_images(bgindices) -> typing.Dict[str, np.ndarray]:
    """
    Load background images as per "$PYTORCH_DATA_HOME/MovingMNIST/background/
    backgrounds.csv". The CSV file should have four columns: INDEX, NPZ, NAME
    and DESCRIPTION. The data type of INDEX is expected to be ``int``.

    :param bgindices: the values from the INDEX field in CSV
    :return: the background images of the original sizes; the traversal order
             of the returned dict is the same as that of ``bgindices``
    :raise KeyError: if one of ``bgindices`` is not found in the first column
           of the CSV file
    """
    bgbasedir = os.path.join(
        os.path.normpath(os.environ['PYTORCH_DATA_HOME']),
        MovingMNIST.__name__,
        'background',
    )
    _bgindices = set(bgindices)
    loaded_imgs = collections.OrderedDict([(k, None) for k in bgindices])
    with open(os.path.join(bgbasedir, 'backgrounds.csv')) as infile:
        for i, line in enumerate(infile):
            if None not in loaded_imgs.values():
                break
            if i > 0:  # skip the header row
                idx, npz, name, _ = line.rstrip('\n').split(',')
                idx = int(idx)
                if idx in _bgindices:
                    with np.load(os.path.join(bgbasedir, npz + '.npz')) as d:
                        loaded_imgs[idx] = d[name]
    for i, img in loaded_imgs.items():
        if img is None:
            raise KeyError('INDEX "{}" not found in backgrounds.csv'
                           .format(i))
    return loaded_imgs


def _generate_one_video(params: MMnistParams,
                        bgimgs: typing.Dict[str, np.ndarray],
                        mnist_dict: LabelImgsDict,
                        cb: typing.Tuple[int, ...]) \
        -> typing.List[np.ndarray]:
    """
    Generate one video as per dataset parameters ``params``.
    :param params: the dataset parameters
    :param bgimgs: background images to select from
    :param mnist_dict: dictionary of MNIST label to images of that label
    :param cb: current combination of digits
    :return: a video
    """
    # to maintain randomness even in multiprocessing setting
    np.random.seed()

    mnist_images = get_random_images(mnist_dict, cb)

    h, w = params.shape
    lims = w - MNIST_IMG_SIZE, h - MNIST_IMG_SIZE

    # randomly generate direction/speed/position,
    # calculate velocity vector
    direcs = np.pi * (np.random.rand(params.nums_per_image) * 2 - 1)
    speeds = np.random.randint(5, size=params.nums_per_image) + 2
    velocs = [(v * math.cos(d), v * math.sin(d))
              for d, v in zip(direcs, speeds)]
    posits = [(np.random.rand() * lims[0], np.random.rand() * lims[1])
              for _ in range(params.nums_per_image)]

    video = []
    for _fid in range(params.seq_len):
        if params.backgrounds:
            bgindex = np.random.choice(params.backgrounds)
            bg = bgimgs[bgindex]
            assert len(bg.shape) == 2, str(bg.shape)
            bg = cv2.resize(bg, (w, h))[np.newaxis]
            canvas = bg.astype(np.float32) / 255
        else:
            canvas = np.zeros((1, h, w), dtype=np.float32)

        for i in range(params.nums_per_image):
            _c = Image.new('L', (w, h))
            _c.paste(Image.fromarray(mnist_images[i]),
                     tuple(map(int, map(round, posits[i]))))
            _c = arr_from_img(_c)  # shape: (1, h, w)
            # Now the order of painting the digits matters, provided
            # that the digits have different colors
            _m = (_c > 0.0)
            canvas[_m] = 0.0
            canvas += _c
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
            .clip(0, 255) \
            .astype(np.uint8) \
            .transpose(1, 2, 0)
        assert image.shape[-1] == 1, str(image.shape)
        image = image[..., 0]  # remove the color channel
        video.append(image)
    return video


class _GenerateOneVideoWrapper:
    def __init__(self, *args):
        self.args = args

    def __call__(self, _):
        return _generate_one_video(*self.args)


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


def generate_moving_mnist(params: MMnistParams) -> tempfile.TemporaryFile:
    """
    Returns the Moving MNIST dataset to dump. The numpy array under each key
    is of shape (N, T, H, W).
    """
    logger = _l(generate_moving_mnist.__name__)
    mnist_dict = load_dataset()
    combs = itertools.combinations(mnist_dict, params.nums_per_image)
    combs = list(map(tuple, map(sorted, combs)))

    # load background images if any
    bgimgs = load_background_images(params.backgrounds)

    # save the generated videos to a temporary file to save memory
    cbuf = tempfile.TemporaryFile()

    logger.info('Generating MovingMNIST')
    with contextlib.closing(utils.IncrementalNpzWriter(cbuf, mode='w')) as out:
        with multiprocessing.Pool(4) as pool:
            for cid, cb in enumerate(combs):
                mnist_cb = pool.map(
                    _GenerateOneVideoWrapper(params, bgimgs, mnist_dict, cb),
                    range(params.seqs_per_class), MP_BATCH_SIZE)
                mnist_cb = np.array(mnist_cb)
                logger.info('Generated %d videos for %s; %d/%d completed',
                            mnist_cb.shape[0], str(cb), cid + 1, len(combs))
                # pylint: disable=no-member
                out.write(comb2str(cb), mnist_cb)
                logger.debug('Dumped %d videos for %s to temp file',
                             mnist_cb.shape[0], str(cb))
    logger.info('Completed generating MovingMNIST')
    cbuf.seek(0)
    return cbuf


def dump_dataset(dataset_tmp: tempfile.TemporaryFile, root: str,
                 params: MMnistParams) -> None:
    """
    Dump dataset in binary format.

    :param dataset_tmp: the generated dataset in tmp file
    :param root: an existing directory to store ``dataset``
    :param params: dataset parameters
    :raise FileNotFoundError: if ``root`` not exists
    """
    with contextlib.closing(dataset_tmp):
        logger = _l(dump_dataset.__name__)
        root = os.path.normpath(root)
        subroot = os.path.join(root, str(params) + '.npz')
        try:
            _ = np.load(subroot)
        # pylint: disable=bare-except
        except FileNotFoundError:
            with open(subroot, 'wb') as outfile:
                shutil.copyfileobj(dataset_tmp, outfile)
        else:
            logger.warning('Sub-dataset {} already exists; aborted'
                           .format(subroot))


# pylint: disable=too-many-instance-attributes
class MovingMNIST(torch.utils.data.Dataset):
    """
    This is a labeled dataset for Moving MNIST.
    """

    # pylint: disable=too-many-locals
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
            params = list(args)
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
                except Exception as err:
                    if generate:
                        try:
                            to_dump = generate_moving_mnist(x)
                        except KeyError as err:
                            logger.error('KeyError: {}'.format(err))
                            raise
                        dump_dataset(to_dump, root, x)
                        data = np.load(os.path.join(root, str(x) + '.npz'))
                        subs.append(data)
                        keys.append(tuple(sorted(data, key=str2comb)))
                    else:
                        logger.error('Failed to load {} -- {}'.format(x, err))
                        raise

            self.params: typing.Tuple[MMnistParams, ...] = tuple(params)
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
            k = 'mnist_test_seq'
            self.data = np.load(os.path.join(
                root, 'testset', '{}.npz'.format(k)))[k]

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

    @property
    def default_normalize(self) -> torchvision.transforms.Normalize:
        if len(self.params) > 1:
            raise AttributeError('no default_normalize when there are more '
                                 'than one ({}) params'
                                 .format(len(self.params)))
        cfg = ConfigParser()
        nmlstats = os.path.join(self.root, 'nml-stats.ini')
        if not cfg.read(nmlstats):
            raise FileNotFoundError('"{}" not found'.format(nmlstats))
        sec_name = str(self.params[0])
        mean = cfg.getfloat(sec_name, 'mean')
        std = cfg.getfloat(sec_name, 'std')
        normalize = torchvision.transforms.Normalize((mean,), (std,))
        return normalize

    def __l(self, method_name: str):
        return _l(self, method_name)


class CausticMovingMNIST(torch.utils.data.Dataset):
    """
    Applies caustic pattern over a ``MovingMNIST`` dataset. It assumes that
    the caustic pattern is independent of Moving MNIST; therefore the
    variance of the transformed MovingMNIST is the sum of the variances of
    plain MovingMNIST and that of the caustic pattern.
    """

    CAUSTIC_SOURCE_BASEDIR = os.path.join(
        os.environ['PYTORCH_DATA_HOME'],
        'MovingMNIST',
        'causticrender',
    )
    CAUSTIC_ASSIGNMENT_BASEDIR = os.path.join(
        CAUSTIC_SOURCE_BASEDIR,
        'assignment',
    )
    DEFAULT_CAUSTIC_SOURCE = 'caustic_int3e-2.npy'

    def __init__(self, *args, caustic_source: str = None,
                 caustic_scale: float = 1.0, **kwargs):
        """
        All positional and keyword arguments will be passed directly
        to ``MovingMNIST`` except for the below listed parameters.
        If ``transform`` and/or ``target_transform`` are in ``kwargs``, they
        are removed from the argument list to ``MovingMNIST`` and passed on
        to that of ``self``.

        :param caustic_source: the caustic pattern source, should be the
               basename of an ``npy`` file under
               ``${PYTORCH_DATA_HOME}/MovingMNIST/causticrender/``;
               default to ``self.DEFAULT_CAUSTIC_SOURCE``
        :param caustic_scale: the scale factor of the caustic pattern
        """
        # load caustic data, its mean and its std
        if not caustic_source:
            caustic_source = self.DEFAULT_CAUSTIC_SOURCE
        caustic_data, c_mean, c_std = self.load_caustic_data(caustic_source)
        caustic_data = (caustic_data.astype(np.float64) * caustic_scale) \
            .astype(np.uint8)
        c_std *= caustic_scale ** 2
        c_mean = (c_mean,)
        c_std = (c_std,)

        # load MNIST data, its mean and its std
        transforms = {}
        try:
            transforms['transform'] = kwargs['transform']
            kwargs['transform'] = np.asarray  # convert PIL image to array
        except KeyError:
            pass
        try:
            transforms['target_transform'] = kwargs['target_transform']
            del kwargs['target_transform']
        except KeyError:
            pass
        dsbackend = MovingMNIST(*args, **kwargs)
        if len(dsbackend.params) != 1:
            raise RuntimeError('Backend MovingMNIST should have only one '
                               'params, but got {}'
                               .format(len(dsbackend.params)))
        m_normalize = dsbackend.default_normalize
        m_mean, m_std = m_normalize.mean, m_normalize.std

        # check compatibility between MovingMNIST and caustic data
        for i, par in enumerate(dsbackend.params):
            if par.shape != caustic_data.shape[1:]:
                raise ValueError('caustic shape ({}) doesn\'t match '
                                 'MovingMNIST sub-dataset {} shape ({})'
                                 .format(caustic_data.shape[1:],
                                         i, par.shape))
            if caustic_data.shape[0] < par.seq_len:
                raise ValueError('caustic period ({}) shorter than '
                                 'MovingMNIST seq_len ({})'
                                 .format(caustic_data.shape[0],
                                         par.seq_len))

        # load caustic data assignment
        c_a = self.load_caustic_assignment(len(dsbackend),
                                           caustic_data.shape[0],
                                           dsbackend.params[0].seq_len)

        self.dsbackend = dsbackend
        self.caustic_data = caustic_data
        self.caustic_assign = c_a
        self.transform = transforms.get('transform', None)
        self.target_transform = transforms.get('target_transform', None)

        # compute final mean and std
        mean = tuple((x + y) for x, y in zip(m_mean, c_mean))
        std = tuple((x ** 2 + y ** 2) for x, y in zip(m_std, c_std))
        self.__normalize = torchvision.transforms.Normalize((mean,), (std,))

    @classmethod
    def load_caustic_data(cls, name: str):
        """
        :param name: the file basename
        :return: the caustic data, mean, and std
        """
        fromfile = os.path.join(cls.CAUSTIC_SOURCE_BASEDIR, name)
        caustic_data = np.load(fromfile)

        cfg = ConfigParser()
        nmlstats = os.path.join(cls.CAUSTIC_SOURCE_BASEDIR, 'nml-stats.ini')
        if not cfg.read(nmlstats):
            raise FileNotFoundError('"{}" not found'.format(nmlstats))
        sec_name = name
        mean = cfg.getfloat(sec_name, 'mean')
        std = cfg.getfloat(sec_name, 'std')
        return caustic_data, mean, std

    @classmethod
    def load_caustic_assignment(cls, n_videos: int, n_caustic: int,
                                seq_len: int) -> np.ndarray:
        """
        :param n_videos: number of videos to apply the patterns
        :param n_caustic: number of caustic patterns in the file
        :param seq_len: sequence length of the video to apply the patterns
        :return: the assignment list
        """
        name = 'assign_T{}_L{}.npy'.format(seq_len, n_caustic)
        a = np.load(os.path.join(cls.CAUSTIC_ASSIGNMENT_BASEDIR, name))
        assert len(a.shape) == 1
        if n_videos > a.shape[0]:
            raise RuntimeError('{} assignments are not enough for {} '
                               'videos'.format(a.shape[0], n_videos))
        a = a[:n_videos]
        return a

    @property
    def default_normalize(self) -> torchvision.transforms.Normalize:
        return self.__normalize

    def __len__(self):
        return len(self.dsbackend)

    def __getitem__(self, index):
        if self.dsbackend.train:
            img, label = self.dsbackend[index]
            img = self.__apply_caustic(img, self.caustic_assign[index])
            if self.transform:
                img = self.transform(img)
            if self.target_transform:
                label = self.target_transform(label)
            return img, label
        img = self.dsbackend[index]
        img = self.__apply_caustic(img, self.caustic_assign[index])
        if self.transform:
            img = self.transform(img)
        return img

    def __apply_caustic(self, img: np.ndarray, cindex: int):
        img = img.astype(np.int64)
        cindices = np.arange(cindex, cindex + img.shape[0])
        cdata = self.caustic_data[cindices]
        cdata = cdata.astype(np.int64)
        img = np.clip(img + cdata, 0, 255).astype(np.uint8)
        return img





def compute_mean_std(filename):
    N_SAMPLE_PER_FILE = 100000
    MIN_N_SAMPLE_PER_KEY = 100
    MAX_MEMORY_BYTES = 4 * 1024 ** 3
    MINSIZE_TO_USE_MMAP = 1024 ** 3

    def sample_img(data, n: int, n_total_keys: int) -> np.ndarray:
        N, T, H, W = data.shape
        n_samples = min(1 + int(MAX_MEMORY_BYTES /
                                (n_total_keys * H * W * 8)), n)
        if n_samples < n:
            logging.warning('reduced n_samples from %d to %d '
                            'due to memory limit',
                            n, n_samples)
        ind = np.random.permutation(N * T)[:n_samples]
        data = np.copy(data.reshape((N * T, H * W))[ind])
        return data.astype(np.float64) / 255

    def _calc():
        alld = []
        with np.load(filename) as zdata:
            n_keys = len(list(zdata))
        n_samples_per_key = int(N_SAMPLE_PER_FILE / n_keys) + 1
        if n_samples_per_key < MIN_N_SAMPLE_PER_KEY:
            logging.warning('n_samples_per_key=%d < MIN_N_SAMPLE_PER_KEY',
                            n_samples_per_key)

        # data shape for each npy: (N, T, H, W)

        with np.load(filename) as zdata:
            npykeys = list(zdata)

        n_bytes = os.path.getsize(filename)
        if n_bytes > self.MINSIZE_TO_USE_MMAP:
            logging.info('switched to mmap mode for file "%s" (%d B)',
                         filename, n_bytes)
            with utils.NpzMMap(filename) as zdata:
                for key in npykeys:
                    with zdata.mmap(key) as d:
                        alld.append(sample_img(d, n_samples_per_key, len(npykeys)))
                        logging.debug('collected %d samples from key=%s',
                                      alld[-1].shape[0], key)
        else:
            logging.info('switched to normal mode for file "%s" (%d B)',
                         filename, n_bytes)
            with np.load(filename) as zdata:
                for key in npykeys:
                    d = zdata[key]
                    alld.append(sample_img(d, n_samples_per_key, len(npykeys)))
                    logging.debug('collected %d samples from key=%s',
                                  alld[-1].shape[0], key)
        alld = np.concatenate(alld, axis=0)
        logging.info('collected %d samples from "%s"', alld.shape[0], filename)
        alld = alld.reshape(-1)
        mean, std = np.mean(alld), np.std(alld)
        logging.info('%s, mean=%s, std=%s', filename, mean, std)
        return mean, std

    def

cfg_bak = configparser.ConfigParser()
if not cfg_bak.read(INI_TOFILE):
    cfg_bak = None

try:
    cfg = configparser.ConfigParser()
    cfg.read(INI_TOFILE)
    logging.info('N_SAMPLE_PER_FILE={}, MIN_N_SAMPLE_PER_KEY={}'
                 .format(N_SAMPLE_PER_FILE, MIN_N_SAMPLE_PER_KEY))
    for filename in args.filenames:
        filename = os.path.normpath(filename)
        secname = os.path.splitext(os.path.basename(filename))[0]
        try:
            cfg.add_section(secname)
        except configparser.DuplicateSectionError:
            logging.info('Skipped "%s" due to duplicate section in "%s"',
                         filename, INI_TOFILE)
        else:
            m, s = calc(filename)
            cfg[secname]['mean'] = str(m)
            cfg[secname]['std'] = str(s)
            logging.debug('cfg-obj written for %s', filename)
    try:
        with open(INI_TOFILE, 'w') as outfile:
            cfg.write(outfile, space_around_delimiters=False)
    except IOError:
        logging.error('Failed to write back to "%s"', INI_TOFILE)
except KeyboardInterrupt:
    if cfg_bak:
        logging.info('Reversing back the modification to "%s"', INI_TOFILE)
        with open(INI_TOFILE, 'w') as outfile:
            cfg_bak.write(outfile, space_around_delimiters=False)
