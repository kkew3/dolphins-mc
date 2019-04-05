"""
Common utilities used in experiments.
"""
import configparser
import inspect
import itertools
import collections
import os
import re
import json
import tempfile
import contextlib
from types import SimpleNamespace
import logging
import typing

import numpy as np
import h5py

import utils
import vmdata

T = typing.TypeVar('T')


def _l(*args):
    return logging.getLogger(utils.loggername(__name__, *args))


class ExperimentLauncher:
    """
    Launch experiment of which the results are to be stored in a single file.
    Returns the results directly if the experiment has been launched before
    with the same arguments; otherwise run the experiment and save the
    results. The experiment parameters are encoded in JSON string.

    Attributes:

        - result_filename: the result filename of the previous launch
    """

    def __init__(self, basedir, prefix='', suffix='', translate_kwargs=str):
        """
        :param basedir: the base directory to load/store results
        :param prefix: the prefix of the stored result and parameter JSON
               file
        :param suffix: the suffix of the stored result file; e.g. the file
               extension name
        :param translate_kwargs: a function that takes in the value of one
               keyword argument of the experiment and returns a JSON
               serializable object; a dictionary of keyword-function pairs is
               also accepted. When either ``translate_kwargs`` or the values
               of ``translate_kwargs`` is not callable, it's regarded as a
               function that always returns that value. Default to ``str``
        """
        self.basedir = os.path.normpath(basedir)
        self.prefix = prefix
        self.suffix = suffix
        self.translate_kwargs = translate_kwargs

        # the result filename of previous launch
        self.result_filename = None

    def load_result(self, filename: str) -> tuple:
        """
        Load experiment result from ``filename``.
        """
        raise NotImplementedError()

    def store_result(self, filename, *args):
        """
        Store anything returned by ``run`` to ``filename``, to be loaded by
        ``load_result`` afterwards.
        """
        raise NotImplementedError()

    def run(self, **kwargs) -> tuple:
        """
        Run the experiment with keyword arguments.
        """
        raise NotImplementedError()

    def encode_kwargs(self, kwdict: dict) -> dict:
        tr = self.translate_kwargs
        if not hasattr(tr, '__getitem__'):
            tr = {k: tr for k in kwdict}

        encdict = {}
        for k, v in kwdict.items():
            try:
                s = tr[k](v)
            except TypeError:
                s = tr[k]
            encdict[k] = s
        return encdict

    def search_history_params(self, enckwargs: dict):
        enckwargs = json.loads(json.dumps(enckwargs))
        found = None
        for filename in iter(os.path.join(self.basedir, x)
                             for x in os.listdir(self.basedir)
                             if x.endswith('.json')):
            try:
                with open(filename) as infile:
                    jo = json.load(infile)
            except (IOError, json.JSONDecodeError):
                pass
            else:
                if jo == enckwargs:
                    found = os.path.splitext(filename)[0]
                    self.result_filename = found
                    break
        if found:
            return self.load_result(found)

    def __call__(self, **kwargs):
        enckwargs = self.encode_kwargs(kwargs)
        result = self.search_history_params(enckwargs)
        if not result:
            new_result = self.run(**kwargs)
            with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                             dir=self.basedir,
                                             prefix=self.prefix,
                                             suffix=self.suffix) as outfile:
                filename = outfile.name
            self.store_result(filename, *new_result)
            self.result_filename = filename
            with open(filename + '.json', 'w') as outfile:
                json.dump(enckwargs, outfile)
            result = new_result
        return result


@contextlib.contextmanager
def fig_as_data(plt, fig, with_alpha=False):
    plt.axis('off')
    ns = SimpleNamespace()
    try:
        yield ns
    except:
        raise
    else:
        fig.canvas.draw()
        # noinspection PyProtectedMember
        data = np.array(fig.canvas.renderer._renderer)
        if not with_alpha:
            data = data[..., :3]
        ns.data = data
    finally:
        plt.close()


def get_runid_from_file(_file_: str, return_prefix=False) \
        -> typing.Union[str, typing.Tuple[str, str]]:
    r"""
    Extract runid from __file__, provided that __file__ is of format
    ``{something}.{runid}.py``, strictly speaking, of regex
    ``^[^\n\t\.]*\.([^\n\t\.]+)\.py$``.

    :param _file_: the ``__file__`` constant
    :param return_prefix: if True, also returns the ``something`` part
    :return: the runid, and possibly ``something``
    """
    pattern = r'^([^\n\t\.]*)\.([^\n\t\.]+)\.(py|ini)$'
    matched = re.match(pattern, os.path.basename(_file_))
    if not matched:
        raise ValueError('"{}" not matching pattern \'{}\''
                         .format(_file_, pattern))
    if return_prefix:
        return matched.group(2), matched.group(1)
    else:
        return matched.group(2)


GridLayoutSpec = typing.Sequence[typing.Sequence[int]]


def make_grid(images: typing.Sequence[np.ndarray], layout: GridLayoutSpec,
              margin: int = 1) -> np.ndarray:
    """
    Make a sequence of images into grid so that they can be plotted in one
    figure. The grid use matrix coordinate, 0-indexed.

    :param images: the images, of dtype either ``np.uint8`` or ``np.float64``
    :param layout: 2D int array to specify the grid layout; the ``layout[i,j]``
           ``images[layout[i,j]]`` should be the image to be shown at
           ``(i,j)`` coordinate of the grid. If ``layout[i,j]`` is -1, that
           cell will be left empty
    :param margin: the width of margin between cells
    :return: the grid
    :raise ValueError: if ``images`` is empty
    """
    if not len(images):
        raise ValueError('No image to make grid')

    layout = np.array(layout, dtype=np.int64)
    cell_sizes = np.ones((2,) + layout.shape, dtype=np.int64)
    for i, j in itertools.product(*map(range, layout.shape)):
        if layout[i, j] >= 0:
            cell_sizes[:, i, j] = images[layout[i, j]].shape[:2]
    nrows = np.max(cell_sizes[0], axis=1)
    ncols = np.max(cell_sizes[1], axis=0)
    nchannels = max(iter((im.shape[2] if len(im.shape) == 3 else 1)
                         for im in images))
    canvas = np.zeros((sum(nrows) + margin * (len(nrows) - 1),
                       sum(ncols) + margin * (len(ncols) - 1),
                       nchannels),
                      dtype=images[0].dtype)
    for i, j in itertools.product(*map(range, layout.shape)):
        if layout[i, j] >= 0:
            loc = (np.sum(nrows[:i]) + i * margin,
                   np.sum(ncols[:j]) + j * margin)
            im = images[layout[i, j]]
            if len(im.shape) == 2:
                im = im[..., np.newaxis]
            if 1 == im.shape[2] < nchannels:
                im = np.repeat(im, nchannels, axis=2)
            canvas[loc[0]:loc[0] + im.shape[0],
            loc[1]:loc[1] + im.shape[1]] = im
    return canvas


class H5ResultSaver(object):
    """
    Save large-scale experiment results in HDF5 file. This helps efficiently
    store up to gigabytes of data without loading all at once to memory when
    reading.

    Assumptions about the experiment results:

        - Each experiment produces k results, each of data type DT_k,
          shape SH_k, and name NAME_k
        - All results are presented as numpy arrays
        - N experiments will be performed in total

    The results are arranged in k databases, named NAME_k. Each database
    is of shape (N, *SH_k). At ``__init__``, (DT_k, SH_k) of each NAME_k
    must be specified.

    It's important to ``close`` the ``H5ResultSaver``.  To close the saver
    automatically, use ``contextlib.closing`` on the ``H5ResultSaver`` object.
    """

    DsSpec = collections.namedtuple('DsSpec', ('dt', 'sh'))
    """Fields: (``dtype``, ``shape``)"""

    def __init__(self, filename, config, **kwargs):
        """
        :param filename: the path of the HDF5 file to write
        :param config: a dict of {NAME_k: (DT_k, SH_k)} as described
               in ``help(H5ResultSaver)``

        Keyword arguments
        +++++++++++++++++

        :param growth: the growth of the size of datasets when appending,
               default to 100
        :param initlen: the initial length of datasets at creation, default
               to twice of ``growth``
        :param overwrite: True to overwrite existing file, default to False
        """
        self.config = {}
        for name, (dt, sh) in config.items():
            dt = np.dtype(dt)
            sh = tuple(map(int, sh))
            if sh and min(sh) < 0:
                raise ValueError('Invalid shape: {}'.format(sh))
            self.config[name] = H5ResultSaver.DsSpec(dt, sh)
        self.growth = int(kwargs.get('growth', 100))
        self.initlen = int(kwargs.get('initlen', 2 * self.growth))
        self.dslen = 0

        # fail if `filename` already exists
        overwrite = bool(kwargs.get('overwrite', False))
        mode = 'w' if overwrite else 'w-'
        self.h5file = h5py.File(filename, mode)

    def save(self, **results):
        """
        Append results to datasets.

        :param results: dict of results {name: data}, where ``name`` must be
               one of the NAME_k specified at ``__init__``,
               ``np.stack(data).shape`` must be equal to SH_k, and
               ``np.stack(data).dtype`` must be equal to DT_k,
               ``len(results)`` must be equal to the size of config at
               ``__init__``; otherwise, raise ``ValueError``
        """
        to_append = set(self.config)
        self._ensure_capacity()
        for name, data in results.items():
            data = np.stack(data)
            spec = self.config[name]
            if data.shape != spec.sh or data.dtype != spec.dt:
                raise ValueError('Invalid shape/dtype of data named "{}"'
                                 .format(name))
            self.h5file[name][self.dslen] = data
            to_append.remove(name)
        if to_append:
            raise ValueError('Missing results: {}'.format(to_append))
        self.dslen += 1

    def close(self):
        self._trim_to_length()
        self.h5file.close()

    def _ensure_capacity(self):
        for name, spec in self.config.items():
            try:
                ds = self.h5file[name]
            except KeyError:
                # noinspection PyTypeChecker
                self.h5file.create_dataset(
                    name, shape=(self.initlen,) + spec.sh,
                    dtype=spec.dt, chunks=True, maxshape=(None,) + spec.sh,
                    compression='gzip')
            else:
                while self.dslen >= len(ds):
                    ds.resize(self.dslen + self.growth, axis=0)

    def _trim_to_length(self):
        for name in self.config:
            try:
                ds = self.h5file[name]
            except KeyError:
                pass
            else:
                ds.resize(self.dslen, axis=0)


class IniFunctionCaller:
    def __init__(self, cfg: configparser.ConfigParser,
                 varparam_policy='raise'):
        self.cfg = cfg
        self.varparam_policy = varparam_policy

    def call(self, f: typing.Callable[[typing.Any], T], **kwargs) -> T:
        """
        :param f: the callable to invoke
        :param scopes: INI sections to search for; if None, search in all
               sections in sequential order until the underlying key is found
        :type scopes: typing.Sequence[str]
        :param argname2inikey: translate certain argument name to INI key name
        :type argname2inikey: typing.Dict[str, str]
        :param argname2ty: translate certain argument name to unary function
               that converts string INI value to appropriate type; if not
               specified, the unary function will be taken from the type
               annotation, and if not found in annotation, default to ``str``
        :type argname2ty: typing.Dict[str, typing.Callable[ [str], typing.Any]]
        :param argname2value: provides default value for certain arguments,
               which will override the default value in function signature
        :type argname2value: typing.Dict[str, typing.Any]
        :return: whatever is returned by ``f``
        """
        logger = _l(self, 'call')
        argname2inikey = kwargs.get('argname2inikey', {})
        argname2ty = kwargs.get('argname2ty', {})
        scopes = kwargs.get('scopes', list(self.cfg.sections()))
        argname2value = kwargs.get('argname2value', {})

        args = collections.OrderedDict()
        kwargs = collections.OrderedDict()
        params = inspect.signature(f).parameters
        for name, par in params.items():
            if par.kind in (inspect.Parameter.VAR_KEYWORD,
                            inspect.Parameter.VAR_POSITIONAL):
                if self.varparam_policy == 'ignore':
                    logger.info('Ignored var parameter in callable {}'
                                .format(f))
                    continue
                else:
                    raise RuntimeError('Unsupported parameter *args and/or '
                                       '**kwargs found in callable {}'
                                       .format(f))
            if par.kind in (inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD):
                param_queue = args
            else:
                param_queue = kwargs

            try:
                param_queue[name] = argname2value[name]
            except KeyError:
                inikey = argname2inikey.get(name, name)
                for sec in scopes:
                    try:
                        opts = self.cfg[sec]
                    except KeyError:
                        logger.debug('section "{}" not found; skipped'
                                     .format(sec))
                        pass
                    else:
                        if inikey in opts:
                            inivalue = opts[inikey]
                            if name in argname2ty:
                                ty = argname2ty[name]
                            elif par.annotation != inspect.Parameter.empty:
                                ty = par.annotation
                            else:
                                ty = str
                            inivalue = ty(inivalue)
                            break
                else:
                    if par.default != inspect.Parameter.empty:
                        inivalue = par.default
                    else:
                        raise KeyError('Argument `{}` (inikey={}) of '
                                       'callable {} not specified in self.cfg'
                                       .format(name, inikey, f))
                param_queue[name] = inivalue
        args = tuple(args.values())
        logger.info('Parsed args: {}'.format(args))
        logger.info('Parsed kwargs: {}'.format(kwargs))
        return f(*args, **kwargs)


class LiteralBoolean:
    def __call__(self, string):
        return string == 'True'


class FixedLenDelimSepList:
    def __init__(self, n: int, delim=',', type_=int):
        self.n = n
        self.delim = delim
        self.type_ = type_

    def __call__(self, string) -> typing.Tuple:
        value = tuple(map(self.type_, string.split(self.delim)))
        if len(value) != self.n:
            raise ValueError('Expecting list of length {} but got {}'
                             .format(self.n, len(value)))
        return value


class VdsetRoot:
    def __init__(self):
        self.parser2 = FixedLenDelimSepList(2)
        self.parser3 = FixedLenDelimSepList(3)

    def __call__(self, string) -> str:
        try:
            value = self.parser2(string)
        except ValueError:
            value = self.parser3(string)
        return vmdata.dataset_root(*value)


class IntRanges:
    def __call__(self, string: str) -> typing.List[typing.Sequence[int]]:
        rngs = []
        for r in string.strip().split(','):
            try:
                start, end = tuple(map(int, r.split('-')))
            except (ValueError, TypeError):
                try:
                    start = int(r)
                except (ValueError, TypeError):
                    raise
                else:
                    end = start + 1
            rngs.append(range(start, end))
        return rngs


def normalize_radiant_angle(x: np.ndarray) -> np.ndarray:
    """
    Normalize angular value to [0, 2pi], with ``np.nan`` intact.
    :param x: the angular value, possibly containing ``np.nan``
    :return: the normalized angles
    """
    x = np.copy(x)
    with utils.suppress_numpy_warning(invalid='ignore'):
        gt0 = (x > 0)
        lt0 = (x < 0)
    mul = np.floor_divide(np.abs(x), 2 * np.pi)
    x += lt0 * (1 + mul) * 2 * np.pi
    x -= gt0 * mul * 2 * np.pi
    return x
