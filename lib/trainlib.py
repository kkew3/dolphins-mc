"""
Common training utilities
"""

import os
import logging
import re
from datetime import datetime
import collections

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Union, Callable, Iterable, Tuple, Dict

from utils import loggername as _l


def action_fired(fired: Union[int, Callable[[Any], bool]]) -> Callable[[Any], bool]:
    """
    Returns a callable that returns a bool indicating whether an action should
    be performed given the progress of an ongoing task.

    :param fired: an int or the callable to be returned. If
           ``fired`` is an int, then ``fired`` will be initialized to
           ``lambda x: x % fired == 0``
    """
    logger = logging.getLogger(_l(__name__, 'action_fired'))
    if isinstance(fired, int):
        if fired < 1:
            logger.warn('Expect `fired` at least 1 if int but got {}; '
                        'converted to 1'.format(fired))
            fired = 1

        def _fired(progress: int):
            return progress % fired == 0

        return _fired
    return fired


class CheckpointSaver(object):
    """
    Save checkpoint periodically.
    """

    def __init__(self, net: nn.Module, savedir: str,
                 checkpoint_tmpl: str = 'checkpoint_{0}.pth',
                 fired: Union[int, Callable[[Any], bool]] = 10):
        """
        :param net: the network to save states; the network should already been
               in the target device
        :param savedir: the directory under which to write checkpoints; if not
               exists, it will be created automatically
        :param fired: a callable that expects one argument and returns a bool
               to indicate whether the checkpoint should be saved. If both
               ``fired`` and ``progress`` is an int, then ``fired`` will be
               initialized to ``lambda x: x % progress == 0``
        :param checkpoint_tmpl: the checkpoint file template
               (by ``str.format``), should accept exactly one positional
               argument the same as that passed to ``fired``
        """
        if not isinstance(net, nn.Module):
            raise TypeError('Expected torch.nn.Module of `net` but got: {}'
                            .format(type(net)))

        savedir = os.path.normpath(savedir)
        os.makedirs(savedir, exist_ok=True)

        self.net = net
        self.fired = action_fired(fired)
        self.checkpoint_tmpl = checkpoint_tmpl
        self.savedir = savedir
        self._logger = logging.getLogger(_l(__name__, self))

    def __call__(self, progress):
        """
        Save checkpoint as needed.

        :param progress: see ``help(type(self).__init__)``
        :return: the file written if fired; otherwise None
        """
        if self.fired(progress):
            name = self.checkpoint_tmpl.format(*progress)
            tofile = os.path.join(self.savedir, name)
            torch.save(self.net.state_dict(), tofile)
            self._logger.debug('{} written at progress {}'
                               .format(tofile, progress))
            return tofile


def load_checkpoint(net: nn.Module, savedir: str, checkpoint_tmpl: str,
                    progress: tuple, map_location: str = 'cpu') -> None:
    """
    Load checkpoint from file.

    :param net: network to load checkpoint
    :param savedir: directory under which checkpoints are saved
    :param checkpoint_tmpl: basename template of the checkpoint files
    :param progress: which checkpoint file to load
    :param map_location: where to load the weights
    """
    logger = logging.getLogger(_l(__name__, 'load_checkpoint'))
    basename = checkpoint_tmpl.format(*progress)
    fromfile = os.path.join(savedir, basename)
    state_dict = torch.load(fromfile, map_location=map_location)
    net.load_state_dict(state_dict)
    logger.debug('progress {} loaded from {}'
                 .format(progress, fromfile))


class StatSaver(object):
    """
    Save (scalar) statistics periodically as npz file.
    """

    def __init__(self, statdir: str,
                 statname_tmpl='stats_{0}.npz',
                 fired: Union[int, Callable[[Any], bool]] = 10):
        """
        :param statdir: the directory under which to write statistics npz
               files; if not exists, it will be created automatically
        :param fired: a callable that expects one argument and returns a bool
               to indicate whether the checkpoint should be saved. If both
               ``fired`` and ``progress`` is an int, then ``fired`` will be
               initialized to ``lambda x: x % progress == 0``
        :param statname_tmpl: the stat npz file basename template
               (by ``str.format``), should accept exactly one positional
               argument the same as that passed to ``fired``
        """
        statdir = os.path.normpath(statdir)
        os.makedirs(statdir, exist_ok=True)

        self.fired = action_fired(fired)
        self.statname_tmpl = statname_tmpl
        self.statdir = statdir
        self.logger = logging.getLogger(_l(__name__, self))

    def __call__(self, progress, **stat_dict):
        """
        Save statistics, which are wrapped into numpy arrays, as needed.

        :param progress: see ``help(type(self).__init__)``
        :param stat_dict: dict of floats or lists of floats, i.e. the
               non-cumulative statistics at the progress
        :return: the file written if fired; otherwise None
        """
        if self.fired(progress):
            name = self.statname_tmpl.format(*progress)
            tofile = os.path.join(self.statdir, name)
            _stat_dict = {k: np.array(stat_dict[k]) for k in stat_dict}
            np.savez(tofile, **_stat_dict)
            self.logger.debug('{} written at progress "{}"'
                              .format(tofile, progress))
            return tofile


def load_stat(statdir: str, statname_tmpl: str, progress: tuple,
              key: str = None) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Load statistics.

    :param statdir: directory under which the statistics are saved
    :param statname_tmpl: stat file basename template
    :param progress: which stat file to load
    :param key: which field to load
    :return: the npz content dict if ``key`` is not specified, otherwise the
             corresponding field data
    """
    logger = logging.getLogger(_l(__name__, 'load_stat'))
    basename = statname_tmpl.format(*progress)
    fromfile = os.path.join(statdir, basename)
    data = np.load(fromfile)
    logger.debug('progress {} loaded from "{}"'
                 .format(progress, fromfile))
    if key is None:
        data = {k: data[k] for k in data.keys()}
    else:
        data = data[key]
    return data


class BasicTrainer(object):
    timestamp_format = '%Y%m%d%H%M'  # e.g. 201811051438

    def __init__(self, net: nn.Module, max_epoch: int = 1, device: str = 'cpu'):
        r"""
        :param net: the network to train
        :param max_epoch: maximum epoch to train, where an epoch is defined as
               a complete traversal of the underlying dataset
        :param device: where to train the network, choices:
               { cpu, cuda(:\d+)? }

        Guideline to override this method:

            - specify where the ``basedir`` of all other directories defined
              below. If a directory has been specified, the ``basedir`` will
              have no effect on that directory. If ``basedir`` is specified
              but not a particular directory, then it will be set to
              ``basedir/{default_directiory_name}``.
            - specify where to store the saved checkpoints (``savedir``) and
              the saved statistics (``statdir``) during training
            - specify where to store the saved statistics (``statdir_eval``)
              during evaluation on validation set
            - specify how the saving of checkpoints and statistics is
              triggered during training (``fired``, of signature
              ``fired(Tuple[int, int]) -> bool``, that accepts a tuple of
              ``(epoch_id, minibatch_id)`` and returns whether or not to
              trigger the saving after current (epoch, minibatch)
            - specify how the saving of statistics is triggered during
              evaluation on validation set (``fired_eval``, definition the same
              as ``fired``)
            - specify the statistics names (``stat_names``), in the same order
              as returned by either ``train_once`` or ``eval_once``; leaving
              it as is makes the statistics anonymous
        """
        self.device = device
        self.net = net.to(device)
        self.max_epoch = max_epoch
        self.train_stages = ('train', 'eval')
        self.stage = None  # current training stage

    @property
    def default_basedir(self):
        """
        The default base directory of checkpoints and statistics, in form of
        "runs-{timestamp}", with ``timestamp`` of datetime format
        ``type(self).timestamp_format``.
        """
        return 'runs-{}'.format(datetime.today().strftime(
                type(self).timestamp_format))

    # noinspection PyUnresolvedReferences,PyAttributeOutsideInit
    def init_monitors(self):
        """
        Initialize CheckpointSaver and StatSaver as per settings in
        ``__init__``.
        """
        basedir = self.basedir if hasattr(
                self, 'basedir') else self.default_basedir
        defaults = {
            'statdir': os.path.join(basedir, 'stat'),
            'savedir': os.path.join(basedir, 'save'),
            'fired': (lambda progress: True),
            'statdir_eval': os.path.join(basedir, 'stat_eval'),
            'fired_eval': (lambda progress: True),
        }
        for k, v in defaults.items():
            if not hasattr(self, k):
                setattr(self, k, v)

        # deferred instantiation
        self._statsaver = lambda: StatSaver(self.statdir, fired=self.fired,
                                            statname_tmpl='stats_{0}_{1}.npz')
        self._checkpointsaver = lambda: CheckpointSaver(self.net, self.savedir, fired=self.fired,
                                                        checkpoint_tmpl='checkpoint_{0}_{1}.pth')
        self._statsaver_eval = lambda: StatSaver(self.statdir_eval, fired=self.fired_eval,
                                                 statname_tmpl='stats_{0}_{1}.npz')
        self.statsaver = None
        self.checkpointsaver = None
        self.statsaver_eval = None

    def train_once(self, inputs, targets) -> Tuple[Any]:
        """
        Train for one minibatch. The return type of this method should be the
        same as that of ``validate_once``.

        :param inputs: the inputs
        :param targets: the targets
        :return: anything that will be saved as statistics
        """
        raise NotImplementedError()

    def eval_once(self, inputs, targets) -> Tuple[Any]:
        """
        Evaluate (in validation set) for one (mini)batch. The return type of
        this method should be the same as that of ``train_once``.

        :param inputs: the inputs
        :param targets: the targets
        :return: anything that will be saved as statistics
        """
        raise NotImplementedError()

    def get_trainloader(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """
        :return: the training set loader of input-target pairs
        """
        raise NotImplementedError()

    def get_evalloader(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """
        :return: the validation set loader of input-target pairs
        """
        raise NotImplementedError()

    def before_batch_train(self):
        """
        Callback before each batch of the training loop.
        """
        pass

    def after_batch_train(self):
        """
        Callback after each batch of the training loop.
        """
        pass

    def before_batch_eval(self):
        """
        Callback before each batch of the evaluation loop.
        """
        pass

    def after_batch_eval(self):
        """
        Callback after each batch of the evaluation loop.
        """
        pass

    def before_epoch(self):
        """
        Callback before each epoch.
        """
        pass

    def after_epoch(self):
        """
        Callback after each epoch.
        """
        pass

    def setup(self):
        """
        Callback before ``run`` and after ``__init__``. Default to initializing
        checkpoing and statistics savers.
        """
        self.init_monitors()

    def teardown(self, error=None):
        """
        Callback before the return of ``run``, whether or not successful return
        or unsuccessful one.

        :param error: the cause of the return, or ``None`` if there's no error
        """
        pass

    def run(self):
        logger = logging.getLogger(_l(__name__, self, 'run'))
        logger.debug('Initializing')
        self.setup()

        try:
            for epoch in range(self.max_epoch):
                self.before_epoch()
                for self.stage in self.train_stages:
                    if self.stage == 'train':
                        self.net.train()
                        for batch, (inputs, targets) in enumerate(self.get_trainloader()):
                            self.before_batch_train()
                            stats = self.train_once(inputs, targets)
                            stats_to_log = self._organize_stats(stats)
                            logger.info('epoch{}/train batch{}: {}'
                                        .format(epoch, batch, list(stats_to_log.items())))
                            if self.statsaver is None:
                                self.statsaver = self._statsaver()
                            self.statsaver((epoch, batch), **stats_to_log)
                            if self.checkpointsaver is None:
                                self.checkpointsaver = self._checkpointsaver()
                            self.checkpointsaver((epoch, batch))
                            self.after_batch_train()
                    else:
                        self.net.eval()
                        for batch, (inputs, targets) in enumerate(self.get_evalloader()):
                            self.before_batch_eval()
                            stats = self.eval_once(inputs, targets)
                            stats_to_log = self._organize_stats(stats)
                            logger.info('epoch{}/eval batch{}: {}'
                                        .format(epoch, batch, list(stats_to_log.items())))
                            if self.statsaver_eval is None:
                                self.statsaver_eval = self._statsaver_eval()
                            self.statsaver_eval((epoch, batch), **stats_to_log)
                            self.after_batch_eval()
                self.after_epoch()
            logger.info('Returns successfully')
            self.teardown()
        except BaseException as err:
            logger.error('Exception raised: {}'.format(err))
            self.teardown(error=err)
            raise

    # noinspection PyUnresolvedReferences
    def _organize_stats(self, stats: Tuple[Any]) -> dict:
        logger = logging.getLogger(_l(__name__, self, '_organize_stats'))
        try:
            stat_names = self.stat_names
        except AttributeError:
            # to conform to the naming policy of ``numpy.savez``
            stat_names = ['arr_{}'.format(i) for i in range(len(stats))]

        if len(self.stat_names) != len(stats):
            logger.warning('len(self.stat_names) is '
                           'different from len(stats): {} != {}'
                           .format(len(self.stat_names), len(stats)))
        return collections.OrderedDict(zip(stat_names, stats))
