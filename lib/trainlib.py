"""
Common training utilities
"""

import os
import contextlib
import logging

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Union, Callable


def action_fired(fired: Union[int, Callable[[Any], bool]]) -> Callable[[Any], bool]:
    """
    Returns a callable that returns a bool indicating whether an action should
    be performed given the progress of an ongoing task.

    :param fired: an int or the callable to be returned. If
           ``fired`` is an int, then ``fired`` will be initialized to
           ``lambda x: x % fired == 0``
    """
    logger = logging.getLogger('.'.join([__name__, 'action_fired']))
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
                 checkpoint_tmpl: str='checkpoint_{0}.pth',
                 fired: Union[int, Callable[[Any], bool]]=10):
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

        os.makedirs(savedir, exist_ok=True)

        self.net = net
        self.fired = action_fired(fired)
        self.checkpoint_tmpl = checkpoint_tmpl
        self.savedir = savedir

    def __call__(self, progress):
        """
        Save checkpoint as needed.

        :param progress: see ``help(type(self).__init__)``
        """
        if self.fired(progress):
            name = self.checkpoint_tmpl.format(*progress)
            tofile = os.path.join(self.savedir, name)
            torch.save(self.net.state_dict(), tofile)


class StatSaver(object):
    """
    Save (scalar) statistics periodically as npz file.
    """

    def __init__(self, statdir: str,
                 statname_tmpl='stats_{0}.npz',
                 fired: Union[int, Callable[[Any], bool]]=10):
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
        os.makedirs(statdir, exist_ok=True)

        self.fired = action_fired(fired)
        self.statname_tmpl = statname_tmpl
        self.statdir = statdir

    def __call__(self, progress, **stat_dict):
        """
        Save statistics, which are wrapped into numpy arrays, as needed.

        :param progress: see ``help(type(self).__init__)``
        :param stat_dict: dict of floats or lists of floats, i.e. the
               non-cumulative statistics at the progress
        """
        if self.fired(progress):
            name = self.statname_tmpl.format(*progress)
            tofile = os.path.join(self.statdir, name)
            _stat_dict = {k: np.array(stat_dict[k]) for k in stat_dict}
            np.savez(tofile, **_stat_dict)
