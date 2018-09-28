"""
Common training utilities
"""

import os

import numpy as np
import torch
import torch.nn as nn

class CheckpointSaver(object):
    """
    Save checkpoint periodically.
    """
    def __init__(self, net, savedir, freq=1, checkpoint_tmpl='checkpoint_{}.pth'):
        """
        :param net: the network to save states; the network should already been
               in the target device
        :type net: torch.nn.Module
        :param freq: the batch/iteration period to save, should be at least 1
        :type freq: int
        :param savedir: the directory under which to write checkpoints; if not
               exists, it will be created automatically
        :type savedir: str
        :param checkpoint_tmpl: the checkpoint file template
               (by ``str.format``), should accept exactly one positional
               argument as the batch/iteration number
        :type checkpoint_tmpl: str
        """
        # sanity checks and auto-corrections
        if not isinstance(net, nn.Module):
            raise TypeError('Expected torch.nn.Module of `net` but got: {}'
                            .format(type(net)))
        freq = max(1, int(freq))
        os.makedirs(savedir, exist_ok=True)
        try:
            _teststr = checkpoint_tmpl.format(0)
            if _teststr == checkpoint_tmpl:
                raise ValueError()
        except:
            raise ValueError('Invalid `checkpoint_tmpl`: {}'
                             .format(checkpoint_tmpl))

        self.net = net
        self.freq = freq
        self.savedir = savedir
        self.cp_tmpl = checkpoint_tmpl

    def __call__(self, batch_id):
        """
        Save checkpoint as needed.

        :param batch_id: current batch/itertion number
        :type batch_id: int
        """
        if batch_id % self.freq == 0:
            tofile = os.path.join(self.savedir, self.cp_tmpl.format(batch_id))
            with open(tofile, 'wb') as outfile:
                torch.save(self.net.state_dict(), outfile)


class StatSaver(object):
    """
    Save (scalar) statistics periodically as npz file.
    """

    def __init__(self, statdir, freq=1, filename_tmpl='stats_{}.npz'):
        freq = max(1, int(freq))
        os.makedirs(statdir, exist_ok=True)
        try:
            _teststr = filename_tmpl.format(0)
            if _teststr == filename_tmpl:
                raise ValueError()
        except:
            raise ValueError('Invalid `filename_tmpl`: {}'
                             .format(filename_tmpl))
        self.freq = freq
        self.statdir = statdir
        self.fn_tmpl = filename_tmpl

    def __call__(self, batch_id, **stat_dict):
        """
        Save statistics, which are wrapped into numpy arrays, as needed.

        :param batch_id: current batch/itertion number
        :type batch_id: int
        :param stat_dict: dict of list of floats, i.e. the statistics
        """
        if batch_id % self.freq == 0:
            tofile = os.path.join(self.statdir, self.fn_tmpl.format(batch_id))
            _stat_dict = {k: np.array(stat_dict[k]) for k in stat_dict}
            np.savez(tofile, **_stat_dict)
