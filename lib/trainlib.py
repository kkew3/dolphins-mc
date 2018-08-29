"""
Common training utilities
"""

import os

import torch
import torch.nn as nn

class CheckpointSaver(object):
    """
    Save checkpoint periodically.
    """
    def __init__(self, net, freq, savedir,
                 checkpoint_tmpl='checkpoint_{}.pth'):
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
        os.makedirs(savedir)
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
