import logging

import torch.nn as nn
import torch.utils.data
from typing import Union

import trainlib
import ezfirstae.loaddata as ld
import ezfirstae.models as ezmodels


def train(net: ezmodels.EzFirstCAE,
          dataset: torch.utils.data.Dataset,
          samplers: ld.TrainValidTestAlternatingSamplers,
          savedir: str, statdir: str,
          temporal_batch_size: int=16,
          max_epoch: int=20,
          lam_nonrigid: float=0.01,
          lam_darkness: float=0.01,
          device: Union[str, torch.device]='cpu'):
    """
    :param net: the autoencoder to train
    :param dataset: the video dataset
    :param samplers: the samplers
    :param savedir: the directory to save checkpoints
    :param statdir: the directory to save scalar statistics
    :param temporal_batch_size: the temporal batch size
    :param max_epoch: the maximal epochs to train
    :param lam_nonrigid: weight of nonrigid penalty
    :param lam_darkness: weight of darkness penalty
    :param device: where to train
    """
    if isinstance(device, str):
        device = torch.device(device)
    net = net.to(device)
    nonrigidp = ezmodels.NonrigidPenalty().to(device)
    darknessp = ezmodels.DarknessPenalty().to(device)
    mse = nn.MSELoss().to(device)

    def criterion(outputs, attention, targets):
        rloss = mse(outputs, targets)
        nloss = nonrigidp(attention)
        dloss = darknessp(attention)
        return rloss + lam_nonrigid * nloss + lam_darkness * dloss

    for epoch in range(max_epoch):
        logging.info('Beginning epoch {}'.format(epoch))

