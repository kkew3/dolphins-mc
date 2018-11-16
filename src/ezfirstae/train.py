import logging
from typing import Union, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as trans

from utils import loggername as _l
import more_sampler
import more_trans
import vmdata
import trainlib
import ezfirstae.basicmodels as basicmodels
import ezfirstae.models.pred9_f1to8 as pred9_f1to8


def train_pred9_f1to8(vdset: vmdata.VideoDataset,
                      trainset: Sequence[int], testset: Sequence[int],
                      savedir: str, statdir: str,
                      device: Union[str, torch.device] = 'cpu',
                      max_epoch: int = 1,
                      lr: float=0.001,
                      lam_dark: float = 1.0,
                      lam_nrgd: float = 0.2):
    logger = logging.getLogger(_l(__name__, 'train_pred9_f1to8'))
    if isinstance(device, str):
        device = torch.device(device)

    encoder = pred9_f1to8.STCAEEncoder()
    decoder = pred9_f1to8.STCAEDecoder()
    attention = pred9_f1to8.STCAEDecoder()
    if isinstance(vdset.transform, trans.Normalize):
        normalize = vdset.transform
    else:
        normalize = next(iter(x for x in vdset.transform.__dict__.values()
                              if isinstance(x, trans.Normalize)))

    ezcae = basicmodels.EzFirstCAE(encoder, decoder, attention).to(device)
    mse = nn.MSELoss().to(device)
    darkp = basicmodels.DarknessPenalty(normalize).to(device)
    nrgdp = basicmodels.NonrigidPenalty().to(device)

    def criterion(_outputs: torch.Tensor, _attns: torch.Tensor,
                  _targets: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        loss1 = mse(_attns * _outputs, _attns * _targets)
        loss2 = darkp(_attns)
        loss3 = nrgdp(_attns.view(-1, 1, *_attns.shape[-2:]))
        _loss = loss1 + lam_dark * loss2 + lam_nrgd * loss3
        _loss123 = np.array([loss1.item(), loss2.item(), loss3.item()],
                            dtype=np.float64)
        return _loss, _loss123

    cpsaver = trainlib.CheckpointSaver(
            ezcae, savedir, checkpoint_tmpl='checkpoint_{0}_{1}.pth',
            fired=lambda pg: True)
    stsaver = trainlib.StatSaver(statdir, statname_tmpl='stats_{0}_{1}.npz',
                                 fired=lambda pg: True)
    alpha = 0.9  # the resistance of the moving average approximation of mean loss
    optimizer = optim.Adam(ezcae.parameters(), lr=lr)

    for epoch in range(max_epoch):
        for stage, dataset in [('train', trainset), ('eval', testset)]:
            swsam = more_sampler.SlidingWindowBatchSampler(dataset, 1 + pred9_f1to8.temporal_batch_size, shuffled=True, batch_size=8)
            dataloader = DataLoader(vdset, batch_sampler=swsam)
            moving_average = None
            getattr(ezcae, stage)()  # ezcae.train() or ezcae.eval()
            torch.set_grad_enabled(stage == 'train')
            for j, inputs in enumerate(dataloader):
                progress = epoch, j
                inputs = more_trans.rearrange_temporal_batch(inputs, 1 + pred9_f1to8.temporal_batch_size)
                inputs, targets = inputs[:, :, :-1, :, :], inputs[:, :, -1:, :, :]
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, attns = ezcae(inputs)
                loss, loss123 = criterion(outputs, attns, targets)

                if stage == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                stat_names = ['loss', 'loss_mse', 'loss_dark', 'loss_nrgd']
                stat_vals = [loss.item()] + list(loss123)
                if stage == 'train':
                    moving_average = loss123 if moving_average is None else \
                        alpha * moving_average + (1 - alpha) * loss123
                    cpsaver(progress)
                    stsaver(progress, **dict(zip(stat_names, stat_vals)))
                logger.info(('[epoch{}/batch{}] '.format(epoch, j) +
                             ' '.join('{}={{:.2f}}'.format(n) for n in stat_names))
                            .format(*stat_vals))


def train_pred9_f1to8_no_attn(vdset: vmdata.VideoDataset,
                              trainset: Sequence[int], testset: Sequence[int],
                              savedir: str, statdir: str,
                              device: Union[str, torch.device] = 'cpu',
                              max_epoch: int = 1,
                              lr: float=0.001):
    logger = logging.getLogger(_l(__name__, 'train_pred9_f1to8_no_attn'))
    if isinstance(device, str):
        device = torch.device(device)

    encoder = pred9_f1to8.STCAEEncoder()
    decoder = pred9_f1to8.STCAEDecoder()

    cae = basicmodels.CAE(encoder, decoder).to(device)
    mse = nn.MSELoss().to(device)

    cpsaver = trainlib.CheckpointSaver(
            cae, savedir, checkpoint_tmpl='checkpoint_{0}_{1}.pth',
            fired=lambda pg: True)
    stsaver = trainlib.StatSaver(statdir, statname_tmpl='stats_{0}_{1}.npz',
                                 fired=lambda pg: True)
    alpha = 0.9  # the resistance of the moving average approximation of mean loss
    optimizer = optim.Adam(cae.parameters(), lr=lr)

    for epoch in range(max_epoch):
        for stage, dataset in [('train', trainset), ('eval', testset)]:
            swsam = more_sampler.SlidingWindowBatchSampler(dataset, 9, shuffled=True, batch_size=8)
            dataloader = DataLoader(vdset, batch_sampler=swsam)
            moving_average = None
            getattr(cae, stage)()  # ezcae.train() or ezcae.eval()
            torch.set_grad_enabled(stage == 'train')
            for j, inputs in enumerate(dataloader):
                progress = epoch, j
                inputs = more_trans.rearrange_temporal_batch(inputs, 9)
                inputs, targets = inputs[:, :, :-1, :, :], inputs[:, :, -1:, :, :]
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = cae(inputs)
                loss = mse(outputs, targets)

                if stage == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_val = loss.item()
                if stage == 'train':
                    moving_average = loss_val if moving_average is None else \
                        alpha * moving_average + (1 - alpha) * loss_val
                    cpsaver(progress)
                    stsaver(progress, loss=loss_val)
                logger.info('[epoch{}/batch{}] loss={:.2f}'
                            .format(epoch, j, loss_val))
