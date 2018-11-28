import importlib
import os
from typing import Tuple, Sequence, List, Iterable
import logging

import torch
from more_sampler import SlidingWindowBatchSampler
import numpy as np
import matplotlib
import vmdata
from more_trans import rearrange_temporal_batch, DeNormalize, clamp_tensor_to_image
from torch import nn
from torch.utils.data import DataLoader
from utils import loggername

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grplaincae.basicmodels import Autoencoder, SpatioTemporalInputShape
from trainlib import load_checkpoint

unit_figsize = (6, 4)


def _l(*args):
    return logging.getLogger(loggername(__name__, *args))


def load_trained_net(model_name: str, savedir: str, progress: Tuple[int, int]):
    m = importlib.import_module('grplaincae.models.' + model_name)
    encoder = m.STCAEEncoder()
    decoder = m.STCAEDecoder()
    net = Autoencoder(encoder, decoder)
    load_checkpoint(net, savedir, 'checkpoint_{}_{}.pth', progress)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net, m


def draw_left_right(tofile: str, arrs: Tuple[np.ndarray, np.ndarray],
                    margin: int = 1, init_color=0) -> None:
    """
    Draw grayscale image.
    """
    arr1, arr2 = arrs
    (h1, w1), (h2, w2) = arr1.shape, arr2.shape
    h, w = max(h1, h2), w1 + margin + w2
    dt1, dt2 = str(arr1.dtype), str(arr2.dtype)
    dt = np.uint8 if ('int' in dt1 and 'int' in dt2) else np.float64

    figsize = (unit_figsize[0], unit_figsize[1] * 2)
    plt.figure(figsize=figsize)
    canvas = (init_color * np.ones((h, w))).astype(dt)
    canvas[:h1, :w1] = arr1
    canvas[:h2, -w2:] = arr2
    plt.imsave(tofile, canvas)
    plt.close()


def draw_up_down(tofile: str, arrs: Tuple[np.ndarray, np.ndarray],
                 margin: int = 1, init_color=0) -> None:
    arr1, arr2 = arrs
    (h1, w1), (h2, w2) = arr1.shape, arr2.shape
    h, w = h1 + margin + h2, max(w1, w2)
    dt1, dt2 = str(arr1.dtype), str(arr2.dtype)
    dt = np.uint8 if ('int' in dt1 and 'int' in dt2) else np.float64

    figsize = (unit_figsize[0] * 2, unit_figsize[1])
    plt.figure(figsize=figsize)
    canvas = (init_color * np.ones((h, w))).astype(dt)
    canvas[:h1, :w1] = arr1
    canvas[-h2:, :w2] = arr2
    plt.imsave(tofile, canvas)
    plt.close()


def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu()


def BCTHW2LoCHW(tensor: torch.Tensor) -> List[torch.Tensor]:
    sh = SpatioTemporalInputShape(*tensor.size())
    return list(tensor.reshape(sh.B * sh.T, sh.C, sh.H, sh.W))


def LoCHW2BTHW(tensors: Iterable[torch.Tensor], T: int) -> torch.Tensor:
    tensor = torch.stack(tuple(tensors))
    tensor = rearrange_temporal_batch(tensor, T)
    tensor = tensor[:, 0, :, :, :]
    return tensor


def postprocess(inputs: torch.Tensor, attns: torch.Tensor,
                outputs: torch.Tensor, targets: torch.Tensor,
                denormalize, tbatch_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inputs = to_cpu(inputs)  # shape: (B, 1, T, H, W)
    attns = to_cpu(attns)  # shape: (B, 1, T, H, W)
    outputs = to_cpu(outputs)  # shape: (B, 1, 1, H, W)
    targets = to_cpu(targets)  # shape: (B, 1, 1, H, W)

    inputs = BCTHW2LoCHW(inputs)
    outputs = BCTHW2LoCHW(outputs)
    targets = BCTHW2LoCHW(targets)

    inputs = map(clamp_tensor_to_image, map(denormalize, inputs))
    outputs = map(clamp_tensor_to_image, map(denormalize, outputs))
    targets = map(clamp_tensor_to_image, map(denormalize, targets))

    inputs = LoCHW2BTHW(inputs, tbatch_size)  # shape: (B, T, H, W)
    attns = attns.squeeze(1)  # shape: (B, T, H, W)
    outputs = LoCHW2BTHW(outputs, 1).squeeze(1)  # shape: (B, H, W)
    targets = LoCHW2BTHW(targets, 1).squeeze(1)  # shape: (B, H, W)

    inputs = inputs.numpy()
    attns = attns.numpy()
    outputs = outputs.numpy()
    targets = targets.numpy()

    return inputs, attns, outputs, targets


def visualize(todir: str, root: str, transform, normalize_stats,
              indices: Sequence[int],
              net: nn.Module, net_module,
              temperature: float = 1.0,
              bwth: float = None,
              device: str = 'cpu',
              batch_size: int = 1,
              predname_tmpl: str = 'pred{}.png',
              attnname_tmpl: str = 'attn{}_pred{}.png') -> None:
    r"""
    Visualize predictions and input gradients.

    :param todir: directory under which to save the visualization
    :param root: the dataset root
    :param transform: transform on dataset inputs
    :param normalize_stats: the mean-std tuple used in normalization
    :param indices: indices of dataset inputs involved in visualization
    :param net: the trained network
    :param net_module: the module from which ``net`` is loaded, as returned by
           ``load_trained_net``
    :param temperature: the larger it is, the more contrast in the attention
           map is
    :param bwth: if not specified, plot the attention map as sigmoidal map,
           where 0.5 means zero gradient, and :math:`0.5 \pn 0.5` means
           positive and negative gradients respectively; if specified as a
           range [0.0, 1.0], get the absolute value of the gradient, multiply
           float in by ``temperature``, take the sigmoid, and remove all values
           lower than the ``bwth`` threshold
    :param device: where to do inference
    :param batch_size: batch size when doing inference; will be set to 1 if
           ``device`` is 'cpu'
    :param predname_tmpl: the basename template of prediction visualization
    :param attnname_tmpl: the basename template of attention (inputs gradient)
           visualization
    """
    if bwth is not None and not (0.0 <= bwth <= 1.0):
        raise ValueError('bwth must be in range [0,1], but got {}'
                         .format(bwth))
    logger = _l('visualize')
    tbatch_size = net_module.temporal_batch_size
    if device == 'cpu':
        batch_size = 1
    sam = SlidingWindowBatchSampler(indices, 1 + tbatch_size, shuffled=False,
                                    batch_size=batch_size, drop_last=True)
    sam_ = SlidingWindowBatchSampler(indices, 1 + tbatch_size, shuffled=False,
                                     batch_size=batch_size, drop_last=True)
    denormalize = DeNormalize(*normalize_stats)
    os.makedirs(todir, exist_ok=True)

    mse = nn.MSELoss().to(device)
    sigmoid = nn.Sigmoid().to(device)
    net = net.to(device)
    with vmdata.VideoDataset(root, transform=transform) as vdset:
        loader = DataLoader(vdset, batch_sampler=sam)
        for frames, iframes in zip(loader, iter(sam_)):
            frames = rearrange_temporal_batch(frames, 1 + tbatch_size)
            iframes = np.array(iframes).reshape((batch_size, 1 + tbatch_size))
            inputs, targets = frames[:, :, :-1, :, :], frames[:, :, -1:, :, :]
            iinputs, itargets = iframes[:, :-1], iframes[:, -1]
            inputs, targets = inputs.to(device), targets.to(device)

            inputs.requires_grad_()
            outputs = net(inputs)
            loss = mse(outputs, targets)
            loss.backward()
            attns = inputs.grad * temperature

            logger.info('[f{}-{}/eval/attn] l1norm={} l2norm={} numel={} max={}'
                        .format(np.min(iframes),
                                np.max(iframes),
                                torch.norm(attns.detach(), 1).item(),
                                torch.norm(attns.detach(), 2).item(),
                                torch.numel(attns.detach()),
                                torch.max(attns.detach())))
            logger.info('[f{}-{}/eval/loss] mse={} B={}'
                        .format(np.min(iframes),
                                np.max(iframes),
                                loss.item(),
                                targets.size(0)))

            if bwth is not None:
                attns = sigmoid(torch.abs(attns)) * 2 - 1
                mask = (attns >= bwth).to(attns.dtype)
                attns = mask * attns
            else:
                attns = sigmoid(attns)

            inputs, attns, outputs, targets = postprocess(
                    inputs, attns, outputs, targets,
                    denormalize, tbatch_size)

            for b in range(batch_size):
                f = os.path.join(todir, predname_tmpl.format(itargets[b]))
                draw_left_right(f, (outputs[b], targets[b]))
                for t in range(tbatch_size):
                    f = os.path.join(todir, attnname_tmpl.format(
                            iinputs[b, t], itargets[b]))
                    draw_up_down(f, (inputs[b, t], attns[b, t]))
