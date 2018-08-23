r"""
Regard frame[t], :math:`x_t`, as the one with dolphin if::

    .. math::
    \|x_{t+1}-x_t\|_p > \mu

where :math:`\mu` is a constant threshold, or a variable that changes *slowly*
with time.
"""
import os

import vdata


def diff_frames(dataset):
    """
    Compute the difference frame tensor in a generator manner.

    :param dataset: the video dataset
    :type dataset: vdata.VideoDataset
    :yield: numpy array of dimension [C x H x W]
    """
    dataloader  # TODO
