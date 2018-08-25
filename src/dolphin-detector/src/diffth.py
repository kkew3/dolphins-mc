"""
Considering the fact that dolphins swim most of the time, we may detect
dolphins sufficiently (yet not necessarily, 'cause dolphins do sometimes stay
still or swim in direction parallel to the orientation of camera) by pickout
out frame[i] if the 90th-percentile of abs(frame[i+1] - frame[i]) is larger
than a threshold.
"""

import argparse
import os

import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import ddlogger

import vhdata
import more_trans

def dataset_diff(dataset):
    """
    dataset[i] dimension: CHW.

    :param dataset: the video dataset
    :type dataset: vhdata.VideoDataset
    """
    dataloader = DataLoader(dataset, batch_size=99, shuffle=False)
    prev_last = None
    for frames in dataloader:
        # frames: numpy array of dimension BCHW
        if prev_last is not None:
            frames = np.concatenate((prev_last, frames), axis=0)
        dframes = np.abs(np.diff(frames, axis=0))
        yield dframes
        prev_last = frames[-1][np.newaxis]

def write_ddataset(dataset, toroot):
    with vhdata.VideoDatasetWriter(toroot, dataset.shape[1:],
                                   **dataset.attrs) as writer:
        with ddlogger.ddlogger() as dl:
            for dframes in dataset_diff(dataset):
                writer.write_batch(dframes)
                dl.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write absolute difference '
                                     'frames to ../runs/root')
    parser.add_argument('-c', type=int, choices=[8, 9],
                        help='camera channel id')
    parser.add_argument('-i', nargs=3, type=int, help='camera id')
    args = parser.parse_args()

    dset = vhdata.VideoDataset(vhdata.prepare_dataset_root(args.c, tuple(args.i)),
                               transform=trans.Compose([
                                   more_trans.ToNumpy(),
                                   more_trans.HWC2CHW(),
                               ]))
    toroot = os.path.join(os.path.dirname(__file__), '..', 'runs',
                          os.path.basename(dset.root) + '_diff')
    toroot = os.path.realpath(toroot)
    if not os.path.isdir(toroot):
        os.mkdir(toroot)
    write_ddataset(dset, toroot)
