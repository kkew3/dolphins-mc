# Lower initial learning rate than main.0.py
# No Sobel filter after attention
import os

_cd = os.path.dirname(os.path.realpath(__file__))
_name = os.path.basename(os.path.realpath(__file__))
_rid = int(_name.split('.')[1])

import logging

logging.basicConfig(level=logging.INFO, format='%(name)s %(asctime)s -- %(message)s',
                    filename='main.{}.log'.format(_rid))

import sys

import torchvision.transforms as trans

import vmdata
import ezfirstae.loaddata as ld
import ezfirstae.train as train

max_epoch = 1
root = vmdata.prepare_dataset_root(9, (8, 0, 0))
normalize = trans.Normalize(*vmdata.get_normalization_stats(root, bw=True))
transform = ld.PreProcTransform(normalize, pool_scale=8, downsample_scale=3)
statdir = 'stat.{}'.format(_rid)
savedir = 'save.{}'.format(_rid)
device = 'cuda'
lr = 5e-5
lam_dark = 0.1
lam_nrgd = 0.0

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.info('Begin training: model=ezfirstae.models.pred9_f1to8 lam_dark={}'
                ' lam_nrgd={}'.format(lam_dark, lam_nrgd))
    with vmdata.VideoDataset(root, transform=transform) as vdset:
        trainset, testset = ld.contiguous_partition_dataset(range(len(vdset)), (5, 1))
        try:
            train.train_pred9_f1to8(vdset, trainset, testset, savedir, statdir,
                                    device, max_epoch, lr, lam_dark, lam_nrgd)
        except KeyboardInterrupt:
            logger.warning('User interrupt')
            print('Cleaning up ...', file=sys.stderr)
