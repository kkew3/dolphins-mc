# Lower initial learning rate than main.2.py
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
root = vmdata.dataset_root(9, (8, 0, 0))
normalize = trans.Normalize(*vmdata.get_normalization_stats(root, bw=True))
transform = ld.PreProcTransform(normalize, pool_scale=8, downsample_scale=3)
statdir = 'stat.{}'.format(_rid)
savedir = 'save.{}'.format(_rid)
device = 'cuda'
lr = 5e-5

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.info('Begin training: model=ezfirstae.models.pred9_f1to8(no-attention)')
    with vmdata.VideoDataset(root, transform=transform, max_mmap=3, max_gzcache=100) as vdset:
        trainset, testset = ld.contiguous_partition_dataset(range(len(vdset)), (5, 1))
        try:
            train.train_pred9_f1to8_no_attn(vdset, trainset, testset,
                                            savedir, statdir, device, max_epoch, lr)
        except KeyboardInterrupt:
            logger.warning('User interrupt')
            print('Cleaning up ...', file=sys.stderr)
