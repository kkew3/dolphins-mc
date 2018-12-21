import logging.config

logging.config.fileConfig('logging.ini')

import torchvision.transforms as trans

from utils import loggername as _l
import exprlib
import vmdata
from grplaincae.train import TrainOnlyAdamTrainer
from more_trans import BWCAEPreprocess
import grplaincae.models.wide_pred9_f1to8 as net_module

root = vmdata.dataset_root(9, (8, 0, 0))
downsample_scale = 3
train_indices = range(10000)
max_epoch = 2
gr_strength = 1.0

if __name__ == '__main__':
    bw = (net_module.input_color == 'gray')
    runid, name = exprlib.get_runid_from_file(__file__, return_prefix=True)
    logger = logging.getLogger(_l(__name__, '{}.{}'.format(name, runid)))
    logger.info('Launched')
    logger.info('root = {}; downsample_scale = {}; bw = {}'
                .format(root, downsample_scale, bw))

    normalize_stats = vmdata.get_normalization_stats(root, bw=bw)
    transform = BWCAEPreprocess(trans.Normalize(*normalize_stats),
                                pool_scale=net_module.pool_scale,
                                downsample_scale=downsample_scale)

    trainer = TrainOnlyAdamTrainer(
            net_module, root, transform,
            train_indices, max_epoch=max_epoch,
            gr_strength=gr_strength, device='cuda')
    trainer.basedir = '{}.{}'.format(name, runid)
    trainer.run()
