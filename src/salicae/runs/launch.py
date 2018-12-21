#!/usr/bin/env python
import torch
import torchvision.transforms as trans

import train
import vmdata
import more_trans
import salicae


if __name__ == '__main__':
    root = vmdata.dataset_root(9, (8, 0, 0))
    normalize = trans.Normalize(*vmdata.get_normalization_stats(root))
    dset = vmdata.VideoDataset(root, max_mmap=4, max_gzcache=10,
                               transform=trans.Compose([
                                   more_trans.MedianBlur(),
                                   trans.ToTensor(),
                                   normalize,
                               ]))
    net = salicae.SaliencyCAE(vgg_arch='Ashallow', batch_norm=False)
    device = torch.device('cuda')
    batch_size = 16
    lasso_strength = 1.
    savedir = 'save'
    statdir = 'stats'
    train.train(net, dset, device, batch_size, lasso_strength,
                statdir, savedir)
