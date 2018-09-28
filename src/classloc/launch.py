#!/usr/bin/env python
import logging
import os
_cd = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(format='[%(levelname)s] [%(asctime)s] (%(name)s) %(message)s',
                    datefmt='%Y-%m-%d %I:%M:%S %p',
                    filename=os.path.join(_cd, '_launch.log'),
                    level=logging.INFO)

import operator as op

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as trans
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

import vmdata
import trainlib
import more_trans
from classloc import squeezenet


root = 'data/CH09-08_00_00'
transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize(*vmdata.get_normalization_stats(root)),
])
labels = 'runs/CH09-08_00_00/global-presence.txt'
savedir = 'runs/CH09-08_00_00/global-presence.0/save'
statdir = 'runs/CH09-08_00_00/global-presence.0/stat'
loaderparams = {'num_workers': 1, 'batch_size': 32}
vparams = {'max_mmap': 5, 'max_gzcache': 20}
max_epochs = 300
device = torch.device('cuda')

if __name__ == '__main__':
    net = squeezenet.SqueezeNet(2, (480, 704)).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    checkpointsaver = trainlib.CheckpointSaver(net, savedir)
    statsaver = trainlib.StatSaver(statdir)
    with vmdata.VideoDataset(root, transform=transform, **vparams) as vdset:
        lvdset = vmdata.LabelledVideoDataset(labels, vdset)
        def getloader(subsetind):
            return data.DataLoader(lvdset, sampler=data.SubsetRandomSampler(subsetind),
                                   **loaderparams)
        perm = np.random.permutation(np.arange(len(lvdset)))
        train_ind, test_ind = perm[:-len(perm)//10], perm[-len(perm)//10:]
        for epoch in range(max_epochs):
            logging.info('Beginning epoch {}/{}'.format(epoch+1, max_epochs))
            loaders = [('train', getloader(train_ind)), ('eval', getloader(test_ind))]
            stats = {'train_loss': [], 'eval_loss': [], 'train_acc': 0.0, 'eval_acc': 0.0}
            for mode, loader in loaders:
                logging.info('- Beginning {} mode'.format(mode))
                getattr(net, mode)()
                torch.set_grad_enabled(mode == 'train')
                running_correct = 0.0
                running_total = 0.0
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    stats[mode + '_loss'].append(loss.detach().item())
                    running_total += inputs.size(0)
                    running_correct += torch.sum(torch.argmax(outputs.detach(), dim=1) == labels.detach()).float()
                    if mode == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                stats[mode + '_acc'] = running_correct / running_total
            checkpointsaver(epoch)
            statsaver(epoch, **stats)
            logging.info('- Stats: train_loss_mean={:.4f} train_acc={:.4f} '
                         'test_loss_mean={:.4f} test_acc={:.4f}'
                         .format(np.mean(stats['train_loss']), stats['train_acc'],
                                 np.mean(stats['eval_loss']), stats['eval_acc']))
    torch.set_grad_enabled(True)
