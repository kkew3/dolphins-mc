import logging
import importlib
import re
import contextlib

import torchvision.transforms as trans
import torch
from torch.utils.data import DataLoader

import vmdata
import ezfirstae.models.pred9_f1to8 as pred9_f1to8
import ezfirstae.basicmodels as basicmodels
import ezfirstae.loaddata as ld


model_name_module_pat = re.compile(r'^import +ezfirstae\.models\.([a-zA-Z0-9_]+)')
root = vmdata.prepare_dataset_root(9, (8, 0, 0))
normalize = trans.Normalize(*vmdata.get_normalization_stats(root, bw=True))


def load_trained_model(model_module, checkpoint_file: str):
    if isinstance(model_module, str):
        model_module = importlib.import_module(model_module)
    encoder = model_module.STCAEEncoder()
    decoder = model_module.STCAEDecoder()
    attention = model_module.STCAEDecoder()
    ezcae = basicmodels.EzFirstCAE(encoder, decoder, attention)
    ezcae.load_state_dict(torch.load(checkpoint_file, map_location='cpu'))
    ezcae.eval()
    return ezcae


@contextlib.contextmanager
def load_dataset_c9(model_module):
    transform = ld.PreProcTransform(normalize, model_module.pool_scale,
                                    model_module.downsample_scale)
    vdset = vmdata.VideoDataset(root, transform=transform)
    trainset, testset = ld.contiguous_partition_dataset(range(len(vdset)), (5, 1))
    try:
        yield vdset, trainset, testset
    finally:
        vdset.release_mmap()


def toimg(tensor):
    assert len(tensor.size()) == 2
    img = tensor.detach().numpy()
    img = img * normalize.std[0] + normalize.mean[0]
    return img


def get_dataloader(model_module, vdset, indices):
    swsam = ld.SlidingWindowBatchSampler(indices, model_module.temporal_batch_size + 1, batch_size=1)
    dataloader = DataLoader(vdset, batch_sampler=swsam)
    for inputs in dataloader:
        inputs = ld.rearrange_temporal_batch(inputs, model_module.temporal_batch_size + 1)
        inputs, targets = inputs[:, :, :-1, :, :], inputs[:, :, -1:, :, :]
        yield inputs, targets
