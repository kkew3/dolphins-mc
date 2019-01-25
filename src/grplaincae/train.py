import logging
from typing import Iterable, Tuple, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
import vmdata
import more_sampler
import more_trans
from gradreg import gradreg, no_grad_params
from grplaincae.basicmodels import Autoencoder
from trainlib import BasicTrainer

def _l(*args):
    return logging.getLogger(utils.loggername(__name__, *args))


class TrainOnlyAdamTrainer(BasicTrainer):
    """
    No evaluation loop. Train for at most one epoch. Use Adam optimizer at
    initial learning rate ``TrainOnlyAdamTrainer.optim_lr`` (1e-4).
    """
    optim_lr = 1e-4

    def __init__(self, net_module, root: str, transform,
                 trainset_indices: Sequence[int], gr_strength=0.0,
                 max_epoch: int = 1, batch_size: int = 8,
                 device='cpu'):
        logger = _l(self, '__init__')
        self.net_module = net_module
        net = Autoencoder(self.net_module.STCAEEncoder(), self.net_module.STCAEDecoder())
        logger.debug('Loaded autoencoder net from module {}'
                     .format(self.net_module.__name__))

        super().__init__(net, max_epoch=max_epoch, device=device)
        self.vdset = vmdata.VideoDataset(root, transform=transform)
        self.criterion = nn.MSELoss().to(device)
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=type(self).optim_lr)
        self.trainset_indices = trainset_indices
        self.gr_strength = gr_strength
        self.batch_size = batch_size

        self.stat_names = ('loss',)
        self.run_stages = ('train',)
        self.train_batch_sampler_ = lambda: more_sampler.SlidingWindowBatchSampler(
                self.trainset_indices, 1 + self.net_module.temporal_batch_size,
                batch_size=batch_size, shuffled=(self.max_epoch > 1))

    # noinspection PyAttributeOutsideInit
    def before_epoch(self):
        self.train_batch_sampler = self.train_batch_sampler_()

    def get_trainloader(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        dataloader = DataLoader(self.vdset,
                                batch_sampler=self.train_batch_sampler)
        for frames in dataloader:
            frames = more_trans.rearrange_temporal_batch(frames, 1 + self.net_module.temporal_batch_size)
            inputs, targets = frames[:, :, :-1, :, :], frames[:, :, -1:, :, :]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            yield inputs, targets

    def train_once(self, inputs, targets):
        """
        Train for one minibatch.
        """
        logger = _l(self, 'train_once')
        loss_part = [0, 0]
        with gradreg(inputs, strength=self.gr_strength) as ns:
            outputs = self.net(inputs)
            ns.loss = self.criterion(outputs, targets)
            loss_part[0] = ns.loss.item()
        self.optimizer.zero_grad()
        ns.loss.backward()
        self.optimizer.step()
        loss_part[1] = ns.loss.item() - loss_part[0]
        logger.debug('loss part: {}'.format(tuple(loss_part)))
        return ns.loss.item(),

    def eval_once(self, inputs, targets):
        """
        Evaluate for one minibatch.
        """
        with no_grad_params(self.net):
            with gradreg(inputs, strength=self.gr_strength, train=False) as ns:
                outputs = self.net(inputs)
                ns.loss = self.criterion(outputs, targets)
        return ns.loss.item(),

    def teardown(self, error=None):
        super().teardown(error)
        self.vdset.release_mmap()
    #
    # def predict(self, indices: Sequence[int], todir: str,
    #             oob_policy: Callable[[torch.Tensor], torch.Tensor]
    #             = more_trans.clamp_tensor_to_image):
    #     """
    #     :param indices: indices used to make predictions from
    #     :param oob_policy: a function that converts each element of a tensor to
    #            within range [0, 1]
    #     :return:
    #     """
    #     logger = logging.getLogger(_l(__name__, self, 'predict'))
    #     sam = more_sampler.SlidingWindowBatchSampler(
    #             indices, self.net_module.temporal_batch_size,
    #             batch_size=self.batch_size, shuffled=False)
    #     os.makedirs(todir, exist_ok=True)
    #     loader = DataLoader(self.vdset, batch_sampler=sam)
    #     try:
    #         with torch.no_grad():
    #             for batch, inputs in enumerate(loader):
    #                 inputs = more_trans.rearrange_temporal_batch(
    #                         inputs, self.net_module.temporal_batch_size)
    #                 inputs = inputs.to(self.device)
    #                 outputs = self.net(inputs)
    #                 preds = self.net_module.outputs_as_images(outputs, oob_policy)
    #                 for j, img in enumerate(preds):
    #                     tofile = os.path.join(todir, 'p{}_{}.png'.format(batch, j))
    #                     plt.imsave(tofile, img.detach().cpu().numpy())
    #                     plt.close()
    #                     logger.debug('Written prediction to file "{}"'
    #                                  .format(tofile))
    #     finally:
    #         self.vdset.release_mmap()
    #
    # def predict_with_attn(self, indices: Sequence[int], todir: str,
    #                       oob_policy: Callable[[torch.Tensor], torch.Tensor]
    #                       = more_trans.clamp_tensor_to_image):
    #     logger = logging.getLogger(_l(__name__, self, 'predict_with_attn'))
    #     sam = more_sampler.SlidingWindowBatchSampler(
    #             indices, self.net_module.temporal_batch_size + 1,
    #             batch_size=self.batch_size, shuffled=False)
    #     os.makedirs(todir, exist_ok=True)
    #     loader = DataLoader(self.vdset, batch_sampler=sam)
    #     sigmoid = nn.Sigmoid()
    #     try:
    #         with no_grad_params(self.net):
    #             for batch, frames in enumerate(loader):
    #                 frames = more_trans.rearrange_temporal_batch(
    #                         frames, self.net_module.temporal_batch_size + 1)
    #                 inputs, targets = frames[:, :, :-1, :, :], frames[:, :, -1:, :, :]
    #                 inputs, targets = inputs.to(self.device), targets.to(self.device)
    #                 inputs.requires_grad_()
    #                 outputs = self.net(inputs)
    #                 loss = self.criterion(outputs, targets)
    #                 loss.backward()
    #                 attns = sigmoid(inputs.grad)
    #
    #                 outputs = self.net_module.outputs_as_images(outputs, oob_policy)
    #                 attns = self.net_module.inputs_as_images(attns, lambda x: x)
    #                 for j, (o, a) in enumerate(zip(outputs, attns)):
    #                     tofile_o = os.path.join(todir, 'p{}_{}.png'.format(batch, j))
    #                     plt.imsave(tofile_o, o.detach().cpu().numpy())
    #                     plt.close()
    #                     logger.debug('Written prediction to file "{}"'
    #                                  .format(tofile_o))
    #
    #                     tofile_a = os.path.join(todir, 'a{}_{}.png'.format(batch, j))
    #                     plt.imsave(tofile_a, a.detach().cpu().numpy())
    #                     plt.close()
    #                     logger.debug('Written predicted attention to file "{}"'
    #                                  .format(tofile_a))
    #     finally:
    #         self.vdset.release_mmap()
