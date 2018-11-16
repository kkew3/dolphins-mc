import time

import torch
import torch.nn as nn

from grplaincae.models.wide_pred9_f1to8 import STCAEEncoder, STCAEDecoder
from gradreg import gradreg


@profile
def backward():
    # firstly a no-op statement
    time.sleep(1)

    device = torch.device('cuda')
    encoder = STCAEEncoder().to(device)
    decoder = STCAEDecoder().to(device)
    criterion = nn.MSELoss().to(device)

    inputs = torch.rand(8, 1, 8, 160, 232).to(device)
    targets = torch.rand(8, 1, 1, 160, 232).to(device)

    with gradreg(inputs) as ns:
        codes = encoder(inputs)
        predictions = decoder(codes)
        ns.loss = criterion(predictions, targets)
    ns.loss.backward()

    # emulating reassignments
    loss = None
    inputs = None
    targets = None


if __name__ == '__main__':
    backward()
